#ifndef INFINI_OPS_KUNLUN_GEMM_KERNEL_H_
#define INFINI_OPS_KUNLUN_GEMM_KERNEL_H_

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "native/cuda/ops/gemm/blas.h"  // BlasGemm: backend-parameterized
#include "native/kunlun/blas.h"
#include "native/kunlun/blas_utils.h"
#include "native/kunlun/elementwise_common.h"    // kunlun_launch::LaunchCast
#include "native/kunlun/launch2_declarations.h"  // LaunchFill
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun GEMM via xBLAS.
//
// xBLAS on the P800 supports f16 and f32 GEMMs natively, and mixed
// AB=bf16 + C=f32 GEMMs, but NOT all-bf16 GEMMs (status 15,
// CUBLAS_STATUS_NOT_SUPPORTED) — same limitation the legacy InfiniCore
// stack worked around. For bf16 outputs we therefore run the BLAS call
// with an f32 temporary output and cast f32 -> bf16 back into y.
template <>
class Operator<Gemm, Device::Type::kKunlun>
    : public BlasGemm<Blas<Device::Type::kKunlun>> {
 public:
  using BlasGemm<Blas<Device::Type::kKunlun>>::BlasGemm;

  bool capture_safe() const override { return true; }

  void operator()(const Tensor a, const Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    // The base class folds beta into EffectiveBeta() which drops it ("C
    // input not supported yet" upstream); the bias arrives here as the `c`
    // argument, so work from the function arguments, not the base members.
    const float beta_arg = beta.value_or(0.0f);
    (void)trans_a;
    (void)trans_b;
    if (!y.IsContiguous()) {
      throw std::runtime_error("kunlun gemm: output must be contiguous");
    }
    if (a.strides().size() != 2 || b.strides().size() != 2) {
      // Batched calls keep 3-d shapes; fold leading batch into strides.
      if (a.strides().size() != 3 || b.strides().size() != 3 ||
          y.strides().size() != 3) {
        throw std::runtime_error("kunlun gemm: unsupported rank (2D/3D only)");
      }
    }
    if (a_type_ != b_type_ || y_type_ != a_type_) {
      throw std::runtime_error(
          "kunlun gemm: mixed dtypes unsupported (except bf16->f32 C)");
    }

    using Backend = Runtime<Device::Type::kKunlun>;
    if (y_type_ != DataType::kBFloat16) {
      // f16/f32: native BLAS path straight into y, then y += beta * c
      // (only beta == 1 with bias c; other betas throw — the fused op is
      //  not worth a second kernel for the 0/1 cases the suite exercises).
      RunBlasCore(a, b, y.data(), /*c_is_f32=*/false);
      if (beta_ != 0.0f && c.has_value()) {
        if (beta_ != 1.0f) {
          throw std::runtime_error(
              "kunlun gemm: f16/f32 supports beta only as 0/1 with bias c");
        }
        const unsigned long long elems = (unsigned long long)m_ *
                                         (unsigned long long)n_ *
                                         (unsigned long long)batch_count_;
        RunFlatAdd(c->data(), y.data(), elems, y_type_);
      }
      return;
    }

    // bf16: GemmEx writes bf16 C straight into y when beta == 0 (no bias
    // pre-processing needed); the f32 workspace round trip (fill + GEMM +
    // cast + malloc/free) is kept only for beta != 0, which needs the
    // up-cast of the bias.
    const unsigned long long out_elems = (unsigned long long)m_ *
                                         (unsigned long long)n_ *
                                         (unsigned long long)batch_count_;
    // The direct bf16 path trips the vendor launcher for very wide outputs
    // (its cluster count scales with n and is capped at 256; observed
    // failing at m=256, n=37888). Gate it to the shape range verified on
    // the P800 and keep the f32-workspace fallback for the rest.
    const long long direct_n = (long long)y_size_n();
    const long long direct_m = (long long)y_size_m();
    // The vendor bf16-C launcher also deadlocks the submission pipeline for
    // the wide-K shapes the 7B model uses (observed hanging at m=7, n=3584,
    // k=18944 inside xblas algo9 after many queued launches), while the
    // f32-C launcher is unaffected. Route those through the workspace path.
    // Wide-K bf16-C direct write ran the whole 7B decode (m=1) end to end
    // in the original PR at 9.4 tok/s — the hang was only ever seen at
    // prefill m>1 (m=7). Keep m>1 wide-K on the slabbed f32 path.
    const bool vendor_shape_ok =
        !(direct_m > 1 && direct_n > 18944) && !(k_ > 8192 && direct_m > 1);
    const bool vendor_direct = vendor_shape_ok;
    // bf16-C GemmEx direct write is verified only on narrow-N shapes; the
    // wide-N gate_up (n=18944) turns it into a multi-second stall, so
    // anything wider than o_proj goes through the f32 workspace below.
    // Decode (m=1) is fast on bf16-C even at wide N (verified end-to-end);
    // the stall only appears at prefill m>1.
    // Decode (m=1) verified direct on the qkv shape (n=4608) end to end;
    // the f32-workspace detour costs 3 extra launches per call.
    const bool bf16c_ok = direct_m == 1 || direct_n <= 4608;
    if (y_type_ == DataType::kBFloat16 &&
        (beta_arg == 0.0f || !c.has_value()) && vendor_direct && bf16c_ok) {
      RunBlasCore(a, b, y.data(), /*c_is_f32=*/false);
      return;
    }
    const bool wide_k = y_type_ == DataType::kBFloat16 && beta_arg == 0.0f &&
                        k_ > 8192 && direct_n <= 18944;
    if (y_type_ == DataType::kBFloat16 && beta_arg == 0.0f && direct_m >= 1 &&
        !wide_k && false && direct_n > 18944) {
      // Chunked wide-N path. Both vendor launchers fail on wide shapes
      // with the full ldc (bf16-C GemmEx half-writes at n = 37888; the
      // f32-C SB rejects mm > 1 with rc = 15). Compute dense chunks
      // [mm, 4608] with ldc = 4608 — the exact shape verified correct —
      // and scatter each chunk into the strided y columns. The A tile is
      // materialized into a contiguous scratch because the vendor also
      // ignores A base-pointer offsets between calls.
      using BL = Blas<Device::Type::kKunlun>;
      constexpr int kChunkN = 3584;
      typename BL::BlasHandle hnd{};
      if (BL::BlasCreate(&hnd) != 0 || hnd == nullptr) {
        throw std::runtime_error("kunlun gemm: fresh handle create failed");
      }
      if (BL::BlasSetStream(hnd, static_cast<typename BL::Stream>(stream_)) !=
          0) {
        BL::BlasDestroy(hnd);
        throw std::runtime_error("kunlun gemm: setting stream failed");
      }
      auto dt_a_ = BlasUtils<Device::Type::kKunlun>::GetDataType(b_type_);
      auto dt_b_ = BlasUtils<Device::Type::kKunlun>::GetDataType(a_type_);
      auto dt_c_ = BlasUtils<Device::Type::kKunlun>::GetDataType(y_type_);
      auto compute_ =
          BlasUtils<Device::Type::kKunlun>::GetComputeType(DataType::kFloat32);
      const bool b_rm = b.stride(-1) == 1;
      auto op_a_ = b_rm ? BL::BLAS_OP_N : BL::BLAS_OP_T;
      const int lda_ = b_rm ? (int)b.stride(-2) : (int)b.stride(-1);
      auto op_b_ = a.stride(-1) == 1 ? BL::BLAS_OP_N : BL::BLAS_OP_T;
      const int ldb_ = (int)a.stride(-2);
      const char* b_base_ = static_cast<const char*>(b.data());
      float alpha1 = 1.0f, beta0 = 0.0f;
      static thread_local char* a_tile = nullptr;
      static thread_local std::size_t a_tile_cap = 0;
      const std::size_t a_tile_bytes = (std::size_t)k_ * kChunkN * 2;
      if (a_tile_cap < a_tile_bytes) {
        if (a_tile) Runtime<Device::Type::kKunlun>::Free(a_tile);
        a_tile = nullptr;
        a_tile_cap = 0;
        if (Runtime<Device::Type::kKunlun>::Malloc(
                reinterpret_cast<void**>(&a_tile), a_tile_bytes) !=
            Runtime<Device::Type::kKunlun>::kSuccess) {
          throw std::runtime_error("kunlun gemm: a-tile alloc failed");
        }
        a_tile_cap = a_tile_bytes;
      }
      static thread_local char* c_chunk = nullptr;
      static thread_local std::size_t c_chunk_cap = 0;
      const std::size_t c_chunk_bytes = (std::size_t)direct_m * kChunkN * 2;
      if (c_chunk_cap < c_chunk_bytes) {
        if (c_chunk) Runtime<Device::Type::kKunlun>::Free(c_chunk);
        c_chunk = nullptr;
        c_chunk_cap = 0;
        if (Runtime<Device::Type::kKunlun>::Malloc(
                reinterpret_cast<void**>(&c_chunk), c_chunk_bytes) !=
            Runtime<Device::Type::kKunlun>::kSuccess) {
          throw std::runtime_error("kunlun gemm: c-chunk alloc failed");
        }
        c_chunk_cap = c_chunk_bytes;
      }
      for (long long n0 = 0; n0 < direct_n; n0 += kChunkN) {
        const int tn = (int)std::min<long long>(kChunkN, direct_n - n0);
        // zero the C chunk: the vendor beta=0 path still reads C, and the
        // scratch starts as unallocated garbage (NaN) which poisons the
        // output.
        {
          long long z_shape[2] = {(long long)direct_m, tn};
          long long z_strides[2] = {tn, 1};
          kunlun_launch::LaunchFill(2, 0, c_chunk, 2, z_shape, z_strides,
                                    (unsigned long long)direct_m * tn, 0.0f,
                                    stream_ ? stream_ : nullptr);
        }
        // Materialize the B tile as [k_, tn] row-major: tile(t, j) at
        // t*tn + j, and tile(t, j) = W[n0+j, t] (B^T column j, element t).
        // The chunk base is `n0 * lda_` elements into b's storage, where
        // lda_ is b's physical row stride (stride(-2) for row-major [N, K]
        // storage, stride(-1) for a transposed [K, N] view — both equal the
        // storage row length). Relative to that base the source element
        // (t, j) sits at t*1 + j*lda_ in EVERY layout the dispatcher admits
        // (row-major b: t*stride(-2) + (n0+j)*1 = t*lda_ + j after the base
        // shift of n0*1... see below; transposed b: t*stride(-2)=1 +
        // (n0+j)*stride(-1) - n0*lda_ = t + j*lda_). The former needs the
        // n0 shift inside the j stride, so spell both out:
        //   row-major b [N, K] (strides (lda_, 1)): element (t, j) at
        //     t*lda_ + n0 + j; base n0*lda_ -> relative t*lda_ + j...
        //   which is NOT {1, lda_}. The generic rule relative to the base
        // is: stride(t) = b.stride(-2) if op_a_==T else 1, stride(j) =
        // b.stride(-1) if op_a_==T else lda_. For transposed b
        // (strides (1, lda_)) that is {1, lda_}; for row-major b
        // (strides (lda_, 1)) the base already folds the n0*lda_ shift and
        // the relative layout is {lda_, 1}. Keep the two branches.
        long long t_shape[2] = {(long long)k_, tn};
        long long t_in_n[2] = {1, lda_};
        long long t_in_t[2] = {(long long)b.stride(-2),
                               (long long)b.stride(-1)};
        const long long* t_in = (op_a_ == BL::BLAS_OP_N) ? t_in_n : t_in_t;
        long long t_out[2] = {tn, 1};
        kunlun_launch::LaunchCopy(
            2, a_tile, b_base_ + (size_t)n0 * (size_t)lda_ * 2, 2, t_shape,
            t_in, t_out, (unsigned long long)k_ * tn,
            stream_ ? stream_ : nullptr);

        // Vendor per-chunk call: each chunk is [mm, tn] <= [mm, 3584] with
        // k_ <= 8192 here (wide-K is routed to the f32 slab path earlier),
        // all shapes the vendor computes correctly — no fallback tile
        // kernel. a_tile is [k_, tn] row-major with tn as leading dim.
        // a_tile is materialized [k_, tn] row-major = col-major [tn, k_],
        // so the vendor's A op is always N regardless of b's layout — the
        // transpose was folded into the tile copy.
        auto stc = BL::BlasGemmEx(
            hnd, BL::BLAS_OP_N, BL::BLAS_OP_N, tn, (int)direct_m, k_, &alpha1,
            const_cast<char*>(a_tile), dt_a_, tn,
            const_cast<char*>(static_cast<const char*>(a.data())), dt_b_,
            (int)a.stride(-2), &beta0, c_chunk, dt_c_, tn, compute_,
            BL::BLAS_GEMM_DEFAULT);
        if (stc != 0) {
          BL::BlasDestroy(hnd);
          throw std::runtime_error("kunlun gemm: chunk GemmEx rc=" +
                                   std::to_string((int)stc));
        }
        long long c_shape[2] = {direct_m, tn};
        long long c_in[2] = {tn, 1};
        long long c_out[2] = {direct_n, 1};
        kunlun_launch::LaunchCopy(2, (char*)y.data() + (size_t)n0 * 2, c_chunk,
                                  2, c_shape, c_in, c_out,
                                  (unsigned long long)direct_m * tn,
                                  stream_ ? stream_ : nullptr);
      }
      BL::BlasDestroy(hnd);
      return;
    }
    // Reuse the workspace across calls: xpu_malloc/free is a synchronous
    // driver round trip that serializes the submission pipeline when it
    // happens inside a long async kernel stream.
    static thread_local void* ws_cache = nullptr;
    static thread_local std::size_t ws_cap = 0;
    void* y_f32 = nullptr;
    if (ws_cap < out_elems * sizeof(float)) {
      if (ws_cache) Runtime<Device::Type::kKunlun>::Free(ws_cache);
      ws_cache = nullptr;
      ws_cap = 0;
      if (Backend::Malloc(&ws_cache, out_elems * sizeof(float)) !=
          Backend::kSuccess) {
        throw std::runtime_error("kunlun gemm: f32 workspace alloc failed");
      }
      ws_cap = out_elems * sizeof(float);
    }
    y_f32 = ws_cache;
    if (beta_arg != 0.0f && c.has_value()) {
      // C = beta * c: up-cast c to the f32 workspace, scaling if beta != 1.
      if (!c->IsContiguous() || (unsigned long long)c->numel() != out_elems) {
        throw std::runtime_error(
            "kunlun gemm: bf16 bias must be contiguous and match y");
      }
      kunlun_launch::LaunchCast(/*bf16 in*/ 2, /*f32 out*/ 0, y_f32, c->data(),
                                out_elems, stream_ ? stream_ : nullptr);
      if (beta_arg != 1.0f) {
        kunlun_launch::LaunchScaleF32(y_f32, out_elems, beta_arg,
                                      stream_ ? stream_ : nullptr);
      }
    } else {
      long long shape[2] = {(long long)out_elems, 1};
      long long strides[2] = {1, 1};
      kunlun_launch::LaunchFill(4, 0, y_f32, 1, shape, strides, out_elems, 0.0f,
                                stream_ ? stream_ : nullptr);
    }
    {
      RunBlasCore(a, b, y_f32, /*c_is_f32=*/true);
      kunlun_launch::LaunchCast(/*f32 in*/ 0, /*bf16 out*/ 2, y.data(), y_f32,
                                out_elems, stream_ ? stream_ : nullptr);
    }
  }

 private:
  // Flat y += c over  (contiguous, same shape). Uploads fresh 1-d
  // metadata each call — negligible next to the GEMM itself.
  void RunFlatAdd(const void* c, void* y, unsigned long long elems,
                  DataType dtype) const {
    using Backend = Runtime<Device::Type::kKunlun>;
    const int tag = detail::KunlunDtypeTag(dtype);
    const std::size_t shape_bytes = sizeof(Tensor::Size);
    const std::size_t stride_bytes = sizeof(Tensor::Stride);
    const std::size_t meta_bytes = 3 * (shape_bytes + stride_bytes);
    const Tensor::Size h_shape = static_cast<Tensor::Size>(elems);
    const Tensor::Stride h_stride = 1;
    std::vector<std::byte> host(meta_bytes);
    std::byte* d_meta = nullptr;
    if (Backend::Malloc(reinterpret_cast<void**>(&d_meta), meta_bytes) !=
        Backend::kSuccess) {
      throw std::runtime_error("kunlun gemm: flat-add meta alloc failed");
    }
    struct FreeGuard {
      void* p;
      ~FreeGuard() {
        if (p != nullptr) Runtime<Device::Type::kKunlun>::Free(p);
      }
    } free_guard{d_meta};
    std::size_t offset = 0;
    for (int i = 0; i < 3; ++i) {
      std::memcpy(host.data() + offset, &h_shape, shape_bytes);
      offset += shape_bytes;
      std::memcpy(host.data() + offset, &h_stride, stride_bytes);
      offset += stride_bytes;
    }
    if (Backend::Memcpy(d_meta, host.data(), meta_bytes,
                        Backend::kMemcpyHostToDevice) != Backend::kSuccess) {
      throw std::runtime_error("kunlun gemm: flat-add meta copy failed");
    }
    kunlun_launch::LaunchAdd(
        tag, y, y, c, reinterpret_cast<const unsigned long long*>(d_meta),
        reinterpret_cast<const unsigned long long*>(d_meta + shape_bytes +
                                                    stride_bytes),
        reinterpret_cast<const unsigned long long*>(
            d_meta + 2 * (shape_bytes + stride_bytes)),
        reinterpret_cast<const long long*>(d_meta + shape_bytes),
        reinterpret_cast<const long long*>(d_meta + shape_bytes * 2 +
                                           stride_bytes),
        reinterpret_cast<const long long*>(d_meta + shape_bytes * 3 +
                                           stride_bytes * 2),
        elems, 1, true, true, true, 1.0, stream_ ? stream_ : nullptr);
  }

  void RunBlasBf16(const Tensor& a, const Tensor& b, void* y_f32) const {
    RunBlasCore(a, b, y_f32, /*c_is_f32=*/true);
  }

  // BLAS call for the InfiniOps Gemm contract y[M,N] = a[M,K] @ b[K,N]
  // (alpha/beta handled by the caller). Column-major view: C[N,M] with
  // ldc = y.stride(-1). With B = a's memory (row-major [M,K] = col-major
  // [K,M], ldb = a.stride(-2)) the product C = op_a(A) @ op_b(B) needs
  // op_a(A) = [N,K] from b's memory:
  //   b row-major [K,N] (stride(-1)==1): memory IS col-major [N,K] ->
  //     op_a = N, lda = b.stride(-2).
  //   b permute view [K,N] with stride(-2)==1 (InfiniLM linear weights,
  //     memory = row-major weight [N,K] = col-major [K,N]): op_a = T turns
  //     it into [N,K], lda = b.stride(-1) = K.
  // B side: a row-major (stride(-1)==1) gives col-major [K,M] with op_b=N;
  // a col-major (stride(-2)==1) gives op_b=T, ldb = a.stride(-1).
  // When c_is_f32, C dtype is R_32F (bf16 A/B path); otherwise it matches
  // the compute dtype (f16/f32 native paths).
  void RunBlasCore(const Tensor& a, const Tensor& b, void* y_out,
                   bool c_is_f32) const {
    using BL = Blas<Device::Type::kKunlun>;
    const auto handle = this->GetHandle();
    if (BL::BlasSetStream(handle, static_cast<typename BL::Stream>(stream_)) !=
        0) {
      throw std::runtime_error("kunlun gemm: setting stream failed");
    }

    // BLAS A carries the [n, k] matrix of b (y = a @ b^T semantics).
    // xBLAS rejects op=N on an [m_arg, k] A whose leading dim is smaller
    // than m_arg (the permute-view case, probed standalone), so A is always
    // expressed as the col-major [k, n] of b's memory with op=T:
    //   b row-major [n, k] (contiguous): ld = stride(-2) = k.
    //   b permute view [k, n] (stride(-2)==1, InfiniLM linear): the memory
    //     as col-major [k, n] has ld = stride(-1) = k. op=T then computes
    //     A^T = the [n, k] matrix read at stride(-1) — verified equal to
    //     a @ b on the P800.
    const bool b_row_major = b.stride(-1) == 1;
    auto op_a = b_row_major ? BL::BLAS_OP_N : BL::BLAS_OP_T;
    const int lda_b = b_row_major ? (int)b.stride(-2) : (int)b.stride(-1);
    // a.stride(-1)==1 means row-major [M,K] = col-major [K,M]: op N.
    // Otherwise a is col-major [M,K] already: op T.
    auto op_b = a.stride(-1) == 1 ? BL::BLAS_OP_N : BL::BLAS_OP_T;

    auto dt_a = BlasUtils<Device::Type::kKunlun>::GetDataType(b_type_);
    auto dt_b = BlasUtils<Device::Type::kKunlun>::GetDataType(a_type_);
    auto dt_c = c_is_f32
                    ? BL::R_32F
                    : BlasUtils<Device::Type::kKunlun>::GetDataType(y_type_);
    auto compute =
        BlasUtils<Device::Type::kKunlun>::GetComputeType(DataType::kFloat32);

    const float alpha = alpha_;
    // c_is_f32: the caller pre-fills the f32 workspace with beta_*c, so the
    // GEMM must ACCUMULATE into it (beta=1). Otherwise (direct bf16/f16/f32
    // write into y) the bias, when present, is added by the caller.
    const float beta = c_is_f32 ? 1.0f : 0.0f;
    const int mm = (int)y_size_m();  // y.size(-2)
    const int nn = (int)y_size_n();  // y.size(-1)
    const int kk = (int)k_;
    const int lda = lda_b;
    const int ldb = (int)a.stride(-2);
    const int ldc = (int)ldy_;
    const long long sa = batch_stride_b_;
    const long long sb = batch_stride_a_;
    const long long sc = batch_stride_y_;

    if (y_type_ == DataType::kBFloat16) {
      // Strided-batched BLAS rejects bf16 entirely; loop GemmEx.
      const auto* b_base = static_cast<const char*>(b.data());
      const auto* a_base = static_cast<const char*>(a.data());
      for (int i = 0; i < (int)batch_count_; ++i) {
        auto status = BL::BlasGemmEx(
            handle, op_a, op_b, nn, mm, kk, &alpha,
            const_cast<char*>(const_cast<const char*>(b_base) +
                              (size_t)sa * 2 * (size_t)i),
            dt_a, lda,
            const_cast<char*>(const_cast<const char*>(a_base) +
                              (size_t)sb * 2 * (size_t)i),
            dt_b, ldb, &beta, (char*)y_out + (size_t)sc * 4 * (size_t)i, dt_c,
            ldc, compute, BL::BLAS_GEMM_DEFAULT);
        if (status != 0) {
          throw std::runtime_error("kunlun bf16 GEMM (GemmEx, batch " +
                                   std::to_string(i) +
                                   ") failed: " + std::to_string((int)status));
        }
      }
      return;
    }
    if (c_is_f32 && kk > 8192) {
      // Vendor GemmEx is wrong at wide K on the P800 even with an f32 C
      // (probe_f32c: 6783/7168 wrong at k=18944), so split K into slabs
      // and accumulate: beta 0 on the first slab, 1 afterwards. Every
      // slab is a verified-shaped vendor call; no fallback kernel.
      constexpr int kKSlab = 8192;
      const char* b_base = static_cast<const char*>(b.data());
      const char* a_base = static_cast<const char*>(a.data());
      // K-axis element stride derived from the actual operand layout:
      // row-major [N,K] and permute [K,N] (stride(-2)==1) both keep K
      // contiguous (element stride 1), but a row-major [K,N] operand
      // strides K by stride(-2). Match the op_a/op_b choice below.
      const size_t b_k_step =
          (op_a == BL::BLAS_OP_N) ? (size_t)b.stride(-2) : 1;
      const size_t a_k_step = (op_b == BL::BLAS_OP_N) ? 1 : (size_t)ldb;
      float beta_acc = 0.0f;
      const float alpha_one = 1.0f;
      for (int k0 = 0; k0 < kk; k0 += kKSlab) {
        const int kc = (int)std::min<long long>(kKSlab, (long long)kk - k0);
        for (int i = 0; i < (int)batch_count_; ++i) {
          auto st = BL::BlasGemmEx(
              handle, op_a, op_b, nn, mm, kc, &alpha_one,
              const_cast<char*>(b_base +
                                ((size_t)k0 * b_k_step + (size_t)sa * i) * 2),
              dt_a, lda,
              const_cast<char*>(a_base +
                                ((size_t)k0 * a_k_step + (size_t)sb * i) * 2),
              dt_b, ldb, &beta_acc, (char*)y_out + (size_t)sc * 4 * i, dt_c,
              ldc, compute, BL::BLAS_GEMM_DEFAULT);
          if (st != 0) {
            throw std::runtime_error("kunlun gemm: K-slab GemmEx rc=" +
                                     std::to_string((int)st));
          }
        }
        beta_acc = 1.0f;
      }
      return;
    }
    auto status = BL::BlasGemmStridedBatchedEx(
        handle, op_a, op_b, nn, mm, kk, &alpha, b.data(), dt_a, lda, sa,
        a.data(), dt_b, ldb, sb, &beta, y_out, dt_c, ldc, sc, (int)batch_count_,
        compute, BL::BLAS_GEMM_DEFAULT);
    if (status != 0) {
      throw std::runtime_error("kunlun GEMM failed: " +
                               std::to_string((int)status));
    }
  }

  std::size_t y_size_m() const { return m_; }
  std::size_t y_size_n() const { return n_; }
};

}  // namespace infini::ops

#endif
