#ifndef INFINI_OPS_KUNLUN_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
#define INFINI_OPS_KUNLUN_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/flash_attn_varlen_func.h"
#include "native/kunlun/elementwise_common.h"  // KunlunDtypeTag
#include "native/kunlun/launch2_declarations.h"

// Vendor `libxpu_flash_attention.so` (torch_xmlir package) implements varlen
// prefill attention (`baidu::xpu::xfa::flash_attention_context_vllm`): q is
// the packed query tokens (`[total_tokens, heads, dim]`, BLHD), per-sequence
// lengths travel as prefix sums (`cu_seqlens` style) in `VectorParam` lod
// vectors — `cpu` points at a host copy (launch planning) and `xpu` at the
// on-card copy (in-kernel reads), exactly how vLLM-XPU passes both copies.
// The bf16/fp16 instantiations ship in the library; the declaration below
// mirrors `xhpc/xfa/include/flash_api.h`.
#include <xpu/refactor/context/newcontext.h>
#include <xpu/refactor/core/dtype.h>
#include <xpu/xdnn_types.h>

namespace baidu {
namespace xpu {
namespace xfa {

template <typename T_Q, typename T_K, typename T_V, typename T_BIAS,
          typename T_O, typename T_GEMM0, typename T_GEMM1, typename T_EW,
          typename TID, int64_t quant_mode>
int flash_attention_context_vllm(
    baidu::xpu::api::Context* ctx, const T_Q* q, const T_K* k, const T_V* v,
    const T_BIAS* mask_bias, T_O* o, std::int64_t batch_num,
    std::int64_t max_seq_q, std::int64_t max_seq_kv, std::int64_t head_num,
    std::int64_t head_dim, std::int64_t kv_head_num, const float* alibi_slopes,
    const baidu::xpu::api::VectorParam<TID>& qlod,
    const baidu::xpu::api::VectorParam<TID>& kvlod,
    const std::vector<std::int64_t>& mask_bias_shape,
    const std::vector<std::int64_t>& alibi_slopes_shape,
    const bool is_causal_mask, const bool is_qkv_fusion,
    std::int64_t qkv_layout, std::int64_t vsl_type, const float* q_maxptr,
    const float* k_maxptr, const float* v_maxptr, float* o_maxptr,
    std::int64_t vo_head_dim, const T_K* k_cache, const T_V* v_cache,
    const float* k_cache_maxptr, const float* v_cache_maxptr,
    std::int64_t block_size, std::int64_t max_blocks_per_seq,
    std::int64_t prefill_len, const TID* block_tables, float alpha,
    std::int64_t swa_left_size, std::int64_t swa_right_size, T_EW* softmax_lse,
    bool unpadded_lse);

}  // namespace xfa
}  // namespace xpu
}  // namespace baidu

namespace infini::ops {

// The vendor ships `flash_attention_context_vllm` instantiations pinned to
// specific accumulate types: `float` GEMMs for bfloat16, `float16` GEMMs for
// float16 (both with a float bias/output-embed width). The trait maps our
// payload type onto the pairing the library actually exports.
template <typename T>
struct CtxVllmGemm;
template <>
struct CtxVllmGemm<bfloat16> {
  using type = float;
};
template <>
struct CtxVllmGemm<float16> {
  using type = float16;
};

template <typename T>
struct CtxVllmBias;
template <>
struct CtxVllmBias<bfloat16> {
  using type = float;
};
template <>
struct CtxVllmBias<float16> {
  using type = float16;
};

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kKunlun>
    : public FlashAttnVarlenFunc {
 public:
  using FlashAttnVarlenFunc::FlashAttnVarlenFunc;

  static bool supports(
      const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
      const int64_t max_seqlen_k, const double dropout_p,
      const std::optional<double> softmax_scale, const bool causal,
      const std::vector<int64_t> window_size, const double softcap,
      const bool deterministic, const bool return_attn_probs, Tensor out,
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) {
    // Varlen prefill over the paged KV cache only (block table + 4D
    // caches, the same contract as vLLM prefill). Dense varlen keeps the
    // caller's unfused path, which carries fp32-accumulation tuning the
    // vendor kernel does not provide.
    if (!block_table) {
      return false;
    }
    // Mirror the option guards enforced in `operator()` so unsupported
    // requests fail at plan time instead of call time — the same
    // supports-precision the nvidia provider applies.
    if (alibi_slopes || dropout_p != 0.0 || softcap != 0.0 || deterministic ||
        return_attn_probs || softmax_lse || s_dmask ||
        window_size != std::vector<int64_t>{-1, -1}) {
      return false;
    }
    if (!FlashAttnVarlenFunc::supports(
            q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
            window_size, softcap, deterministic, return_attn_probs, out,
            softmax_lse, s_dmask)) {
      return false;
    }
    // The vendor kernel writes a dense packed output; a strided output view
    // would be silently misplaced (the base only pins the last stride).
    const bool out_dense = out.stride(0) == out.size(1) * out.size(2) &&
                           out.stride(1) == static_cast<int64_t>(out.size(2));
    return out_dense && (q.dtype() == DataType::kBFloat16 ||
                         q.dtype() == DataType::kFloat16);
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<Tensor> block_table,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  const double dropout_p,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool deterministic, const bool return_attn_probs,
                  Tensor out, std::optional<Tensor> softmax_lse,
                  std::optional<Tensor> s_dmask) const override {
    if (alibi_slopes || dropout_p != 0.0 || softcap != 0.0 || deterministic ||
        return_attn_probs || softmax_lse || s_dmask ||
        window_size != std::vector<int64_t>{-1, -1}) {
      throw std::invalid_argument(
          "kunlun FlashAttnVarlenFunc received an unsupported option");
    }

    const int64_t batch = cu_seqlens_q.size(0) - 1;
    const int64_t head_num = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t kv_head_num = k.size(-2);
    const float scale = static_cast<float>(
        softmax_scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim))));

    baidu::xpu::api::Context* ctx = VendorContext();

    int rc;
    if (q.dtype() == DataType::kBFloat16) {
      const float alpha = scale * std::sqrt(static_cast<double>(head_dim));
      rc = Run<bfloat16>(stream_, q, k, v, cu_seqlens_q, cu_seqlens_k,
                         block_table, out, batch, max_seqlen_q, max_seqlen_k,
                         head_num, kv_head_num, head_dim, alpha, causal);
    } else {
      const float alpha = scale * std::sqrt(static_cast<double>(head_dim));
      rc = Run<float16>(stream_, q, k, v, cu_seqlens_q, cu_seqlens_k,
                        block_table, out, batch, max_seqlen_q, max_seqlen_k,
                        head_num, kv_head_num, head_dim, alpha, causal);
    }
    if (rc != 0) {
      throw std::runtime_error(
          "kunlun flash_attention_context_vllm failed, rc=" +
          std::to_string(rc));
    }
  }

 private:
  template <typename T>
  static int Run(void* stream, const Tensor& q, const Tensor& k,
                 const Tensor& v, const Tensor& cu_seqlens_q,
                 const Tensor& cu_seqlens_k,
                 const std::optional<Tensor>& block_table, const Tensor& out,
                 int64_t batch, int64_t max_seq_q, int64_t max_seq_kv,
                 int64_t head_num, int64_t kv_head_num, int64_t head_dim,
                 float alpha, bool causal) {
    // The vendor `flash_attention_context_vllm` requires DENSE k/v with
    // vsl_type=NOVSL. Its paged path (ATTN_LVSL + cache pointers) reads only
    // KV-cache head 0 (GQA unsupported there) and deadlocks at multi-head
    // shapes — the vendor's own evaluation scripts always drive the dense
    // mode. Gather the paged cache into a dense [total_kv, kv_heads, dim]
    // scratch, then submit the dense call. Prefill KV lengths are bounded,
    // so the gather cost is small relative to attention itself.
    const int page_size = static_cast<int>(k.size(1));
    const int bt_stride = static_cast<int>(block_table->size(1));

    // total kv tokens = last kvlod entry; read via a tiny D2H (host launch
    // planning needs the size anyway).
    thread_local std::vector<int> kvlod_host;
    kvlod_host.assign(cu_seqlens_k.numel(), 0);
    xpu_wait(static_cast<XPUStream>(stream));
    if (xpu_memcpy(kvlod_host.data(), cu_seqlens_k.data(),
                   kvlod_host.size() * sizeof(int), XPU_DEVICE_TO_HOST) != 0) {
      return -1000;
    }
    const int64_t kv_tokens = kvlod_host.back();
    const int64_t kv_elems = kv_tokens * kv_head_num * head_dim;
    thread_local void* kv_scratch = nullptr;
    thread_local int64_t kv_scratch_cap = 0;
    if (kv_scratch_cap < kv_elems) {
      if (kv_scratch) Runtime<Device::Type::kKunlun>::Free(kv_scratch);
      kv_scratch = nullptr;
      kv_scratch_cap = 0;
      void* raw = nullptr;
      // K and V both live in this scratch: 2 caches x 2 bytes/elem (the
      // f32 path allocates separately — supports() admits only bf16/f16).
      if (xpu_malloc(&raw, kv_elems * 2 * 2) != 0) {
        return -1002;
      }
      kv_scratch = raw;
      kv_scratch_cap = kv_elems;
    }
    void* k_dense = kv_scratch;
    void* v_dense = static_cast<char*>(kv_scratch) +
                    kv_elems * (q.dtype() == DataType::kFloat32 ? 4 : 2);
    // f32 path needs separate sizing; the supports() contract admits only
    // bf16/f16 here, so 2 bytes per element is exact.
    const int tag = detail::KunlunDtypeTag(q.dtype());

    // Caller's stream throughout: the QKV GEMM producing k/v, the cache
    // write (reshape_and_cache, now bound to this stream) and the vendor
    // context attention (ctx re-bound below) all serialize here.
    kunlun_launch::LaunchGatherKv(tag, k_dense, k.data(), cu_seqlens_k.data(),
                                  block_table->data(), (int)kv_head_num,
                                  (int)head_dim, page_size, bt_stride,
                                  (int)batch, (int)kv_tokens, stream);
    kunlun_launch::LaunchGatherKv(tag, v_dense, v.data(), cu_seqlens_k.data(),
                                  block_table->data(), (int)kv_head_num,
                                  (int)head_dim, page_size, bt_stride,
                                  (int)batch, (int)kv_tokens, stream);
    xpu_wait(static_cast<XPUStream>(stream));

    // The vendor kernel reads a dense [tokens, heads, dim] query; the caller
    // may pass a split view of the fused-QKV output (token stride spans
    // Q+K+V), so compact it first — same treatment the decode wrapper applies.
    const T* qp = static_cast<const T*>(q.data());
    if (q.stride(0) != q.size(1) * q.size(2) ||
        q.stride(1) != static_cast<int64_t>(q.size(2)) || q.stride(2) != 1) {
      const int64_t tokens = q.size(0);
      const int64_t heads = q.size(1);
      const int64_t dim = q.size(2);
      const int64_t q_elems = tokens * heads * dim;
      thread_local void* q_scratch = nullptr;
      thread_local int64_t q_scratch_cap = 0;
      if (q_scratch_cap < q_elems) {
        if (q_scratch) Runtime<Device::Type::kKunlun>::Free(q_scratch);
        q_scratch = nullptr;
        q_scratch_cap = 0;
        void* raw = nullptr;
        if (xpu_malloc(&raw, q_elems * sizeof(T)) != 0) {
          return -1002;
        }
        q_scratch = raw;
        q_scratch_cap = q_elems;
      }
      long long shape[3] = {tokens, heads, dim};
      long long in_strides[3] = {q.stride(0), q.stride(1), q.stride(2)};
      long long out_strides[3] = {heads * dim, dim, 1};
      kunlun_launch::LaunchCopy(2, q_scratch, q.data(), 3, shape, in_strides,
                                out_strides, q_elems, stream);
      xpu_wait(static_cast<XPUStream>(stream));
      qp = static_cast<const T*>(q_scratch);
    }

    int kvh_vendor = (int)kv_head_num;
    if (head_num != kv_head_num) {
      const int64_t ext_elems = kv_tokens * head_num * head_dim;
      thread_local void* kv_ext = nullptr;
      thread_local int64_t kv_ext_cap = 0;
      if (kv_ext_cap < ext_elems * 2) {
        if (kv_ext) Runtime<Device::Type::kKunlun>::Free(kv_ext);
        kv_ext = nullptr;
        kv_ext_cap = 0;
        void* raw = nullptr;
        if (xpu_malloc(&raw, ext_elems * 2 * 2) != 0) {
          return -1002;
        }
        kv_ext = raw;
        kv_ext_cap = ext_elems * 2;
      }
      char* k_ext = static_cast<char*>(kv_ext);
      char* v_ext = k_ext + ext_elems * 2;
      kunlun_launch::LaunchGqaBroadcast(k_ext, k_dense, (int)kv_tokens,
                                        (int)kv_head_num, (int)head_num,
                                        (int)head_dim, stream);
      kunlun_launch::LaunchGqaBroadcast(v_ext, v_dense, (int)kv_tokens,
                                        (int)kv_head_num, (int)head_num,
                                        (int)head_dim, stream);
      xpu_wait(static_cast<XPUStream>(stream));
      k_dense = k_ext;
      v_dense = v_ext;
      kvh_vendor = (int)head_num;
    }

    baidu::xpu::api::Context* ctx = VendorContext();
    const int* qlod_dev = static_cast<const int*>(cu_seqlens_q.data());
    const int* kvlod_dev = static_cast<const int*>(cu_seqlens_k.data());
    thread_local std::vector<int> qlod_host;
    qlod_host.assign(cu_seqlens_q.numel(), 0);
    xpu_memcpy(qlod_host.data(), qlod_dev, qlod_host.size() * sizeof(int),
               XPU_DEVICE_TO_HOST);
    baidu::xpu::api::VectorParam<int> qlod{qlod_host.data(),
                                           (int)cu_seqlens_q.numel(),
                                           const_cast<int*>(qlod_dev)};
    baidu::xpu::api::VectorParam<int> kvlod{kvlod_host.data(),
                                            (int)cu_seqlens_k.numel(),
                                            const_cast<int*>(kvlod_dev)};
    ctx->set_stream(stream);

    const int rc = baidu::xpu::xfa::flash_attention_context_vllm<
        T, T, T, typename CtxVllmBias<T>::type, T,
        typename CtxVllmGemm<T>::type, typename CtxVllmGemm<T>::type, float,
        int, /*quant_mode=*/0>(
        ctx, qp, static_cast<const T*>(k_dense), static_cast<const T*>(v_dense),
        /*mask_bias=*/nullptr, static_cast<T*>(const_cast<void*>(out.data())),
        static_cast<std::int64_t>(batch), max_seq_q, max_seq_kv,
        static_cast<std::int64_t>(head_num),
        static_cast<std::int64_t>(head_dim),
        static_cast<std::int64_t>(kvh_vendor), /*alibi_slopes=*/nullptr, qlod,
        kvlod, /*mask_bias_shape=*/{}, /*alibi_slopes_shape=*/{}, causal,
        /*is_qkv_fusion=*/false,
        /*qkv_layout=*/0x10 /*ATTN_BLHD*/, /*vsl_type=*/0x00 /*ATTN_NOVSL*/,
        /*q_maxptr=*/nullptr, /*k_maxptr=*/nullptr, /*v_maxptr=*/nullptr,
        /*o_maxptr=*/nullptr, /*vo_head_dim=*/-1,
        /*k_cache=*/nullptr, /*v_cache=*/nullptr,
        /*k_cache_maxptr=*/nullptr, /*v_cache_maxptr=*/nullptr,
        /*block_size=*/-1, /*max_blocks_per_seq=*/-1, /*prefill_len=*/-1,
        /*block_tables=*/nullptr, alpha, /*swa_left_size=*/-1,
        /*swa_right_size=*/-1, /*softmax_lse=*/nullptr, /*unpadded_lse=*/false);

    return rc;
  }

  static baidu::xpu::api::Context* VendorContext() {
    thread_local baidu::xpu::api::Context* ctx = nullptr;
    if (ctx == nullptr) {
      ctx = baidu::xpu::api::create_context();
      // The context is reused across calls; its stream is bound per call
      // (`ctx->set_stream(stream)` before each vendor launch) so vendor
      // kernels order against the caller's InfiniRT submissions.
    }
    return ctx;
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_KUNLUN_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
