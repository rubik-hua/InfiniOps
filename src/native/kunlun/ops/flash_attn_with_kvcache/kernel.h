#ifndef INFINI_OPS_KUNLUN_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
#define INFINI_OPS_KUNLUN_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "base/flash_attn_with_kvcache.h"
#include "native/kunlun/launch2_declarations.h"

// Vendor `libxpu_flash_attention.so` (torch_xmlir package) implements paged
// decode attention over a block-tabled KV cache — the same cache contract
// InfiniOps uses (`[num_blocks, block_size, num_kv_heads, head_size]`, int32
// block table, per-sequence context lengths). The vendor header
// (`xhpc/xfa/include/flash_api.h`) declares the entry as
// `baidu::xpu::xfa::paged_attention_xft<T, TKVCACHE, TO, TL, TGEMM0, TGEMM1,
// TEW, TID, v_transed, quant_mode>`; `libxpu_flash_attention.so` ships the
// bf16/fp16 instantiations this wrapper binds. The context is the same
// `baidu::xpu::api::Context` family used by `libapiinfer.so`.
#include <xpu/refactor/context/newcontext.h>
#include <xpu/refactor/core/dtype.h>
#include <xpu/xdnn_types.h>

namespace baidu {
namespace xpu {
namespace xfa {

// The older non-xft paged entry (`flash_api.h` line ~298): context_len as a
// raw device pointer, no host-side VectorParam.
template <typename TQ, typename TKV, typename TO, typename TL, typename TGEMM0,
          typename TGEMM1, typename TEW, typename TID = int, int quant_mode = 0>
int paged_attention(
    api::Context* ctx, TO* out, const TQ* query, const TKV* key_cache,
    const TKV* value_cache, std::int64_t num_kv_heads, float scale,
    const TID* block_table, const TID* context_len, std::int64_t block_size,
    std::int64_t max_context_len, const float* alibi_slopes,
    std::int64_t batch_num, std::int64_t head_num, std::int64_t head_dim,
    std::int64_t max_num_blocks_per_seq, const float* query_maxptr = nullptr,
    const float* key_cache_maxptr = nullptr,
    const float* value_cache_maxptr = nullptr, float* out_maxptr = nullptr,
    const float* nonlinear_distances = nullptr, std::int64_t v_head_dim = -1);

// Dispatch to a shipped instantiation: bf16 caches use float softmax/gemm
// accumulators; fp16 caches are all-fp16.
template <typename T>
struct VendorPagedAttn;

template <>
struct VendorPagedAttn<bfloat16> {
  static int call(api::Context* ctx, bfloat16* out, const bfloat16* query,
                  const bfloat16* key_cache, const bfloat16* value_cache,
                  std::int64_t num_kv_heads, float scale,
                  const int* block_table, const int* context_len,
                  std::int64_t block_size, std::int64_t max_context_len,
                  std::int64_t batch, std::int64_t head_num,
                  std::int64_t head_dim, std::int64_t max_blocks) {
    return paged_attention<bfloat16, bfloat16, bfloat16, float, float, float,
                           float, int, 0>(
        ctx, out, query, key_cache, value_cache, num_kv_heads, scale,
        block_table, context_len, block_size, max_context_len, nullptr, batch,
        head_num, head_dim, max_blocks);
  }
};

template <>
struct VendorPagedAttn<float16> {
  static int call(api::Context* ctx, float16* out, const float16* query,
                  const float16* key_cache, const float16* value_cache,
                  std::int64_t num_kv_heads, float scale,
                  const int* block_table, const int* context_len,
                  std::int64_t block_size, std::int64_t max_context_len,
                  std::int64_t batch, std::int64_t head_num,
                  std::int64_t head_dim, std::int64_t max_blocks) {
    return paged_attention<float16, float16, float16, float16, float16, float16,
                           float16, int, 0>(
        ctx, out, query, key_cache, value_cache, num_kv_heads, scale,
        block_table, context_len, block_size, max_context_len, nullptr, batch,
        head_num, head_dim, max_blocks);
  }
};

template <typename T, typename TKVCACHE, typename TO, typename TL,
          typename TGEMM0, typename TGEMM1, typename TEW, typename TID,
          bool v_transed, int quant_mode>
int paged_attention_xft(
    baidu::xpu::api::Context* ctx, TO* out, const T* query, const T* key,
    const T* value, const TKVCACHE* key_cache, const TKVCACHE* value_cache,
    std::int64_t num_kv_heads, float scale, const TID* block_table,
    const baidu::xpu::api::VectorParam<TID>& context_len,
    const baidu::xpu::api::VectorParam<TID>& valid_batch,
    std::int64_t block_size, std::int64_t max_context_len,
    const float* alibi_slopes, std::int64_t batch_num, std::int64_t head_num,
    std::int64_t head_dim, std::int64_t max_num_blocks_per_seq, const TO* shift,
    const TO* smooth, const float* query_maxptr, const float* key_cache_maxptr,
    const float* value_cache_maxptr, const T* key_cache_scales,
    const T* key_cache_zeros, const T* value_cache_scales,
    const T* value_cache_zeros, float* out_maxptr, std::int64_t v_head_dim);

}  // namespace xfa
}  // namespace xpu
}  // namespace baidu

namespace infini::ops {

// The vendor ships `paged_attention_xft` instantiations pinned to specific
// accumulate types: `tfloat32` for bfloat16, `float16` (matching the data
// type) for float16. The trait maps our payload type onto the pairing the
// library actually exports.
// The vendor pins the accumulate types per data type: for bfloat16 the
// local accumulator TL is `float` with `tfloat32` GEMMs; for float16 all
// three are `float16` (TEW stays float in both).
template <typename T>
struct GemmAccum;
template <>
struct GemmAccum<bfloat16> {
  using type = tfloat32;
};
template <>
struct GemmAccum<float16> {
  using type = float16;
};

template <typename T>
struct LocalAccum;
template <>
struct LocalAccum<bfloat16> {
  using type = float;
};
template <>
struct LocalAccum<float16> {
  using type = float16;
};

template <>
class Operator<FlashAttnWithKvcache, Device::Type::kKunlun>
    : public FlashAttnWithKvcache {
 public:
  using FlashAttnWithKvcache::FlashAttnWithKvcache;

  static bool supports_impl(const Tensor q, Tensor k_cache, Tensor v_cache,
                            const std::optional<Tensor> k,
                            const std::optional<Tensor> v,
                            const std::optional<Tensor> rotary_cos,
                            const std::optional<Tensor> rotary_sin,
                            const std::optional<Tensor> cache_seqlens,
                            const std::optional<Tensor> cache_batch_idx,
                            const std::optional<Tensor> cache_leftpad,
                            const std::optional<Tensor> block_table,
                            const std::optional<Tensor> alibi_slopes,
                            const std::optional<double> softmax_scale,
                            const bool causal,
                            const std::vector<int64_t> window_size,
                            const double softcap, const bool rotary_interleaved,
                            const int64_t num_splits,
                            const bool return_softmax_lse, Tensor out,
                            std::optional<Tensor> softmax_lse) {
    // The vendor kernel takes none of the optional extensions below;
    // reject them at plan time instead of call time — the same
    // supports-precision the nvidia provider applies. `causal` is not
    // gated: the vendor kernel has no causal flag, and with one query
    // token per sequence (enforced below) attention over the cache is
    // causal regardless, so either flag computes the same result.
    if (k || v || rotary_cos || rotary_sin || cache_batch_idx ||
        cache_leftpad || alibi_slopes ||
        window_size != std::vector<int64_t>{-1, -1} || softcap != 0.0 ||
        num_splits != 0 || return_softmax_lse || softmax_lse ||
        !cache_seqlens) {
      return false;
    }
    if (!FlashAttnWithKvcache::supports(
            q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
            cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
            softmax_scale, causal, window_size, softcap, rotary_interleaved,
            num_splits, return_softmax_lse, out, softmax_lse)) {
      return false;
    }
    // Paged decode only: the vendor kernel reads KV exclusively through
    // the block table and only supports its pinned page sizes; the static
    // backend's whole-slot pseudo pages (block_size == cache capacity,
    // e.g. 4096) keep their unfused path. The vendor kernel takes no
    // layout arguments, so the cache blocks must be fully dense
    // (the base only pins the last stride).
    const bool paged = block_table.has_value();
    const int64_t page_size = static_cast<int64_t>(k_cache.size(1));
    const bool cache_dense = k_cache.stride(2) == k_cache.size(3) &&
                             v_cache.stride(2) == v_cache.size(3);
    // The vendor kernel writes a dense [batch, 1, heads, dim] output; a
    // strided output view would be silently misplaced.
    // out may be an unsqueeze singleton view ([batch, 1, heads, dim] with
    // stride(1) == 0) — size(1)==1 makes the sequence stride irrelevant.
    const bool out_dense =
        (out.size(1) == 1 ||
         out.stride(1) == static_cast<int64_t>(out.size(2) * out.size(3))) &&
        out.stride(2) == static_cast<int64_t>(out.size(3));
    const bool ok =
        paged && cache_dense && out_dense && q.size(1) == 1 && page_size > 0 &&
        page_size <= 512 &&
        (q.dtype() == DataType::kBFloat16 || q.dtype() == DataType::kFloat16);
    return ok;
  }

  static bool supports(const Tensor q, Tensor k_cache, Tensor v_cache,
                       const std::optional<Tensor> k,
                       const std::optional<Tensor> v,
                       const std::optional<Tensor> rotary_cos,
                       const std::optional<Tensor> rotary_sin,
                       const std::optional<Tensor> cache_seqlens,
                       const std::optional<Tensor> cache_batch_idx,
                       const std::optional<Tensor> cache_leftpad,
                       const std::optional<Tensor> block_table,
                       const std::optional<Tensor> alibi_slopes,
                       const std::optional<double> softmax_scale,
                       const bool causal,
                       const std::vector<int64_t> window_size,
                       const double softcap, const bool rotary_interleaved,
                       const int64_t num_splits, const bool return_softmax_lse,
                       Tensor out, std::optional<Tensor> softmax_lse) {
    try {
      const bool ok = supports_impl(
          q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
          cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
          softmax_scale, causal, window_size, softcap, rotary_interleaved,
          num_splits, return_softmax_lse, out, softmax_lse);
      return ok;
    } catch (const std::exception& e) {
      throw;
    }
  }

  static bool supports(const Tensor q, Tensor k_cache, Tensor v_cache,
                       const std::optional<Tensor> k,
                       const std::optional<Tensor> v,
                       const std::optional<Tensor> rotary_cos,
                       const std::optional<Tensor> rotary_sin,
                       const int64_t cache_seqlens,
                       const std::optional<Tensor> cache_batch_idx,
                       const std::optional<Tensor> cache_leftpad,
                       const std::optional<Tensor> block_table,
                       const std::optional<Tensor> alibi_slopes,
                       const std::optional<double> softmax_scale,
                       const bool causal,
                       const std::vector<int64_t> window_size,
                       const double softcap, const bool rotary_interleaved,
                       const int64_t num_splits, const bool return_softmax_lse,
                       Tensor out, std::optional<Tensor> softmax_lse) {
    return supports(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
                    std::optional<Tensor>{}, cache_batch_idx, cache_leftpad,
                    block_table, alibi_slopes, softmax_scale, causal,
                    window_size, softcap, rotary_interleaved, num_splits,
                    return_softmax_lse, out, softmax_lse);
  }

  // The base declares one virtual overload taking a scalar
  // `cache_seqlens` and one taking an optional tensor; both must be
  // overridden, otherwise the class stays abstract and dispatch silently
  // skips it. The scalar form forwards to the tensor form.
  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const int64_t cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override {
    (*this)(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
            std::optional<Tensor>{}, cache_batch_idx, cache_leftpad,
            block_table, alibi_slopes, softmax_scale, causal, window_size,
            softcap, rotary_interleaved, num_splits, return_softmax_lse, out,
            softmax_lse);
  }

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const std::optional<Tensor> cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override {
    Validate(k, v, rotary_cos, rotary_sin, cache_batch_idx, cache_leftpad,
             alibi_slopes, window_size, softcap, num_splits, return_softmax_lse,
             softmax_lse);

    // [batch, 1, heads, dim] vendor BLHD decode layout.
    const int64_t batch = q.size(0);
    const int64_t head_num = q.size(2);
    const int64_t head_dim = q.size(3);
    const int64_t kv_head_num = k_cache.size(2);
    const int64_t block_size = k_cache.size(1);
    const float scale = static_cast<float>(
        softmax_scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim))));

    // Reserved for vendor entry points that inherit their stream from the
    // context; the calls below create dedicated contexts bound to the
    // caller's InfiniRT stream instead.
    baidu::xpu::api::Context* ctx = VendorContext();
    (void)ctx;
    // Null `valid_batch` means one query per batch entry (one-to-one
    // mapping); InfiniLM decode steps submit exactly that, so the kernel
    // gets an empty vector there.

    int rc;
    if (q.dtype() == DataType::kBFloat16) {
      rc = Run<bfloat16>(ctx, stream_, q, k_cache, v_cache, cache_seqlens,
                         block_table, out, batch, head_num, kv_head_num,
                         head_dim, block_size, scale);
    } else {
      rc = Run<float16>(ctx, stream_, q, k_cache, v_cache, cache_seqlens,
                        block_table, out, batch, head_num, kv_head_num,
                        head_dim, block_size, scale);
    }
    if (rc != 0) {
      throw std::runtime_error("kunlun paged_attention_xft failed, rc=" +
                               std::to_string(rc));
    }
  }

 private:
  template <typename T>
  static int Run(baidu::xpu::api::Context* ctx, void* stream, const Tensor& q,
                 const Tensor& k_cache, const Tensor& v_cache,
                 const std::optional<Tensor>& cache_seqlens,
                 const std::optional<Tensor>& block_table, const Tensor& out,
                 int64_t batch, int64_t head_num, int64_t kv_head_num,
                 int64_t head_dim, int64_t block_size, float scale) {
    // The vendor kernel reads a dense [batch, 1, heads, dim] query. The
    // caller may pass a split view of the fused-QKV output whose sequence
    // stride spans Q+K+V, so compact non-dense inputs first.
    const T* query = static_cast<const T*>(q.data());
    if (q.stride(1) != q.size(2) * q.size(3) ||
        q.stride(2) != static_cast<int64_t>(q.size(3)) || q.stride(3) != 1) {
      const int64_t batch = q.size(0);
      const int64_t heads = q.size(2);
      const int64_t dim = q.size(3);
      // The row copy below assumes a unit inner stride; the base contract
      // (`q.stride(-1) == 1`) guarantees it, but fail loudly if a direct
      // caller ever violates it.
      if (q.stride(3) != 1) {
        return -1005;
      }
      // Device-side strided compaction (LaunchCopy takes HOST metadata
      // and uploads it inside MakeMeta — no stream-ordered hazard, unlike
      // the 28 blocking D2H xpu_memcpy calls this replaces).
      thread_local T* q_dense_dev = nullptr;
      thread_local int64_t q_dense_cap = 0;
      const int64_t q_elems = batch * heads * dim;
      if (q_dense_cap < q_elems) {
        if (q_dense_dev) xpu_free(q_dense_dev);
        q_dense_dev = nullptr;
        q_dense_cap = 0;
        void* raw = nullptr;
        if (xpu_malloc(&raw, q_elems * sizeof(T)) != 0) {
          return -1002;
        }
        q_dense_dev = static_cast<T*>(raw);
        q_dense_cap = q_elems;
      }
      {
        // Treat q as [batch, heads, dim] with per-axis strides.
        const long long qc_shape[3] = {batch, heads, dim};
        const long long qc_in[3] = {(long long)q.stride(0),
                                    (long long)q.stride(2), 1};
        const long long qc_out[3] = {heads * dim, dim, 1};
        kunlun_launch::LaunchCopy(sizeof(T), q_dense_dev, q.data(), 3, qc_shape,
                                  qc_in, qc_out, (unsigned long long)q_elems,
                                  stream);
      }
      query = q_dense_dev;
    }
    // The vendor decode kernel addresses the cache as
    // `[num_blocks, num_kv_heads, page_size, head_dim]` (head-major — the
    // view vLLM-XPU's own paged_attn.py takes), while the InfiniOps/FA
    // contract is `[num_blocks, page_size, num_kv_heads, head_dim]`
    // (token-major, what reshape_and_cache_flash writes and the unit
    // tests verify). MQA (kv_head_num == 1) makes the two layouts
    // coincide; GQA needs the in-block axes swapped, so transpose each
    // cache into a scratch half first (one device pass per cache per
    // decode step; unwritten pages hold garbage in either layout and are
    // never read because context_len bounds the window).
    // Per-sequence context lengths: the vendor reads `VectorParam.cpu` on
    // the host for launch planning and `VectorParam.xpu` on-card, so
    // materialize the host copy via the vendor runtime's device-to-host
    // helper.
    thread_local std::vector<int> context_len_host;
    const int* context_len_dev =
        cache_seqlens ? static_cast<const int*>(cache_seqlens->data())
                      : nullptr;
    const int64_t context_len_count =
        cache_seqlens ? static_cast<int64_t>(cache_seqlens->size(0)) : batch;
    // The engine H2D-writes cache_seqlens once per decode step and every
    // layer of the step then calls this op with the same device tensor.
    // The first call of a step does the full-device wait + D2H; the other
    // 27 reuse the cached host copy (the engine never rewrites the tensor
    // between layers of a step). The call counter below resets whenever a
    // DIFFERENT device pointer shows up, so a new batch/step allocation
    // re-reads. If the same pointer were rewritten mid-step the copy would
    // go stale — the engine's contract (metadata written once per step,
    // before layer 0) is what makes this sound.
    if (context_len_dev) {
      thread_local const int* cached_ptr = nullptr;
      thread_local int cached_count = 0;
      thread_local long long step_calls = 0;
      if (cached_ptr != context_len_dev || cached_count != context_len_count ||
          step_calls == 0 || step_calls >= 64) {
        context_len_host.assign(context_len_count, 0);
        // The engine writes cache_seqlens via its own submission path; only a
        // full-device wait guarantees the H2D landed before this D2H read
        // (stream-scoped waits have raced it — read stale lengths, decode
        // attends over a fixed window, output repeats one token).
        xpu_wait(nullptr);
        if (xpu_memcpy(context_len_host.data(), context_len_dev,
                       context_len_host.size() * sizeof(int),
                       XPU_DEVICE_TO_HOST) != 0) {
          return -1000;
        }
        cached_ptr = context_len_dev;
        cached_count = (int)context_len_count;
      }
      step_calls = (step_calls + 1) % 64;
    }
    const int64_t max_context_len =
        block_table ? block_table->size(1) * block_size : k_cache.size(1);
    const int64_t max_blocks_per_seq = block_table ? block_table->size(1) : 1;
    int* context_len_host_ptr =
        context_len_dev ? context_len_host.data() : nullptr;
    (void)context_len_host_ptr;
    auto* o = static_cast<T*>(const_cast<void*>(out.data()));
    const int* block_table_ptr =
        block_table ? static_cast<const int*>(block_table->data()) : nullptr;
    const auto* key_cache = static_cast<const T*>(k_cache.data());
    const auto* value_cache = static_cast<const T*>(v_cache.data());
    // Vendor `paged_attention` (the non-xft entry): device-side context_len
    // pointer, no host D2H. The vendor indexes the cache head-major
    // ([num_blocks, kv_heads, page, dim]) while reshape_and_cache writes
    // token-major ([num_blocks, page, kv_heads, dim]); folding that axis
    // swap plus the GQA group broadcast into one TransposeBroadcast pass
    // per cache per step lets the single vendor kernel serve all q heads
    // (kv_head_num = head_num, the pairing the vendor is verified under).
    // Vendor `paged_attention` over a dense active-window copy: gather the
    // first `clen` tokens of each sequence into a contiguous scratch
    // (broadcast to q heads), back it with an identity block table, and let
    // the single vendor kernel serve all q heads (kv_head_num = head_num —
    // the pairing the prefill vendor path is verified under). 3 launches
    // replace the decomposed chain's 6; falls back to the chain when the
    // vendor call reports failure.
#ifndef KLATTN_NO_VENDOR
    {
      // Host-side context lens are cached from the D2H above.
      bool lens_ok = true;
      int clen0 = (int)max_context_len;
      if (context_len_host_ptr != nullptr && batch > 0) {
        clen0 = context_len_host_ptr[0];
      } else if (context_len_dev != nullptr) {
        lens_ok = false;  // no host copy available
      }
      if (lens_ok && block_table_ptr != nullptr && clen0 > 0) {
        const int64_t page_words = block_size * head_num * head_dim;
        const int64_t pages = (clen0 + block_size - 1) / block_size;
        const int64_t scratch_elems = pages * page_words;
        thread_local void* dscr = nullptr;
        thread_local int64_t dcap = 0;
        if (dcap < scratch_elems) {
          if (dscr) xpu_free(dscr);
          dscr = nullptr;
          dcap = 0;
          void* raw = nullptr;
          if (xpu_malloc(&raw, 2 * scratch_elems * sizeof(T)) != 0) {
            return -1002;
          }
          dscr = raw;
          dcap = scratch_elems;
        }
        // Identity block table [0..pages) as a device buffer (kernel
        // pointer args must be device memory).
        thread_local int* ident = nullptr;
        thread_local int64_t ident_cap = 0;
        if (ident_cap < pages) {
          if (ident) xpu_free(ident);
          ident = nullptr;
          ident_cap = 0;
          void* raw = nullptr;
          if (xpu_malloc(&raw, pages * sizeof(int)) != 0) {
            return -1002;
          }
          ident = static_cast<int*>(raw);
          ident_cap = pages;
        }
        {
          std::vector<int> ident_h(pages);
          for (int64_t i = 0; i < pages; ++i) ident_h[i] = (int)i;
          xpu_memcpy(ident, ident_h.data(), pages * sizeof(int),
                     XPU_HOST_TO_DEVICE);
        }
        T* k_scr = static_cast<T*>(dscr);
        T* v_scr = k_scr + scratch_elems;
        for (int64_t b = 0; b < batch; ++b) {
          const int clen_b =
              context_len_host_ptr ? context_len_host_ptr[b] : clen0;
          const int* bt_row = block_table_ptr + b * max_blocks_per_seq;
          T* kb = k_scr + b * pages * page_words;
          T* vb = v_scr + b * pages * page_words;
          kunlun_launch::LaunchGatherDenseBroadcastKV(
              kb, vb, key_cache, value_cache, bt_row, (int)max_blocks_per_seq,
              (int)block_size, (int)kv_head_num, (int)head_num, (int)head_dim,
              clen_b, stream);
        }
        thread_local baidu::xpu::api::Context* vctx = nullptr;
        if (vctx == nullptr) {
          vctx = baidu::xpu::api::create_context();
        }
        vctx->set_stream(static_cast<XPUStream>(stream));
        int rc = baidu::xpu::xfa::VendorPagedAttn<T>::call(
            vctx, o, query, k_scr, v_scr, (std::int64_t)head_num, scale, ident,
            context_len_dev, (std::int64_t)block_size,
            (std::int64_t)(pages * block_size), (std::int64_t)batch,
            (std::int64_t)head_num, (std::int64_t)head_dim,
            (std::int64_t)pages);
        if (rc == 0) {
          return 0;
        }
        // rc != 0: fall through to the decomposed chain.
      }
    }
#endif
    // Fallback decomposed chain: `paged_attention_xft` mis-resolves GQA at
    // context_len > 1 and the raw-cache `paged_attention` variant hangs —
    // the chain below is verified against a torch reference.
    // Decomposed decode path: the vendor paged kernel both mis-resolves GQA
    // (all q heads read one kv group at ctx>1) and stalls at the broadcast
    // kv_head_num == head_num shape, so gather the pages, broadcast kv heads
    // to q heads, and run QK -> softmax -> Pv with our own kernels — the
    // chain verified against a torch reference in the C++ probes.
    const int clen_max = static_cast<int>(max_context_len);
    const int64_t kv_gather_elems =
        (int64_t)batch * clen_max * head_num * head_dim;
    thread_local void* dec_scratch = nullptr;
    thread_local int64_t dec_cap = 0;
    if (dec_cap < kv_gather_elems * 4) {
      if (dec_scratch) xpu_free(dec_scratch);
      dec_scratch = nullptr;
      dec_cap = 0;
      void* raw = nullptr;
      // k_ext + v_ext (bf16) + scores f32
      if (xpu_malloc(&raw, (size_t)kv_gather_elems * 4 * 2 +
                               (size_t)batch * head_num * clen_max * 4) != 0) {
        return -1002;
      }
      dec_scratch = raw;
      dec_cap = kv_gather_elems * 4;
    }
    // Gather lands in a kv-sized staging half; the GQA broadcast writes
    // the q-head-expanded half. In-place broadcast would overwrite unread
    // source rows (out grows kv->q heads inside the same buffer).
    auto* k_tmp = static_cast<T*>(dec_scratch);
    auto* v_tmp = k_tmp + kv_gather_elems;
    auto* k_ext = v_tmp + kv_gather_elems;
    auto* v_ext = k_ext + kv_gather_elems;
    float* scores = reinterpret_cast<float*>(v_ext + kv_gather_elems);
    for (int64_t b = 0; b < batch; ++b) {
      const int clen =
          context_len_host_ptr ? context_len_host_ptr[b] : (int)max_context_len;

      const int* bt_row = block_table_ptr + b * max_blocks_per_seq;
      T* k_b = k_ext + b * clen_max * head_num * head_dim;
      T* v_b = v_ext + b * clen_max * head_num * head_dim;
      T* o_b = o + b * head_num * head_dim;
      kunlun_launch::LaunchGatherPage(
          k_tmp, key_cache, bt_row, (int)max_blocks_per_seq, (int)block_size,
          (int)kv_head_num, (int)head_dim, clen, 2, stream);
      kunlun_launch::LaunchGatherPage(
          v_tmp, value_cache, bt_row, (int)max_blocks_per_seq, (int)block_size,
          (int)kv_head_num, (int)head_dim, clen, 2, stream);
      kunlun_launch::LaunchGqaBroadcast(k_b, k_tmp, clen, (int)kv_head_num,
                                        (int)head_num, (int)head_dim, stream);
      kunlun_launch::LaunchGqaBroadcast(v_b, v_tmp, clen, (int)kv_head_num,
                                        (int)head_num, (int)head_dim, stream);

      kunlun_launch::LaunchQK(scores + b * head_num * clen_max, query, k_b,
                              scale, clen, (int)head_num, (int)head_dim, 2,
                              stream);
      kunlun_launch::LaunchPv(o_b, scores + b * head_num * clen_max, v_b, clen,
                              (int)head_num, (int)head_dim, 2, stream);
    }
    xpu_wait(static_cast<XPUStream>(stream));
    return 0;
  }

  static void Validate(const std::optional<Tensor>& k,
                       const std::optional<Tensor>& v,
                       const std::optional<Tensor>& rotary_cos,
                       const std::optional<Tensor>& rotary_sin,
                       const std::optional<Tensor>& cache_batch_idx,
                       const std::optional<Tensor>& cache_leftpad,
                       const std::optional<Tensor>& alibi_slopes,
                       const std::vector<int64_t>& window_size, double softcap,
                       int64_t num_splits, bool return_softmax_lse,
                       const std::optional<Tensor>& softmax_lse) {
    if (k.has_value() || v.has_value() || rotary_cos.has_value() ||
        rotary_sin.has_value() || cache_batch_idx.has_value() ||
        cache_leftpad.has_value() || alibi_slopes.has_value() ||
        window_size != std::vector<int64_t>{-1, -1} || softcap != 0.0 ||
        num_splits != 0 || return_softmax_lse || softmax_lse.has_value()) {
      throw std::invalid_argument(
          "kunlun FlashAttnWithKvcache received an unsupported option");
    }
  }

  static baidu::xpu::api::Context* VendorContext() {
    thread_local baidu::xpu::api::Context* ctx = nullptr;
    if (ctx == nullptr) {
      ctx = baidu::xpu::api::create_context();
      // Reserved for vendor entry points that inherit their stream from the
      // context (left null = device default stream). The decode path does
      // NOT go through this context: Run() creates a dedicated context per
      // vendor call and binds it to the caller's InfiniRT stream so the
      // gather kernels and the vendor kernel order against InfiniRT.
      ctx->set_stream(nullptr);
    }
    return ctx;
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_KUNLUN_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
