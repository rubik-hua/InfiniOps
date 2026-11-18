#ifndef INFINI_OPS_KUNLUN_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_KUNLUN_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/reshape_and_cache_flash.h"
#include "native/kunlun/launch2_declarations.h"

// Vendor `libapiinfer.so` (Kunlun inference kernel library; shipped in the
// vLLM-XPU wheel, `vllm_xpu/so/`) exposes a paged KV-cache scatter with the
// same layout contract InfiniOps uses: key_cache/value_cache are
// `[num_blocks, block_size, num_heads, head_size]` and `slot_mapping` holds
// flat slots (`block_idx * block_size + offset`). The vendor header
// (`infer_ops.h`) is not installed alongside the library, so the entry point
// is declared here with the exact vendor template signature and resolved at
// link time. Three vendor specifics:
//   * It takes a `baidu::xpu::api::Context` — the "old" refactor context
//     (factory in `xpu/refactor/context/newcontext.h`, implemented in
//     `libxpuapi.so`), not the `xpytorch::xpu::api` one.
//   * The library only instantiates the vendor `float16`/`bfloat16` class
//     types (global-scope, declared by `xpu/refactor/util/{float16,
//     bfloat16}.h` via `core/dtype.h`), which are 16-bit bit-compatible with
//     fp16/bf16 storage, so payloads are passed through a reinterpret cast.
//   * The slot type is pinned to `int` (int32), while the InfiniOps contract
//     carries int64 slots — the wrapper narrows them on the host first.
#include <xpu/refactor/context/newcontext.h>
#include <xpu/refactor/core/dtype.h>

// The vendor entry point lives in `namespace infer_ops` (mangled exactly as
// `infer_ops::reshape_and_cache<T, TID>` in `libapiinfer.so`).
namespace infer_ops {

template <typename T, typename TID>
int reshape_and_cache(baidu::xpu::api::Context* ctx, const T* k, const T* v,
                      T* k_cache, T* v_cache, TID* slot_mapping,
                      std::int64_t num_tokens, std::int64_t num_blocks,
                      std::int64_t num_heads, std::int64_t k_head_size,
                      std::int64_t v_head_size, std::int64_t block_size);

}  // namespace infer_ops

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kKunlun>
    : public ReshapeAndCacheFlash {
 public:
  using ReshapeAndCacheFlash::ReshapeAndCacheFlash;

  bool capture_safe() const override { return true; }

  static bool supports(const Tensor& key, const Tensor& value,
                       const Tensor& slots, const Tensor& k_scale,
                       const Tensor& v_scale, const std::string& mode,
                       const Tensor& key_cache, const Tensor& value_cache) {
    // The base `SupportsCommon` already enforces the canonical FA layout
    // (`[blocks, block, heads, dim]`, contiguous, unit last stride,
    // int64 slots, float32 scales). Kunlun adds a dtype restriction to the
    // 16-bit cache types the vendor kernel instantiates.
    return SupportsCommon(key, value, slots, k_scale, v_scale, mode, key_cache,
                          value_cache) &&
           (key.dtype() == DataType::kBFloat16 ||
            key.dtype() == DataType::kFloat16);
  }

  void operator()(const Tensor key, const Tensor value, const Tensor slots,
                  const Tensor k_scale, const Tensor v_scale,
                  const std::string kv_cache_dtype, Tensor key_cache,
                  Tensor value_cache) const override {
    (void)k_scale;
    (void)v_scale;
    (void)kv_cache_dtype;
    if (num_tokens_ == 0) {
      return;
    }

    baidu::xpu::api::Context* ctx = VendorContext();

    // The vendor kernel reads int32 slots; the InfiniOps contract hands over
    // int64. Narrow on-device so the submission path stays free of blocking
    // host round trips.
    thread_local int* slots_dev = nullptr;
    thread_local std::int64_t slots_dev_cap = 0;
    if (slots_dev_cap < num_tokens_) {
      if (slots_dev) {
        xpu_free(slots_dev);
        slots_dev = nullptr;
      }
      slots_dev_cap = 0;
      void* raw = nullptr;
      if (xpu_malloc(&raw, sizeof(int) * num_tokens_) != 0) {
        throw std::runtime_error(
            "kunlun reshape_and_cache slot buffer "
            "allocation failed");
      }
      slots_dev = static_cast<int*>(raw);
      slots_dev_cap = num_tokens_;
    }
    kunlun_launch::LaunchNarrowI64ToI32(slots_dev, slots.data(), num_tokens_,
                                        stream_);

    // The caller may hand over split views of the fused-QKV GEMM output
    // ([tokens, heads, dim] with a sequence stride spanning Q+K+V); the
    // vendor kernel reads dense rows, so compact non-dense inputs first.
    const auto dense_rows = [](const Tensor& t) {
      const auto& st = t.strides();
      return st[0] == static_cast<ptrdiff_t>(t.size(1) * t.size(2)) &&
             st[1] == static_cast<ptrdiff_t>(t.size(2)) && st[2] == 1;
    };
    const int64_t tokens = key.size(0);
    const int64_t kv_heads = key.size(1);
    const int64_t dim = key.size(2);
    const bool k_dense = dense_rows(key);
    const bool v_dense = dense_rows(value);
    const char* kp = static_cast<const char*>(key.data());
    const char* vp = static_cast<const char*>(value.data());
    if (!k_dense || !v_dense) {
      // Compact the split/padded fused-QKV views on-device. A host-side
      // D2H->H2D staging would race the caller's InfiniRT stream (the
      // synchronous-looking xpu_memcpy D2H does not order against kernels
      // already queued on that stream — observed reading the pre-GEMM
      // workspace as NaN and writing garbage into the KV cache), while a
      // device-side copy launched on the same stream orders naturally and
      // stays graph-capture-safe. Honor every stride: a padded fused-QKV
      // view also spreads its heads apart (`head_stride > dim`).
      const int64_t total_elems = 2 * tokens * kv_heads * dim;
      thread_local void* staging_dev = nullptr;
      thread_local int64_t staging_cap = 0;
      if (staging_cap < total_elems) {
        if (staging_dev) xpu_free(staging_dev);
        staging_dev = nullptr;
        staging_cap = 0;
        void* raw = nullptr;
        if (xpu_malloc(&raw, total_elems * sizeof(uint16_t)) != 0) {
          throw std::runtime_error(
              "kunlun reshape_and_cache staging "
              "allocation failed");
        }
        staging_dev = raw;
        staging_cap = total_elems;
      }
      auto* st = static_cast<uint16_t*>(staging_dev);
      const int tag = key.dtype() == DataType::kBFloat16 ? 2 : 1;
      if (!k_dense) {
        long long shape[3] = {(long long)tokens, (long long)kv_heads,
                              (long long)dim};
        long long in_strides[3] = {(long long)key.strides()[0],
                                   (long long)key.strides()[1], 1};
        long long out_strides[3] = {(long long)kv_heads * dim, (long long)dim,
                                    1};
        kunlun_launch::LaunchCopy(tag, st, key.data(), 3, shape, in_strides,
                                  out_strides, total_elems / 2,
                                  stream_ ? stream_ : nullptr);
        kp = static_cast<const char*>(staging_dev);
      }
      if (!v_dense) {
        long long shape[3] = {(long long)tokens, (long long)kv_heads,
                              (long long)dim};
        long long in_strides[3] = {(long long)value.strides()[0],
                                   (long long)value.strides()[1], 1};
        long long out_strides[3] = {(long long)kv_heads * dim, (long long)dim,
                                    1};
        kunlun_launch::LaunchCopy(tag, st + tokens * kv_heads * dim,
                                  value.data(), 3, shape, in_strides,
                                  out_strides, total_elems / 2,
                                  stream_ ? stream_ : nullptr);
        vp =
            static_cast<const char*>(staging_dev) + tokens * kv_heads * dim * 2;
      }
    }

    // Bind the vendor context to the caller's stream: InfiniRT submits its
    // kernels on its own XRE stream, and this op must order against them.
    ctx->set_stream(stream_);

    // The vendor kernel addresses the cache as
    // `[num_blocks, num_heads, block_size, head_size]` (head-major), while
    // the InfiniOps/FA contract (and the tensors the tests and the runtime
    // hand over) is `[num_blocks, block_size, num_heads, head_size]`
    // (token-major). When the cache is a plain contiguous tensor of the
    // token-major physical layout, folding the head axis into the head size
    // (num_heads=1, head_size=H*D) makes the vendor head-major indexing
    // collapse to plain row-major addressing, which lands every row in the
    // right place for both physical layouts.
    const bool token_major_cache =
        key_cache.strides()[3] == 1 &&
        key_cache.strides()[2] == static_cast<ptrdiff_t>(key_cache.size(3)) &&
        key_cache.strides()[1] ==
            static_cast<ptrdiff_t>(key_cache.size(2) * key_cache.size(3));
    const int64_t vendor_heads = token_major_cache ? 1 : num_heads_;
    const int64_t vendor_k_dim =
        token_major_cache ? head_size_ * num_heads_ : head_size_;
    const int64_t vendor_v_dim =
        token_major_cache ? value_head_size_ * num_heads_ : value_head_size_;

    int rc;
    if (key.dtype() == DataType::kBFloat16) {
      rc = infer_ops::reshape_and_cache<bfloat16, int>(
          ctx, reinterpret_cast<const bfloat16*>(kp),
          reinterpret_cast<const bfloat16*>(vp),
          static_cast<bfloat16*>(key_cache.data()),
          static_cast<bfloat16*>(value_cache.data()), slots_dev,
          static_cast<std::int64_t>(num_tokens_),
          static_cast<std::int64_t>(key_cache.size(0)),
          static_cast<std::int64_t>(vendor_heads),
          static_cast<std::int64_t>(vendor_k_dim),
          static_cast<std::int64_t>(vendor_v_dim),
          static_cast<std::int64_t>(block_size_));
    } else {
      rc = infer_ops::reshape_and_cache<float16, int>(
          ctx, reinterpret_cast<const float16*>(kp),
          reinterpret_cast<const float16*>(vp),
          static_cast<float16*>(key_cache.data()),
          static_cast<float16*>(value_cache.data()), slots_dev,
          static_cast<std::int64_t>(num_tokens_),
          static_cast<std::int64_t>(key_cache.size(0)),
          static_cast<std::int64_t>(vendor_heads),
          static_cast<std::int64_t>(vendor_k_dim),
          static_cast<std::int64_t>(vendor_v_dim),
          static_cast<std::int64_t>(block_size_));
    }

    if (rc != 0) {
      throw std::runtime_error("kunlun reshape_and_cache failed, rc=" +
                               std::to_string(rc));
    }
  }

 private:
  static baidu::xpu::api::Context* VendorContext() {
    thread_local baidu::xpu::api::Context* ctx = nullptr;
    if (ctx == nullptr) {
      ctx = baidu::xpu::api::create_context();
      // The vendor context stream is bound per call (ctx->set_stream(stream_))
      // so the cache write orders against the caller's InfiniRT submissions
      // — the decode/prefill attention wrappers read this cache from kernels
      // launched on the same caller stream.
      ctx->set_stream(nullptr);
    }
    return ctx;
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_KUNLUN_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
