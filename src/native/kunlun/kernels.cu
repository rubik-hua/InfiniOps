// Kunlun XPU device kernels + their host launchers.
//
// The `*Kernel` templates below are compiled by the XTDK clang++ into XPU
// machine code; the `Launch*` functions at the bottom are host C++ and
// submit them through the `<<<cluster_num, cores_per_cluster, stream>>>`
// launch syntax, which the XTDK compiles to `xpu_launch_config`.
//
// Design note: XPU infers plain pointer parameters as global-address-space-1
// qualified pointers, and the XTDK `half`/`bfloat16_t` types cannot be
// copied between address spaces implicitly. All half/bf16 data therefore
// flows as `unsigned short` payloads with explicit bit conversion to float
// for compute (the same approach the legacy backend used for bf16).

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// clang-format off
#include <xpu/runtime.h>
#include <xpu/kernel/xtdk.h>
#include <xpu/kernel/xtdk_atomic_sm_xpu3.h>
#include <xpu/kernel/xtdk_bf16.h>
#include <xpu/kernel/xtdk_fp16.h>
#include <xpu/kernel/xtdk_math.h>
#include <xpu/kernel/xtdk_simd.h>
#include <xpu/kernel/xtdk_trigonometric.h>
// clang-format on

namespace infini::ops {
namespace {

constexpr int kTagF32 = 0;
constexpr int kTagF16 = 1;
constexpr int kTagBF16 = 2;
constexpr int kTagI32 = 3;
constexpr int kTagI64 = 4;
constexpr int kTagI16 = 5;

// Host-side bit converters (bit-exact round-to-nearest-even; used by the
// binary launchers for alpha and by LaunchFill). Mirrors the kl2 ones.
inline unsigned short FloatToF16BitsHost(float f) {
  unsigned int bits;
  std::memcpy(&bits, &f, 4);
  const unsigned int sign = (bits >> 16) & 0x8000;
  int exponent = (int)((bits >> 23) & 0xFF) - 127;
  unsigned int mantissa = bits & 0x7FFFFF;
  if (exponent >= 16) {
    return (unsigned short)(sign | 0x7C00 |
                            ((exponent == 128 && mantissa) ? 0x200 : 0));
  }
  if (exponent >= -14) {
    unsigned int m = mantissa + 0x1000;
    if (m & 0x800000) {
      m >>= 1;
      ++exponent;
      if (exponent >= 16) return (unsigned short)(sign | 0x7C00);
    }
    return (unsigned short)(sign | ((exponent + 15) << 10) | (m >> 13));
  }
  if (exponent >= -25) {
    unsigned int m = (mantissa | 0x800000) >> (-14 - exponent);
    m = (m + 0xFFF) >> 13;
    return (unsigned short)(sign | (m < 0x400 ? m : 0x3FF));
  }
  return (unsigned short)sign;
}

inline unsigned short FloatToBF16BitsHost(float f) {
  unsigned int bits;
  std::memcpy(&bits, &f, 4);
  return (unsigned short)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// ---------------------------------------------------------------------------
// dtype views: fp16/bf16 as raw 16-bit payloads.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float F16ToFloat(unsigned short bits) {
  __half_raw hr;
  hr.x = bits;
  return __half2float(__half(hr));
}

__device__ __forceinline__ unsigned short FloatToF16Bits(float f) {
  __half h = __float2half(f);
  __half_raw hr = static_cast<__half_raw>(h);
  return hr.x;
}

__device__ __forceinline__ float BF16ToFloat(unsigned short bits) {
  __xpu_bfloat16_raw br;
  br.x = bits;
  return static_cast<float>(__xpu_bfloat16(br));
}

__device__ __forceinline__ unsigned short FloatToBF16Bits(float f) {
  __xpu_bfloat16 h = __float2bfloat16(f);
  __xpu_bfloat16_raw br = static_cast<__xpu_bfloat16_raw>(h);
  return br.x;
}

// ---------------------------------------------------------------------------
// Elementwise op functors over the three compute types:
//   float          (kTagF32)
//   F16View<T>     (kTagF16)  -- 16-bit payload wrapper
//   BF16View<T>    (kTagBF16)
// ---------------------------------------------------------------------------

template <int kTag>
struct DtTraits;

template <>
struct DtTraits<kTagF32> {
  using Storage = float;
  static __device__ __forceinline__ float ToFloat(Storage v) { return v; }
  static __device__ __forceinline__ Storage FromFloat(float f) { return f; }
};

template <>
struct DtTraits<kTagF16> {
  using Storage = unsigned short;
  static __device__ __forceinline__ float ToFloat(Storage v) {
    return F16ToFloat(v);
  }
  static __device__ __forceinline__ Storage FromFloat(float f) {
    return FloatToF16Bits(f);
  }
};

template <>
struct DtTraits<kTagBF16> {
  using Storage = unsigned short;
  static __device__ __forceinline__ float ToFloat(Storage v) {
    return BF16ToFloat(v);
  }
  static __device__ __forceinline__ Storage FromFloat(float f) {
    return FloatToBF16Bits(f);
  }
};

// Integer payloads pass through unchanged; ToFloat/FromFloat are unused for
// the pure pass-through ops (Add) but must exist for the templates.
template <>
struct DtTraits<kTagI16> {
  using Storage = short;
  static __device__ __forceinline__ float ToFloat(Storage v) {
    return static_cast<float>(v);
  }
  static __device__ __forceinline__ Storage FromFloat(float f) {
    return static_cast<short>(f);
  }
};

template <>
struct DtTraits<kTagI32> {
  using Storage = int;
  static __device__ __forceinline__ float ToFloat(Storage v) {
    return static_cast<float>(v);
  }
  static __device__ __forceinline__ Storage FromFloat(float f) {
    return static_cast<int>(f);
  }
};

template <>
struct DtTraits<kTagI64> {
  using Storage = long long;
  static __device__ __forceinline__ float ToFloat(Storage v) {
    return static_cast<float>(v);
  }
  static __device__ __forceinline__ Storage FromFloat(float f) {
    return static_cast<long long>(f);
  }
};

// Per-element unary transforms expressed in float.
template <template <typename> class F, int kTag, typename... Args>
__device__ __forceinline__ typename DtTraits<kTag>::Storage ApplyUnary(
    typename DtTraits<kTag>::Storage x, Args... args) {
  return DtTraits<kTag>::FromFloat(
      F<float>{}(DtTraits<kTag>::ToFloat(x), args...));
}

template <template <typename> class F, int kTag>
__device__ __forceinline__ typename DtTraits<kTag>::Storage ApplyBinary(
    typename DtTraits<kTag>::Storage a, typename DtTraits<kTag>::Storage b) {
  return DtTraits<kTag>::FromFloat(
      F<float>{}(DtTraits<kTag>::ToFloat(a), DtTraits<kTag>::ToFloat(b)));
}

// Alpha-scaled variant: a + alpha * b semantics via an extra scale on b.
template <template <typename> class F, int kTag>
__device__ __forceinline__ typename DtTraits<kTag>::Storage ApplyBinary(
    typename DtTraits<kTag>::Storage a, typename DtTraits<kTag>::Storage b,
    float alpha) {
  return DtTraits<kTag>::FromFloat(F<float>{}(
      DtTraits<kTag>::ToFloat(a), alpha * DtTraits<kTag>::ToFloat(b)));
}

// Float-domain elementwise ops.
template <typename T>
struct SigmoidF {
  __device__ T operator()(T x) const {
    return static_cast<T>(1.0f / (1.0f + exp(-static_cast<float>(x))));
  }
};

template <typename T>
struct ReluF {
  __device__ T operator()(T x) const {
    // NaN must pass through (IEEE semantics, torch.relu(nan) == nan).
    // XTDK may not honor (x != x) reliably, so test the bit pattern:
    // NaN has exponent all-ones AND nonzero mantissa.
    unsigned int bits = __builtin_bit_cast(unsigned int, x);
    const unsigned int isNaN =
        ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0u);
    if (isNaN) return x;
    // Return x as-is for >= 0 so that signed zeros keep their sign
    // (torch.relu(-0.0) == -0.0, and the suite asserts signbit equality).
    return (x < T(0)) ? T(0) : x;
  }
};

// XTDK exposes no device-side erf; approximate it with the Abramowitz &
// Stegun 7.1.26 rational polynomial (abs error < 1.5e-7 in erf, accumulated
// in double so the GELU result stays far below the 1e-6 test tolerance).
__device__ __forceinline__ float KlErf(float x) {
  const double t = 1.0 / (1.0 + 0.3275911 * (x < 0 ? -x : x));
  const double y =
      1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
              0.284496736) *
                 t +
             0.254829592) *
                t * exp(-(double)x * (double)x);
  return (float)(x < 0 ? -y : y);
}

template <typename T>
struct GeluF {
  __device__ T operator()(T x) const {
    const float f = static_cast<float>(x);
    const float cdf = 0.5f * (1.0f + KlErf(f * 0.7071067811865475f));
    return static_cast<T>(f * cdf);
  }
};

template <typename T>
struct GeluTanhF {
  __device__ T operator()(T x) const {
    const float f = static_cast<float>(x);
    const float cdf =
        0.5f * (1.0f + static_cast<float>(tanh(0.7978845608028654 *
                                               (f + 0.044715 * f * f * f))));
    return static_cast<T>(f * cdf);
  }
};

template <typename T>
struct SiluF {
  __device__ T operator()(T x) const {
    const float f = static_cast<float>(x);
    return static_cast<T>(f / (1.0f + static_cast<float>(exp(-f))));
  }
};

template <typename T>
struct AddF {
  __device__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct AddAlphaF {
  T alpha;
  __device__ T operator()(T a, T b) const { return a + alpha * b; }
};

template <typename T>
struct SubF {
  __device__ T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct MulF {
  __device__ T operator()(T a, T b) const { return a * b; }
};

// ---------------------------------------------------------------------------
// Kernels. XPU infers plain pointer params as address-space-1 pointers;
// 16-bit payloads and floats copy fine between AS1 and generic via plain
// assignment, so storage types stay builtin.
// ---------------------------------------------------------------------------

template <typename T, int kTag>
__global__ void UnaryElementwiseKernel(
    T* out, const T* input, const unsigned long long* out_shape,
    const unsigned long long* input_shape, const long long* out_strides,
    const long long* input_strides, unsigned long long output_size,
    unsigned long long ndim, bool out_contiguous, bool input_contiguous) {
  const std::size_t thread_id = cluster_id() * 64 + core_id();
  const std::size_t nthreads = cluster_num() * 64;

  for (std::size_t idx = thread_id; idx < output_size; idx += nthreads) {
    std::size_t out_idx = idx;
    std::size_t in_idx = idx;
    if (!out_contiguous) {
      std::size_t rem = idx;
      for (std::size_t i = ndim; i-- > 0;) {
        out_idx +=
            (rem % out_shape[i]) * static_cast<std::size_t>(out_strides[i]);
        rem /= out_shape[i];
      }
    }
    if (!input_contiguous) {
      std::size_t rem = idx;
      for (std::size_t i = ndim; i-- > 0;) {
        in_idx +=
            (rem % input_shape[i]) * static_cast<std::size_t>(input_strides[i]);
        rem /= input_shape[i];
      }
    }
    out[out_idx] = ApplyUnary<ReluF, kTag>(input[in_idx]);
    (void)sizeof(DtTraits<kTag>);  // force kTag dependency
  }
}

template <typename T, int kTag, template <typename> class Op>
__global__ void UnaryEWKernel(
    T* out, const T* input, const unsigned long long* out_shape,
    const unsigned long long* input_shape, const long long* out_strides,
    const long long* input_strides, unsigned long long output_size,
    unsigned long long ndim, bool out_contiguous, bool input_contiguous) {
  const std::size_t thread_id = cluster_id() * 64 + core_id();
  const std::size_t nthreads = cluster_num() * 64;

  for (std::size_t idx = thread_id; idx < output_size; idx += nthreads) {
    std::size_t out_idx = 0;
    std::size_t in_idx = 0;
    if (!out_contiguous) {
      std::size_t rem = idx;
      for (std::size_t i = ndim; i-- > 0;) {
        out_idx +=
            (rem % out_shape[i]) * static_cast<std::size_t>(out_strides[i]);
        rem /= out_shape[i];
      }
    } else {
      out_idx = idx;
    }
    if (!input_contiguous) {
      std::size_t rem = idx;
      for (std::size_t i = ndim; i-- > 0;) {
        in_idx +=
            (rem % input_shape[i]) * static_cast<std::size_t>(input_strides[i]);
        rem /= input_shape[i];
      }
    } else {
      in_idx = idx;
    }
    out[out_idx] = ApplyUnary<Op, kTag>(input[in_idx]);
  }
}

template <typename T, int kTag, template <typename> class Op>
__global__ void BinaryEWKernel(
    T* out, const T* a, const T* b, const unsigned long long* out_shape,
    const unsigned long long* a_shape, const unsigned long long* b_shape,
    const long long* out_strides, const long long* a_strides,
    const long long* b_strides, unsigned long long output_size,
    unsigned long long ndim, bool out_contiguous, bool a_contiguous,
    bool b_contiguous, float alpha) {
  const std::size_t thread_id = cluster_id() * 64 + core_id();
  const std::size_t nthreads = cluster_num() * 64;

  for (std::size_t idx = thread_id; idx < output_size; idx += nthreads) {
    std::size_t out_idx = out_contiguous ? idx : 0;
    std::size_t a_idx = a_contiguous ? idx : 0;
    std::size_t b_idx = b_contiguous ? idx : 0;
    if (!out_contiguous) {
      std::size_t rem = idx;
      for (std::size_t i = ndim; i-- > 0;) {
        out_idx +=
            (rem % out_shape[i]) * static_cast<std::size_t>(out_strides[i]);
        rem /= out_shape[i];
      }
    }
    if (!a_contiguous) {
      // a_strides are broadcast strides relative to the output layout, so
      // the decomposition divides by out_shape (never a_shape: broadcast
      // operands may have fewer dims than the output).
      std::size_t rem = idx;
      for (std::size_t i = ndim; i-- > 0;) {
        a_idx += (rem % out_shape[i]) * static_cast<std::size_t>(a_strides[i]);
        rem /= out_shape[i];
      }
    }
    if (!b_contiguous) {
      std::size_t rem = idx;
      for (std::size_t i = ndim; i-- > 0;) {
        b_idx += (rem % out_shape[i]) * static_cast<std::size_t>(b_strides[i]);
        rem /= out_shape[i];
      }
    }
    out[out_idx] = ApplyBinary<Op, kTag>(a[a_idx], b[b_idx], alpha);
  }
}

// Contiguous binary fast path: 4 bf16/f16 elems per 64-bit word. The generic
// BinaryEWKernel's scalar u16 loop runs at ~5x slower device rate for the same
// bytes (measured 39.6us vs 7.65us on a 3584-elem add, 8 clusters). Decode is
// dominated by contiguous [1, 3584] adds/muls, so route them here.
template <int kTag, template <typename> class Op>
__global__ void BinaryContigPackedKernel(unsigned long long* out,
                                         const unsigned long long* a,
                                         const unsigned long long* b,
                                         unsigned long long words,
                                         float alpha) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * 64 + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * 64;
  for (unsigned long long w = tid; w < words; w += nth) {
    unsigned long long av = a[w];
    unsigned long long bv = b[w];
    unsigned long long rv = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const unsigned short ae = (unsigned short)((av >> (e * 16)) & 0xffffu);
      const unsigned short be = (unsigned short)((bv >> (e * 16)) & 0xffffu);
      const unsigned short re =
          (unsigned short)ApplyBinary<Op, kTag>(ae, be, alpha);
      rv |= (unsigned long long)re << (e * 16);
    }
    out[w] = rv;
  }
}

// f32 variant: 1 elem per word, direct.
template <template <typename> class Op>
__global__ void BinaryContigPackedKernelF32(float* out, const float* a,
                                            const float* b,
                                            unsigned long long elems,
                                            float alpha) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * 64 + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * 64;
  for (unsigned long long i = tid; i < elems; i += nth) {
    out[i] = Op<float>{}(a[i], alpha * b[i]);
  }
}

template <typename T, int kTag>
__global__ void SiluAndMulKernel(T* out, const T* in, unsigned long long rows,
                                 unsigned long long cols,
                                 long long in_row_stride,
                                 long long out_row_stride, bool in_c,
                                 bool out_c) {
  const std::size_t thread_id = cluster_id() * 64 + core_id();
  const std::size_t nthreads = cluster_num() * 64;
  const std::size_t total = rows * cols;
  for (std::size_t idx = thread_id; idx < total; idx += nthreads) {
    const std::size_t row = idx / cols;
    const std::size_t col = idx % cols;
    // in row stride covers a full gate+up pair (2*cols logical elements)
    const std::size_t in_row_base =
        in_c ? row * cols * 2 : static_cast<std::size_t>(row * in_row_stride);
    const T g = in[in_row_base + col];
    const T u = in[in_row_base + cols + col];
    const std::size_t out_idx =
        out_c ? idx : static_cast<std::size_t>(row * out_row_stride) + col;
    out[out_idx] = ApplyBinary<MulF, kTag>(ApplyUnary<SiluF, kTag>(g), u);
  }
}

// Contiguous packed SiluAndMul: rows of [gate | up] with 4 bf16/f16 elems per
// 64-bit word. Scalar u16 loops run ~5x slower for the same bytes on XPU;
// the decode MLP shape [1, 18944] is contiguous, so route it here from the
// launcher. One thread handles one output word (4 elems).
template <int kTag>
__global__ void SiluAndMulPackedKernel(unsigned long long* out,
                                       const unsigned long long* in,
                                       unsigned long long rows,
                                       unsigned long long words) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * 64 + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * 64;
  const unsigned long long in_row_words = words * 2;
  for (unsigned long long idx = tid; idx < rows * words; idx += nth) {
    const unsigned long long row = idx / words;
    const unsigned long long w = idx % words;
    const unsigned long long gw = in[row * in_row_words + w];
    const unsigned long long uw = in[row * in_row_words + words + w];
    unsigned long long rv = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const unsigned short ge = (unsigned short)((gw >> (e * 16)) & 0xffffu);
      const unsigned short ue = (unsigned short)((uw >> (e * 16)) & 0xffffu);
      const unsigned short re = (unsigned short)ApplyBinary<MulF, kTag>(
          ApplyUnary<SiluF, kTag>(ge), ue);
      rv |= (unsigned long long)re << (e * 16);
    }
    out[idx] = rv;
  }
}

// ---------------------------------------------------------------------------
// Host launchers.
// ---------------------------------------------------------------------------

constexpr unsigned int kClustersPerLaunch = 64;
constexpr unsigned int kCoresPerCluster = 64;

template <typename T, template <typename> class Op>
void LaunchUnaryTag(int tag, void* out, const void* in,
                    const unsigned long long* out_shape,
                    const unsigned long long* in_shape,
                    const long long* out_strides, const long long* in_strides,
                    unsigned long long output_size, unsigned long long ndim,
                    bool out_c, bool in_c, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  if (tag == kTagF32) {
    UnaryEWKernel<float, kTagF32, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<float*>(out), static_cast<const float*>(in), out_shape,
            in_shape, out_strides, in_strides, output_size, ndim, out_c, in_c);
  } else if (tag == kTagF16) {
    UnaryEWKernel<unsigned short, kTagF16, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            reinterpret_cast<unsigned short*>(out),
            reinterpret_cast<const unsigned short*>(in), out_shape, in_shape,
            out_strides, in_strides, output_size, ndim, out_c, in_c);
  } else if (tag == kTagBF16) {
    UnaryEWKernel<unsigned short, kTagBF16, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            reinterpret_cast<unsigned short*>(out),
            reinterpret_cast<const unsigned short*>(in), out_shape, in_shape,
            out_strides, in_strides, output_size, ndim, out_c, in_c);
  } else if (tag == kTagI16) {
    UnaryEWKernel<short, kTagI16, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<short*>(out), static_cast<const short*>(in), out_shape,
            in_shape, out_strides, in_strides, output_size, ndim, out_c, in_c);
  } else if (tag == kTagI32) {
    UnaryEWKernel<int, kTagI32, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<int*>(out), static_cast<const int*>(in), out_shape,
            in_shape, out_strides, in_strides, output_size, ndim, out_c, in_c);
  } else if (tag == kTagI64) {
    UnaryEWKernel<long long, kTagI64, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<long long*>(out), static_cast<const long long*>(in),
            out_shape, in_shape, out_strides, in_strides, output_size, ndim,
            out_c, in_c);
  } else {
    // Unreachable: wrappers gate dtypes/paths before launching.
    return;
  }
}

template <template <typename> class Op>
void LaunchBinaryTag(int tag, void* out, const void* a, const void* b,
                     const unsigned long long* out_shape,
                     const unsigned long long* a_shape,
                     const unsigned long long* b_shape,
                     const long long* out_strides, const long long* a_strides,
                     const long long* b_strides, unsigned long long output_size,
                     unsigned long long ndim, bool out_c, bool a_c, bool b_c,
                     double alpha, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  // Contiguous fast path (decode hot shapes); strided/broadcast fall through
  // to the generic metadata-driven kernel below.
  if (out_c && a_c && b_c && output_size % 4 == 0) {
    const unsigned long long words = output_size / 4;
    if (tag == kTagBF16) {
      BinaryContigPackedKernel<kTagBF16, Op>
          <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
              reinterpret_cast<unsigned long long*>(out),
              reinterpret_cast<const unsigned long long*>(a),
              reinterpret_cast<const unsigned long long*>(b), words,
              static_cast<float>(alpha));
      return;
    }
    if (tag == kTagF16) {
      BinaryContigPackedKernel<kTagF16, Op>
          <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
              reinterpret_cast<unsigned long long*>(out),
              reinterpret_cast<const unsigned long long*>(a),
              reinterpret_cast<const unsigned long long*>(b), words,
              static_cast<float>(alpha));
      return;
    }
    if (tag == kTagF32) {
      BinaryContigPackedKernelF32<Op>
          <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
              static_cast<float*>(out), static_cast<const float*>(a),
              static_cast<const float*>(b), output_size,
              static_cast<float>(alpha));
      return;
    }
  }
  if (tag == kTagF32) {
    BinaryEWKernel<float, kTagF32, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<float*>(out), static_cast<const float*>(a),
            static_cast<const float*>(b), out_shape, a_shape, b_shape,
            out_strides, a_strides, b_strides, output_size, ndim, out_c, a_c,
            b_c, static_cast<float>(alpha));
  } else if (tag == kTagF16) {
    BinaryEWKernel<unsigned short, kTagF16, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            reinterpret_cast<unsigned short*>(out),
            reinterpret_cast<const unsigned short*>(a),
            reinterpret_cast<const unsigned short*>(b), out_shape, a_shape,
            b_shape, out_strides, a_strides, b_strides, output_size, ndim,
            out_c, a_c, b_c, static_cast<float>(alpha));
  } else if (tag == kTagBF16) {
    BinaryEWKernel<unsigned short, kTagBF16, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            reinterpret_cast<unsigned short*>(out),
            reinterpret_cast<const unsigned short*>(a),
            reinterpret_cast<const unsigned short*>(b), out_shape, a_shape,
            b_shape, out_strides, a_strides, b_strides, output_size, ndim,
            out_c, a_c, b_c, static_cast<float>(alpha));
  } else if (tag == kTagI16) {
    BinaryEWKernel<short, kTagI16, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<short*>(out), static_cast<const short*>(a),
            static_cast<const short*>(b), out_shape, a_shape, b_shape,
            out_strides, a_strides, b_strides, output_size, ndim, out_c, a_c,
            b_c, static_cast<float>(alpha));
  } else if (tag == kTagI32) {
    BinaryEWKernel<int, kTagI32, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<int*>(out), static_cast<const int*>(a),
            static_cast<const int*>(b), out_shape, a_shape, b_shape,
            out_strides, a_strides, b_strides, output_size, ndim, out_c, a_c,
            b_c, static_cast<int>(alpha));
  } else if (tag == kTagI64) {
    BinaryEWKernel<long long, kTagI64, Op>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<long long*>(out), static_cast<const long long*>(a),
            static_cast<const long long*>(b), out_shape, a_shape, b_shape,
            out_strides, a_strides, b_strides, output_size, ndim, out_c, a_c,
            b_c, static_cast<long long>(alpha));
  } else {
    // Unreachable: wrappers gate dtypes/paths before launching.
    return;
  }
}

}  // namespace

namespace kunlun_launch {

void LaunchRelu(int tag, void* out, const void* in,
                const unsigned long long* out_shape,
                const unsigned long long* in_shape,
                const long long* out_strides, const long long* in_strides,
                unsigned long long output_size, unsigned long long ndim,
                bool out_c, bool in_c, void* stream) {
  LaunchUnaryTag<float, ReluF>(tag, out, in, out_shape, in_shape, out_strides,
                               in_strides, output_size, ndim, out_c, in_c,
                               stream);
}

void LaunchGelu(int tag, void* out, const void* in,
                const unsigned long long* out_shape,
                const unsigned long long* in_shape,
                const long long* out_strides, const long long* in_strides,
                unsigned long long output_size, unsigned long long ndim,
                bool out_c, bool in_c, int approximate_tanh, void* stream) {
  if (approximate_tanh) {
    LaunchUnaryTag<float, GeluTanhF>(tag, out, in, out_shape, in_shape,
                                     out_strides, in_strides, output_size, ndim,
                                     out_c, in_c, stream);
  } else {
    LaunchUnaryTag<float, GeluF>(tag, out, in, out_shape, in_shape, out_strides,
                                 in_strides, output_size, ndim, out_c, in_c,
                                 stream);
  }
}

void LaunchSilu(int tag, void* out, const void* in,
                const unsigned long long* out_shape,
                const unsigned long long* in_shape,
                const long long* out_strides, const long long* in_strides,
                unsigned long long output_size, unsigned long long ndim,
                bool out_c, bool in_c, void* stream) {
  LaunchUnaryTag<float, SiluF>(tag, out, in, out_shape, in_shape, out_strides,
                               in_strides, output_size, ndim, out_c, in_c,
                               stream);
}

void LaunchAdd(int tag, void* out, const void* a, const void* b,
               const unsigned long long* out_shape,
               const unsigned long long* a_shape,
               const unsigned long long* b_shape, const long long* out_strides,
               const long long* a_strides, const long long* b_strides,
               unsigned long long output_size, unsigned long long ndim,
               bool out_c, bool a_c, bool b_c, double alpha, void* stream) {
  LaunchBinaryTag<AddF>(tag, out, a, b, out_shape, a_shape, b_shape,
                        out_strides, a_strides, b_strides, output_size, ndim,
                        out_c, a_c, b_c, alpha, stream);
}

void LaunchSub(int tag, void* out, const void* a, const void* b,
               const unsigned long long* out_shape,
               const unsigned long long* a_shape,
               const unsigned long long* b_shape, const long long* out_strides,
               const long long* a_strides, const long long* b_strides,
               unsigned long long output_size, unsigned long long ndim,
               bool out_c, bool a_c, bool b_c, void* stream) {
  LaunchBinaryTag<SubF>(tag, out, a, b, out_shape, a_shape, b_shape,
                        out_strides, a_strides, b_strides, output_size, ndim,
                        out_c, a_c, b_c, 1.0, stream);
}

void LaunchMul(int tag, void* out, const void* a, const void* b,
               const unsigned long long* out_shape,
               const unsigned long long* a_shape,
               const unsigned long long* b_shape, const long long* out_strides,
               const long long* a_strides, const long long* b_strides,
               unsigned long long output_size, unsigned long long ndim,
               bool out_c, bool a_c, bool b_c, void* stream) {
  LaunchBinaryTag<MulF>(tag, out, a, b, out_shape, a_shape, b_shape,
                        out_strides, a_strides, b_strides, output_size, ndim,
                        out_c, a_c, b_c, 1.0, stream);
}

void LaunchSiluAndMul(int tag, void* out, const void* in,
                      unsigned long long rows, unsigned long long cols,
                      long long in_row_stride, long long out_row_stride,
                      bool in_c, bool out_c, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  if (in_c && out_c && (cols & 3) == 0 && (tag == kTagBF16 || tag == kTagF16)) {
    if (tag == kTagBF16) {
      SiluAndMulPackedKernel<kTagBF16>
          <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
              reinterpret_cast<unsigned long long*>(out),
              reinterpret_cast<const unsigned long long*>(in), rows, cols / 4);
    } else {
      SiluAndMulPackedKernel<kTagF16>
          <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
              reinterpret_cast<unsigned long long*>(out),
              reinterpret_cast<const unsigned long long*>(in), rows, cols / 4);
    }
    return;
  }
  if (tag == kTagF32) {
    SiluAndMulKernel<float, kTagF32>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<float*>(out), static_cast<const float*>(in), rows, cols,
            in_row_stride, out_row_stride, in_c, out_c);
  } else if (tag == kTagF16) {
    SiluAndMulKernel<unsigned short, kTagF16>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            reinterpret_cast<unsigned short*>(out),
            reinterpret_cast<const unsigned short*>(in), rows, cols,
            in_row_stride, out_row_stride, in_c, out_c);
  } else if (tag == kTagBF16) {
    SiluAndMulKernel<unsigned short, kTagBF16>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            reinterpret_cast<unsigned short*>(out),
            reinterpret_cast<const unsigned short*>(in), rows, cols,
            in_row_stride, out_row_stride, in_c, out_c);
  } else {
    // Unreachable: wrappers gate dtypes/paths before launching.
    return;
  }
}

// Full n-d strided variant: idx decomposed by out_shape (right-aligned).
// in_strides refer to the input element grid; the caller passes strides
// computed for a [gate,up] pair grid where up sits at gate + cols*gate_stride.
template <typename T, int kTag>
__global__ void SiluAndMulKernelNd(T* out, const T* in,
                                   const unsigned long long* out_shape,
                                   const long long* out_strides,
                                   const long long* in_strides,
                                   unsigned long long ndim,
                                   unsigned long long cols) {
  const std::size_t thread_id = cluster_id() * 64 + core_id();
  const std::size_t nthreads = cluster_num() * 64;
  std::size_t total = 1;
  for (std::size_t i = 0; i < ndim; ++i) total *= out_shape[i];
  // Decompose idx by the OUTPUT's outer dims (ndim-1 dims); the last dim
  // (cols) is folded into `inner`. The input's layout: for outer coords the
  // caller passes in_strides[0..ndim-2] (in element units); within a row the
  // gate sits at inner*2 and up at inner*2+cols (pairs packed, stride 1).
  // When the last input dim is NOT packed (caller guarantees packed pairs,
  // see wrapper), the gate stride within the pair is 1.
  for (std::size_t idx = thread_id; idx < total; idx += nthreads) {
    std::size_t out_off = 0;
    std::size_t in_off = 0;
    std::size_t rem = idx / cols;
    for (std::size_t i = ndim - 1; i-- > 0;) {
      const std::size_t coord = rem % out_shape[i];
      rem /= out_shape[i];
      out_off += coord * static_cast<std::size_t>(out_strides[i]);
      in_off += coord * static_cast<std::size_t>(in_strides[i]);
    }
    const std::size_t inner = idx % cols;
    out_off += inner * static_cast<std::size_t>(out_strides[ndim - 1]);
    // gate/up pair packed in the input's last dim (caller guarantees the
    // pair stride is 1): gate at inner, up at inner + cols — matching
    // torch's input[..., :cols] / input[..., cols:] views.
    in_off += inner;
    const T g = in[in_off];
    const T u = in[in_off + cols];
    out[out_off] = ApplyBinary<MulF, kTag>(ApplyUnary<SiluF, kTag>(g), u);
  }
}

void LaunchSiluAndMulNd(int tag, void* out, const void* in,
                        const unsigned long long* out_shape,
                        const long long* out_strides,
                        const long long* in_strides, unsigned long long ndim,
                        unsigned long long cols, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  if (tag == kTagF32) {
    SiluAndMulKernelNd<float, kTagF32>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            static_cast<float*>(out), static_cast<const float*>(in), out_shape,
            out_strides, in_strides, ndim, cols);
  } else if (tag == kTagF16) {
    SiluAndMulKernelNd<unsigned short, kTagF16>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            reinterpret_cast<unsigned short*>(out),
            reinterpret_cast<const unsigned short*>(in), out_shape, out_strides,
            in_strides, ndim, cols);
  } else if (tag == kTagBF16) {
    SiluAndMulKernelNd<unsigned short, kTagBF16>
        <<<kClustersPerLaunch, kCoresPerCluster, s>>>(
            reinterpret_cast<unsigned short*>(out),
            reinterpret_cast<const unsigned short*>(in), out_shape, out_strides,
            in_strides, ndim, cols);
  } else {
    // Unreachable: wrappers gate dtypes/paths before launching.
    return;
  }
}

}  // namespace kunlun_launch

}  // namespace infini::ops
// Kunlun XPU device kernels for the Qwen2-style inference path (batch 2):
// Sigmoid, RmsNorm, Softmax, Embedding, Copy, Cast, Fill, Triu/Tril, Argmax,
// SeparateTableRope.
//
// Conventions:
//   - fp16/bf16 flow as `unsigned short` payloads; compute happens in float
//     through the Dt bit-conversion traits.
//   - Launch syntax: <<<clusters, cores_per_cluster, stream>>>.
//   - XTDK device code has a 32-bit size_t; metadata crossing into device
//     code is carried explicitly (int by value, long long for strides).
//   - Row-reduction kernels read global memory directly, so any row length
//     works.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// clang-format off
#include <xpu/runtime.h>
#include <xpu/kernel/xtdk.h>
#include <xpu/kernel/xtdk_atomic_sm_xpu3.h>
#include <xpu/kernel/xtdk_bf16.h>
#include <xpu/kernel/xtdk_fp16.h>
#include <xpu/kernel/xtdk_math.h>
#include <xpu/kernel/xtdk_simd.h>
#include <xpu/kernel/xtdk_trigonometric.h>
// clang-format on

namespace infini::ops {
namespace kl2 {

constexpr int kTagF32 = 0;
constexpr int kTagF16 = 1;
constexpr int kTagBF16 = 2;
constexpr int kTagI16 = 5;
constexpr int kTagI32 = 3;
constexpr int kTagI64 = 4;
constexpr unsigned int kClusters = 64;
constexpr unsigned int kCores = 64;

__device__ __forceinline__ float F16ToFloat(unsigned short bits) {
  __half_raw hr;
  hr.x = bits;
  return __half2float(__half(hr));
}

__device__ __forceinline__ unsigned short FloatToF16Bits(float f) {
  __half h = __float2half(f);
  __half_raw hr = static_cast<__half_raw>(h);
  return hr.x;
}

__device__ __forceinline__ float BF16ToFloat(unsigned short bits) {
  __xpu_bfloat16_raw br;
  br.x = bits;
  return static_cast<float>(__xpu_bfloat16(br));
}

__device__ __forceinline__ unsigned short FloatToBF16Bits(float f) {
  __xpu_bfloat16 h = __float2bfloat16(f);
  __xpu_bfloat16_raw br = static_cast<__xpu_bfloat16_raw>(h);
  return br.x;
}

template <int kTag>
struct Dt;

template <>
struct Dt<kTagF32> {
  using Storage = float;
  static __device__ __forceinline__ float ToFloat(Storage v) { return v; }
  static __device__ __forceinline__ Storage FromFloat(float f) { return f; }
};

template <>
struct Dt<kTagF16> {
  using Storage = unsigned short;
  static __device__ __forceinline__ float ToFloat(Storage v) {
    return F16ToFloat(v);
  }
  static __device__ __forceinline__ Storage FromFloat(float f) {
    return FloatToF16Bits(f);
  }
};

template <>
struct Dt<kTagBF16> {
  using Storage = unsigned short;
  static __device__ __forceinline__ float ToFloat(Storage v) {
    return BF16ToFloat(v);
  }
  static __device__ __forceinline__ Storage FromFloat(float f) {
    return FloatToBF16Bits(f);
  }
};

// Host-side bit converters (mirror the device ones; used by LaunchFill2).
inline unsigned short FloatToF16BitsHost(float f) {
  // __half host conversion via XTDK fp16 header is device-only; do the
  // conversion in software (round-to-nearest-even), same as __float2half.
  unsigned int bits;
  std::memcpy(&bits, &f, 4);
  const unsigned int sign = (bits >> 16) & 0x8000;
  int exponent = (int)((bits >> 23) & 0xFF) - 127;
  unsigned int mantissa = bits & 0x7FFFFF;
  if (exponent >= 16) {
    return (unsigned short)(sign | 0x7C00 |
                            ((exponent == 128 && mantissa) ? 0x200 : 0));
  }
  if (exponent >= -14) {
    unsigned int m = mantissa + 0x1000;  // round to nearest
    if (m & 0x800000) {                  // mantissa overflow
      m >>= 1;
      ++exponent;
      if (exponent >= 16) return (unsigned short)(sign | 0x7C00);
    }
    return (unsigned short)(sign | ((exponent + 15) << 10) | (m >> 13));
  }
  if (exponent >= -25) {
    unsigned int m = (mantissa | 0x800000) >> (-14 - exponent);
    m = (m + 0xFFF) >> 13;  // subnormal rounding
    return (unsigned short)(sign | (m < 0x400 ? m : 0x3FF));
  }
  return (unsigned short)sign;
}

inline unsigned short FloatToBF16BitsHost(float f) {
  unsigned int bits;
  std::memcpy(&bits, &f, 4);
  return (unsigned short)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// 32-bit shape/stride metadata, passed to kernels by value. Strides are
// element strides; offsets accumulate in unsigned long long.
struct ShapeMeta {
  int ndim;
  int rows;  // last-two-dims row count for triangular ops
  unsigned long long total;
  int shape[8];
  int strides[8];
  int out_strides[8];
};

__device__ __forceinline__ unsigned long long FlatToOffset(
    unsigned long long flat, int ndim, const int* shape, const int* strides) {
  unsigned long long off = 0;
  for (int i = ndim; i-- > 0;) {
    off +=
        (flat % (unsigned long long)shape[i]) * (unsigned long long)strides[i];
    flat /= (unsigned long long)shape[i];
  }
  return off;
}

// ===========================================================================
// 1. Sigmoid (flat).
// ===========================================================================

template <int kTag>
__global__ void SigmoidKernel(typename Dt<kTag>::Storage* out,
                              const typename Dt<kTag>::Storage* in,
                              unsigned long long total) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < total; i += nth) {
    float x = Dt<kTag>::ToFloat(in[i]);
    out[i] = Dt<kTag>::FromFloat(1.0f / (1.0f + exp(-x)));
  }
}

// ===========================================================================
// 2. RmsNorm: one cluster per row; row data read straight from global memory.
//    grid = rows; each cluster handles one row of length `dim`.
// ===========================================================================

template <int kTag>
__global__ void RmsNormKernel(typename Dt<kTag>::Storage* out,
                              const typename Dt<kTag>::Storage* in,
                              const typename Dt<kTag>::Storage* w, int dim,
                              float eps, long long in_row_stride) {
  const int row = cluster_id();
  const unsigned long long x_base =
      (unsigned long long)row * (unsigned long long)in_row_stride;
  const unsigned long long y_base =
      (unsigned long long)row * (unsigned long long)dim;

  float local = 0.0f;
  for (int i = core_id(); i < dim; i += kCores) {
    float xi = Dt<kTag>::ToFloat(in[x_base + i]);
    local += xi * xi;
  }
  __shared__ float red[kCores];
  red[core_id()] = local;
  sync_cluster();
  if (core_id() == 0) {
    float ss = 0.0f;
    for (int i = 0; i < kCores; ++i) ss += red[i];
    red[0] = rsqrt(ss / (float)dim + eps);
  }
  sync_cluster();
  const float rms = red[0];

  for (int i = core_id(); i < dim; i += kCores) {
    float xi = Dt<kTag>::ToFloat(in[x_base + i]);
    float wi = Dt<kTag>::ToFloat(w[i]);
    out[y_base + i] = Dt<kTag>::FromFloat(xi * wi * rms);
  }
  sync_cluster();
}

// ===========================================================================
// 2b. FusedAddRmsNorm: one cluster per row.
//     residual = input + residual (in place); input = norm(residual) * w.
//     input row stride may differ from `dim` (fused-QKV split views).
// ===========================================================================

template <int kTag>
__global__ void FusedAddRmsNormKernel(typename Dt<kTag>::Storage* in_out,
                                      typename Dt<kTag>::Storage* residual,
                                      const typename Dt<kTag>::Storage* w,
                                      int dim, float eps,
                                      long long in_row_stride,
                                      long long residual_row_stride) {
  const int row = cluster_id();
  const unsigned long long x_base =
      (unsigned long long)row * (unsigned long long)in_row_stride;
  const unsigned long long r_base =
      (unsigned long long)row * (unsigned long long)residual_row_stride;

  // Pass 1 only accumulates the sum of squares — `in_out` and `residual`
  // stay untouched, so pass 2 never reads memory written by another core
  // (global read-after-write across cores within a cluster is not safely
  // visible with mfence+sync_cluster alone).
  float local = 0.0f;
  for (int i = core_id(); i < dim; i += kCores) {
    float xi = Dt<kTag>::ToFloat(in_out[x_base + i]);
    float ri = Dt<kTag>::ToFloat(residual[r_base + i]);
    local += (xi + ri) * (xi + ri);
  }
  __shared__ float red[kCores];
  red[core_id()] = local;
  sync_cluster();
  if (core_id() == 0) {
    float ss = 0.0f;
    for (int i = 0; i < kCores; ++i) ss += red[i];
    red[0] = rsqrt(ss / (float)dim + eps);
  }
  sync_cluster();
  const float rms = red[0];
  for (int i = core_id(); i < dim; i += kCores) {
    float xi = Dt<kTag>::ToFloat(in_out[x_base + i]);
    float ri = Dt<kTag>::ToFloat(residual[r_base + i]);
    const float sum = xi + ri;
    residual[r_base + i] = Dt<kTag>::FromFloat(sum);
    in_out[x_base + i] =
        Dt<kTag>::FromFloat(sum * Dt<kTag>::ToFloat(w[i]) * rms);
  }
  sync_cluster();
}

template <int kTag>
__global__ void FusedAddRmsNormPackedKernel(unsigned long long* in_out,
                                            unsigned long long* residual,
                                            const unsigned long long* w,
                                            int words, float eps,
                                            long long in_row_wstride,
                                            long long residual_row_wstride) {
  // Word-typed twin of FusedAddRmsNormKernel: rows of dim/4 64-bit words
  // (4 bf16/f16 elems each). Scalar u16 loops run ~5x slower for the same
  // bytes on XPU. Same two-pass structure: pass 1 only accumulates.
  const int row = cluster_id();
  const unsigned long long x_base =
      (unsigned long long)row * (unsigned long long)in_row_wstride;
  const unsigned long long r_base =
      (unsigned long long)row * (unsigned long long)residual_row_wstride;
  float local = 0.0f;
  for (int i = core_id(); i < words; i += kCores) {
    const unsigned long long iw = in_out[x_base + i];
    const unsigned long long rw = residual[r_base + i];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const unsigned short x16 = (unsigned short)((iw >> (e * 16)) & 0xffffu);
      const unsigned short r16 = (unsigned short)((rw >> (e * 16)) & 0xffffu);
      const float xi = Dt<kTag>::ToFloat(x16);
      const float ri = Dt<kTag>::ToFloat(r16);
      local += (xi + ri) * (xi + ri);
    }
  }
  __shared__ float red[kCores];
  red[core_id()] = local;
  sync_cluster();
  if (core_id() == 0) {
    float ss = 0.0f;
    for (int i = 0; i < kCores; ++i) ss += red[i];
    red[0] = rsqrt(ss / (float)(words * 4) + eps);
  }
  sync_cluster();
  const float rms = red[0];
  for (int i = core_id(); i < words; i += kCores) {
    const unsigned long long iw = in_out[x_base + i];
    const unsigned long long rw = residual[r_base + i];
    const unsigned long long ww = w[i];
    unsigned long long out_r = 0;
    unsigned long long out_x = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const unsigned short x16 = (unsigned short)((iw >> (e * 16)) & 0xffffu);
      const unsigned short r16 = (unsigned short)((rw >> (e * 16)) & 0xffffu);
      const unsigned short w16 = (unsigned short)((ww >> (e * 16)) & 0xffffu);
      const float xi = Dt<kTag>::ToFloat(x16);
      const float ri = Dt<kTag>::ToFloat(r16);
      const float sum = xi + ri;
      out_r |= (unsigned long long)(unsigned short)Dt<kTag>::FromFloat(sum)
               << (e * 16);
      out_x |= (unsigned long long)(unsigned short)Dt<kTag>::FromFloat(
                   sum * Dt<kTag>::ToFloat(w16) * rms)
               << (e * 16);
    }
    in_out[x_base + i] = out_x;
    residual[r_base + i] = out_r;
  }
  sync_cluster();
}

// ===========================================================================
// 3. Softmax over the last dimension: one cluster per row, three passes over
//    global memory (max, exp+sum, normalize). Any row length.
// ===========================================================================

template <int kTag>
__global__ void SoftmaxKernel(typename Dt<kTag>::Storage* out,
                              const typename Dt<kTag>::Storage* in, int dim,
                              long long in_row_stride,
                              long long out_row_stride) {
  const int row = cluster_id();
  const unsigned long long x_base =
      (unsigned long long)row * (unsigned long long)in_row_stride;
  const unsigned long long y_base =
      (unsigned long long)row * (unsigned long long)out_row_stride;

  float local_max = -3.0e38f;
  for (int i = core_id(); i < dim; i += kCores) {
    local_max = fmax(local_max, Dt<kTag>::ToFloat(in[x_base + i]));
  }
  __shared__ float red[kCores];
  red[core_id()] = local_max;
  sync_cluster();
  if (core_id() == 0) {
    float m = red[0];
    for (int i = 1; i < kCores; ++i) m = fmax(m, red[i]);
    red[0] = m;
  }
  sync_cluster();
  const float row_max = red[0];

  float local_sum = 0.0f;
  for (int i = core_id(); i < dim; i += kCores) {
    local_sum += exp(Dt<kTag>::ToFloat(in[x_base + i]) - row_max);
  }
  red[core_id()] = local_sum;
  sync_cluster();
  if (core_id() == 0) {
    float s = 0.0f;
    for (int i = 0; i < kCores; ++i) s += red[i];
    red[0] = s;
  }
  sync_cluster();
  const float row_sum = red[0];

  for (int i = core_id(); i < dim; i += kCores) {
    float e = exp(Dt<kTag>::ToFloat(in[x_base + i]) - row_max);
    out[y_base + i] = Dt<kTag>::FromFloat(e / row_sum);
  }
  sync_cluster();
}

// ===========================================================================
// 4. Embedding: flat row gather.
// ===========================================================================

template <int kTag, typename IndexT>
__global__ void EmbeddingKernel(typename Dt<kTag>::Storage* out,
                                const IndexT* indices,
                                const typename Dt<kTag>::Storage* w,
                                unsigned long long total, int emb_dim,
                                unsigned long long vocab) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < total; i += nth) {
    unsigned long long token_pos = i / (unsigned long long)emb_dim;
    unsigned long long col = i - token_pos * (unsigned long long)emb_dim;
    IndexT token = indices[token_pos];
    if ((long long)token >= 0 && (unsigned long long)token < vocab) {
      out[i] = w[(unsigned long long)token * (unsigned long long)emb_dim + col];
    }
  }
}

// ===========================================================================
// 5. Copy: broadcast-aware, same dtype, 2- or 4-byte lanes.
// ===========================================================================

template <typename T>
__global__ void CopyKernel(T* out, const T* in, ShapeMeta meta) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < meta.total; i += nth) {
    unsigned long long in_off =
        FlatToOffset(i, meta.ndim, meta.shape, meta.strides);
    unsigned long long out_off =
        FlatToOffset(i, meta.ndim, meta.shape, meta.out_strides);
    out[out_off] = in[in_off];
  }
}

// ===========================================================================
// 5b. NarrowI64ToI32: slot-mapping narrowing on-device (decode path keeps
// this off the host round trip).
// ===========================================================================

__global__ void NarrowI64ToI32Kernel(int* out, const long long* in,
                                     unsigned long long total) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < total; i += nth) {
    out[i] = static_cast<int>(in[i]);
  }
}

// ===========================================================================
// 6. Cast: 16-bit <-> 16-bit (any direction), plus f32 endpoints. All flat.
// ===========================================================================

template <int kTagIn, int kTagOut>
__global__ void CastKernel16(unsigned short* out, const unsigned short* in,
                             unsigned long long total) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < total; i += nth) {
    float v = Dt<kTagIn>::ToFloat(in[i]);
    out[i] = Dt<kTagOut>::FromFloat(v);
  }
}

template <int kTagOut>
__global__ void CastFromF32Kernel(unsigned short* out, const float* in,
                                  unsigned long long total) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < total; i += nth) {
    out[i] = Dt<kTagOut>::FromFloat(in[i]);
  }
}

template <int kTagIn>
__global__ void CastToF32Kernel(float* out, const unsigned short* in,
                                unsigned long long total) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < total; i += nth) {
    out[i] = Dt<kTagIn>::ToFloat(in[i]);
  }
}

// ===========================================================================
// 6b. Scale: flat y *= alpha (f32 lanes via 4-byte payload, 16-bit via tag).
// ===========================================================================

template <typename T>
__global__ void ScaleKernel(T* y, unsigned long long total, T alpha) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < total; i += nth) {
    y[i] = static_cast<T>(static_cast<float>(y[i]) * static_cast<float>(alpha));
  }
}

// ===========================================================================
// 7. Fill: broadcast-aware scalar fill.
// ===========================================================================

template <typename T>
__global__ void FillKernel(T* out, ShapeMeta meta, T value) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  for (unsigned long long i = tid; i < meta.total; i += nth) {
    unsigned long long off =
        FlatToOffset(i, meta.ndim, meta.shape, meta.out_strides);
    out[off] = value;
  }
}

// ===========================================================================
// 8. Triu/Tril: mask over the last two dims; upper keeps col-row >= d.
// ===========================================================================

template <typename T, bool kUpper>
__global__ void TriangularKernel(T* out, const T* in, ShapeMeta meta,
                                 int diagonal) {
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;
  const int rows = meta.rows;
  const unsigned long long cols = (unsigned long long)meta.shape[meta.ndim - 1];
  for (unsigned long long i = tid; i < meta.total; i += nth) {
    unsigned long long in_off =
        FlatToOffset(i, meta.ndim, meta.shape, meta.strides);
    unsigned long long out_off =
        FlatToOffset(i, meta.ndim, meta.shape, meta.out_strides);
    const int col = (int)(i % cols);
    const int row = (int)((i / cols) % (unsigned long long)rows);
    const long long rel = (long long)col - (long long)row;
    const bool keep =
        kUpper ? (rel >= (long long)diagonal) : (rel <= (long long)diagonal);
    out[out_off] = keep ? in[in_off] : (T)0;
  }
}

// ===========================================================================
// 9. Argmax over the last dim: one cluster per row -> int64. Ties resolve to
//    the smallest index (PyTorch semantics).
// ===========================================================================

template <int kTag>
__global__ void ArgmaxKernel(long long* out,
                             const typename Dt<kTag>::Storage* in, int dim,
                             long long row_stride, unsigned long long rows) {
  const int row = cluster_id();
  if ((unsigned long long)row >= rows) return;
  const unsigned long long x_base =
      (unsigned long long)row * (unsigned long long)row_stride;

  float local_best = -3.0e38f;
  int local_idx = 0;
  for (int i = core_id(); i < dim; i += kCores) {
    float v = Dt<kTag>::ToFloat(in[x_base + i]);
    if (v > local_best) {
      local_best = v;
      local_idx = i;
    }
  }
  __shared__ float best_v[kCores];
  __shared__ int best_i[kCores];
  best_v[core_id()] = local_best;
  best_i[core_id()] = local_idx;
  sync_cluster();
  if (core_id() == 0) {
    float bv = best_v[0];
    int bi = best_i[0];
    for (int i = 1; i < kCores; ++i) {
      // Strict > keeps the smallest index on ties.
      if (best_v[i] > bv) {
        bv = best_v[i];
        bi = best_i[i];
      }
    }
    out[row] = (long long)bi;
  }
  sync_cluster();
}

// ===========================================================================
// 10. SeparateTableRope. Layout mirrors the CPU reference:
//   x/y rank 3: [seq, heads, 2*half]; rank 4: [batch, seq, heads, 2*half]
//   positions: [seq] or [batch, seq] (int64, contiguous)
//   tables:    [table_rows, half] (row stride == half)
// Flat thread index t decodes (b, s, h); each thread walks the half pairs.
// The host guarantees batched_input == batched_pos for rank 4 (it flattens
// otherwise); batch > 1 implies both are batched.
// ===========================================================================

// Batch component of the flat offset; 0 when the tensor is rank 3.
__device__ __forceinline__ unsigned long long batched_input_offset_(
    unsigned long long b, bool batched, long long stride_batch) {
  return batched ? b * (unsigned long long)stride_batch : 0ULL;
}

// ===========================================================================
// Causal prefill attention over a paged KV cache with GQA, one fused kernel.
// Replaces the vendor `flash_attention_context_vllm` (reads only KV-cache
// head 0 — GQA unsupported — and deadlocks/overruns at multi-head shapes).
// Payload conventions as other kernels: bf16/fp16 as unsigned short with
// explicit bit conversion; fp32 accumulation.
//
//   q:   [tokens, H, D], token stride q_tok_stride, head stride D
//   kc:  [blocks, PAGE, KVH, D] contiguous; block bt[b*bt_stride + k/PAGE]
//   out: [tokens, H, D]
//   qlod/kvlod: int32 [batch+1] device prefix sums
// One cluster = one (token, head) pair; cores split over keys (passes 1-2)
// then over dims (pass 3, sequential keys — reuses per-key scores? recomputed).
// ===========================================================================

template <int kTag, bool kInterleaved>
__global__ void RopeKernel(
    typename Dt<kTag>::Storage* y, const typename Dt<kTag>::Storage* x,
    const long long* pos, const typename Dt<kTag>::Storage* sin_t,
    const typename Dt<kTag>::Storage* cos_t, int half, long long x_stride_seq,
    long long x_stride_head, long long y_stride_seq, long long y_stride_head,
    unsigned long long seq, unsigned long long heads, unsigned long long batch,
    long long x_stride_batch, long long y_stride_batch, bool batched_pos) {
  const unsigned long long pairs = seq * heads;
  const unsigned long long total = batch * pairs;
  const unsigned long long tid =
      (unsigned long long)cluster_id() * kCores + core_id();
  const unsigned long long nth = (unsigned long long)cluster_num() * kCores;

  for (unsigned long long t = tid; t < total; t += nth) {
    const unsigned long long h = t % heads;
    const unsigned long long s = (t / heads) % seq;
    const unsigned long long b = t / pairs;

    const long long p =
        pos[batched_pos ? (long long)(b * seq + s) : (long long)s];
    const unsigned long long table_row = (unsigned long long)p;

    const unsigned long long x_base =
        batched_input_offset_(b, batched_pos, x_stride_batch) +
        s * (unsigned long long)x_stride_seq +
        h * (unsigned long long)x_stride_head;
    const unsigned long long y_base =
        batched_input_offset_(b, batched_pos, y_stride_batch) +
        s * (unsigned long long)y_stride_seq +
        h * (unsigned long long)y_stride_head;

    for (int i = 0; i < half; ++i) {
      const float s_v =
          Dt<kTag>::ToFloat(sin_t[table_row * (unsigned long long)half + i]);
      const float c_v =
          Dt<kTag>::ToFloat(cos_t[table_row * (unsigned long long)half + i]);
      const int a = kInterleaved ? 2 * i : i;
      const int z = kInterleaved ? 2 * i + 1 : i + half;
      // Both pair members read before either write (in-place safe).
      const float x0 = Dt<kTag>::ToFloat(x[x_base + a]);
      const float x1 = Dt<kTag>::ToFloat(x[x_base + z]);
      y[y_base + a] = Dt<kTag>::FromFloat(x0 * c_v - x1 * s_v);
      y[y_base + z] = Dt<kTag>::FromFloat(x0 * s_v + x1 * c_v);
    }
  }
}

template <int kTag>
__global__ void GatherKvKernel(typename Dt<kTag>::Storage* out,
                               const typename Dt<kTag>::Storage* cache,
                               const int* kvlod, const int* bt, int kv_heads,
                               int dim, int page_size, int bt_stride, int batch,
                               int kv_tokens) {
  // One cluster owns one (token, kv_head) row; the cluster's cores split
  // the dim columns. Row selection must not depend on core_id — a core
  // is a lane of the row's copy, not a separate row worker.
  // kv_tokens arrives as a host scalar: a device-side kvlod[batch] load
  // proved unreliable under the launcher's cluster fan-out.
  const unsigned long long total_rows =
      (unsigned long long)kv_tokens * (unsigned long long)kv_heads;
  for (unsigned long long t = cluster_id(); t < total_rows;
       t += cluster_num()) {
    const unsigned long long token = t / (unsigned long long)kv_heads;
    const unsigned long long g = t % (unsigned long long)kv_heads;
    int bi = 0;
    while (bi < batch - 1 && kvlod[bi + 1] <= (int)token) ++bi;
    const int local = (int)token - kvlod[bi];
    const int blk = bt[bi * bt_stride + local / page_size];
    const unsigned long long src =
        ((unsigned long long)blk * page_size + (local % page_size)) *
            (unsigned long long)kv_heads * (unsigned long long)dim +
        g * (unsigned long long)dim;
    const unsigned long long dst =
        token * (unsigned long long)kv_heads * (unsigned long long)dim +
        g * (unsigned long long)dim;
    for (int d = core_id(); d < dim; d += kCores) {
      out[dst + d] = cache[src + d];
    }
  }
}

// TransposeCacheBlocksKernel: permute the in-block axes of the paged KV
// cache from the InfiniOps/FA token-major physical layout
// `[num_blocks, page_size, kv_heads, dim]` to the head-major layout the
// vendor decode kernel (`paged_attention_xft`) addresses:
// `[num_blocks, kv_heads, page_size, dim]`. One cluster handles one
// (block, kv_head) row: it copies that head-slice of every page, cores
// splitting over (page, dim). A full cache transpose would be prohibitive,
// so the caller transposes only the resident-window blocks on a scratch
// buffer per decode step.
template <int kTag>
__global__ void TransposeCacheBlocksKernel(typename Dt<kTag>::Storage* out,
                                           const typename Dt<kTag>::Storage* in,
                                           int num_blocks, int page_size,
                                           int kv_heads, int dim) {
  // One cluster owns one (block, kv_head) row; the cluster's cores split
  // the (page, dim) span. Row selection must not depend on core_id.
  const unsigned long long rows =
      (unsigned long long)num_blocks * (unsigned long long)kv_heads;
  for (unsigned long long t = cluster_id(); t < rows; t += cluster_num()) {
    const unsigned long long blk = t / (unsigned long long)kv_heads;
    const unsigned long long g = t % (unsigned long long)kv_heads;
    // in:  in[((blk*page_size + p)*kv_heads + g)*dim + d]
    // out: out[((blk*kv_heads + g)*page_size + p)*dim + d]
    const unsigned long long in_base =
        (blk * (unsigned long long)page_size * (unsigned long long)kv_heads +
         g) *
        (unsigned long long)dim;
    const unsigned long long out_base =
        ((blk * (unsigned long long)kv_heads + g) *
         (unsigned long long)page_size) *
        (unsigned long long)dim;
    const unsigned long long span =
        (unsigned long long)page_size * (unsigned long long)dim;
    for (unsigned long long i = core_id(); i < span; i += kCores) {
      const unsigned long long p = i / (unsigned long long)dim;
      const unsigned long long d = i - p * (unsigned long long)dim;
      out[out_base + i] =
          in[in_base +
             p * (unsigned long long)kv_heads * (unsigned long long)dim + d];
    }
  }
}

// Transpose the paged KV cache in-block axes AND broadcast kv_heads to
// q_heads (GQA): out[NB, q_heads, PS, D] where head h reads group
// h / (q_heads/kv_heads). One cluster per (block, out-head) row.
template <int kTag>
__global__ void TransposeBroadcastKernel(typename Dt<kTag>::Storage* out,
                                         const typename Dt<kTag>::Storage* in,
                                         int num_blocks, int page_size,
                                         int kv_heads, int q_heads, int dim) {
  const int group = q_heads / kv_heads;
  const unsigned long long rows =
      (unsigned long long)num_blocks * (unsigned long long)q_heads;
  for (unsigned long long t = cluster_id(); t < rows; t += cluster_num()) {
    const unsigned long long blk = t / (unsigned long long)q_heads;
    const unsigned long long h = t % (unsigned long long)q_heads;
    const unsigned long long g = h / (unsigned long long)group;
    const unsigned long long in_base =
        (blk * (unsigned long long)page_size * (unsigned long long)kv_heads +
         g) *
        (unsigned long long)dim;
    const unsigned long long out_base =
        ((blk * (unsigned long long)q_heads + h) *
         (unsigned long long)page_size) *
        (unsigned long long)dim;
    const unsigned long long span =
        (unsigned long long)page_size * (unsigned long long)dim;
    for (unsigned long long i2 = 0; i2 < span; ++i2) {
      const unsigned long long p = i2 / (unsigned long long)dim;
      const unsigned long long d = i2 - p * (unsigned long long)dim;
      out[out_base + i2] =
          in[in_base +
             p * (unsigned long long)kv_heads * (unsigned long long)dim + d];
    }
  }
}
}  // namespace kl2

// ===========================================================================
// Host launchers.
// ===========================================================================

namespace kunlun_launch {

using kl2::kClusters;
using kl2::kCores;
using kl2::kTagBF16;
using kl2::kTagF16;
using kl2::kTagF32;
using kl2::ShapeMeta;

namespace {

// Build the by-value metadata struct from host arrays.
ShapeMeta MakeMeta(int ndim, const long long* shape, const long long* strides,
                   const long long* out_strides, unsigned long long total,
                   int rows) {
  ShapeMeta meta;
  meta.ndim = ndim;
  meta.rows = rows;
  meta.total = total;
  for (int i = 0; i < ndim && i < 8; ++i) {
    meta.shape[i] = (int)shape[i];
    meta.strides[i] = (int)strides[i];
    meta.out_strides[i] = (int)out_strides[i];
  }
  return meta;
}

}  // namespace

void LaunchSigmoid(int tag, void* out, const void* in, unsigned long long total,
                   void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  auto* o = static_cast<unsigned short*>(out);
  auto* i = static_cast<const unsigned short*>(in);
  if (tag == kTagF32) {
    kl2::SigmoidKernel<kTagF32><<<kClusters, kCores, s>>>(
        static_cast<float*>(out), static_cast<const float*>(in), total);
  } else if (tag == kTagF16) {
    kl2::SigmoidKernel<kTagF16><<<kClusters, kCores, s>>>(o, i, total);
  } else {
    kl2::SigmoidKernel<kTagBF16><<<kClusters, kCores, s>>>(o, i, total);
  }
}

void LaunchSigmoidStrided(
    int tag, void* out, const void* in, const unsigned long long* out_shape,
    const unsigned long long* in_shape, const long long* out_strides,
    const long long* in_strides, unsigned long long output_size,
    unsigned long long ndim, bool out_c, bool in_c, void* stream) {
  LaunchUnaryTag<float, SigmoidF>(tag, out, in, out_shape, in_shape,
                                  out_strides, in_strides, output_size, ndim,
                                  out_c, in_c, stream);
}

void LaunchRmsNorm(int tag, void* out, const void* in, const void* w,
                   unsigned int rows, int dim, float eps,
                   long long in_row_stride, void* stream) {
  if (rows == 0 || dim == 0) return;
  auto* s = static_cast<XPUStream>(stream);
  auto* o = static_cast<unsigned short*>(out);
  auto* i = static_cast<const unsigned short*>(in);
  auto* wp = static_cast<const unsigned short*>(w);
  if (tag == kTagF32) {
    kl2::RmsNormKernel<kTagF32><<<rows, kCores, s>>>(
        static_cast<float*>(out), static_cast<const float*>(in),
        static_cast<const float*>(w), dim, eps, in_row_stride);
  } else if (tag == kTagF16) {
    kl2::RmsNormKernel<kTagF16>
        <<<rows, kCores, s>>>(o, i, wp, dim, eps, in_row_stride);
  } else {
    kl2::RmsNormKernel<kTagBF16>
        <<<rows, kCores, s>>>(o, i, wp, dim, eps, in_row_stride);
  }
}

void LaunchFusedAddRmsNorm(int tag, void* in_out, void* residual, const void* w,
                           unsigned int rows, int dim, float eps,
                           long long in_row_stride,
                           long long residual_row_stride, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  auto* io = static_cast<unsigned short*>(in_out);
  auto* r = static_cast<unsigned short*>(residual);
  if ((tag == kTagBF16 || tag == kTagF16) && (dim % 4 == 0) &&
      (in_row_stride % 4 == 0) && (residual_row_stride % 4 == 0)) {
    const int words = dim / 4;
    const long long io_wstride = in_row_stride / 4;
    const long long r_wstride = residual_row_stride / 4;
    if (tag == kTagBF16) {
      kl2::FusedAddRmsNormPackedKernel<kTagBF16>
          <<<rows, kCores, s>>>(reinterpret_cast<unsigned long long*>(in_out),
                                reinterpret_cast<unsigned long long*>(residual),
                                reinterpret_cast<const unsigned long long*>(w),
                                words, eps, io_wstride, r_wstride);
    } else {
      kl2::FusedAddRmsNormPackedKernel<kTagF16>
          <<<rows, kCores, s>>>(reinterpret_cast<unsigned long long*>(in_out),
                                reinterpret_cast<unsigned long long*>(residual),
                                reinterpret_cast<const unsigned long long*>(w),
                                words, eps, io_wstride, r_wstride);
    }
    return;
  }
  if (tag == kTagF32) {
    kl2::FusedAddRmsNormKernel<kTagF32><<<rows, kCores, s>>>(
        static_cast<float*>(in_out), static_cast<float*>(residual),
        static_cast<const float*>(w), dim, eps, in_row_stride,
        residual_row_stride);
  } else if (tag == kTagF16) {
    kl2::FusedAddRmsNormKernel<kTagF16>
        <<<rows, kCores, s>>>(io, r, static_cast<const unsigned short*>(w), dim,
                              eps, in_row_stride, residual_row_stride);
  } else {
    kl2::FusedAddRmsNormKernel<kTagBF16>
        <<<rows, kCores, s>>>(io, r, static_cast<const unsigned short*>(w), dim,
                              eps, in_row_stride, residual_row_stride);
  }
}

void LaunchSoftmax(int tag, void* out, const void* in, unsigned int rows,
                   int dim, long long in_row_stride, long long out_row_stride,
                   void* stream) {
  if (rows == 0 || dim == 0) return;
  auto* s = static_cast<XPUStream>(stream);
  auto* o = static_cast<unsigned short*>(out);
  auto* i = static_cast<const unsigned short*>(in);
  if (tag == kTagF32) {
    kl2::SoftmaxKernel<kTagF32><<<rows, kCores, s>>>(
        static_cast<float*>(out), static_cast<const float*>(in), dim,
        in_row_stride, out_row_stride);
  } else if (tag == kTagF16) {
    kl2::SoftmaxKernel<kTagF16>
        <<<rows, kCores, s>>>(o, i, dim, in_row_stride, out_row_stride);
  } else {
    kl2::SoftmaxKernel<kTagBF16>
        <<<rows, kCores, s>>>(o, i, dim, in_row_stride, out_row_stride);
  }
}

void LaunchEmbedding(int tag, int index_is_64, void* out, const void* indices,
                     const void* w, unsigned long long total, int emb_dim,
                     unsigned long long vocab, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  auto* o = static_cast<unsigned short*>(out);
  auto* wp = static_cast<const unsigned short*>(w);
  if (index_is_64) {
    auto* idx = static_cast<const long long*>(indices);
    if (tag == kTagF32) {
      kl2::EmbeddingKernel<kTagF32, long long><<<kClusters, kCores, s>>>(
          static_cast<float*>(out), idx, static_cast<const float*>(w), total,
          emb_dim, vocab);
    } else if (tag == kTagF16) {
      kl2::EmbeddingKernel<kTagF16, long long>
          <<<kClusters, kCores, s>>>(o, idx, wp, total, emb_dim, vocab);
    } else {
      kl2::EmbeddingKernel<kTagBF16, long long>
          <<<kClusters, kCores, s>>>(o, idx, wp, total, emb_dim, vocab);
    }
  } else {
    auto* idx = static_cast<const int*>(indices);
    if (tag == kTagF32) {
      kl2::EmbeddingKernel<kTagF32, int><<<kClusters, kCores, s>>>(
          static_cast<float*>(out), idx, static_cast<const float*>(w), total,
          emb_dim, vocab);
    } else if (tag == kTagF16) {
      kl2::EmbeddingKernel<kTagF16, int>
          <<<kClusters, kCores, s>>>(o, idx, wp, total, emb_dim, vocab);
    } else {
      kl2::EmbeddingKernel<kTagBF16, int>
          <<<kClusters, kCores, s>>>(o, idx, wp, total, emb_dim, vocab);
    }
  }
}

void LaunchCopy(int element_size, void* out, const void* in, int ndim,
                const long long* shape, const long long* in_strides,
                const long long* out_strides, unsigned long long total,
                void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  ShapeMeta meta = MakeMeta(ndim, shape, in_strides, out_strides, total, 0);
  if (element_size == 8) {
    kl2::CopyKernel<unsigned long long><<<kClusters, kCores, s>>>(
        static_cast<unsigned long long*>(out),
        static_cast<const unsigned long long*>(in), meta);
    return;
  }
  if (element_size == 4) {
    kl2::CopyKernel<unsigned int>
        <<<kClusters, kCores, s>>>(static_cast<unsigned int*>(out),
                                   static_cast<const unsigned int*>(in), meta);
  } else {
    kl2::CopyKernel<unsigned short><<<kClusters, kCores, s>>>(
        static_cast<unsigned short*>(out),
        static_cast<const unsigned short*>(in), meta);
  }
}

void LaunchNarrowI64ToI32(void* out, const void* in, unsigned long long total,
                          void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  kl2::NarrowI64ToI32Kernel<<<kClusters, kCores, s>>>(
      static_cast<int*>(out), static_cast<const long long*>(in), total);
}

void LaunchCast(int tag_in, int tag_out, void* out, const void* in,
                unsigned long long total, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  auto* o = static_cast<unsigned short*>(out);
  auto* i = static_cast<const unsigned short*>(in);
  if (tag_in == kTagF32 && tag_out == kTagF32) {
    // f32 -> f32: plain flat copy through the 4-byte Copy kernel.
    ShapeMeta meta;
    meta.ndim = 1;
    meta.rows = 0;
    meta.total = total;
    meta.shape[0] = (int)total;
    meta.strides[0] = 1;
    meta.out_strides[0] = 1;
    kl2::CopyKernel<unsigned int>
        <<<kClusters, kCores, s>>>(static_cast<unsigned int*>(out),
                                   static_cast<const unsigned int*>(in), meta);
    return;
  }
  if (tag_in == kTagF32) {
    if (tag_out == kTagF16) {
      kl2::CastFromF32Kernel<kTagF16>
          <<<kClusters, kCores, s>>>(o, static_cast<const float*>(in), total);
    } else {
      kl2::CastFromF32Kernel<kTagBF16>
          <<<kClusters, kCores, s>>>(o, static_cast<const float*>(in), total);
    }
    return;
  }
  if (tag_out == kTagF32) {
    if (tag_in == kTagF16) {
      kl2::CastToF32Kernel<kTagF16>
          <<<kClusters, kCores, s>>>(static_cast<float*>(out), i, total);
    } else {
      kl2::CastToF32Kernel<kTagBF16>
          <<<kClusters, kCores, s>>>(static_cast<float*>(out), i, total);
    }
    return;
  }
  if (tag_in == kTagF16 && tag_out == kTagBF16) {
    kl2::CastKernel16<kTagF16, kTagBF16><<<kClusters, kCores, s>>>(o, i, total);
  } else if (tag_in == kTagBF16 && tag_out == kTagF16) {
    kl2::CastKernel16<kTagBF16, kTagF16><<<kClusters, kCores, s>>>(o, i, total);
  } else if (tag_in == kTagF16) {
    kl2::CastKernel16<kTagF16, kTagF16><<<kClusters, kCores, s>>>(o, i, total);
  } else {
    kl2::CastKernel16<kTagBF16, kTagBF16>
        <<<kClusters, kCores, s>>>(o, i, total);
  }
}

void LaunchScaleF32(void* y, unsigned long long total, float alpha,
                    void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  kl2::ScaleKernel<float>
      <<<kClusters, kCores, s>>>(static_cast<float*>(y), total, alpha);
}

void LaunchFill(int element_size, int tag, void* out, int ndim,
                const long long* shape, const long long* out_strides,
                unsigned long long total, float value, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  static const long long kZero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  ShapeMeta meta = MakeMeta(ndim, shape, kZero, out_strides, total, 0);
  if (tag == kTagI16) {
    const short v = static_cast<short>(static_cast<int>(value));
    unsigned short bits;
    std::memcpy(&bits, &v, sizeof(bits));
    kl2::FillKernel<unsigned short><<<kClusters, kCores, s>>>(
        static_cast<unsigned short*>(out), meta, bits);
    return;
  }
  if (tag == kTagI64) {
    const long long v = static_cast<long long>(value);
    kl2::FillKernel<long long>
        <<<kClusters, kCores, s>>>(static_cast<long long*>(out), meta, v);
    return;
  }
  if (element_size == 4) {
    unsigned int bits;
    if (tag == kTagI32) {
      const int v = static_cast<int>(value);
      std::memcpy(&bits, &v, sizeof(bits));
    } else {
      std::memcpy(&bits, &value, sizeof(bits));
    }
    kl2::FillKernel<unsigned int>
        <<<kClusters, kCores, s>>>(static_cast<unsigned int*>(out), meta, bits);
  } else {
    const unsigned short bits = tag == kTagF16
                                    ? kl2::FloatToF16BitsHost(value)
                                    : kl2::FloatToBF16BitsHost(value);
    kl2::FillKernel<unsigned short><<<kClusters, kCores, s>>>(
        static_cast<unsigned short*>(out), meta, bits);
  }
}

void LaunchTriangular(bool upper, int element_size, void* out, const void* in,
                      int ndim, const long long* shape,
                      const long long* in_strides, const long long* out_strides,
                      int rows, unsigned long long total, int diagonal,
                      void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  ShapeMeta meta = MakeMeta(ndim, shape, in_strides, out_strides, total, rows);
  if (element_size == 4) {
    if (upper) {
      kl2::TriangularKernel<unsigned int, true><<<kClusters, kCores, s>>>(
          static_cast<unsigned int*>(out), static_cast<const unsigned int*>(in),
          meta, diagonal);
    } else {
      kl2::TriangularKernel<unsigned int, false><<<kClusters, kCores, s>>>(
          static_cast<unsigned int*>(out), static_cast<const unsigned int*>(in),
          meta, diagonal);
    }
  } else {
    if (upper) {
      kl2::TriangularKernel<unsigned short, true><<<kClusters, kCores, s>>>(
          static_cast<unsigned short*>(out),
          static_cast<const unsigned short*>(in), meta, diagonal);
    } else {
      kl2::TriangularKernel<unsigned short, false><<<kClusters, kCores, s>>>(
          static_cast<unsigned short*>(out),
          static_cast<const unsigned short*>(in), meta, diagonal);
    }
  }
}

void LaunchArgmax(int tag, void* out, const void* in, unsigned int rows,
                  int dim, long long row_stride, void* stream) {
  if (rows == 0 || dim == 0) return;
  auto* s = static_cast<XPUStream>(stream);
  auto* o = static_cast<long long*>(out);
  auto* i = static_cast<const unsigned short*>(in);
  if (tag == kTagF32) {
    kl2::ArgmaxKernel<kTagF32><<<rows, kCores, s>>>(
        o, static_cast<const float*>(in), dim, row_stride, rows);
  } else if (tag == kTagF16) {
    kl2::ArgmaxKernel<kTagF16>
        <<<rows, kCores, s>>>(o, i, dim, row_stride, rows);
  } else {
    kl2::ArgmaxKernel<kTagBF16>
        <<<rows, kCores, s>>>(o, i, dim, row_stride, rows);
  }
}

void LaunchRope(int tag, bool interleaved, void* out, const void* in,
                const void* positions, const void* sin_table,
                const void* cos_table, int half, long long x_stride_seq,
                long long x_stride_head, long long y_stride_seq,
                long long y_stride_head, unsigned long long seq,
                unsigned long long heads, unsigned long long batch,
                long long x_stride_batch, long long y_stride_batch,
                bool batched_pos, void* stream) {
  if (seq == 0 || heads == 0 || half == 0 || batch == 0) return;
  auto* s = static_cast<XPUStream>(stream);
  auto* o = static_cast<unsigned short*>(out);
  auto* i = static_cast<const unsigned short*>(in);
  auto* p = static_cast<const long long*>(positions);
  auto* st = static_cast<const unsigned short*>(sin_table);
  auto* ct = static_cast<const unsigned short*>(cos_table);
#define KL_ROPE(TAG, IL, O, I, ST, CT)                                  \
  kl2::RopeKernel<TAG, IL><<<kClusters, kCores, s>>>(                   \
      O, I, p, ST, CT, half, x_stride_seq, x_stride_head, y_stride_seq, \
      y_stride_head, seq, heads, batch, x_stride_batch, y_stride_batch, \
      batched_pos)
  if (tag == kTagF32) {
    auto* of = static_cast<float*>(out);
    auto* inf = static_cast<const float*>(in);
    auto* stf = static_cast<const float*>(sin_table);
    auto* ctf = static_cast<const float*>(cos_table);
    if (interleaved) {
      KL_ROPE(kTagF32, true, of, inf, stf, ctf);
    } else {
      KL_ROPE(kTagF32, false, of, inf, stf, ctf);
    }
  } else if (tag == kTagF16) {
    if (interleaved) {
      KL_ROPE(kTagF16, true, o, i, st, ct);
    } else {
      KL_ROPE(kTagF16, false, o, i, st, ct);
    }
  } else {
    if (interleaved) {
      KL_ROPE(kTagBF16, true, o, i, st, ct);
    } else {
      KL_ROPE(kTagBF16, false, o, i, st, ct);
    }
  }
#undef KL_ROPE
}

// ===========================================================================
// Paged KV cache -> dense gather: out[k, g, d] = cache[bt[b*bt_stride+k/PAGE],
// k%PAGE, g, d]. One cluster handles one (token, kv_head) row; cores split
// over dims. Restores the dense layout the vendor context kernel requires
// (vsl_type=NOVSL) without a full-cache copy.
// ===========================================================================
void LaunchGatherKv(int tag, void* out, const void* cache, const void* kvlod,
                    const void* block_tables, int kv_heads, int dim,
                    int page_size, int bt_stride, int batch, int kv_tokens,
                    void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  if (tag == kTagBF16) {
    kl2::GatherKvKernel<kTagBF16><<<kClusters, kCores, s>>>(
        static_cast<unsigned short*>(out),
        static_cast<const unsigned short*>(cache),
        static_cast<const int*>(kvlod), static_cast<const int*>(block_tables),
        kv_heads, dim, page_size, bt_stride, batch, kv_tokens);
  } else if (tag == kTagF16) {
    kl2::GatherKvKernel<kTagF16><<<kClusters, kCores, s>>>(
        static_cast<unsigned short*>(out),
        static_cast<const unsigned short*>(cache),
        static_cast<const int*>(kvlod), static_cast<const int*>(block_tables),
        kv_heads, dim, page_size, bt_stride, batch, kv_tokens);
  } else {
    kl2::GatherKvKernel<kTagF32><<<kClusters, kCores, s>>>(
        static_cast<float*>(out), static_cast<const float*>(cache),
        static_cast<const int*>(kvlod), static_cast<const int*>(block_tables),
        kv_heads, dim, page_size, bt_stride, batch, kv_tokens);
  }
}

void LaunchTransposeCacheBlocks(int tag, void* out, const void* in,
                                int num_blocks, int page_size, int kv_heads,
                                int dim, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  if (tag == kTagBF16) {
    kl2::TransposeCacheBlocksKernel<kTagBF16>
        <<<kClusters, kCores, s>>>(static_cast<unsigned short*>(out),
                                   static_cast<const unsigned short*>(in),
                                   num_blocks, page_size, kv_heads, dim);
  } else if (tag == kTagF16) {
    kl2::TransposeCacheBlocksKernel<kTagF16>
        <<<kClusters, kCores, s>>>(static_cast<unsigned short*>(out),
                                   static_cast<const unsigned short*>(in),
                                   num_blocks, page_size, kv_heads, dim);
  } else {
    kl2::TransposeCacheBlocksKernel<kTagF32><<<kClusters, kCores, s>>>(
        static_cast<float*>(out), static_cast<const float*>(in), num_blocks,
        page_size, kv_heads, dim);
  }
}

void LaunchTransposeBroadcast(int tag, void* out, const void* in,
                              int num_blocks, int page_size, int kv_heads,
                              int q_heads, int dim, void* stream) {
  auto* s = static_cast<XPUStream>(stream);
  if (tag == kTagBF16) {
    kl2::TransposeBroadcastKernel<kTagBF16><<<kClusters, kCores, s>>>(
        static_cast<unsigned short*>(out),
        static_cast<const unsigned short*>(in), num_blocks, page_size, kv_heads,
        q_heads, dim);
  } else if (tag == kTagF16) {
    kl2::TransposeBroadcastKernel<kTagF16><<<kClusters, kCores, s>>>(
        static_cast<unsigned short*>(out),
        static_cast<const unsigned short*>(in), num_blocks, page_size, kv_heads,
        q_heads, dim);
  } else {
    kl2::TransposeBroadcastKernel<kTagF32><<<kClusters, kCores, s>>>(
        static_cast<float*>(out), static_cast<const float*>(in), num_blocks,
        page_size, kv_heads, q_heads, dim);
  }
}

}  // namespace kunlun_launch

}  // namespace infini::ops

namespace klgemm {
struct KlGemmMeta {
  int m;
  int n;
  int k;
  int ldb;
};

__global__ void KlGemmTileKernel(const unsigned short* a,
                                 const unsigned short* b, unsigned short* y,
                                 KlGemmMeta meta) {
  const int n0 = cluster_id() * 64;
  if (n0 >= meta.n) return;
  const int n_lim = min(64, meta.n - n0);
  for (int row = 0; row < meta.m; ++row) {
    auto arow = a + (std::size_t)row * meta.k;
    for (int j = core_id(); j < n_lim; j += 64) {
      auto bcol = b + n0 + j;
      float acc = 0.0f;
      for (int t = 0; t < meta.k; ++t) {
        acc += infini::ops::BF16ToFloat(arow[t]) *
               infini::ops::BF16ToFloat(bcol[(std::size_t)t * meta.ldb]);
      }
      y[(std::size_t)row * meta.n + n0 + j] = infini::ops::FloatToBF16Bits(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// Fused decode-attention kernels: 6 launches -> 2.
//
// GatherBroadcastKvKernel: per (token, q_head) row, resolve the page from
// the block table and copy the token's kv group row into the q-head
// expanded buffer (folds LaunchGatherPage + LaunchGqaBroadcast; the
// broadcast-in-gather removes the staging half).
// DecodeFlashKernel: per q_head cluster, one pass over the token-major
// k_ext rows computes the online-softmax weighted sum of v_ext (folds
// LaunchQK + softmax + LaunchPv).
// ---------------------------------------------------------------------------
}  // namespace klgemm

namespace infini::ops {
namespace kunlun_launch {

using kl2::kClusters;
using kl2::kCores;

void LaunchKlGemmTile(void* y, const void* a, const void* b, int m, int n,
                      int k, int lda, int ldb, void* stream) {
  (void)lda;
  auto* s = static_cast<XPUStream>(stream);
  klgemm::KlGemmMeta meta{m, n, k, ldb};
  klgemm::KlGemmTileKernel<<<kClusters, kCores, s>>>(
      static_cast<const unsigned short*>(a),
      static_cast<const unsigned short*>(b), static_cast<unsigned short*>(y),
      meta);
}

}  // namespace kunlun_launch

namespace klgemm {

// Gathers <clen> tokens of a paged kv cache into a DENSE token-major scratch
// (rows 0..clen-1), broadcasting kv_heads -> q_heads on the way out. The
// vendor paged kernel then reads the scratch through an identity block
// table (blocks 0..n-1 of the scratch are rows [b*page, (b+1)*page)).

// Gathers both K and V active windows in ONE launch: rows over 2*clen*q_heads
// (first half K, second half V), writing head-major
// [block, head, page_off, dim] scratch for the vendor paged kernel.
// bf16 rows are copied as 8-byte words: scalar u16 stores take a slow
// store path on XPU (measured 4.8x slower than packed 64-bit stores for the
// same bytes). head_dim=128 elems = 256B per row and every row base in both
// layouts is a multiple of dim elements, so word addressing is exact.
__global__ void GatherDenseBroadcastKVKernel(
    long long* k_out, long long* v_out, const long long* k_cache,
    const long long* v_cache, const int* block_table, int bt_stride,
    int page_size, int kv_heads, int q_heads, int dim, int clen) {
  const int words = dim / 4;  // 4 bf16 elems (8 bytes) per 64-bit word
  const unsigned long long rows =
      (unsigned long long)clen * (unsigned long long)q_heads;
  for (unsigned long long r = cluster_id(); r < rows; r += cluster_num()) {
    const int token = (int)(r / q_heads);
    const int h = (int)(r % q_heads);
    const int blk = block_table[(token / page_size) * bt_stride];
    const int off = token % page_size;
    const int g = h / (q_heads / kv_heads);
    const unsigned long long dst_off =
        ((unsigned long long)(token / page_size) * q_heads * page_size * dim +
         (unsigned long long)h * page_size * dim +
         (unsigned long long)off * dim) /
        4;
    const unsigned long long src_off =
        (((unsigned long long)blk * page_size + off) * kv_heads * dim +
         (unsigned long long)g * dim) /
        4;
    for (int w = core_id(); w < words; w += 64) {
      k_out[dst_off + w] = k_cache[src_off + w];
      v_out[dst_off + w] = v_cache[src_off + w];
    }
  }
}

__global__ void GatherDenseBroadcastKernel(long long* out,
                                           const long long* cache,
                                           const int* block_table,
                                           int bt_stride, int page_size,
                                           int kv_heads, int q_heads, int dim,
                                           int clen) {
  const int words = dim / 4;
  const unsigned long long rows =
      (unsigned long long)clen * (unsigned long long)q_heads;
  for (unsigned long long t = cluster_id(); t < rows; t += cluster_num()) {
    const int token = (int)(t / q_heads);
    const int h = (int)(t % q_heads);
    const int blk = block_table[(token / page_size) * bt_stride];
    const int off = token % page_size;
    auto src =
        cache + (((unsigned long long)blk * page_size + off) * kv_heads * dim +
                 (unsigned long long)(h / (q_heads / kv_heads)) * dim) /
                    4;
    // Head-major out: [block, head, page_off, dim] — the vendor paged
    // kernel's cache layout.
    auto dst =
        out +
        (((unsigned long long)(token / page_size) * q_heads * page_size * dim +
          (unsigned long long)h * page_size * dim +
          (unsigned long long)off * dim) /
         4);
    for (int w = core_id(); w < words; w += 64) {
      dst[w] = src[w];
    }
  }
}

__global__ void BroadcastGqaKernel(const unsigned short* in,
                                   unsigned short* out, int tokens,
                                   int kv_heads, int q_heads, int dim) {
  const int group = q_heads / kv_heads;
  const unsigned long long total = (unsigned long long)tokens * q_heads * dim;
  const unsigned long long nth = (unsigned long long)cluster_num() * 64;
  for (unsigned long long i = (unsigned long long)cluster_id() * 64 + core_id();
       i < total; i += nth) {
    const unsigned long long d = i % dim;
    const unsigned long long h = (i / dim) % q_heads;
    const unsigned long long t = i / ((unsigned long long)q_heads * dim);
    out[i] = in[(t * kv_heads + h / group) * dim + d];
  }
}

__global__ void GatherPageKernel(long long* out, const long long* cache,
                                 const int* block_table, int bt_stride,
                                 int page_size, int kv_heads, int dim,
                                 int clen) {
  const int words = dim / 4;
  const unsigned long long rows =
      (unsigned long long)clen * (unsigned long long)kv_heads;
  for (unsigned long long t = cluster_id(); t < rows; t += cluster_num()) {
    const int token = (int)(t / kv_heads);
    const int g = (int)(t % kv_heads);
    const int blk = block_table[(token / page_size) * bt_stride];
    const int off = token % page_size;
    auto src =
        cache + (((unsigned long long)blk * page_size + off) * kv_heads * dim +
                 (unsigned long long)g * dim) /
                    4;
    auto dst = out + t * (unsigned long long)words;
    for (int w = core_id(); w < words; w += 64) {
      dst[w] = src[w];
    }
  }
}

// Dot product over bf16 rows via 64-bit word loads: u16 scalar loads take
// a slow path on XPU (measured ~4x vs 8-byte packed loads).
__global__ void QKKernel(float* scores, const long long* q, const long long* k,
                         float scale, int clen, int heads, int dim) {
  const int words = dim / 4;  // dim bf16 elems = dim/4 64-bit words
  for (int h = cluster_id(); h < heads; h += cluster_num()) {
    auto qh = q + (unsigned long long)h * words;
    // k is [clen, heads, dim] (token-major, GQA-broadcast); per-head row
    // stride spans the whole head count.
    for (int s = core_id(); s < clen; s += 64) {
      auto ks = k + (unsigned long long)s * heads * words +
                (unsigned long long)h * words;
      auto qh_w = qh;
      auto ks_w = ks;
      float acc = 0.0f;
      for (int w = 0; w < words; ++w) {
        const unsigned long long qw = static_cast<unsigned long long>(qh_w[w]);
        const unsigned long long kw = static_cast<unsigned long long>(ks_w[w]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const unsigned short qv =
              static_cast<unsigned short>((qw >> (e * 16)) & 0xffffu);
          const unsigned short kv =
              static_cast<unsigned short>((kw >> (e * 16)) & 0xffffu);
          acc += infini::ops::BF16ToFloat(qv) * infini::ops::BF16ToFloat(kv);
        }
      }
      scores[(unsigned long long)h * clen + s] = acc * scale;
    }
  }
}

// Packed 8-byte v rows: each (s, h) row is dim bf16 elems (256B at dim=128),
// 8B-aligned in the token-major layout (heads*dim and dim are multiples of 4),
// so cores slice rows into 64-bit words (4 bf16 elems each).
__global__ void PvKernel(long long* out, const float* scores,
                         const long long* v, int clen, int heads, int dim) {
  const int words = dim / 4;
  for (int h = cluster_id(); h < heads; h += cluster_num()) {
    auto sc = scores + (unsigned long long)h * clen;
    float mx = -3.0e38f;
    for (int s = 0; s < clen; ++s) mx = fmax(mx, sc[s]);
    float denom = 0.0f;
    for (int s = 0; s < clen; ++s) denom += exp(sc[s] - mx);
    // v is [clen, heads, dim] (token-major, GQA-broadcast), addressed in words.
    auto vh = v + (unsigned long long)h * words;
    auto oh = out + (unsigned long long)h * words;
    for (int w = core_id(); w < words; w += 64) {
      float acc[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = 0.0f;
      for (int s = 0; s < clen; ++s) {
        const float p = exp(sc[s] - mx) / denom;
        const unsigned long long vw = static_cast<unsigned long long>(
            vh[(unsigned long long)s * heads * (unsigned long long)words +
               (unsigned long long)w]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const unsigned short vv =
              static_cast<unsigned short>((vw >> (e * 16)) & 0xffffu);
          acc[e] += p * infini::ops::BF16ToFloat(vv);
        }
      }
      unsigned long long outw = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        outw |= static_cast<unsigned long long>(
                    infini::ops::FloatToBF16Bits(acc[e]))
                << (e * 16);
      }
      oh[w] = static_cast<long long>(outw);
    }
  }
}

__global__ void GatherBroadcastKvKernel(unsigned short* out,
                                        const unsigned short* cache,
                                        const int* block_table, int bt_stride,
                                        int page_size, int kv_heads,
                                        int q_heads, int dim, int clen) {
  const unsigned long long rows =
      (unsigned long long)clen * (unsigned long long)q_heads;
  for (unsigned long long t = cluster_id(); t < rows; t += cluster_num()) {
    const int token = (int)(t / q_heads);
    const int h = (int)(t % q_heads);
    const int blk = block_table[(token / page_size) * bt_stride];
    const int off = token % page_size;
    auto src = cache +
               ((unsigned long long)blk * page_size + off) * kv_heads * dim +
               (unsigned long long)(h / (q_heads / kv_heads)) * dim;
    auto dst = out + (unsigned long long)token * q_heads * dim +
               (unsigned long long)h * dim;
    for (int d = core_id(); d < dim; d += 64) {
      dst[d] = src[d];
    }
  }
}

}  // namespace klgemm

namespace kunlun_launch {

void LaunchGatherBroadcastKv(void* out, const void* cache,
                             const void* block_table, int bt_stride,
                             int page_size, int kv_heads, int q_heads, int dim,
                             int clen, void* stream) {
  klgemm::GatherBroadcastKvKernel<<<64u, 64u, static_cast<XPUStream>(stream)>>>(
      static_cast<unsigned short*>(out),
      static_cast<const unsigned short*>(cache),
      static_cast<const int*>(block_table), bt_stride, page_size, kv_heads,
      q_heads, dim, clen);
}

void LaunchGatherDenseBroadcast(void* out, const void* cache,
                                const void* block_table, int bt_stride,
                                int page_size, int kv_heads, int q_heads,
                                int dim, int clen, void* stream) {
  klgemm::
      GatherDenseBroadcastKernel<<<8u, 64u, static_cast<XPUStream>(stream)>>>(
          static_cast<long long*>(out), static_cast<const long long*>(cache),
          static_cast<const int*>(block_table), bt_stride, page_size, kv_heads,
          q_heads, dim, clen);
}

void LaunchGatherDenseBroadcastKV(void* k_out, void* v_out, const void* k_cache,
                                  const void* v_cache, const void* block_table,
                                  int bt_stride, int page_size, int kv_heads,
                                  int q_heads, int dim, int clen,
                                  void* stream) {
  klgemm::
      GatherDenseBroadcastKVKernel<<<8u, 64u, static_cast<XPUStream>(stream)>>>(
          static_cast<long long*>(k_out), static_cast<long long*>(v_out),
          static_cast<const long long*>(k_cache),
          static_cast<const long long*>(v_cache),
          static_cast<const int*>(block_table), bt_stride, page_size, kv_heads,
          q_heads, dim, clen);
}

void LaunchGqaBroadcast(void* out, const void* in, int tokens, int kv_heads,
                        int head_num, int head_dim, void* stream) {
  klgemm::BroadcastGqaKernel<<<8u, 64u, static_cast<XPUStream>(stream)>>>(
      static_cast<const unsigned short*>(in), static_cast<unsigned short*>(out),
      tokens, kv_heads, head_num, head_dim);
}

void LaunchGatherPage(void* out, const void* cache, const void* block_table,
                      int bt_stride, int page_size, int kv_heads, int dim,
                      int clen, int tag, void* stream) {
  (void)tag;
  klgemm::GatherPageKernel<<<8u, 64u, static_cast<XPUStream>(stream)>>>(
      static_cast<long long*>(out), static_cast<const long long*>(cache),
      static_cast<const int*>(block_table), bt_stride, page_size, kv_heads, dim,
      clen);
}

void LaunchBroadcastGqa(void* out, const void* in, int tokens, int kv_heads,
                        int head_num, int head_dim, int tag, void* stream) {
  (void)tag;
  LaunchGqaBroadcast(out, in, tokens, kv_heads, head_num, head_dim, stream);
}

void LaunchQK(void* scores, const void* q, const void* k, float scale, int clen,
              int heads, int dim, int tag, void* stream) {
  (void)tag;
  klgemm::QKKernel<<<8u, 64u, static_cast<XPUStream>(stream)>>>(
      static_cast<float*>(scores), static_cast<const long long*>(q),
      static_cast<const long long*>(k), scale, clen, heads, dim);
}

void LaunchPv(void* out, const void* scores, const void* v, int clen, int heads,
              int dim, int tag, void* stream) {
  (void)tag;
  klgemm::PvKernel<<<8u, 64u, static_cast<XPUStream>(stream)>>>(
      static_cast<long long*>(out), static_cast<const float*>(scores),
      static_cast<const long long*>(v), clen, heads, dim);
}

}  // namespace kunlun_launch
}  // namespace infini::ops