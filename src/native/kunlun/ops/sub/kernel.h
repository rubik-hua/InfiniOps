#ifndef INFINI_OPS_KUNLUN_SUB_KERNEL_H_
#define INFINI_OPS_KUNLUN_SUB_KERNEL_H_

#include <cstdint>
#include <cstring>

#include "base/sub.h"
#include "host_range_profiler.h"
#include "native/kunlun/binary_broadcast_helper.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/ops/relu/kernel.h"  // detail::KunlunUploadBinaryMeta
#include "native/kunlun/runtime_.h"

namespace infini::ops {

template <>
class Operator<Sub, Device::Type::kKunlun> : public Sub {
 public:
  using Sub::Sub;

  // Dtype contract mirror of add.
  static bool supports(const Tensor input, const Tensor other,
                       const double alpha, Tensor out) {
    (void)input;
    (void)other;
    (void)alpha;
    switch (out.dtype()) {
      case DataType::kFloat32:
      case DataType::kFloat16:
      case DataType::kBFloat16:
      case DataType::kInt16:
      case DataType::kInt32:
      case DataType::kInt64:
        return true;
      default:
        return false;
    }
  }

  ~Operator() {
    if (d_metadata_ != nullptr) {
      Runtime<Device::Type::kKunlun>::Free(d_metadata_);
    }
  }

  void operator()(const Tensor input, const Tensor other, const double alpha,
                  Tensor out) const override {
    if (input.numel() == 0) {
      return;
    }
    [[maybe_unused]] HostRangeScope host_range_backend_submit{
        HostRangeLayer::kBackendSubmit};
    if (alpha != 1.0) {
      throw std::runtime_error("kunlun sub: only alpha == 1.0 is supported");
    }

    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(out_type_);

    // Upload shape/stride metadata for the (possibly broadcast/strided)
    // operands. Broadcasted strides are relative to the output layout.
    const Tensor::Shape out_shape{out.shape()};
    // Right-align-pad lower-rank operands: metadata buffer is sized by
    // out.ndim() and broadcast strides are relative to the output layout.
    const Tensor::Shape in_shape =
        detail::KunlunPadShape(input.shape(), out.ndim());
    const Tensor::Shape other_shape =
        detail::KunlunPadShape(other.shape(), out.ndim());
    const Tensor::Strides out_strides{out.strides()};
    const Tensor::Strides in_strides =
        detail::KunlunBroadcastStrides(input, out);
    const Tensor::Strides other_strides =
        detail::KunlunBroadcastStrides(other, out);
    if (meta_valid_ &&
        (cached_out_shape_ != out_shape || cached_in_shape_ != in_shape ||
         cached_other_shape_ != other_shape ||
         cached_out_strides_ != out_strides ||
         cached_in_strides_ != in_strides ||
         cached_other_strides_ != other_strides)) {
      Runtime<Device::Type::kKunlun>::Free(d_metadata_);
      d_metadata_ = nullptr;
      meta_valid_ = false;
    }
    if (!meta_valid_) {
      d_metadata_ = detail::KunlunUploadBinaryMeta(
          out_shape, in_shape, other_shape, out_strides, in_strides,
          other_strides, d_out_shape_, d_in_shape_, d_other_shape_,
          d_out_strides_, d_in_strides_, d_other_strides_);
      cached_out_shape_ = out_shape;
      cached_in_shape_ = in_shape;
      cached_other_shape_ = other_shape;
      cached_out_strides_ = out_strides;
      cached_in_strides_ = in_strides;
      cached_other_strides_ = other_strides;
      meta_valid_ = true;
    }

    kunlun_launch::LaunchSub(
        tag, out.data(), input.data(), other.data(),
        reinterpret_cast<const unsigned long long*>(d_out_shape_),
        reinterpret_cast<const unsigned long long*>(d_in_shape_),
        reinterpret_cast<const unsigned long long*>(d_other_shape_),
        reinterpret_cast<const long long*>(d_out_strides_),
        reinterpret_cast<const long long*>(d_in_strides_),
        reinterpret_cast<const long long*>(d_other_strides_), out.numel(),
        out.ndim(), out.IsContiguous(),
        input.shape() == out.shape() && input.IsContiguous(),
        other.shape() == out.shape() && other.IsContiguous(), stream);
  }

  // The base declares three pure-virtual call forms; all must be overridden
  // for the specialization to be concrete (an abstract class is silently
  // skipped by the dispatcher's `is_base_of` probe). The deprecated no-out
  // form runs through a temporary contiguous output.
  void operator()(Tensor input, const Tensor other,
                  const double alpha) const override {
    if (input.numel() == 0) {
      return;
    }
    void* out_data = AllocateContiguousLike(input);
    const Tensor::Shape shape{input.shape()};
    Tensor out(out_data, shape, input_type_,
               Device(Device::Type::kKunlun, device_index_));
    (*this)(input, other, alpha, out);
    Runtime<Device::Type::kKunlun>::Free(out_data);
  }

  // The scalar-`other` form materializes the scalar as a single-element
  // device tensor of the output's dtype; the tensor form's broadcast path
  // (stride 0 for size-1 dims) then handles the rest.
  void operator()(const Tensor input, const double other, const double alpha,
                  Tensor out) const override {
    if (input.numel() == 0) {
      return;
    }
    void* scalar_data = AllocateContiguousLike(out);
    try {
      UploadScalar(scalar_data, other * alpha);
    } catch (...) {
      Runtime<Device::Type::kKunlun>::Free(scalar_data);
      throw;
    }
    Tensor scalar(scalar_data, {1}, out_type_, out.device(), {0});
    (*this)(input, scalar, 1.0, out);
    Runtime<Device::Type::kKunlun>::Free(scalar_data);
  }

 private:
  // Allocates uninitialized device storage matching `like`'s numel/dtype.
  static void* AllocateContiguousLike(const Tensor& like) {
    void* data = nullptr;
    const std::size_t bytes = like.numel() * like.element_size();
    if (Runtime<Device::Type::kKunlun>::Malloc(&data, bytes) !=
        Runtime<Device::Type::kKunlun>::kSuccess) {
      throw std::runtime_error("kunlun sub: temporary allocation failed");
    }
    return data;
  }

  // Encodes `value` in the output dtype's host representation and uploads it
  // as a single device element.
  void UploadScalar(void* data, double value) const {
    std::vector<std::byte> host(out_type_ == DataType::kFloat64
                                    ? sizeof(double)
                                    : sizeof(std::uint32_t));
    const std::size_t bytes = EncodeScalar(out_type_, host.data(), value);
    if (Runtime<Device::Type::kKunlun>::Memcpy(
            data, host.data(), bytes,
            Runtime<Device::Type::kKunlun>::kMemcpyHostToDevice) !=
        Runtime<Device::Type::kKunlun>::kSuccess) {
      throw std::runtime_error("kunlun sub: scalar upload failed");
    }
  }

  // Encodes `value` in the output dtype's host representation.
  static std::size_t EncodeScalar(DataType dtype, void* dst, double value) {
    const float v = static_cast<float>(value);
    switch (dtype) {
      case DataType::kFloat32:
        std::memcpy(dst, &v, sizeof(v));
        return sizeof(v);
      case DataType::kFloat64: {
        std::memcpy(dst, &value, sizeof(value));
        return sizeof(value);
      }
      case DataType::kFloat16: {
        // IEEE-754 binary16 bits (host representation matches the device
        // half payload bit-for-bit); same rounding as caster_.h.
        std::uint32_t fbits;
        std::memcpy(&fbits, &v, sizeof(fbits));
        const std::uint32_t sign = (fbits >> 16) & 0x8000u;
        const std::int32_t exp =
            static_cast<std::int32_t>((fbits >> 23) & 0xFFu) - 127 + 15;
        const std::uint32_t man = fbits & 0x007FFFFFu;
        std::uint16_t bits;
        if ((fbits & 0x7FFFFFFFu) == 0) {
          bits = static_cast<std::uint16_t>(sign);
        } else if (exp >= 0x1F) {
          bits = static_cast<std::uint16_t>(sign | 0x7C00u);
        } else if (exp <= 0) {
          bits = static_cast<std::uint16_t>(
              exp < -10 ? sign
                        : sign | ((man | 0x00800000u) >> (1 - exp)) >> 13);
        } else {
          bits = static_cast<std::uint16_t>(sign | (exp << 10) | (man >> 13));
        }
        std::memcpy(dst, &bits, sizeof(bits));
        return sizeof(bits);
      }
      case DataType::kBFloat16: {
        // bf16 payloads are truncated fp32 in memory (computed as f32
        // on device); see caster_.h.
        std::uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        const std::uint16_t b = static_cast<std::uint16_t>(bits >> 16);
        std::memcpy(dst, &b, sizeof(b));
        return sizeof(b);
      }
      case DataType::kInt32: {
        const std::int32_t i = static_cast<std::int32_t>(value);
        std::memcpy(dst, &i, sizeof(i));
        return sizeof(i);
      }
      case DataType::kInt64: {
        const std::int64_t i = static_cast<std::int64_t>(value);
        std::memcpy(dst, &i, sizeof(i));
        return sizeof(i);
      }
      default:
        throw std::runtime_error("kunlun sub: unsupported scalar dtype");
    }
  }

  // Mutable: metadata upload happens on first invocation and is cached until
  // the operand layouts change; the buffer is freed in the destructor.
  mutable std::byte* d_metadata_{nullptr};
  mutable Tensor::Size* d_out_shape_{nullptr};
  mutable Tensor::Size* d_in_shape_{nullptr};
  mutable Tensor::Size* d_other_shape_{nullptr};
  mutable Tensor::Stride* d_out_strides_{nullptr};
  mutable Tensor::Stride* d_in_strides_{nullptr};
  mutable Tensor::Stride* d_other_strides_{nullptr};
  mutable bool meta_valid_{false};
  mutable Tensor::Shape cached_out_shape_{};
  mutable Tensor::Shape cached_in_shape_{};
  mutable Tensor::Shape cached_other_shape_{};
  mutable Tensor::Strides cached_out_strides_{};
  mutable Tensor::Strides cached_in_strides_{};
  mutable Tensor::Strides cached_other_strides_{};
};

}  // namespace infini::ops

#endif
