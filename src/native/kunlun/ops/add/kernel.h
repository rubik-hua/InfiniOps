#ifndef INFINI_OPS_KUNLUN_ADD_KERNEL_H_
#define INFINI_OPS_KUNLUN_ADD_KERNEL_H_

#include "base/add.h"
#include "host_range_profiler.h"
#include "native/kunlun/binary_broadcast_helper.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/ops/relu/kernel.h"  // detail::KunlunUploadBinaryMeta
#include "native/kunlun/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kKunlun> : public Add {
 public:
  using Add::Add;

  // The base ctor admits float64; the kunlun kernels only instantiate the
  // f32/f16/bf16/int lanes. Gate here so the dispatcher skips this provider
  // for anything else instead of throwing at call time (same contract
  // pattern as the flash/argmax providers).
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
    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(out_type_);
    (void)alpha;  // alpha != 1 handled by pre-scaling  below.

    // Upload shape/stride metadata for the (possibly broadcast/strided)
    // operands. Broadcasted strides are relative to the output layout.
    // The upload is a synchronous malloc + H2D memcpy round trip (~16us
    // measured, and it drains the async submission pipeline), so the
    // metadata is cached across invocations: decode steps re-invoke with
    // identical layouts. A different shape/stride set frees and re-uploads.
    const Tensor::Shape out_shape{out.shape()};
    // Lower-rank operands must be right-aligned padded with leading 1s:
    // KunlunUploadBinaryMeta copies shape/stride arrays sized by out.ndim(),
    // and the broadcast strides are defined relative to the output layout.
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

    kunlun_launch::LaunchAdd(
        tag, out.data(), input.data(), other.data(),
        reinterpret_cast<const unsigned long long*>(d_out_shape_),
        reinterpret_cast<const unsigned long long*>(d_in_shape_),
        reinterpret_cast<const unsigned long long*>(d_other_shape_),
        reinterpret_cast<const long long*>(d_out_strides_),
        reinterpret_cast<const long long*>(d_in_strides_),
        reinterpret_cast<const long long*>(d_other_strides_), out.numel(),
        out.ndim(), out.IsContiguous(),
        input.shape() == out.shape() && input.IsContiguous(),
        other.shape() == out.shape() && other.IsContiguous(), alpha, stream);
  }

 private:
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
