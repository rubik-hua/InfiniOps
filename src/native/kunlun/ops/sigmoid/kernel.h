#ifndef INFINI_OPS_KUNLUN_SIGMOID_KERNEL_H_
#define INFINI_OPS_KUNLUN_SIGMOID_KERNEL_H_

#include <stdexcept>
#include <vector>

#include "base/sigmoid.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/ops/relu/kernel.h"  // detail::KunlunUploadUnaryMeta
#include "native/kunlun/runtime_.h"
#include "native/kunlun/runtime_utils.h"

namespace infini::ops {

// Kunlun sigmoid: flat, contiguous, f32/f16/bf16.
template <>
class Operator<Sigmoid, Device::Type::kKunlun> : public Sigmoid {
 public:
  using Sigmoid::Sigmoid;

  bool capture_safe() const override { return true; }

  static bool supports(const Tensor input, Tensor out) {
    (void)input;
    switch (out.dtype()) {
      case DataType::kFloat32:
      case DataType::kFloat16:
      case DataType::kBFloat16:
        return true;
      default:
        return false;
    }
  }

  Operator(const Tensor input, Tensor out) : Sigmoid{input, out} {
    if (!(is_input_contiguous_ && is_out_contiguous_)) {
      // Kernels dereference these pointers on-device: host stack arrays are
      // not readable from the cluster, so the metadata must live in device
      // memory (same scheme as the relu/gelu/silu wrappers).
      d_metadata_ = detail::KunlunUploadUnaryMeta(
          input_shape_, out_shape_, input_strides_, out_strides_,
          d_input_shape_, d_out_shape_, d_input_strides_, d_out_strides_);
    }
  }

  ~Operator() {
    if (d_metadata_ != nullptr) {
      Runtime<Device::Type::kKunlun>::Free(d_metadata_);
    }
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }
    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(out_type_);
    if (is_input_contiguous_ && is_out_contiguous_) {
      kunlun_launch::LaunchSigmoid(tag, out.data(), input.data(),
                                   (unsigned long long)output_size_, stream);
      return;
    }
    if (ndim_ > 8) {
      throw std::runtime_error("kunlun sigmoid: rank exceeds 8");
    }
    kunlun_launch::LaunchSigmoidStrided(
        tag, out.data(), input.data(),
        reinterpret_cast<const unsigned long long*>(d_out_shape_),
        reinterpret_cast<const unsigned long long*>(d_input_shape_),
        reinterpret_cast<const long long*>(d_out_strides_),
        reinterpret_cast<const long long*>(d_input_strides_),
        (unsigned long long)output_size_, ndim_, is_out_contiguous_,
        is_input_contiguous_, stream);
  }

 private:
  std::byte* d_metadata_{nullptr};
  Tensor::Size* d_input_shape_{nullptr};
  Tensor::Size* d_out_shape_{nullptr};
  Tensor::Stride* d_input_strides_{nullptr};
  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
