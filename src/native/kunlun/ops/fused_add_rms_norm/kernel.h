#ifndef INFINI_OPS_KUNLUN_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_KUNLUN_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <stdexcept>

#include "base/fused_add_rms_norm.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun FusedAddRmsNorm: one XTDK kernel per call — `residual` becomes
// input + residual in place and `input` becomes norm(residual) * weight.
// Both tensors may be strided views (fused-QKV splits), so per-row strides
// come from the base constructor and the last dimension must be unit.
template <>
class Operator<FusedAddRmsNorm, Device::Type::kKunlun>
    : public FusedAddRmsNorm {
 public:
  using FusedAddRmsNorm::FusedAddRmsNorm;

  bool capture_safe() const override { return true; }

  static bool supports(const Tensor input, const Tensor residual,
                       const std::optional<Tensor> weight, float epsilon) {
    if (input.dtype() != DataType::kFloat32 &&
        input.dtype() != DataType::kFloat16 &&
        input.dtype() != DataType::kBFloat16) {
      return false;
    }
    if (input.stride(-1) != 1 || residual.stride(-1) != 1) {
      return false;
    }
    // Uniform-row contract (see softmax/rms_norm): the per-row kernel walks
    // rows at r * stride(-2); rank >= 3 views with non-uniform outer row
    // spacing would be silently misread.
    if (input.shape().size() >= 3 &&
        (input.stride(-2) !=
             static_cast<Tensor::Stride>(input.shape().back()) ||
         residual.stride(-2) !=
             static_cast<Tensor::Stride>(residual.shape().back()))) {
      return false;
    }
    return true;
  }

  void operator()(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight,
                  float epsilon) const override {
    if (num_tokens_ == 0 || dim_ == 0) {
      return;
    }
    const Tensor::Stride in_row_stride =
        input_strides_.size() >= 2 ? input_strides_[input_strides_.size() - 2]
                                   : 0;
    const Tensor::Stride res_row_stride =
        residual_strides_.size() >= 2
            ? residual_strides_[residual_strides_.size() - 2]
            : 0;

    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(input.dtype());
    unsigned int rows;
    if (input.shape().size() >= 3) {
      rows = 1;
      for (std::size_t i = 0; i + 1 < input.shape().size(); ++i) {
        rows *= input.shape()[i];
      }
    } else {
      rows = input.shape().size() == 2
                 ? static_cast<unsigned int>(input.shape()[0])
                 : 1;
    }

    kunlun_launch::LaunchFusedAddRmsNorm(
        tag, input.data(), residual.data(),
        weight.has_value() ? weight->data() : nullptr, rows, (int)dim_, epsilon,
        (long long)in_row_stride, (long long)res_row_stride, stream);
  }
};

}  // namespace infini::ops

#endif