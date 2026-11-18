#ifndef INFINI_OPS_KUNLUN_BINARY_BROADCAST_HELPER_H_
#define INFINI_OPS_KUNLUN_BINARY_BROADCAST_HELPER_H_

// Shared by the elementwise binary operators (add/mul/sub): broadcast
// `input` strides against `out`'s shape (right-aligned broadcast).

#include "native/kunlun/runtime_.h"
#include "tensor.h"

namespace infini::ops {

namespace detail {

inline Tensor::Strides KunlunBroadcastStrides(const Tensor& input,
                                              const Tensor& out) {
  const Tensor::Shape out_shape{out.shape()};
  const Tensor::Strides input_strides{input.strides()};
  Tensor::Strides result(out_shape.size(), 0);
  if (input.ndim() > out_shape.size()) {
    // Broadcast requires input rank <= output rank; guard the unsigned
    // subtraction below from underflowing on a rank-mismatched pair.
    throw std::runtime_error(
        "kunlun broadcast: input rank exceeds output rank");
  }
  const Tensor::Size offset = out_shape.size() - input.ndim();
  for (Tensor::Size i = 0; i < input.ndim(); ++i) {
    result[offset + i] = input.size(i) == 1 ? 0 : input_strides[i];
  }
  return result;
}

}  // namespace detail

}  // namespace infini::ops

#endif
