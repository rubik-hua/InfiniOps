import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randn_strided,
)

_FLOAT_DTYPE_CASES = (
    (torch.float32, 1e-6, 1e-6),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 5e-3),
)

_SHAPES = (
    (13, 4),
    (13, 4, 4),
    (16, 5632),
    (4, 4, 5632),
)


_SHAPES_SUB = _SHAPES + ((13, 4),)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, other_shape, out_shape, input_strides, other_strides, out_strides",
    (
        ((13, 4), (13, 4), (13, 4), None, None, None),
        ((13, 4), (13, 4), (13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4, 4), (13, 4, 4), (13, 4, 4), None, None, None),
        ((16, 5632), (16, 5632), (16, 5632), None, None, None),
        ((4, 4, 5632), (4, 4, 5632), (4, 4, 5632), None, None, None),
        ((13, 4), (4,), (13, 4), None, None, None),
        ((0,), (0,), (0,), None, None, None),
    ),
)
@pytest.mark.parametrize("dtype, rtol, atol", _FLOAT_DTYPE_CASES)
def test_sub(
    input_shape,
    other_shape,
    out_shape,
    input_strides,
    other_strides,
    out_strides,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    other = randn_strided(other_shape, other_strides, dtype=dtype, device=device)
    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _sub,
        _torch_sub,
        (input, other, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _sub(input, other, out, **kwargs):
    infini.ops.sub(
        input, other, 1.0, out, stream=get_stream(input.device), **kwargs
    )

    return out


def _torch_sub(input, other, out):
    out.copy_(torch.sub(input, other))

    return out
