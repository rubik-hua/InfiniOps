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


_PAD_CASES = (
    ((13, 4), (1, 2), (14, 7)),
    ((13, 4, 4), (0, 0, 1, 1), (14, 6, 4)),
    ((16, 5632), (0, 8), (16, 5648)),
    ((2, 3), (1, 1, 2, 2), (5, 6)),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("input_shape, pad, out_shape", _PAD_CASES)
@pytest.mark.parametrize("dtype, rtol, atol", _FLOAT_DTYPE_CASES)
def test_pad(input_shape, pad, out_shape, dtype, device, rtol, atol):
    if device == "cuda":
        pytest.skip(
            "kunlun pad is a stub in this commit; implementation lands with 021"
        )
    input = randn_strided(input_shape, None, dtype=dtype, device=device)
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        _pad,
        _torch_pad,
        (input, out),
        {"pad": pad},
        rtol=rtol,
        atol=atol,
    )


def _pad(input, out, *, pad, **kwargs):
    infini.ops.pad(
        input, pad, "constant", 0.0, out, stream=get_stream(input.device), **kwargs
    )

    return out


def _torch_pad(input, out, *, pad):
    out.copy_(torch.nn.functional.pad(input, tuple(pad), mode="constant", value=0.0))

    return out
