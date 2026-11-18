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


_DIAGONAL_CASES = (0, 1, -2)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape", _SHAPES + ((7, 7),))
@pytest.mark.parametrize("diagonal", _DIAGONAL_CASES)
@pytest.mark.parametrize("dtype, rtol, atol", _FLOAT_DTYPE_CASES)
def test_triu(shape, diagonal, dtype, device, rtol, atol):
    input = randn_strided(shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)

    return Payload(
        lambda input, out: _triu(input, out, diagonal),
        lambda input, out: _torch_triu(input, out, diagonal),
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _triu(input, out, diagonal=0, **kwargs):
    infini.ops.triu(input, diagonal, out, stream=get_stream(input.device), **kwargs)

    return out


def _torch_triu(input, out, diagonal):
    out.copy_(torch.triu(input, diagonal))

    return out
