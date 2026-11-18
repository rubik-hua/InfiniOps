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


_HALF_CASES = (32, 64)


def _reference_rope(x, positions, sin_table, cos_table, is_neox):
    # x: [seq, heads, 2*half]; positions: [seq] int64
    half = x.shape[-1] // 2
    seq, heads = x.shape[0], x.shape[1]
    x32 = x.float()
    out = torch.empty_like(x32)
    for s in range(seq):
        row = int(positions[s].item())
        sin_t = sin_table[row].float()
        cos_t = cos_table[row].float()
        for h in range(heads):
            if is_neox:
                x1 = x32[s, h, :half]
                x2 = x32[s, h, half:]
                out[s, h, :half] = x1 * cos_t - x2 * sin_t
                out[s, h, half:] = x1 * sin_t + x2 * cos_t
            else:
                for i in range(half):
                    x0 = x32[s, h, 2 * i]
                    x1 = x32[s, h, 2 * i + 1]
                    out[s, h, 2 * i] = x0 * cos_t[i] - x1 * sin_t[i]
                    out[s, h, 2 * i + 1] = x0 * sin_t[i] + x1 * cos_t[i]
    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("seq, heads, half", ((4, 2, 32), (1, 8, 64), (3, 1, 16)))
@pytest.mark.parametrize("is_neox", (True, False))
@pytest.mark.parametrize("dtype, rtol, atol", _FLOAT_DTYPE_CASES)
def test_separate_table_rope(seq, heads, half, is_neox, dtype, device, rtol, atol):
    torch.manual_seed(7)
    table_rows = 128
    x = randn_strided((seq, heads, 2 * half), None, dtype=dtype, device=device)
    positions = torch.arange(seq, device=device, dtype=torch.int64)
    sin_table = torch.randn(table_rows, half, device=device, dtype=dtype)
    cos_table = torch.randn(table_rows, half, device=device, dtype=dtype)
    out = empty_strided((seq, heads, 2 * half), None, dtype=dtype, device=device)

    return Payload(
        lambda x, positions, sin_table, cos_table, out: _separate_table_rope(
            x, positions, sin_table, cos_table, is_neox, out
        ),
        lambda x, positions, sin_table, cos_table, out: _torch_rope(
            x, positions, sin_table, cos_table, is_neox, out
        ),
        (x, positions, sin_table, cos_table, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _separate_table_rope(x, positions, sin_table, cos_table, is_neox, out, **kwargs):
    infini.ops.separate_table_rope(
        x,
        positions,
        sin_table,
        cos_table,
        is_neox,
        out,
        stream=get_stream(x.device),
        **kwargs,
    )

    return out


def _torch_rope(x, positions, sin_table, cos_table, is_neox, out):
    res = _reference_rope(x, positions, sin_table, cos_table, is_neox)
    out.copy_(res.to(out.dtype))

    return out
