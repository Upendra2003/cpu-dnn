import random

import pytest
import torch

from roba import reference
from tiny import qnet as Q


def _layer(seed, bits, c=3, o=4, k=3, n=2, hw=6):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, Q.QA[bits] + 1, (n, c, hw, hw), generator=g)
    w = torch.randint(-Q.QW[bits], Q.QW[bits] + 1, (o, c, k, k), generator=g)
    return x, w


def _loop_conv(x, w, mul, stride, pad):
    """Scalar convolution with an arbitrary product, the way hw/sw/tiny.c loops."""
    n, c, h, wd = x.shape
    o, _, k, _ = w.shape
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    oh, ow = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
    out = torch.zeros(n, o, oh, ow, dtype=torch.long)
    for b in range(n):
        for q in range(o):
            for i in range(oh):
                for j in range(ow):
                    patch = xp[b, :, i * stride:i * stride + k, j * stride:j * stride + k].flatten().tolist()
                    out[b, q, i, j] = mul(patch, w[q].flatten().tolist())
    return out


@pytest.mark.parametrize("bits", Q.BITS)
@pytest.mark.parametrize("method", ["exact", "dn1", "dn2"])
@pytest.mark.parametrize("stride,pad", [(1, 1), (2, 1), (2, 0)])
def test_int_conv_matches_scalar_reference(bits, method, stride, pad):
    x, w = _layer(bits + stride + pad, bits)
    mult = Q.ROBA[method]
    want = _loop_conv(x, w, lambda a, b: sum(reference.mul(p, q, mult) for p, q in zip(a, b)), stride, pad)
    assert torch.equal(Q.int_conv(x, w, method, stride, pad), want)


@pytest.mark.parametrize("bits", (4, 8))
def test_square_table_identity_is_exact(bits):
    x, w = _layer(7, bits)
    sq = lambda v: v * v

    def pol(a, b):
        return (sum(sq(p + q) for p, q in zip(a, b)) - sum(map(sq, a)) - sum(map(sq, b))) >> 1

    assert torch.equal(_loop_conv(x, w, pol, 1, 1), Q.int_conv(x, w, "exact", 1, 1))


def _sq_apx_scalar(v):
    a = abs(v)
    if a == 0:
        return 0
    k = a.bit_length() - 1
    return (a << (k + 1)) - (1 << (2 * k))


def test_approximate_square_conv():
    x, w = _layer(11, 8)

    def pol(a, b):
        d = sum(_sq_apx_scalar(p + q) for p, q in zip(a, b)) - sum(map(_sq_apx_scalar, a)) - sum(
            map(_sq_apx_scalar, b))
        return d >> 1

    assert torch.equal(Q.int_conv(x, w, "sqapx", 1, 1), _loop_conv(x, w, pol, 1, 1))
    v = torch.tensor([random.Random(3).randint(-382, 382) for _ in range(500)] + [0, 1, -1, 2, 255, 382])
    assert Q.sq_apx(v).tolist() == [_sq_apx_scalar(int(t)) for t in v]


@pytest.mark.parametrize("bits", Q.BITS)
def test_quantize_input_rounds(bits):
    x = torch.arange(256, dtype=torch.uint8)
    q = Q.qa_in(bits)
    assert Q.quantize_input(x, bits).tolist() == [int(v * q / 255 + 0.5) for v in range(256)]
