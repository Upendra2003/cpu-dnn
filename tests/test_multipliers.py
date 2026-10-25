import ctypes
import pathlib
import random
import subprocess

import numpy as np
import pytest
import torch

from roba import reference
from roba.matmul import approx_conv2d, approx_linear
from roba.multipliers import MULTIPLIERS, approx_mul

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _rand_signed(n, bits, rng):
    lim = (1 << (bits - 1)) - 1
    return [rng.randint(-lim, lim) for _ in range(n)]


@pytest.fixture(scope="module")
def operands():
    rng = random.Random(0)
    a, b = [], []
    for bits in (4, 8, 12, 16):
        a += _rand_signed(3000, bits, rng)
        b += _rand_signed(3000, bits, rng)
    edge = [0, 1, -1, 2, 3, -3, 5, 6, 7, 127, -127, 128, 96, 192, 32767, -32767, 24576]
    a += [x for x in edge for _ in edge]
    b += [y for _ in edge for y in edge]
    return a, b


@pytest.fixture(scope="module")
def libroba():
    subprocess.run(["make", "-s", "-C", str(ROOT / "csrc")], check=True)
    return ctypes.CDLL(str(ROOT / "csrc" / "libroba.so"))


@pytest.mark.parametrize("mult", MULTIPLIERS)
def test_torch_matches_reference(operands, mult):
    a, b = operands
    want = [reference.mul(x, y, mult) for x, y in zip(a, b)]
    got = approx_mul(torch.tensor(a, dtype=torch.float64),
                     torch.tensor(b, dtype=torch.float64), mult)
    assert got.long().tolist() == want


@pytest.mark.parametrize("mult", MULTIPLIERS)
def test_c_matches_reference(operands, libroba, mult):
    a, b = operands
    want = [reference.mul(x, y, mult) for x, y in zip(a, b)]
    ca = np.array(a, dtype=np.int32)
    cb = np.array(b, dtype=np.int32)
    out = np.zeros(len(a), dtype=np.int32)
    p = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    libroba.roba_batch_s(MULTIPLIERS.index(mult), p(ca), p(cb), p(out), len(a))
    assert out.tolist() == want


@pytest.mark.parametrize("mult", MULTIPLIERS)
def test_matmul_matches_elementwise(mult):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(-127, 128, (37, 50), generator=g).double()
    w = torch.randint(-127, 128, (23, 50), generator=g).double()
    want = approx_mul(x[:, None, :], w[None], mult).sum(-1)
    assert torch.equal(approx_linear(x, w, mult), want)


@pytest.mark.parametrize("mult", ["exact", "roba_updn2", "mitchell"])
def test_conv_matches_loop(mult):
    g = torch.Generator().manual_seed(2)
    x = torch.randint(-127, 128, (2, 3, 6, 6), generator=g).double()
    w = torch.randint(-127, 128, (4, 3, 3, 3), generator=g).double()
    out = approx_conv2d(x, w, mult, padding=1)
    xp = torch.nn.functional.pad(x, (1, 1, 1, 1))
    for n, o, i, j in [(0, 0, 0, 0), (1, 3, 5, 5), (0, 2, 3, 1)]:
        patch = xp[n, :, i:i + 3, j:j + 3]
        assert out[n, o, i, j] == approx_mul(patch, w[o], mult).sum()
    if mult == "exact":
        assert torch.equal(out, torch.nn.functional.conv2d(x, w, padding=1))
