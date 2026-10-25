"""Approximate matrix multiplication: out[n, o] = sum_k approx_mul(x[n, k], w[o, k])."""
import torch
import torch.nn.functional as F

from .multipliers import approx_mul, factors, is_separable

# Max elements of the (N, O, K) product tensor materialised at once for Mitchell.
_CHUNK_ELEMS = 1 << 25


def approx_linear(x, w, mult, integer=True):
    """x: (N, K), w: (O, K), both float64. Returns (N, O) float64."""
    if is_separable(mult):
        fx = factors(x, mult, "a", integer)
        fw = factors(w, mult, "b", integer)
        return sum(a @ b.T for a, b in zip(fx, fw))
    n, k = x.shape
    rows = max(1, _CHUNK_ELEMS // (w.shape[0] * k))
    out = [approx_mul(x[i:i + rows, None, :], w[None], mult, integer).sum(-1)
           for i in range(0, n, rows)]
    return torch.cat(out)


def approx_conv2d(x, w, mult, stride=1, padding=0, integer=True):
    """x: (N, C, H, W), w: (O, C, kh, kw). Convolution via unfold + approx_linear."""
    n, _, h, wd = x.shape
    o, _, kh, kw = w.shape
    cols = F.unfold(x, (kh, kw), padding=padding, stride=stride)      # (N, C*kh*kw, L)
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (wd + 2 * padding - kw) // stride + 1
    cols = cols.transpose(1, 2).reshape(-1, cols.shape[1])
    out = approx_linear(cols, w.reshape(o, -1), mult, integer)        # (N*L, O)
    return out.reshape(n, oh * ow, o).transpose(1, 2).reshape(n, o, oh, ow)
