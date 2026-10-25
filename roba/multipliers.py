"""Approximate multipliers from Rajiv & Kedia, "Software-Based Approximate Multiplication on
Multiplierless CPUs Using Custom Instruction" (DATE'26), plus the baselines they compare against.

All routines operate on float64 tensors. For integer-valued inputs (|x| < 2^26) every
intermediate is an integer below 2^53, so the results are bit-exact with the integer C routines
in csrc/roba.c. For real-valued inputs the same formulas give "float-domain" RoBA, i.e. the
rounding to a power of two is applied to the float itself (equivalent to working on the mantissa).

Signed operands follow the paper: multiply |a| and |b|, then restore the sign.

Every multiplier except Mitchell can be written as a sum of products of per-operand factors,
    approx(a, b) = sum_i fa_i(a) * fb_i(b)
e.g. RoBA-Dn1: Ar*b + a*Br - Ar*Br = Ar*b + (a - Ar)*Br.
That lets an approximate matmul be computed exactly as a few ordinary matmuls (see matmul.py).
"""
import torch

MULTIPLIERS = ["exact", "mitchell", "drum6", "roba_dn1", "roba_updn",
               "roba_dn2", "roba_updndn", "roba_updn2"]
APPROX_MULTIPLIERS = MULTIPLIERS[1:]


# ---------------------------------------------------------------- rounding to a power of two

def _pow2(e, like):
    return torch.ldexp(torch.ones_like(like), e)


def round_dn(u):
    """Largest power of two <= u (u >= 0); 0 for u == 0. Equivalent to 1 << LOD(u)."""
    m, e = torch.frexp(u)                      # u = m * 2^e, m in [0.5, 1)
    return torch.where(u > 0, _pow2(e - 1, u), torch.zeros_like(u))


def round_updn(u):
    """Nearest power of two as in RoBA: round up iff the bit after the leading one is set."""
    m, e = torch.frexp(u)
    e = torch.where(m >= 0.75, e, e - 1)       # m >= 0.75 <=> bit (LOD-1) is 1
    return torch.where(u > 0, _pow2(e, u), torch.zeros_like(u))


def _signed_round(x, rnd):
    return torch.sign(x) * rnd(x.abs())


# ---------------------------------------------------------------- factor decompositions

# Each RoBA variant is a list of rounding functions, one per iteration. Iteration i is applied
# to the residual left by the previous iteration: r_0 = x, r_{i+1} = r_i - sign(r_i)*R_i(|r_i|).
# The residual product (a - Ar)(b - Br) carries its own sign, which is how (UpDn)2 handles the
# signed error term.
ROBA_ITERATIONS = {
    "roba_dn1":    [round_dn],
    "roba_updn":   [round_updn],
    "roba_dn2":    [round_dn, round_dn],
    "roba_updndn": [round_updn, round_dn],
    "roba_updn2":  [round_updn, round_updn],
}

DRUM_BITS = {"drum6": 6}


def drum_truncate(x, k, integer=True):
    """DRUM: keep k bits starting at the leading one, force the lowest kept bit to 1.
    Integers with fewer than k+1 significant bits are kept exact."""
    u = x.abs()
    m, e = torch.frexp(u)
    lod = e - 1
    shift = lod - k + 1
    t = torch.floor(torch.ldexp(u, -shift))
    t = torch.floor(t / 2) * 2 + 1                         # t | 1
    d = torch.ldexp(t, shift)
    if integer:
        d = torch.where(lod >= k, d, u)
    d = torch.where(u > 0, d, torch.zeros_like(u))
    return torch.sign(x) * d


def factors(x, mult, side, integer=True):
    """Per-operand factor list for `mult`; side is 'a' or 'b'."""
    if mult == "exact":
        return [x]
    if mult in DRUM_BITS:
        return [drum_truncate(x, DRUM_BITS[mult], integer)]
    if mult not in ROBA_ITERATIONS:
        raise ValueError(f"{mult} has no factor decomposition")
    out, r = [], x
    for rnd in ROBA_ITERATIONS[mult]:
        rr = _signed_round(r, rnd)
        # Ar*b + (a - Ar)*Br
        out += [rr, r - rr] if side == "a" else [r, rr]
        r = r - rr
    return out


def is_separable(mult):
    return mult != "mitchell"


# ---------------------------------------------------------------- elementwise products

def mitchell(a, b):
    ua, ub = a.abs(), b.abs()
    ra, rb = round_dn(ua), round_dn(ub)
    p = ra * rb
    m = (ua - ra) * rb + (ub - rb) * ra        # 2^(ka+kb) * (xa + xb)
    res = torch.where(m < p, p + m, 2 * m)     # carry into the next characteristic
    res = torch.where((ua > 0) & (ub > 0), res, torch.zeros_like(res))
    return torch.sign(a) * torch.sign(b) * res


def approx_mul(a, b, mult, integer=True):
    """Elementwise approximate product (broadcasts like a * b)."""
    if mult == "mitchell":
        return mitchell(a, b)
    fa = factors(a, mult, "a", integer)
    fb = factors(b, mult, "b", integer)
    return sum(x * y for x, y in zip(fa, fb))
