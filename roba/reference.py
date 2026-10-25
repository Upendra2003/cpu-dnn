"""Scalar pure-Python reference for the integer routines, written with bit operations the way
the paper's C code is (LOD = position of the leading one). Used only to test multipliers.py and
csrc/roba.c against an independent implementation."""


def lod(x):
    return x.bit_length() - 1


def _pos_updn(x):
    p = lod(x)
    if p > 0 and (x >> (p - 1)) & 1:
        p += 1
    return p


def _roba1(a, b, pos):
    if a == 0 or b == 0:
        return 0
    pa, pb = pos(a), pos(b)
    return (b << pa) + (a << pb) - (1 << (pa + pb))


def _roba2(a, b, pos1, pos2):
    if a == 0 or b == 0:
        return 0
    pa, pb = pos1(a), pos1(b)
    r = (b << pa) + (a << pb) - (1 << (pa + pb))
    ea, eb = a - (1 << pa), b - (1 << pb)
    e = _roba1(abs(ea), abs(eb), pos2)
    return r - e if (ea < 0) != (eb < 0) else r + e


def _drum_trunc(x, k):
    p = lod(x)
    if p < k:
        return x
    s = p - k + 1
    return ((x >> s) | 1) << s


def _mitchell(a, b):
    if a == 0 or b == 0:
        return 0
    pa, pb = lod(a), lod(b)
    m = ((a - (1 << pa)) << pb) + ((b - (1 << pb)) << pa)
    p = 1 << (pa + pb)
    return p + m if m < p else m << 1


UNSIGNED = {
    "exact": lambda a, b: a * b,
    "mitchell": _mitchell,
    "drum6": lambda a, b: _drum_trunc(a, 6) * _drum_trunc(b, 6),
    "roba_dn1": lambda a, b: _roba1(a, b, lod),
    "roba_updn": lambda a, b: _roba1(a, b, _pos_updn),
    "roba_dn2": lambda a, b: _roba2(a, b, lod, lod),
    "roba_updndn": lambda a, b: _roba2(a, b, _pos_updn, lod),
    "roba_updn2": lambda a, b: _roba2(a, b, _pos_updn, _pos_updn),
}


def mul(a, b, mult):
    r = UNSIGNED[mult](abs(a), abs(b))
    return -r if (a < 0) != (b < 0) else r
