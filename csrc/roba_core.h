/* Inline copies of the routines in roba.c, for code that fuses the multiply into its own
 * multiply-accumulate loop (hw/sw/dnn.c). roba.c keeps its own definitions so that the measured
 * call-per-product programs stay instruction-for-instruction the same; tests/test_multipliers.py
 * checks that both give identical products (through roba_core_test.c).
 *
 * roba_<name>_nz(a, b) requires a != 0 and b != 0: the caller has already skipped zero operands,
 * so the check is not repeated for every product. Exact products (the exact and DRUM6 routines)
 * use the paper's shift-and-add loop over b (Listing 1) inline instead of calling libgcc's
 * __mulsi3, which runs the same loop.
 */
#ifndef ROBA_CORE_H
#define ROBA_CORE_H

#include <stdint.h>

#ifdef ROBA_LOD_INSN
/* R-type custom-0 instruction; match funct7/funct3 to the Ibex RTL implementation. */
static inline uint32_t LOD(uint32_t x)
{
    uint32_t r;
    asm volatile(".insn r 0x0b, 0, 0, %0, %1, x0" : "=r"(r) : "r"(x));
    return r;
}
#else
static inline uint32_t LOD(uint32_t x) { return 31u - (uint32_t)__builtin_clz(x); }
#endif

static inline uint32_t roba_shift_add(uint32_t a, uint32_t b)
{
    uint32_t r = 0;
    while (b) {
        if (b & 1u)
            r += a;
        a <<= 1;
        b >>= 1;
    }
    return r;
}

/* Leading-one position, rounded up when the next bit is set (RoBA nearest power of two). */
static inline uint32_t pos_updn(uint32_t x)
{
    uint32_t p = LOD(x);
    if (p > 0 && ((x >> (p - 1)) & 1u))
        p++;
    return p;
}

static inline uint32_t roba1(uint32_t a, uint32_t b, uint32_t pa, uint32_t pb)
{
    return (b << pa) + (a << pb) - ((1u << pa) << pb);
}

static inline uint32_t roba_exact_nz(uint32_t a, uint32_t b) { return roba_shift_add(a, b); }

static inline uint32_t roba_mitchell_nz(uint32_t a, uint32_t b)
{
    uint32_t pa = LOD(a), pb = LOD(b);
    uint32_t m = ((a - (1u << pa)) << pb) + ((b - (1u << pb)) << pa);
    uint32_t p = (1u << pa) << pb;
    return m < p ? p + m : m << 1;
}

static inline uint32_t drum_trunc(uint32_t x, uint32_t *s)
{
    uint32_t p = LOD(x);
    if (p < 6) {
        *s = 0;
        return x;
    }
    *s = p - 5;
    return (x >> *s) | 1u;
}

static inline uint32_t roba_drum6_nz(uint32_t a, uint32_t b)
{
    uint32_t sa, sb;
    uint32_t ta = drum_trunc(a, &sa), tb = drum_trunc(b, &sb);
    return roba_shift_add(ta, tb) << (sa + sb); /* 6x6-bit product */
}

static inline uint32_t roba_dn1_nz(uint32_t a, uint32_t b) { return roba1(a, b, LOD(a), LOD(b)); }
static inline uint32_t roba_dn1_z(uint32_t a, uint32_t b) { return (a == 0 || b == 0) ? 0 : roba_dn1_nz(a, b); }

static inline uint32_t roba_updn_nz(uint32_t a, uint32_t b) { return roba1(a, b, pos_updn(a), pos_updn(b)); }
static inline uint32_t roba_updn_z(uint32_t a, uint32_t b) { return (a == 0 || b == 0) ? 0 : roba_updn_nz(a, b); }

/* The second iteration's operands (the residuals) can be zero, so it uses the checked routine. */
static inline uint32_t roba_dn2_nz(uint32_t a, uint32_t b)
{
    uint32_t pa = LOD(a), pb = LOD(b);
    return roba1(a, b, pa, pb) + roba_dn1_z(a - (1u << pa), b - (1u << pb));
}

/* Second iteration on the signed error term (A - Ar)(B - Br). */
static inline uint32_t roba_updn_second(uint32_t a, uint32_t b, uint32_t (*second)(uint32_t, uint32_t))
{
    uint32_t pa = pos_updn(a), pb = pos_updn(b);
    uint32_t r = roba1(a, b, pa, pb);
    int32_t ea = (int32_t)(a - (1u << pa)), eb = (int32_t)(b - (1u << pb));
    uint32_t ua = ea < 0 ? -(uint32_t)ea : (uint32_t)ea;
    uint32_t ub = eb < 0 ? -(uint32_t)eb : (uint32_t)eb;
    uint32_t e = second(ua, ub);
    return ((ea ^ eb) < 0) ? r - e : r + e;
}

static inline uint32_t roba_updndn_nz(uint32_t a, uint32_t b) { return roba_updn_second(a, b, roba_dn1_z); }
static inline uint32_t roba_updn2_nz(uint32_t a, uint32_t b) { return roba_updn_second(a, b, roba_updn_z); }

#endif
