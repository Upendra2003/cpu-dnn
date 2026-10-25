#include "roba.h"

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

uint32_t roba_exact_u(uint32_t a, uint32_t b) { return a * b; }

uint32_t roba_mitchell_u(uint32_t a, uint32_t b)
{
    if (a == 0 || b == 0)
        return 0;
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

uint32_t roba_drum6_u(uint32_t a, uint32_t b)
{
    if (a == 0 || b == 0)
        return 0;
    uint32_t sa, sb;
    uint32_t ta = drum_trunc(a, &sa), tb = drum_trunc(b, &sb);
    return (ta * tb) << (sa + sb); /* 6x6-bit product: shift-and-add on RV32I */
}

uint32_t roba_dn1_u(uint32_t a, uint32_t b)
{
    if (a == 0 || b == 0)
        return 0;
    return roba1(a, b, LOD(a), LOD(b));
}

uint32_t roba_updn_u(uint32_t a, uint32_t b)
{
    if (a == 0 || b == 0)
        return 0;
    return roba1(a, b, pos_updn(a), pos_updn(b));
}

uint32_t roba_dn2_u(uint32_t a, uint32_t b)
{
    if (a == 0 || b == 0)
        return 0;
    uint32_t pa = LOD(a), pb = LOD(b);
    return roba1(a, b, pa, pb) + roba_dn1_u(a - (1u << pa), b - (1u << pb));
}

/* Second iteration on the signed error term (A - Ar)(B - Br). */
static inline uint32_t roba_updn_second(uint32_t a, uint32_t b, uint32_t (*second)(uint32_t, uint32_t))
{
    if (a == 0 || b == 0)
        return 0;
    uint32_t pa = pos_updn(a), pb = pos_updn(b);
    uint32_t r = roba1(a, b, pa, pb);
    int32_t ea = (int32_t)(a - (1u << pa)), eb = (int32_t)(b - (1u << pb));
    uint32_t ua = ea < 0 ? -(uint32_t)ea : (uint32_t)ea;
    uint32_t ub = eb < 0 ? -(uint32_t)eb : (uint32_t)eb;
    uint32_t e = second(ua, ub);
    return ((ea ^ eb) < 0) ? r - e : r + e;
}

uint32_t roba_updndn_u(uint32_t a, uint32_t b) { return roba_updn_second(a, b, roba_dn1_u); }
uint32_t roba_updn2_u(uint32_t a, uint32_t b) { return roba_updn_second(a, b, roba_updn_u); }

static uint32_t (*const UNSIGNED[ROBA_NUM_MULT])(uint32_t, uint32_t) = {
    roba_exact_u, roba_mitchell_u, roba_drum6_u, roba_dn1_u,
    roba_updn_u, roba_dn2_u, roba_updndn_u, roba_updn2_u,
};

int32_t roba_mul_s(enum roba_mult m, int32_t a, int32_t b)
{
    uint32_t ua = a < 0 ? -(uint32_t)a : (uint32_t)a;
    uint32_t ub = b < 0 ? -(uint32_t)b : (uint32_t)b;
    uint32_t r = UNSIGNED[m](ua, ub);
    return ((a ^ b) < 0) ? -(int32_t)r : (int32_t)r;
}

int64_t roba_dot_s(enum roba_mult m, const int32_t *a, const int32_t *b, int n)
{
    int64_t acc = 0;
    for (int i = 0; i < n; i++)
        acc += roba_mul_s(m, a[i], b[i]);
    return acc;
}

void roba_batch_s(int m, const int32_t *a, const int32_t *b, int32_t *out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = roba_mul_s((enum roba_mult)m, a[i], b[i]);
}
