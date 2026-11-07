/* Integer kernels for the MLPerf Tiny networks on the multiplierless Ibex (used by hw/sw/tiny.c).
 *
 * Generated net.h defines BITS (4, 8 or 16) and the network; one of these selects the dot-product method:
 *   -DMULT_NZ=<roba_core.h routine>   fused multiply-accumulate: the routine (roba_exact_nz, roba_dn1_nz,
 *                                     roba_dn2_nz) inlined behind one zero check, weights as sign-magnitude words
 *   -DSQTAB                           exact products from a table of squares (polarization identity)
 *   -DSQAPX                           the same identity with approximate squares 2^(k+1)|v| - 2^(2k), k = LOD(|v|)
 * Layouts: activations HWC (unsigned, 0..QA), weights OHWI (signed, -QW..QW); a depthwise conv's weights are
 * C x KH x KW. The network input is unsigned pixels or, for audio features, signed values -QW..QW (in_t).
 * tiny/qnet.py (int_forward) computes the same integers in Python; every logit must match bit for bit.
 */
#ifndef TINY_KERNELS_H
#define TINY_KERNELS_H

#include <stdint.h>

#if BITS == 16
typedef uint16_t act_t;
typedef int16_t sin_t;                       /* signed network input */
typedef int16_t wq_t;
typedef uint16_t wsm_t;
typedef int64_t acc_t;                       /* int16 sums exceed 32 bits */
#define W_SIGN 15
#else
typedef uint8_t act_t;
typedef int8_t sin_t;
typedef int8_t wq_t;
typedef uint8_t wsm_t;
typedef int32_t acc_t;
#define W_SIGN 7
#endif
#define W_MAG(w) ((uint32_t)(w) & ((1u << W_SIGN) - 1u))

#if defined(SQTAB) || defined(SQAPX)
#define TABLE_METHOD 1
typedef wq_t wk_t;                           /* the identity needs the signed weight */
#else
#include "roba_core.h"
#ifndef MULT_NZ
#error "define MULT_NZ, SQTAB or SQAPX"
#endif
typedef wsm_t wk_t;                          /* sign-magnitude: bit W_SIGN = sign, below = |w| */
#endif

/* ---------------------------------------------------------------- squares */

#if defined(SQTAB)
/* S(v) = v^2 for v = -2QW .. QA + QW (x in 0..QA, or -QW..QW for a signed input; |w| <= QW), built at startup
 * without multiplying. */
static uint32_t STAB[QA + 3 * QW + 1];
static const uint32_t *SQ0;                  /* STAB + 2QW: SQ0[v] = v^2 */
#define SQ(v) ((int32_t)SQ0[(v)])
static void squares_init(void)
{
    uint32_t s = 0;
    SQ0 = STAB + 2 * QW;
    for (int n = 0; n <= QA + QW; n++) {
        STAB[2 * QW + n] = s;
        if (n <= 2 * QW)
            STAB[2 * QW - n] = s;
        s += 2u * (uint32_t)n + 1u;          /* (n+1)^2 = n^2 + 2n + 1 */
    }
}
#elif defined(SQAPX)
static inline uint32_t lod(uint32_t x)
{
    uint32_t r;
    asm volatile(".insn r 0x0b, 0, 0, %0, %1, x0" : "=r"(r) : "r"(x));
    return r;
}
/* 2^(k+1)|v| - 2^(2k) = (2|v| - 2^k) << k, k = LOD(|v|); 0 for v = 0 */
static inline int32_t sq_apx(int32_t v)
{
    uint32_t a = v < 0 ? -(uint32_t)v : (uint32_t)v;
    if (!a)
        return 0;
    uint32_t k = lod(a);
    return (int32_t)((2u * a - (1u << k)) << k);
}
#define SQ(v) sq_apx(v)
static void squares_init(void) {}
#else
static void squares_init(void) {}
#endif

/* ---------------------------------------------------------------- multiply-accumulate */

#if !defined(TABLE_METHOD)
/* acc + a*w, a >= 0 (every layer input is a pixel or a ReLU output) */
static inline __attribute__((always_inline)) acc_t mac(acc_t acc, uint32_t a, wk_t w)
{
    uint32_t ub = W_MAG(w);
    if (a == 0 || ub == 0)
        return acc;
    int32_t r = (int32_t)MULT_NZ(a, ub);
    int32_t s = -(int32_t)((uint32_t)w >> W_SIGN);        /* -1 if w < 0 */
    return acc + (acc_t)((r ^ s) - s);
}

/* acc + a*w for a signed a (the audio feature input) */
static inline __attribute__((always_inline)) acc_t mac_g(acc_t acc, int32_t a, wk_t w)
{
    uint32_t ub = W_MAG(w);
    if (a == 0 || ub == 0)
        return acc;
    uint32_t ua = a < 0 ? -(uint32_t)a : (uint32_t)a;
    int32_t r = (int32_t)MULT_NZ(ua, ub);
    int32_t s = (a >> 31) ^ -(int32_t)((uint32_t)w >> W_SIGN);
    return acc + (acc_t)((r ^ s) - s);
}
#endif

/* Weight preparation at startup (outside the timed region): sign-magnitude words, or the per-channel sum
 * of S(w) for the table methods. */
_Static_assert(sizeof(wk_t) == sizeof(wq_t), "in-place weight prep");
static void prep_weights(const wq_t *w, int O, int K, wk_t *wk, int32_t *sw)
{
    for (int o = 0; o < O; o++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++) {
            int32_t v = w[o * K + k];
#if defined(TABLE_METHOD)
            s += SQ(v);
            wk[o * K + k] = (wk_t)v;
#else
            wk[o * K + k] = (wk_t)((v < 0 ? (1u << W_SIGN) : 0u) | (uint32_t)(v < 0 ? -v : v));
#endif
        }
        if (sw)
            sw[o] = s;
    }
}

/* ---------------------------------------------------------------- rescale */

/* Exact 64-bit shift-and-add product; the loop runs over the bits of the smaller operand. */
static int64_t mul_wide(int64_t a, int64_t b)
{
    uint64_t ua = a < 0 ? (uint64_t)(-a) : (uint64_t)a;
    uint64_t ub = b < 0 ? (uint64_t)(-b) : (uint64_t)b;
    if (ub > ua) {
        uint64_t t = ua;
        ua = ub;
        ub = t;
    }
    uint64_t r = 0;
    while (ub) {
        if (ub & 1u)
            r += ua;
        ua <<= 1;
        ub >>= 1;
    }
    return ((a < 0) != (b < 0)) ? -(int64_t)r : (int64_t)r;
}

/* ---------------------------------------------------------------- convolution (also fully connected) */

/* Largest padded input and per-position square sums any layer needs; net.h sets PAD_MAX, PADS_MAX, POS_MAX. */
static act_t XPAD[PAD_MAX];
static sin_t XPADS[PADS_MAX];
#if defined(TABLE_METHOD)
static int32_t TSQ[POS_MAX];
#endif

/* y = layer(x). x: H x W x C, unsigned act_t, or signed sin_t when XS. Weights: O x KH x KW x C, or C x KH x KW
 * for a depthwise conv (DW: O == C, one filter per channel). Stride S, zero padding PH / PW. Output OH x OW x O:
 *   acc_out != 0  raw accumulators (a projection shortcut, consumed by the next conv's skip)
 *   logits != 0   int64 values without clamping (last layer)
 *   otherwise     clamp(rescale, 0, QA) as act_t (ReLU)
 * skip_act: identity shortcut (same shape as y, scaled by MS); skip_acc: projection accumulators. */
static inline __attribute__((always_inline)) void conv(
    const void *xv, const int XS, const int H, const int W, const int C, const int O, const int KH, const int KW,
    const int S, const int PH, const int PW, const int DW, const int OH, const int OW, const wk_t *w,
    const int32_t *sw, const int32_t *M, const int64_t *Cb, const uint8_t *SHf, const int64_t *MS,
    const act_t *skip_act, const acc_t *skip_acc, acc_t *acc_out, act_t *y, int64_t *logits)
{
    const int HP = H + 2 * PH, WP = W + 2 * PW;
    const act_t *xu = (const act_t *)xv;
    const sin_t *xs = (const sin_t *)xv;
    if (PH || PW) {
        for (int i = 0; i < HP * WP * C; i++) {
            if (XS)
                XPADS[i] = 0;
            else
                XPAD[i] = 0;
        }
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                for (int c = 0; c < C; c++) {
                    const int d = ((i + PH) * WP + (j + PW)) * C + c, s = (i * W + j) * C + c;
                    if (XS)
                        XPADS[d] = xs[s];
                    else
                        XPAD[d] = xu[s];
                }
        xu = XPAD;
        xs = XPADS;
    }
#define XAT(i) (XS ? (int32_t)xs[(i)] : (int32_t)xu[(i)])
#if defined(TABLE_METHOD)
    if (DW) {                                              /* per element: S(x) */
        for (int p = 0; p < HP * WP * C; p++)
            TSQ[p] = SQ(XAT(p));
    } else {                                               /* per pixel: sum over channels of S(x) */
        for (int p = 0; p < HP * WP; p++) {
            int32_t s = 0;
            for (int c = 0; c < C; c++)
                s += SQ(XAT(p * C + c));
            TSQ[p] = s;
        }
    }
#endif
    for (int oy = 0; oy < OH; oy++)
        for (int ox = 0; ox < OW; ox++) {
#if defined(TABLE_METHOD)
            int32_t sa = 0;                                /* sum S(x) over the window, shared by all o */
            if (!DW)
                for (int ky = 0; ky < KH; ky++)
                    for (int kx = 0; kx < KW; kx++)
                        sa += TSQ[(oy * S + ky) * WP + ox * S + kx];
#endif
            const wk_t *wo = w;
            for (int o = 0; o < O; o++) {
                acc_t acc = 0;
#if defined(TABLE_METHOD)
                int32_t d = 0;
                if (DW)
                    sa = 0;
#endif
                for (int ky = 0; ky < KH; ky++)
                    for (int kx = 0; kx < KW; kx++) {
                        const int base = ((oy * S + ky) * WP + ox * S + kx) * C;
                        if (DW) {
                            const int32_t a = XAT(base + o);
#if defined(TABLE_METHOD)
                            d += SQ(a + (int32_t)wo[0]);
                            sa += TSQ[base + o];
#else
                            acc = XS ? mac_g(acc, a, wo[0]) : mac(acc, (uint32_t)a, wo[0]);
#endif
                            wo += 1;
                        } else {
#if defined(TABLE_METHOD)
                            for (int c = 0; c < C; c++)
                                d += SQ(XAT(base + c) + (int32_t)wo[c]);
#else
                            if (XS)
                                for (int c = 0; c < C; c++)
                                    acc = mac_g(acc, xs[base + c], wo[c]);
                            else
                                for (int c = 0; c < C; c++)
                                    acc = mac(acc, xu[base + c], wo[c]);
#endif
                            wo += C;
                        }
                    }
#if defined(TABLE_METHOD)
                acc = (d - sa - sw[o]) >> 1;
#endif
                const int q = (oy * OW + ox) * O + o;
                if (acc_out) {
                    acc_out[q] = acc;
                    continue;
                }
                const int sh = SHf[o];
                int64_t v = mul_wide(acc, M[o]) + Cb[o] + ((int64_t)1 << (sh - 1));
                if (skip_act)
                    v += mul_wide(skip_act[q], MS[o]);
                if (skip_acc)
                    v += mul_wide(skip_acc[q], MS[o]);
                v >>= sh;
                if (logits)
                    logits[q] = v;
                else
                    y[q] = (act_t)(v < 0 ? 0 : (v > QA ? QA : v));
            }
        }
#undef XAT
}

/* KH x KW average pool (stride = kernel), then rescale to the next layer's scale. */
static inline __attribute__((always_inline)) void avgpool(const act_t *x, const int H, const int W, const int C,
                                                          const int KH, const int KW, const int32_t M, const int sh,
                                                          act_t *y)
{
    for (int oy = 0; oy < H / KH; oy++)
        for (int ox = 0; ox < W / KW; ox++)
            for (int c = 0; c < C; c++) {
                int32_t s = 0;
                for (int ky = 0; ky < KH; ky++)
                    for (int kx = 0; kx < KW; kx++)
                        s += x[((oy * KH + ky) * W + ox * KW + kx) * C + c];
                int64_t v = (mul_wide(s, M) + ((int64_t)1 << (sh - 1))) >> sh;
                y[(oy * (W / KW) + ox) * C + c] = (act_t)(v < 0 ? 0 : (v > QA ? QA : v));
            }
}

#endif
