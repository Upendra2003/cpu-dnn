/* Integer-only int8 inference of the MLP / LeNet-5 on the multiplierless Ibex.
 *
 * Built once per (dataset, model, configuration, image batch) by hw/run_ibex.py, which generates
 *   model.h   int8 weights W<l>, fixed-point rescale M<l> (int32) and offset C<l> (int64)
 *   images.h  N_IMG pre-quantized int8 test images, their labels and test-set indices
 * and picks how every dot product is computed:
 *   -DMULT_NZ=<roba_core.h routine>   fused MAC (default): the multiplier is inlined into the loop
 *                                     behind one zero check; weights are stored as sign-magnitude
 *                                     bytes, and layers fed by ReLU skip the activation's sign
 *   -DMAC_CALL -DMULT_U=<roba.c routine>   the original wrapper: mul_s() calls the routine per product
 *   -DSQTAB                           no multiplier: the polarization identity with a square table
 *         sum_k x_k w_k = ( sum_k S(x_k + w_k) - sum_k S(x_k) - sum_k S(w_k) ) / 2,   S(v) = v^2
 *       sum S(x) is computed once per layer input (per output position in a conv) and shared by
 *       all output channels, so it costs 1 + 1/O table lookups per multiply-accumulate
 *   -DZERO_SKIP                       ablation: fully connected layers loop only over the nonzero
 *                                     activations, gathered once per layer input; conv taps with a
 *                                     zero input are skipped (the multiplier paths always do this)
 * Tables built from the weights (sign-magnitude bytes, S, sum S(w)) are made at startup, outside the
 * timed region; on a real device they would be stored that way.
 *
 * model.h may define INPUT_NONNEG (raw pixels with the input normalization folded into layer 1)
 * and C0_PER_POSITION (LeNet-5: that fold gives conv1 a bias per output position, because zero
 * padding of raw pixels is not zero padding of normalized ones).
 *
 * Layer output (before the next layer's int8 input):
 *   acc = sum_k mul(x_k, w_k)                       exact int32 sum of (approximate) products
 *   y   = (acc * M + C + 2^(SH-1)) >> SH            = round(acc * sx*sw / s_next + b / s_next)
 *   x'  = clamp(y, 0, 127)                          ReLU + int8 saturation
 * The last layer keeps y as an int64 logit. hw/run_ibex.py repeats this arithmetic in Python,
 * so every logit must match bit for bit.
 *
 * Prints one line per image:
 *   IMG,<test index>,<label>,<pred>,<logit hash>,<total cycles>,<cycles layer 0>,<layer 1>,...
 */
#include <stdint.h>

#include "util.h"
#include "model.h"
#include "images.h"

/* hw/profile_ibex.py traces a single image (an instruction trace is ~1 GB per image). */
#ifdef PROFILE_ONE_IMAGE
#undef N_IMG
#define N_IMG 1
#endif

/* ---------------------------------------------------------------- multiply-accumulate */

#if defined(SQTAB)

static uint16_t STAB[509];                          /* n^2 for n = -254..254 (int8 + int8) */
/* Indexing through a pointer set at startup lets GCC fold the +254 into the load's offset
 * (lhu 508(a)); with STAB[254 + v] it adds 254 on every lookup. */
static const uint16_t *SQ0;                         /* STAB + 254, so SQ0[v] = v^2 */
#define SQ(v) ((int32_t)SQ0[v])

static void tables_init(void)
{
    uint32_t s = 0;
    SQ0 = STAB + 254;
    for (int n = 0; n <= 254; n++) {
        STAB[254 + n] = STAB[254 - n] = (uint16_t)s;
        s += 2u * (uint32_t)n + 1u;                 /* (n+1)^2 = n^2 + 2n + 1 */
    }
}

/* sum_k S(w[o][k]) per output channel */
static void sw_init(const int8_t *w, int K, int O, int32_t *sw)
{
    for (int o = 0; o < O; o++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++)
            s += SQ(*w++);
        sw[o] = s;
    }
}

#define WEIGHTS(l) W##l

#elif defined(MAC_CALL)

#include "roba.h"

#ifndef MULT_U
#error "define MULT_U, e.g. -DMULT_U=roba_dn1_u"
#endif

/* Sign-magnitude wrapper used for every multiplier, exact included; zero operands are skipped. */
static inline int32_t mul_s(int32_t a, int32_t b)
{
    if (a == 0 || b == 0)
        return 0;
    uint32_t ua = a < 0 ? -(uint32_t)a : (uint32_t)a;
    uint32_t ub = b < 0 ? -(uint32_t)b : (uint32_t)b;
    uint32_t r = MULT_U(ua, ub);
    return ((a ^ b) < 0) ? -(int32_t)r : (int32_t)r;
}

typedef int8_t wt_t;
#define MAC_G(acc, a, w) ((acc) + mul_s((a), (w)))
#define MAC_R(acc, a, w) ((acc) + mul_s((a), (w)))
#define WEIGHTS(l) W##l
static void tables_init(void) {}

#else

#include "roba_core.h"

#ifndef MULT_NZ
#error "define MULT_NZ, e.g. -DMULT_NZ=roba_dn1_nz (or -DMAC_CALL / -DSQTAB)"
#endif

/* Weights as sign-magnitude bytes: bit 7 = sign, bits 6..0 = |w| (int8 weights are in -127..127). */
typedef uint8_t wt_t;
#define W_MAG(w)  ((uint32_t)((w) & 0x7Fu))
#define W_MASK(w) (-(int32_t)((uint32_t)(w) >> 7))  /* -1 if negative, else 0 */

/* acc + a*w for a signed activation (the network input) */
static inline __attribute__((always_inline)) int32_t mac_g(int32_t acc, int32_t a, wt_t w)
{
    uint32_t ub = W_MAG(w);
    if (a == 0 || ub == 0)
        return acc;
    uint32_t ua = a < 0 ? -(uint32_t)a : (uint32_t)a;
    int32_t r = (int32_t)MULT_NZ(ua, ub);
    int32_t s = (a >> 31) ^ W_MASK(w);
    return acc + ((r ^ s) - s);
}

/* acc + a*w for an activation known to be >= 0 (after ReLU) */
static inline __attribute__((always_inline)) int32_t mac_r(int32_t acc, int32_t a, wt_t w)
{
    uint32_t ub = W_MAG(w);
    if (a == 0 || ub == 0)
        return acc;
    int32_t r = (int32_t)MULT_NZ((uint32_t)a, ub);
    int32_t s = W_MASK(w);
    return acc + ((r ^ s) - s);
}

#define MAC_G(acc, a, w) mac_g((acc), (a), (w))
#define MAC_R(acc, a, w) mac_r((acc), (a), (w))
#define WEIGHTS(l) WS##l

#if defined(MODEL_MLP)
static uint8_t WS0[sizeof(W0)], WS1[sizeof(W1)], WS2[sizeof(W2)];
#define SM_LAYERS SM(W0, WS0); SM(W1, WS1); SM(W2, WS2)
#else
static uint8_t WS0[sizeof(W0)], WS1[sizeof(W1)], WS2[sizeof(W2)], WS3[sizeof(W3)], WS4[sizeof(W4)];
#define SM_LAYERS SM(W0, WS0); SM(W1, WS1); SM(W2, WS2); SM(W3, WS3); SM(W4, WS4)
#endif
#define SM(W, WS) for (unsigned i = 0; i < sizeof(W); i++) \
        WS[i] = (uint8_t)((W[i] < 0 ? 0x80u : 0u) | (uint32_t)(W[i] < 0 ? -W[i] : W[i]))
static void tables_init(void) { SM_LAYERS; }

#endif

/* The network input is signed (normalized pixels) unless the normalization was folded away. */
#if defined(INPUT_NONNEG)
#define MAC_IN MAC_R
#else
#define MAC_IN MAC_G
#endif

/* ---------------------------------------------------------------- rescale */

/* Exact 32x32 -> 64-bit shift-and-add product, used only for the per-output rescale. */
static int64_t mul_wide(int32_t a, int32_t b)
{
    uint64_t ua = a < 0 ? (uint64_t)(-(int64_t)a) : (uint64_t)a;
    uint32_t ub = b < 0 ? -(uint32_t)b : (uint32_t)b;
    uint64_t r = 0;
    while (ub) {
        if (ub & 1u)
            r += ua;
        ua <<= 1;
        ub >>= 1;
    }
    return ((a < 0) != (b < 0)) ? -(int64_t)r : (int64_t)r;
}

static inline int64_t rescale(int32_t acc, int32_t m, int64_t c)
{
    return (mul_wide(acc, m) + c + ((int64_t)1 << (SH - 1))) >> SH;
}

static inline int8_t relu_q(int64_t y)
{
    return (int8_t)(y < 0 ? 0 : (y > 127 ? 127 : y));
}

#define STORE(y, o, v, logits)                                           \
    do {                                                                 \
        if (logits) ((int64_t *)(y))[o] = (v);                           \
        else ((int8_t *)(y))[o] = relu_q(v);                             \
    } while (0)

/* ---------------------------------------------------------------- fully connected layers */

/* Nonzero activations of the current layer input (ZERO_SKIP): values and positions. */
#if defined(ZERO_SKIP)
static int8_t NZV[784];
static uint16_t NZI[784];
static inline int gather_nz(const int8_t *x, int K)
{
    int n = 0;
    for (int k = 0; k < K; k++)
        if (x[k]) {
            NZV[n] = x[k];
            NZI[n] = (uint16_t)k;
            n++;
        }
    return n;
}
#endif

/* y[o] = layer(x); logits != 0 keeps the int64 outputs instead. */
#if defined(SQTAB) && defined(ZERO_SKIP)
/* A zero activation adds S(w) - S(0) - S(w) = 0, so only nonzero ones are visited; their S(w) is
 * subtracted in the loop because the per-channel sum S(w) would include the skipped ones. */
#define LINEAR(x, K, W, SW, O, M, C, y, logits, MAC)                     \
    do {                                                                 \
        int n_ = gather_nz((x), (K));                                    \
        int32_t sa_ = 0;                                                 \
        for (int j_ = 0; j_ < n_; j_++)                                  \
            sa_ += SQ(NZV[j_]);                                          \
        const int8_t *w_ = (W);                                          \
        for (int o_ = 0; o_ < (O); o_++) {                               \
            int32_t d_ = 0;                                              \
            for (int j_ = 0; j_ < n_; j_++) {                            \
                int32_t wk_ = w_[NZI[j_]];                               \
                d_ += SQ(NZV[j_] + wk_) - SQ(wk_);                       \
            }                                                            \
            w_ += (K);                                                   \
            STORE(y, o_, rescale((d_ - sa_) >> 1, (M)[o_], (C)[o_]), logits); \
        }                                                                \
    } while (0)
#elif defined(SQTAB)
#define LINEAR(x, K, W, SW, O, M, C, y, logits, MAC)                     \
    do {                                                                 \
        int32_t sa_ = 0;                                                 \
        for (int k_ = 0; k_ < (K); k_++)                                 \
            sa_ += SQ((x)[k_]);                                          \
        const int8_t *w_ = (W);                                          \
        for (int o_ = 0; o_ < (O); o_++) {                               \
            int32_t d_ = 0;                                              \
            for (int k_ = 0; k_ < (K); k_++)                             \
                d_ += SQ((x)[k_] + w_[k_]);                              \
            w_ += (K);                                                   \
            STORE(y, o_, rescale((d_ - sa_ - (SW)[o_]) >> 1, (M)[o_], (C)[o_]), logits); \
        }                                                                \
    } while (0)
#elif defined(ZERO_SKIP)
#define LINEAR(x, K, W, SW, O, M, C, y, logits, MAC)                     \
    do {                                                                 \
        int n_ = gather_nz((x), (K));                                    \
        const wt_t *w_ = (W);                                            \
        for (int o_ = 0; o_ < (O); o_++) {                               \
            int32_t acc_ = 0;                                            \
            for (int j_ = 0; j_ < n_; j_++)                              \
                acc_ = MAC(acc_, NZV[j_], w_[NZI[j_]]);                  \
            w_ += (K);                                                   \
            STORE(y, o_, rescale(acc_, (M)[o_], (C)[o_]), logits);       \
        }                                                                \
    } while (0)
#else
#define LINEAR(x, K, W, SW, O, M, C, y, logits, MAC)                     \
    do {                                                                 \
        const wt_t *w_ = (W);                                            \
        for (int o_ = 0; o_ < (O); o_++) {                               \
            int32_t acc_ = 0;                                            \
            for (int k_ = 0; k_ < (K); k_++)                             \
                acc_ = MAC(acc_, (x)[k_], w_[k_]);                       \
            w_ += (K);                                                   \
            STORE(y, o_, rescale(acc_, (M)[o_], (C)[o_]), logits);       \
        }                                                                \
    } while (0)
#endif

/* Layer l of the model: weights, per-channel square sums (SQTAB), rescale and offset. */
#define FC(x, K, l, O, y, logits, MAC) LINEAR(x, K, WEIGHTS(l), SW##l, O, M##l, C##l, y, logits, MAC)

/* ---------------------------------------------------------------- output helpers */

static uint32_t logit_hash(const int64_t *z, int n)
{
    uint32_t h = 0x811C9DC5u;
    for (int i = 0; i < n; i++) {
        h = (h << 5) | (h >> 27);
        h ^= (uint32_t)z[i];
        h = (h << 5) | (h >> 27);
        h ^= (uint32_t)((uint64_t)z[i] >> 32);
    }
    return h;
}

static int argmax(const int64_t *z, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (z[i] > z[best])
            best = i;
    return best;
}

/* ---------------------------------------------------------------- models */

#if defined(MODEL_MLP)

#define N_LAYERS 3
static int8_t h1[256], h2[128];
static int64_t z[10];
#if defined(SQTAB)
static int32_t SW0[256], SW1[128], SW2[10];
#endif

static void model_init(void)
{
#if defined(SQTAB)
    sw_init(W0, 784, 256, SW0);
    sw_init(W1, 256, 128, SW1);
    sw_init(W2, 128, 10, SW2);
#endif
}

static void forward(const int8_t *x, uint32_t *lc)
{
    uint32_t t = (uint32_t)cycles(), u;
    FC(x, 784, 0, 256, h1, 0, MAC_IN);
    u = (uint32_t)cycles(); lc[0] = u - t; t = u;
    FC(h1, 256, 1, 128, h2, 0, MAC_R);
    u = (uint32_t)cycles(); lc[1] = u - t; t = u;
    FC(h2, 128, 2, 10, z, 1, MAC_R);
    u = (uint32_t)cycles(); lc[2] = u - t;
}

#elif defined(MODEL_LENET5)

#define N_LAYERS 5
static int8_t c1[6][28][28], p1[6][14][14], c2[16][10][10], p2[16 * 5 * 5], f1[120], f2[84];
static int64_t z[10];
#if defined(SQTAB)
static int32_t SW0[6], SW1[16], SW2[120], SW3[84], SW4[10];
#endif

#if defined(C0_PER_POSITION)
#define C0_AT(o, i, j) C0[((o) * 28 + (i)) * 28 + (j)]
#else
#define C0_AT(o, i, j) C0[o]
#endif

static void model_init(void)
{
#if defined(SQTAB)
    sw_init(W0, 25, 6, SW0);
    sw_init(W1, 150, 16, SW1);
    sw_init(W2, 400, 120, SW2);
    sw_init(W3, 120, 84, SW3);
    sw_init(W4, 84, 10, SW4);
#endif
}

#if defined(SQTAB)

/* With the table, sum S(x) over each 5x5 window is shared by all output channels: a box sum of the
 * per-pixel squares T, computed once per layer. Without ZERO_SKIP every window has all its taps, so
 * conv1 reads a zero-padded copy of the image; zero inputs add exactly 0 either way. */
static int8_t xp[32][32];
static int32_t T[32][32], R[32][28], SA[28][28];

/* SA[i][j] = sum of T over the 5x5 window at (i, j); T is n x n, SA is (n-4) x (n-4). */
static void box5(int n)
{
    for (int y = 0; y < n; y++)
        for (int j = 0; j < n - 4; j++)
            R[y][j] = T[y][j] + T[y][j + 1] + T[y][j + 2] + T[y][j + 3] + T[y][j + 4];
    for (int i = 0; i < n - 4; i++)
        for (int j = 0; j < n - 4; j++)
            SA[i][j] = R[i][j] + R[i + 1][j] + R[i + 2][j] + R[i + 3][j] + R[i + 4][j];
}

#if defined(ZERO_SKIP)
#define TAP(d, a, w) do { int32_t a_ = (a), w_ = (w); if (a_) d += SQ(a_ + w_) - SQ(w_); } while (0)
#define CONV_ACC(d, i, j, o, SW) (((d) - SA[i][j]) >> 1)
#else
#define TAP(d, a, w) (d += SQ((a) + (w)))
#define CONV_ACC(d, i, j, o, SW) (((d) - SA[i][j] - (SW)[o]) >> 1)
#endif

/* conv1: 1x28x28, 5x5, pad 2 -> 6x28x28 */
static void conv1(const int8_t *x)
{
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++) {
            int v = (i >= 2 && i < 30 && j >= 2 && j < 30) ? x[(i - 2) * 28 + (j - 2)] : 0;
            xp[i][j] = (int8_t)v;
            T[i][j] = SQ(v);
        }
    box5(32);
    for (int o = 0; o < 6; o++) {
        const int8_t *w = W0 + o * 25;
        for (int i = 0; i < 28; i++)
            for (int j = 0; j < 28; j++) {
                int32_t d = 0;
                for (int di = 0; di < 5; di++)
                    for (int dj = 0; dj < 5; dj++)
                        TAP(d, xp[i + di][j + dj], w[di * 5 + dj]);
                c1[o][i][j] = relu_q(rescale(CONV_ACC(d, i, j, o, SW0), M0[o], C0_AT(o, i, j)));
            }
    }
}

/* conv2: 6x14x14, 5x5, no pad -> 16x10x10 */
static void conv2(void)
{
    for (int y = 0; y < 14; y++)
        for (int x = 0; x < 14; x++) {
            int32_t s = 0;
            for (int c = 0; c < 6; c++)
                s += SQ(p1[c][y][x]);
            T[y][x] = s;
        }
    box5(14);
    for (int o = 0; o < 16; o++) {
        const int8_t *w = W1 + o * 150;
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++) {
                int32_t d = 0;
                for (int c = 0; c < 6; c++)
                    for (int di = 0; di < 5; di++)
                        for (int dj = 0; dj < 5; dj++)
                            TAP(d, p1[c][i + di][j + dj], w[c * 25 + di * 5 + dj]);
                c2[o][i][j] = relu_q(rescale(CONV_ACC(d, i, j, o, SW1), M1[o], C1[o]));
            }
    }
}

#else

/* conv1: 1x28x28, 5x5, pad 2 -> 6x28x28 */
static void conv1(const int8_t *x)
{
    for (int o = 0; o < 6; o++)
        for (int i = 0; i < 28; i++)
            for (int j = 0; j < 28; j++) {
                int32_t acc = 0;
                const wt_t *w = WEIGHTS(0) + o * 25;
                for (int di = 0; di < 5; di++) {
                    int ii = i + di - 2;
                    if (ii < 0 || ii >= 28) continue;
                    for (int dj = 0; dj < 5; dj++) {
                        int jj = j + dj - 2;
                        if (jj < 0 || jj >= 28) continue;
                        acc = MAC_IN(acc, x[ii * 28 + jj], w[di * 5 + dj]);
                    }
                }
                c1[o][i][j] = relu_q(rescale(acc, M0[o], C0_AT(o, i, j)));
            }
}

/* conv2: 6x14x14, 5x5, no pad -> 16x10x10 */
static void conv2(void)
{
    for (int o = 0; o < 16; o++)
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++) {
                int32_t acc = 0;
                const wt_t *w = WEIGHTS(1) + o * 150;
                for (int c = 0; c < 6; c++)
                    for (int di = 0; di < 5; di++)
                        for (int dj = 0; dj < 5; dj++)
                            acc = MAC_R(acc, p1[c][i + di][j + dj], w[c * 25 + di * 5 + dj]);
                c2[o][i][j] = relu_q(rescale(acc, M1[o], C1[o]));
            }
}

#endif

static inline int8_t max4(int8_t a, int8_t b, int8_t c, int8_t d)
{
    int8_t m = a > b ? a : b;
    m = m > c ? m : c;
    return m > d ? m : d;
}

static void forward(const int8_t *x, uint32_t *lc)
{
    uint32_t t = (uint32_t)cycles(), u;
    conv1(x);
    for (int c = 0; c < 6; c++)
        for (int i = 0; i < 14; i++)
            for (int j = 0; j < 14; j++)
                p1[c][i][j] = max4(c1[c][2 * i][2 * j], c1[c][2 * i][2 * j + 1],
                                   c1[c][2 * i + 1][2 * j], c1[c][2 * i + 1][2 * j + 1]);
    u = (uint32_t)cycles(); lc[0] = u - t; t = u;
    conv2();
    for (int c = 0; c < 16; c++)
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                p2[c * 25 + i * 5 + j] = max4(c2[c][2 * i][2 * j], c2[c][2 * i][2 * j + 1],
                                              c2[c][2 * i + 1][2 * j], c2[c][2 * i + 1][2 * j + 1]);
    u = (uint32_t)cycles(); lc[1] = u - t; t = u;
    FC(p2, 400, 2, 120, f1, 0, MAC_R);
    u = (uint32_t)cycles(); lc[2] = u - t; t = u;
    FC(f1, 120, 3, 84, f2, 0, MAC_R);
    u = (uint32_t)cycles(); lc[3] = u - t; t = u;
    FC(f2, 84, 4, 10, z, 1, MAC_R);
    u = (uint32_t)cycles(); lc[4] = u - t;
}

#else
#error "model.h must define MODEL_MLP or MODEL_LENET5"
#endif

int main(void)
{
    uint32_t lc[N_LAYERS];
    tables_init();
    model_init();
    pcount_enable(1);
    for (int n = 0; n < N_IMG; n++) {
        uint32_t t0 = (uint32_t)cycles();
        forward(IMG[n], lc);
        uint32_t total = (uint32_t)cycles() - t0;
        puts("IMG,");
        put_u32(IMG_INDEX[n]);
        putchar(',');
        put_u32(LABEL[n]);
        putchar(',');
        put_u32((uint32_t)argmax(z, 10));
        putchar(',');
        puthex(logit_hash(z, 10));
        putchar(',');
        put_u32(total);
        for (int l = 0; l < N_LAYERS; l++) {
            putchar(',');
            put_u32(lc[l]);
        }
        putchar('\n');
    }
    puts("DNN_DONE\n");
    return 0;
}
