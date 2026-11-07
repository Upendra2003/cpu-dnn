/* MLPerf Tiny networks as integer programs on the multiplierless Ibex.
 *
 * Built once per (dataset, model, seed, precision, method, image batch) by hw/run_tiny.py, which generates
 *   net.h     the network: BITS, buffers, weights, per-channel rescale, forward() as kernel calls
 *   images.h  N_IMG pre-quantized test inputs (HWC), their labels and test-set indices
 * and selects the dot-product method with -DMULT_NZ=..., -DSQTAB or -DSQAPX (see tiny_kernels.h).
 *
 * Prints one line per input:
 *   IMG,<test index>,<label>,<pred>,<logit hash>,<total cycles>,<cycles of op 0>,<op 1>,...
 */
#include <stdint.h>

#include "util.h"
#include "net.h"
#include "images.h"

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

int main(void)
{
    uint32_t lc[N_OPS];
    squares_init();
    net_init();
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
        put_u32((uint32_t)argmax(LOGITS, N_OUT));
        putchar(',');
        puthex(logit_hash(LOGITS, N_OUT));
        putchar(',');
        put_u32(total);
        for (int l = 0; l < N_OPS; l++) {
            putchar(',');
            put_u32(lc[l]);
        }
        putchar('\n');
    }
    puts("DNN_DONE\n");
    return 0;
}
