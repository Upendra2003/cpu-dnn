/* MULT kernel from the paper: 10K unsigned multiplications per routine and operand width,
 * timed with mcycle. Prints one CSV line per (routine, width):
 *   MULT,<routine>,<bits>,<cycles for 10K>,<instructions for 10K>,<checksum>
 */
#include <stdint.h>

#include "roba.h"
#include "util.h"

#define N 10000

static uint32_t A[N], B[N];

static const char *const NAMES[ROBA_NUM_MULT] = {
    "exact", "mitchell", "drum6", "roba_dn1", "roba_updn", "roba_dn2", "roba_updndn", "roba_updn2",
};

static inline uint32_t instret(void)
{
    uint32_t v;
    PCOUNT_READ(minstret, v);
    return v;
}

/* One timed loop per routine so each call site is a direct call, as in a real kernel. */
#define TIMED_LOOP(fn)                          \
    do {                                        \
        t0 = (uint32_t)cycles();                \
        i0 = instret();                         \
        for (int i = 0; i < N; i++)             \
            sum += fn(A[i], B[i]);              \
        i1 = instret();                         \
        t1 = (uint32_t)cycles();                \
    } while (0)

int main(void)
{
    static const uint32_t widths[] = {4, 8, 16, 31};
    pcount_enable(1);
    for (unsigned w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
        uint32_t mask = (1u << widths[w]) - 1u;
        for (int i = 0; i < N; i++) {
            A[i] = xorshift32() & mask;
            B[i] = xorshift32() & mask;
            if (!A[i]) A[i] = 1;
            if (!B[i]) B[i] = 1;
        }
        for (int m = 0; m < ROBA_NUM_MULT; m++) {
            uint32_t t0, t1, i0, i1, sum = 0;
            switch (m) {
            case ROBA_EXACT:    TIMED_LOOP(roba_exact_u); break;
            case ROBA_MITCHELL: TIMED_LOOP(roba_mitchell_u); break;
            case ROBA_DRUM6:    TIMED_LOOP(roba_drum6_u); break;
            case ROBA_DN1:      TIMED_LOOP(roba_dn1_u); break;
            case ROBA_UPDN:     TIMED_LOOP(roba_updn_u); break;
            case ROBA_DN2:      TIMED_LOOP(roba_dn2_u); break;
            case ROBA_UPDNDN:   TIMED_LOOP(roba_updndn_u); break;
            default:            TIMED_LOOP(roba_updn2_u); break;
            }
            puts("MULT,");
            puts(NAMES[m]);
            putchar(',');
            put_u32(widths[w]);
            putchar(',');
            put_u32(t1 - t0);
            putchar(',');
            put_u32(i1 - i0);
            putchar(',');
            puthex(sum);
            putchar('\n');
        }
    }
    puts("MULT_DONE\n");
    return 0;
}
