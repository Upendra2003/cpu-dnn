/* Checks the LOD custom instruction against a software leading-one search. */
#include <stdint.h>

#include "util.h"

static inline uint32_t lod_insn(uint32_t x)
{
    uint32_t r;
    asm volatile(".insn r 0x0b, 0, 0, %0, %1, x0" : "=r"(r) : "r"(x));
    return r;
}

static uint32_t lod_ref(uint32_t x)
{
    uint32_t p = 0;
    for (uint32_t i = 0; i < 32; i++)
        if (x & (1u << i))
            p = i;
    return p;
}

static uint32_t checked, failed;

static void check(uint32_t x)
{
    uint32_t got = lod_insn(x), want = lod_ref(x);
    checked++;
    if (got != want) {
        if (failed < 8) {
            puts("MISMATCH x=");
            puthex(x);
            puts(" got=");
            put_u32(got);
            puts(" want=");
            put_u32(want);
            putchar('\n');
        }
        failed++;
    }
}

int main(void)
{
    check(0);
    for (uint32_t i = 0; i < 32; i++) {
        check(1u << i);                  /* single bit */
        check((1u << i) | 1u);           /* leading bit plus LSB */
        check(0xFFFFFFFFu >> (31 - i));  /* all ones below */
    }
    for (uint32_t i = 0; i < 20000; i++)
        check(xorshift32() >> (xorshift32() & 31u));

    /* Cycle cost of one LOD in a dependent chain. */
    uint32_t x = 0x12345678u;
    uint64_t t0 = cycles();
    for (int i = 0; i < 1000; i++)
        x = lod_insn(x | 0x80000000u) + x;
    uint64_t t1 = cycles();

    puts("LOD_TEST checked=");
    put_u32(checked);
    puts(" failed=");
    put_u32(failed);
    puts(" loop_cycles_per_iter_x100=");
    put_u32((uint32_t)(t1 - t0) / 10u);
    puts(" sink=");
    puthex(x);
    putchar('\n');
    puts(failed ? "LOD_TEST FAIL\n" : "LOD_TEST PASS\n");
    return 0;
}
