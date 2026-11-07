/* Small helpers shared by the Ibex programs: decimal printing, cycle counter, PRNG. */
#ifndef ROBA_UTIL_H
#define ROBA_UTIL_H

#include <stdint.h>

#include "simple_system_common.h"

static inline void put_u32(uint32_t v)
{
    char buf[11];
    int n = 0;
    do {
        buf[n++] = (char)('0' + v % 10u);
        v /= 10u;
    } while (v);
    while (n)
        putchar(buf[--n]);
}

static inline void put_i32(int32_t v)
{
    if (v < 0) {
        putchar('-');
        put_u32((uint32_t)(-(int64_t)v));
    } else {
        put_u32((uint32_t)v);
    }
}

/* 64-bit cycle counter (mcycle/mcycleh). */
static inline uint64_t cycles(void)
{
    uint32_t lo, hi, hi2;
    do {
        PCOUNT_READ(mcycleh, hi);
        PCOUNT_READ(mcycle, lo);
        PCOUNT_READ(mcycleh, hi2);
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

static inline void put_u64(uint64_t v)
{
    /* Split to avoid a 64-bit division routine. */
    if (v >= 1000000000ull) {
        uint32_t hi = 0;
        while (v >= 1000000000ull) {
            v -= 1000000000ull;
            hi++;
        }
        put_u32(hi);
        uint32_t lo = (uint32_t)v;
        for (uint32_t d = 100000000u; d; d /= 10u) {
            putchar((char)('0' + (lo / d) % 10u));
        }
    } else {
        put_u32((uint32_t)v);
    }
}

static uint32_t rng_state = 0x2545F491u;
static inline uint32_t xorshift32(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng_state = x;
}

#endif
