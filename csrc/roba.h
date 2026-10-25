/* Approximate multiplication routines (Rajiv & Kedia, DATE'26) for multiplierless RV32I cores.
 *
 * LOD(x) returns the bit position of the leading one of x (x != 0). On the Ibex core with the
 * custom LOD instruction, define ROBA_LOD_INSN and adjust the .insn encoding in roba.c to match
 * the opcode/funct fields used in the RTL. Otherwise __builtin_clz is used (host builds; on plain
 * RV32I this becomes a slow libgcc routine, which is exactly the bottleneck LOD removes).
 *
 * Overflow is not handled (same as the paper): products must fit in 32 bits.
 */
#ifndef ROBA_H
#define ROBA_H

#include <stdint.h>

enum roba_mult {
    ROBA_EXACT = 0,
    ROBA_MITCHELL,
    ROBA_DRUM6,
    ROBA_DN1,
    ROBA_UPDN,
    ROBA_DN2,
    ROBA_UPDNDN,
    ROBA_UPDN2,
    ROBA_NUM_MULT
};

uint32_t roba_exact_u(uint32_t a, uint32_t b);
uint32_t roba_mitchell_u(uint32_t a, uint32_t b);
uint32_t roba_drum6_u(uint32_t a, uint32_t b);
uint32_t roba_dn1_u(uint32_t a, uint32_t b);
uint32_t roba_updn_u(uint32_t a, uint32_t b);
uint32_t roba_dn2_u(uint32_t a, uint32_t b);
uint32_t roba_updndn_u(uint32_t a, uint32_t b);
uint32_t roba_updn2_u(uint32_t a, uint32_t b);

/* Signed: multiply magnitudes, then restore the sign. */
int32_t roba_mul_s(enum roba_mult m, int32_t a, int32_t b);

/* Approximate dot product with 64-bit accumulation (int8/int16 DNN layers). */
int64_t roba_dot_s(enum roba_mult m, const int32_t *a, const int32_t *b, int n);

/* Batch helper used by the Python tests. */
void roba_batch_s(int m, const int32_t *a, const int32_t *b, int32_t *out, int n);

#endif
