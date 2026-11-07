/* Host-side twin of roba_batch_s() built from the inline routines in roba_core.h, so the tests
 * can check them against roba.c and the Python reference. */
#include "roba.h"
#include "roba_core.h"

static uint32_t (*const CORE_NZ[ROBA_NUM_MULT])(uint32_t, uint32_t) = {
    roba_exact_nz, roba_mitchell_nz, roba_drum6_nz, roba_dn1_nz,
    roba_updn_nz, roba_dn2_nz, roba_updndn_nz, roba_updn2_nz,
};

void roba_core_batch_s(int m, const int32_t *a, const int32_t *b, int32_t *out, int n)
{
    for (int i = 0; i < n; i++) {
        uint32_t ua = a[i] < 0 ? -(uint32_t)a[i] : (uint32_t)a[i];
        uint32_t ub = b[i] < 0 ? -(uint32_t)b[i] : (uint32_t)b[i];
        uint32_t r = (ua == 0 || ub == 0) ? 0 : CORE_NZ[m](ua, ub);
        out[i] = ((a[i] ^ b[i]) < 0) ? -(int32_t)r : (int32_t)r;
    }
}
