"""Cycle and energy *estimates* for running the multiplications of a DNN on Ibex RV32I+LOD.

These numbers are placeholders until we measure on Ibex in Verilator:
  * Cycles per multiplication are read off Fig. 4 of the paper (MULT kernel, 10K unsigned
    multiplications, x10^6 cycles), so they include the kernel's loop/load/store overhead
    and carry roughly +-1 cycle of read-off error.
  * Power is relative to RV32MSingleCycle and is derived from the paper's text: RV32MNone+LOD
    uses 17.1% less power than single-cycle, and LOD adds 4.5% over RV32MNone.
  * The operands are treated as unsigned. Signed operands cost more (Fig. 3), especially for
    exact shift-and-add when a negative value lands in the loop operand.
"""

CYCLES_PER_MULT = {  # operand width -> multiplier -> cycles
    4:  {"exact": 30, "mitchell": 20, "drum6": 33, "roba_dn1": 17, "roba_updn": 22,
         "roba_dn2": 21, "roba_updn2": 31, "roba_updndn": 30},
    8:  {"exact": 44, "mitchell": 23, "drum6": 50, "roba_dn1": 19, "roba_updn": 26,
         "roba_dn2": 27, "roba_updn2": 43, "roba_updndn": 39},
    16: {"exact": 77, "mitchell": 25, "drum6": 65, "roba_dn1": 20, "roba_updn": 29,
         "roba_dn2": 32, "roba_updn2": 53, "roba_updndn": 46},
    32: {"exact": 108, "mitchell": 26, "drum6": 68, "roba_dn1": 20, "roba_updn": 30,
         "roba_dn2": 32, "roba_updn2": 55, "roba_updndn": 48},
}

POWER_NONE_LOD = 1 - 0.171
POWER_NONE = POWER_NONE_LOD / 1.045


def estimate(macs, mult, bits):
    """Multiplication cycles and relative energy for one inference."""
    cycles = macs * CYCLES_PER_MULT[bits][mult]
    # Exact runs on the plain multiplierless core; approximate routines need the LOD core.
    power = POWER_NONE if mult == "exact" else POWER_NONE_LOD
    return {"mult_cycles": cycles, "energy_rel": cycles * power}
