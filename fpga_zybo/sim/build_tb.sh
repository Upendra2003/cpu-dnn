#!/usr/bin/env bash
# Build the Verilator testbench of the Zybo SoC (optional; Vivado is not needed for this).
#   sim/build_tb.sh [nolod]   -> sim/obj_dir/Vibex_zybo_soc
set -euo pipefail
cd "$(dirname "$0")/.."
FILES=$(sed 's|^|rtl/|' rtl/files.f)
if [ "${1:-}" = "nolod" ]; then
  FILES=$(echo "$FILES" | sed -e 's|rtl/ibex/lowrisc_ibex_ibex_core_0.1/rtl/ibex_alu.sv|rtl/ibex_base_nolod/ibex_alu.sv|' \
                              -e 's|rtl/ibex/lowrisc_ibex_ibex_core_0.1/rtl/ibex_decoder.sv|rtl/ibex_base_nolod/ibex_decoder.sv|' \
                              -e 's|rtl/ibex/lowrisc_ibex_ibex_pkg_0.1/rtl/ibex_pkg.sv|rtl/ibex_base_nolod/ibex_pkg.sv|')
fi
INC=$(find rtl/ibex -name "*.svh" -o -name "prim_assert.sv" | xargs -n1 dirname | sort -u | sed 's|^|-I|')
verilator ${TRACE:+--trace -CFLAGS -DTRACE} --cc --exe --build -j "$(nproc)" -O3 --top-module ibex_zybo_soc -Wno-fatal -Wno-lint -Wno-style \
  -Wno-UNOPTFLAT -Wno-MULTIDRIVEN --x-assign fast --x-initial unique \
  $INC $FILES rtl/ibex_zybo_soc.sv "$PWD/sim/tb_soc.cpp" -Mdir sim/obj_dir${TRACE:+_trace} -CFLAGS "-O2" > sim/build_tb.log 2>&1 \
  || { grep -E "Error|error:" sim/build_tb.log | head -20; exit 1; }
echo "built sim/obj_dir${TRACE:+_trace}/Vibex_zybo_soc"
