#!/usr/bin/env bash
# Run an ELF on the Ibex Simple System model in its own directory.
# Usage: hw/run_sim.sh <program.elf> <run_dir>   -> <run_dir>/ibex_simple_system.log holds the program output
set -euo pipefail
HW="$(cd "$(dirname "$0")" && pwd)"
ELF="$(realpath "$1")"
mkdir -p "$2" && cd "$2"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/ibex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$HW/ibex/build/lowrisc_ibex_ibex_simple_system_0/sim-verilator/Vibex_simple_system" +ibex_tracer_enable=0 --meminit=ram,"$ELF" > sim_stdout.log 2>&1
cat ibex_simple_system.log
