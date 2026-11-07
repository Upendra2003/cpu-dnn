#!/usr/bin/env bash
# Build the Ibex Simple System Verilator model, multiplierless (RV32I, RV32M=None).
# Usage: hw/build_sim.sh            -> hw/ibex/build/lowrisc_ibex_ibex_simple_system_0/sim-verilator/Vibex_simple_system
set -euo pipefail
HW="$(cd "$(dirname "$0")" && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ibex
export PATH="$HOME/tools/riscv/bin:$PATH"
# libelf comes from the conda env (no system libelf-dev).
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAKEFLAGS="-j$(nproc)"

cd "$HW/ibex"
# Simple System has no BaseIsa parameter at this commit (RV32E=0 already means RV32I).
OPTS=$(util/ibex_config.py small fusesoc_opts \
  | sed -e 's/--RV32M=ibex_pkg::RV32MFast/--RV32M=ibex_pkg::RV32MNone/' -e 's/--BaseIsa=[^ ]* //')
echo "fusesoc options: $OPTS"
# The core file compiles the model with -g only; optimise the generated C++ for speed.
fusesoc --cores-root=. run --target=sim --setup --build lowrisc:ibex:ibex_simple_system \
  --make_options="OPT_FAST=-O3 OPT_SLOW=-O2 OPT_GLOBAL=-O2" $OPTS
