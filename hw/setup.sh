#!/usr/bin/env bash
# One-time setup of the Ibex simulation flow (no sudo needed):
#   conda env "ibex" (Verilator 4.228, elfutils, zlib, FuseSoC), lowRISC RV32 GCC in ~/tools/riscv,
#   Ibex pinned to the last commit before the CHERIoT rebase, plus our LOD patch.
set -euo pipefail
HW="$(cd "$(dirname "$0")" && pwd)"
IBEX_COMMIT=7b5df75a041affe56e8c235260f98a09b3319008
TOOLCHAIN=lowrisc-toolchain-gcc-rv32imcb-20220210-1

source ~/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q '^ibex '; then
  conda create -y -n ibex -c conda-forge python=3.11 verilator=4.228 elfutils zlib make
fi

if [ ! -x ~/tools/riscv/bin/riscv32-unknown-elf-gcc ]; then
  mkdir -p ~/tools/riscv
  curl -sfL "https://github.com/lowRISC/lowrisc-toolchains/releases/download/20220210-1/$TOOLCHAIN.tar.xz" \
    | tar -C ~/tools/riscv -xJ --strip-components=1
fi

if [ ! -d "$HW/ibex" ]; then
  git clone https://github.com/lowRISC/ibex.git "$HW/ibex"
  git -C "$HW/ibex" checkout -b roba-lod "$IBEX_COMMIT"
  git -C "$HW/ibex" apply "$HW/patches/0001-ibex-lod-instruction.patch"
fi

conda activate ibex
pip install -q -r "$HW/ibex/python-requirements.txt"
"$HW/build_sim.sh"
