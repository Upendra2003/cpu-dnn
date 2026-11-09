# Context for Claude Code on the FPGA machine

This folder was prepared on the training server and unzipped here, on the machine with Vivado/Vitis and the
Zybo board. Its job: build the FPGA design, run the prebuilt RISC-V programs on the board, and bring back
`results/` and `reports/`. `README.md` has the user-facing steps; this file is the background you need.

## The study in three lines

Approximate multiplication (RoBA-Dn1/Dn2, using a new LOD = leading-one-detect instruction) against exact
shift-and-add multiplication and an exact table of squares, on a RISC-V core with **no hardware multiplier**
(Ibex RV32IC). Neural networks from MLPerf Tiny (ResNet-8, DS-CNN, an FC autoencoder) at int4/int8/int16.
On Verilator all of this is done; the board must reproduce it on real hardware and add area/power numbers.

## What must hold (the acceptance checks)

1. `python3 host/run_board.py --port <port> --selftest` prints `SELFTEST PASS`.
2. Every row of `results/*/*_board_images.csv` has `logits_match = 1`: the board's outputs equal the bit-exact
   Python model (hashes in `programs/manifest.json`).
3. For `mem = bram` rows: `cycles_equal_sim = 1`. The block RAM has the same 1-cycle timing as the Verilator
   Simple System and the programs are byte-identical in their code, so the cycle counts must be equal. A
   difference means the RTL or the build differs from the simulation; investigate, do not paper over it.
4. `reports/lod/timing_summary.rpt` has positive WNS at the chosen clock.

## Rules

* **Do not edit `programs/`.** They are compiled on the training server from the study's code and checked
  against its Python model. Rebuilding needs `host/make_programs.py` with the server's checkpoints (not here).
  If a program really must change, say so and stop; do not recompile with a local toolchain.
* Keep the Ibex configuration as in `rtl/ibex_zybo_soc.sv` (RV32IC, `RV32MNone`, no icache, no writeback
  stage, no branch predictor). Changing it changes the cycle counts and breaks the comparison.
* The RTL was checked here only on Verilator (`sim/`): the DDR path with a modelled 20-cycle latency, all
  outputs bit-exact; the console, loader and register map through `host/run_board.py` against
  `sim/fake_board.py`. **Vivado, Vitis and the board itself were not available on the training server**, so
  the Tcl/Python build scripts are untested there: expect to fix small tool-version issues (IP versions,
  Vitis 2024 Python API names, board-file version). Fix the scripts, not the design, and note what you changed.
* The ARM program keeps a 16 MB staging buffer in its own DDR and gives the Ibex the window at physical
  `0x1000_0000`. If the Vitis linker script ever places ARM data above `0x1000_0000`, it prints a warning:
  then shrink the ARM's DDR region in `lscript.ld` to end below `0x1000_0000`.
* Results files are appended to; rerunning resumes. Use `--redo` only on purpose.

## Useful facts

* Board: original Zybo, `xc7z010clg400-1`, board part `digilentinc.com:zybo:part0:*` (Digilent board files
  required). PS UART1 (MIO 48/49) is the USB-UART. LED LD0 (M14) = Ibex done.
* PL clock FCLK0 = 50 MHz (`vivado/build.tcl` second argument changes it). Ibex control registers at
  `0x43C0_0000`, Ibex block RAM 192 KB, DDR window 16 MB at physical `0x1000_0000` (Ibex address `0x8000_0000`).
* UART: 460800 baud after start-up (115200 for the first banner). Protocol in `vitis/src/main.c` header.
* Verilator testbench (optional, needs Verilator >= 4.2): `sim/build_tb.sh`, then
  `sim/obj_dir/Vibex_zybo_soc programs/selftest/lod_test_bram.elf`. `python3 host/run_board.py --backend tb ...`
  runs the whole flow on it; `python3 sim/fake_board.py` + `--port /dev/pts/N` tests the UART path.
* Energy: `host/collect_reports.py` derives the LOD power factor P(core with LOD)/P(core without) from the two
  Vivado builds; `run_board.py --summary-only` then adds `energy_vs_exact_fpga` next to the paper-based
  `energy_vs_exact`. Vivado's power here is vectorless (default toggle rates); say so when reporting it.

## When done

Report: the summary table (`results/board_summary.csv`), the four checks above, the WNS and clock, LUT/FF/BRAM
of both cores and the LOD power factor, and any script changes you had to make.
