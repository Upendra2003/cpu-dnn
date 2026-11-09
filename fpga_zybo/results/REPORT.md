# Zybo board results: Ibex RV32IC + LOD (RoBA) on XC7Z010, 2026-11-09

Everything here is a copy of the run's output; the hardware/software sources stay in `../fpga_zybo/`.

| file | content |
|---|---|
| `acceptance.txt` | the four acceptance checks (all PASS) |
| `summary_table.md` | per dataset / precision / method / memory: cycles, time, speed-up, energy |
| `results/board_summary.csv` | same, machine-readable |
| `results/<dataset>/*_board_images.csv`, `*_board_runs.csv` | per input / per batch raw data |
| `results/logs/run_board.log` | full log of the 190-batch run (+2 smoke-test batches before it) |
| `reports/{lod,nolod}/*.rpt` | Vivado utilization, timing, power (incl. `power_hier_deep.rpt`) |
| `reports/fpga_cores.csv` | LUT / FF / power of the Ibex with and without LOD |

## Acceptance checks

| check | result |
|---|---|
| 1. `--selftest` | **SELFTEST PASS** (LOD_TEST checked=20097 failed=0) |
| 2. `logits_match = 1`, all rows | **960 / 960** (every input of every program bit-exact with the Python model) |
| 3. `cycles_equal_sim = 1`, bram rows | **240 / 240** (cycle-exact with Verilator); batches ok=1: 192 / 192 |
| 4. WNS at 50 MHz | **+1.027 ns** (LOD core), +1.083 ns (no-LOD core); TNS = THS = 0 |

DDR rows show `cycles_equal_sim = 0` by design: the reference counts are the BRAM (1-cycle) simulation, and DDR adds
real memory latency. CIFAR-10 and ToyADMOS have no BRAM programs in this manifest (too large for 192 KB).

## Hardware: area and power (Vivado 2024.2, xc7z010clg400-1, FCLK0 = 50 MHz)

| | Ibex core LUTs | core FFs | Ibex top LUTs | SoC BRAM | core power | Ibex top power |
|---|---|---|---|---|---|---|
| with LOD | 2471 | 723 | 3110 | 48 RAMB36 + 1 RAMB18 | 0.016 W | 0.020 W |
| without LOD | 2475 | 724 | 3117 | 48 RAMB36 + 1 RAMB18 | 0.016 W | 0.019 W |

* The LOD costs no measurable area (the ALU grows from 236 to 393 cells, the rest of the core absorbs it).
* Power is Vivado's **vectorless** estimate (default toggle rates), reported to 1 mW only: the LOD power factor is
  1.000 (core) or 1.053 (Ibex top), both within rounding of the paper's 1.045. `energy_vs_exact_fpga` uses the core
  factor (1.000); `energy_vs_exact` uses the paper's 0.829 / 0.793. Treat the FPGA power factor as "≈1, below 1 mW
  resolution", not as a precise measurement.

## Headline numbers (speed-up of RoBA-Dn1 vs exact shift-and-add)

| | int4 | int8 | int16 |
|---|---|---|---|
| Speech Commands, BRAM | 1.20x | 1.81x | 2.93x |
| Speech Commands, DDR | 1.07x | 1.36x | 1.97x |
| CIFAR-10 (ResNet-8), DDR | 1.06x | 1.31x | 1.95x |
| ToyADMOS (FC-AE), DDR | 1.04x | 1.40x | 2.21x |

The square table wins in BRAM at int8 (1.95x) but loses most of its lead in DDR (1.04x) and is slower than exact
at int4 in DDR (0.77x), as the study predicted.

## Changes made on the FPGA machine (scripts / mapping only; `programs/` untouched, Ibex config unchanged)

1. `vivado/build.tcl`: add the XHub board store (`~/.Xilinx/Vivado/<ver>/xhub/...`) to `board.repoPaths` before
   `create_project`; batch mode did not find the installed Digilent Zybo board files otherwise.
2. `vivado/build.tcl`: add the `` `include``d files (`prim_assert.sv`, `prim_flop_macros.sv`, `*.svh`) as Verilog
   headers; the block-design module reference failed without them.
3. `rtl/ibex_zybo_soc.sv`: `ram_decomp = "power"` on the 192 KB block RAM (attribute only). Without it Vivado 2024.2
   built it from 64 RAMB36 (> 60 on the XC7Z010) and placement failed; with it, byte-write mode, 48 RAMB36. Same
   logic and timing; the 240/240 cycle-exact BRAM results confirm the behaviour is unchanged.
4. `vivado/build.tcl`: `-jobs 4` instead of 8 (8 parallel out-of-context syntheses ran out of 15 GB RAM).
5. `vitis/create_app_vitis.py`: explicit `domain_name`, workspace recreated on each run, and the app's `lscript.ld`
   DDR region shrunk to end at 0x1000_0000 (Vitis default ran to 0x2000_0000, over the Ibex DDR window).
6. `vitis/src/main.c` (Vitis 2024.2 SDT flow): `xiltimer.h` instead of `xtime_l.h`; start the A9 global timer
   (the SDT boot code leaves it off, `arm_us` was 0); wrap `COUNTS_PER_SECOND`, which 2024.2 defines without
   parentheses (`.../2`, made times 4 too small); compute FCLK0 from the SLCR clock registers (no
   `XPAR_PS7_FPGA0_CLK_FREQ_HZ` in SDT), with the PS oscillator derived from the CPU clock (50 MHz on the
   original Zybo).
7. `host/collect_reports.py`: skip empty table cells (crashed on them); prefer `power_hier_deep.rpt`.
8. Added `vivado/power_detail.tcl` (deeper / XML power reports) and `host/make_board_report.py` (this folder).

Notes for rerunning: close any serial terminal on `/dev/ttyUSB1` first (a `picocom` session was stealing the
replies); run the host scripts with `env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH /usr/bin/python3`, because
the Xilinx settings point Python at Vivado's own 3.8 without `termios`/pyserial.
