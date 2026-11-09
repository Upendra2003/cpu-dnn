# RoBA on a real FPGA: Ibex + LOD on the Zybo (Zynq XC7Z010)

This folder runs the MLPerf Tiny experiments of the study (CIFAR-10 / ResNet-8, Speech Commands / DS-CNN,
ToyADMOS / autoencoder; int4, int8, int16; exact, square table, RoBA-Dn1, RoBA-Dn2, approximate square) on a
real multiplierless Ibex RISC-V core with the LOD instruction, built into the programmable logic of the Zybo.
The ARM processor on the same chip loads the programs, starts the Ibex and sends the results to the PC.

```
 PC  --USB-UART-->  ARM Cortex-A9 (vitis/src/main.c)
                       |  AXI4-Lite (control, program loading, console)        programmable logic
                       v
                    Ibex RV32IC + LOD, no multiplier  (rtl/ibex_zybo_soc.sv, 50 MHz)
                       |-- 192 KB block RAM, 1 cycle (code; and data when it fits)
                       '-- AXI4 -> HP0 -> DDR3 window (data of programs too big for the block RAM)
```

The programs are the study's own C code (`hw/sw/tiny.c`), already compiled (`programs/`), with the expected
outputs of the bit-exact Python model in `programs/manifest.json`. Nothing needs the training server.

## What you get at the end

| file | content |
|---|---|
| `results/<dataset>/<model>_board_images.csv` | one row per input: prediction, logit-hash check, cycles per layer |
| `results/<dataset>/<model>_board_runs.csv` | one row per batch: PL cycles, wall time, DDR reads/writes/wait |
| `results/board_summary.csv` | per task / precision / method / memory: cycles, speed-up, energy, checks |
| `reports/{lod,nolod}/*.rpt` | Vivado utilization, timing, power (hierarchical) of both cores |
| `reports/fpga_cores.csv` | LUTs / FFs / power of the Ibex with and without the LOD instruction |

Two memory settings are measured:

* **bram**: the whole program in the block RAM. Same timing as the Verilator simulation, so the cycle counts
  must be *exactly* the simulated ones (`cycles_equal_sim = 1`). These are the programs that fit in 192 KB:
  Speech Commands at every precision and most CIFAR-10 int4/int8 programs.
* **ddr**: code in the block RAM, all data (weights, activations, tables) in the board's DDR3. Every program
  runs this way, including those too big for the block RAM (all of ToyADMOS, CIFAR-10 int16). This is what a
  real microcontroller with external memory would see; the square table, for example, loses most of its lead.

## Steps

Everything is run from this folder (`cd fpga_zybo`).

### 1. Prerequisites (once)

* Vivado + Vitis 2023.x or 2024.x (Linux or Windows).
* **Digilent board files**, so Vivado knows the Zybo's DDR3 and MIO settings:
  download https://github.com/Digilent/vivado-boards (Code -> Download ZIP) and copy
  `new/board_files/*` into `<Vivado install>/data/boards/board_files/` (create the folder if needed).
  Check in Vivado's Tcl console: `get_board_parts *zybo*` must list `digilentinc.com:zybo:part0:...`.
* Python 3 with `pyserial` (`pip install pyserial`).
* The board: original Zybo (XC7Z010-1CLG400C, 512 MB DDR3). Jumper **JP5 on JTAG**, power jumper on USB (or a
  5 V supply), one micro-USB cable on PROG/UART. The USB serial port shows up as `/dev/ttyUSB1` on Linux
  (the second of two FTDI ports) or a `COM` port on Windows. On Linux you may need to be in the `dialout` group.
* Optional: Verilator 4.2+ to run the testbench in `sim/` (not needed for the board).

### 2. Build the hardware (about 20-40 minutes each)

```
vivado -mode batch -source vivado/build.tcl                   # Ibex with LOD: bitstream, XSA, reports
vivado -mode batch -source vivado/build.tcl -tclargs nolod    # stock Ibex: reports only (for area/power)
python3 host/collect_reports.py                               # -> reports/fpga_cores.csv
```

Check `reports/lod/timing_summary.rpt`: the worst negative slack (WNS) must be positive at 50 MHz.
If it is negative, rebuild at a lower clock: `-tclargs lod 40`. The cycle counts do not depend on the clock.

### 3. Build the ARM program

```
xsct vitis/create_app_xsct.tcl          # Vitis 2023.x
vitis -s vitis/create_app_vitis.py      # Vitis 2024.x (Unified IDE)
```

Either one ends with an `ibex_host.elf` under `build/vitis_ws/`. If the 2024 Python API differs on your
release, do the same in the IDE: platform component from `build/zybo_ibex_lod.xsa` (standalone,
ps7_cortexa9_0, stdout = ps7_uart_1), an empty application on it, add `vitis/src/main.c`, build.

### 4. Program the board

```
xsct vitis/run_app.tcl
```

(or Run in the Vitis IDE). A serial terminal at 115200 baud would show `#ZYBO-IBEX id=1be70103 ...`; the ARM
then switches the UART to 460800 baud, which is what the PC script uses. Close any terminal before step 5.

### 5. Run the experiments

```
python3 host/run_board.py --port /dev/ttyUSB1 --selftest                          # 1 s: LOD works?
python3 host/run_board.py --port /dev/ttyUSB1 --only speech_commands int8 dn1 --limit 1   # one batch
python3 host/run_board.py --port /dev/ttyUSB1                                     # everything
```

The full run is about 300 batches (5 inputs each). At 50 MHz the BRAM part takes roughly half an hour and the
DDR part a few hours (the first program of each configuration sends up to 600 KB over the UART; the other
batches only send their new inputs). The script appends results as it goes: if it stops, run the same command
again and it continues where it was. At the end it prints and writes `results/board_summary.csv`.

What to check:
* every row `logits_match = 1` (the board computes exactly what the Python model computes);
* for **bram** rows `cycles_equal_sim = 1` (the board's Ibex runs the same number of cycles as Verilator);
* `results/*/<model>_board_runs.csv` has `ok = 1` everywhere.

### 6. Bring the results back

Copy `results/` and `reports/` back to the project (they go in `fpga_zybo/results`, `fpga_zybo/reports`).

## The register map (for debugging)

The ARM sees the Ibex control block at `0x43C0_0000` (see `rtl/ibex_zybo_soc.sv` for details):

| offset | name | |
|---|---|---|
| 0x00 | CTRL | bit 0 = run (0 holds the Ibex in reset and clears done/counters/FIFO) |
| 0x04 | STATUS | bit 0 done, 1 FIFO empty, 2 FIFO full, 3 run, 4 bus error |
| 0x08 | FIFO | bit 31 valid, bits 7:0 next console byte |
| 0x0C/0x10 | CYCLES | PL cycles from run to done (64 bit) |
| 0x14/0x18 | LOAD_ADDR / LOAD_DATA | block RAM access while run = 0 (auto-increment) |
| 0x1C | DDR_BASE | physical address of the Ibex DDR window (0x1000_0000) |
| 0x20 | ID | 0x1BE7_0103 = LOD core, 192 KB; 0x1BE7_0003 = no-LOD core |
| 0x28/0x2C | DDR_READS / DDR_WRITES | Ibex data accesses that went to DDR |
| 0x30/0x34 | DDR_WAIT | cycles the Ibex waited for DDR (64 bit) |

The Ibex address map is the Verilator Simple System's (console at 0x20000, RAM at 0x100000), plus the DDR window
at 0x8000_0000, so the same C programs run unchanged.

## Folder contents

```
rtl/        ibex_zybo_soc.sv (the system), ibex_zybo_wrap.v (block-design wrapper), files.f,
            ibex/ (Ibex + lowRISC primitives, with the LOD patch), ibex_base_nolod/ (stock files for the
            comparison core), prim_xilinx/ (BUFGCE clock gate for the FPGA)
vivado/     build.tcl (project, block design, bitstream, reports), zybo.xdc
vitis/      src/main.c (ARM program), create_app_xsct.tcl / create_app_vitis.py, run_app.tcl
sw/         link_zybo_bram.ld, link_zybo_ddr.ld (the two memory layouts)
programs/   prebuilt RISC-V programs + manifest.json; selftest/lod_test_bram.elf
host/       run_board.py (PC side), collect_reports.py, make_programs.py (rebuilds programs/; needs the
            training server's checkpoints and code, not needed here)
sim/        Verilator testbench of the same RTL (build_tb.sh, tb_soc.cpp) and fake_board.py, which speaks
            the UART protocol on a pseudo-terminal: lets you test host/run_board.py without a board
```

## Troubleshooting

* **`Digilent Zybo board files not found`**: step 1.
* **No `#PONG`**: wrong port (try the other ttyUSB), a terminal still open, or the ARM program is not running
  (step 4 again). `--baud 115200` only works if `BAUD` in `vitis/src/main.c` is changed too.
* **`logits_match = 0`**: the board computed something else than the Python model. Run `--selftest`; check
  that `reports/lod/timing_summary.rpt` has positive slack; rebuild at a lower clock.
* **bram cycles differ from the simulation**: should not happen (same RTL, same program). Report the first
  differing row; the per-layer columns show where.
* **Timeouts on DDR programs**: raise `--timeout` (seconds per batch).
