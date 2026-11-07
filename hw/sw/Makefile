# Build bare-metal programs for the multiplierless Ibex Simple System (RV32IC, no M).
#   make PROG=lod_test LOD=1     -> build/lod_test_lod.elf  (LOD custom instruction)
#   make PROG=mult_bench LOD=0   -> build/mult_bench_sw.elf (LOD via libgcc __clzsi2)
PROG ?= mult_bench
LOD  ?= 1
EXTRA_SRCS ?=

ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
IBEX_SW := $(ROOT)/hw/ibex/examples/sw/simple_system/common
CSRC   := $(ROOT)/csrc

CC      := $(HOME)/tools/riscv/bin/riscv32-unknown-elf-gcc
OBJDUMP := $(HOME)/tools/riscv/bin/riscv32-unknown-elf-objdump

ARCH   := rv32ic
SUFFIX := $(if $(filter 1,$(LOD)),lod,sw)
CFLAGS := -march=$(ARCH) -mabi=ilp32 -static -mcmodel=medany -Wall -O3 \
          -fvisibility=hidden -nostdlib -nostartfiles -ffreestanding \
          -I$(IBEX_SW) -I$(CSRC) -I$(ROOT)/hw/sw $(if $(filter 1,$(LOD)),-DROBA_LOD_INSN)

SRCS := $(PROG).c $(CSRC)/roba.c $(IBEX_SW)/simple_system_common.c $(IBEX_SW)/crt0.S $(EXTRA_SRCS)
OUT  := build/$(PROG)_$(SUFFIX).elf

# libgcc here is built for rv32imcb: its __mulsi3 is the shift-and-add routine (the paper's
# baseline), but some C helpers (e.g. 64-bit division) contain M-extension instructions that
# trap on this core. The disassembly check fails the build if any of them got linked in.
$(OUT): $(SRCS) $(CSRC)/roba.h $(CSRC)/roba_core.h $(wildcard *.h) | build
	$(CC) $(CFLAGS) -T $(ROOT)/hw/sw/link_1mb.ld $(SRCS) -o $@ -lgcc
	$(OBJDUMP) -d $@ > $(@:.elf=.dis)
	@if grep -qP '\t(mul|mulh|mulhsu|mulhu|div|divu|rem|remu)\t' $(@:.elf=.dis); then \
	  echo "ERROR: $@ contains M-extension instructions:"; \
	  grep -P '\t(mul|mulh|mulhsu|mulhu|div|divu|rem|remu)\t' $(@:.elf=.dis) | head -5; rm -f $@; exit 1; fi

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: clean
