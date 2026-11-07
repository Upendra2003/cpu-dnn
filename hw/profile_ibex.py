"""Where do the inference cycles go on Ibex?

For each (model, multiplier) this runs one test image of the original wrapper build (hw/sw/dnn.c with
MAC_CALL: one routine call per product, as in Fig. 11) with the Ibex instruction tracer on, then
charges every retired instruction's cycles (gap to the previous retirement, so stalls included)
to a category, using the debug info to map each PC to its (possibly inlined) C function and line:

  Multiplier routine        csrc/roba.c (+ roba_core.h) routines and libgcc __mulsi3 / __clzsi2
  Sign & zero handling      the inlined mul_s wrapper (operand signs, zero skip, call)
  Operand loads & stores    loads/stores in the MAC and conv loops
  Loop control & accumulate the remaining MAC/conv loop instructions (indices, branches, acc +=)
  Rescale & ReLU            fixed-point requantization (mul_wide), ReLU/int8 clamp
  Pooling                   2x2 max pooling (LeNet-5)
  Bookkeeping               cycle counters, argmax, logit hash, main loop
Printing and startup code are reported separately and excluded from the inference total.

The program is the measured one compiled with -g (the same machine code) and N_IMG = 1.
Traces are deleted right after parsing. Output: results/ibex/profile.csv, profile_mult_insns.csv

  python3 hw/profile_ibex.py --dataset fashion --jobs 16
"""
import argparse
import bisect
import collections
import concurrent.futures as cf
import csv
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hw"))

import run_ibex as ri  # noqa: E402

CATEGORIES = ["Multiplier routine", "Sign & zero handling", "Operand loads & stores",
              "Loop control & accumulate", "Rescale & ReLU", "Pooling", "Bookkeeping"]
EXCLUDED = ["Printing", "Startup"]
LOADS = {"lb", "lbu", "lh", "lhu", "lw", "c.lw", "c.lwsp"}
STORES = {"sb", "sh", "sw", "c.sw", "c.swsp"}
PRINT_FUNCS = {"putchar", "puts", "puthex", "put_u32", "put_i32", "put_u64"}
BOOK_FUNCS = {"argmax", "logit_hash", "cycles", "main", "pcount_enable"}
RESCALE_FUNCS = {"rescale", "mul_wide", "relu_q"}
MULT_SYMBOLS = {"__mulsi3", "__clzsi2"}
TOOLS = ri.TOOLS


def symbols(elf):
    out = subprocess.run([str(TOOLS / "riscv32-unknown-elf-nm"), "-n", str(elf)],
                         capture_output=True, text=True, check=True).stdout
    syms = [(int(a, 16), n) for a, t, n in (l.split() for l in out.splitlines() if len(l.split()) == 3)
            if t.lower() == "t"]
    return [a for a, _ in syms], [n for _, n in syms]


def addr2line(elf, pcs):
    """PC -> (innermost function, file, line) using inline info."""
    res = subprocess.run([str(TOOLS / "riscv32-unknown-elf-addr2line"), "-a", "-f", "-i", "-e", str(elf)],
                         input="\n".join(f"{p:x}" for p in pcs), capture_output=True, text=True, check=True).stdout
    info, cur, lines = {}, None, res.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("0x"):
            cur = int(lines[i], 16)
            func, loc = lines[i + 1], lines[i + 2]
            f, _, ln = loc.rpartition(":")
            ln = re.match(r"\d+", ln)
            info[cur] = (func, f, int(ln.group()) if ln else 0)   # first frame = innermost
            i += 3
        else:
            i += 1                                                  # outer inline frames
    return info


def pooling_lines():
    """Line numbers of the max-pooling blocks in dnn.c's LeNet-5 forward()."""
    src = (ri.SW / "dnn.c").read_text().splitlines()
    lines, inside = set(), False
    for n, text in enumerate(src, 1):
        if re.search(r"^\s*conv[12]\((x)?\);", text):
            inside = True
            continue
        if inside and "cycles()" in text:
            inside = False
        if inside:
            lines.add(n)
    return lines


def classify(pc, mnemonic, insn, info, sym_addr, sym_name, pool):
    k = bisect.bisect_right(sym_addr, pc) - 1
    sym = sym_name[k] if k >= 0 else ""
    func, path, line = info.get(pc, ("??", "", 0))
    if sym in MULT_SYMBOLS or path.endswith(("roba.c", "roba_core.h")):
        return "Multiplier routine"
    if sym in PRINT_FUNCS or path.endswith("simple_system_common.c") and sym != "pcount_enable":
        return "Printing"
    if sym in ("_start", "reset_handler", "default_exc_handler", "_init") or path.endswith("crt0.S") or func == "??":
        return "Startup"
    if func == "mul_s":
        return "Sign & zero handling"
    if func in RESCALE_FUNCS:
        return "Rescale & ReLU"
    if func == "max4" or (func == "forward" and line in pool):
        return "Pooling"
    if func in BOOK_FUNCS:
        return "Bookkeeping"
    if func in ("forward", "conv1", "conv2"):
        if mnemonic in LOADS or mnemonic in STORES:
            return "Operand loads & stores"
        return "Loop control & accumulate"
    return "Bookkeeping"


def insn_class(mnemonic, insn_hex):
    if len(insn_hex) == 8 and (int(insn_hex, 16) & 0x7F) == 0x0B:
        return "LOD"
    m = mnemonic.removeprefix("c.")
    if m in {"lb", "lbu", "lh", "lhu", "lw", "lwsp", "sb", "sh", "sw", "swsp"}:
        return "load/store"
    if m.startswith(("b", "j")) or m in {"jal", "jalr", "ret", "jr"}:
        return "branch/jump"
    if m in {"sll", "slli", "srl", "srli", "sra", "srai"}:
        return "shift"
    if m in {"add", "addi", "sub", "neg", "mv", "li", "lui", "addi16sp", "addi4spn"}:
        return "add/move"
    return "logic/compare"


def profile_one(task):
    ds, model, mult = task
    gen = ri.HW / "gen" / ds / model / mult
    run = ri.HW / "runs" / "profile" / ds / model / mult
    run.mkdir(parents=True, exist_ok=True)
    elf = run / "dnn_g.elf"
    ri.compile_elf(ri.SW / "dnn.c", elf, include_dirs=[gen / "b00000", gen],
                   defines=[*ri.config_defines(f"{mult}+call"), "PROFILE_ONE_IMAGE"], extra_flags=["-g"])
    env = dict(os.environ, LD_LIBRARY_PATH=f"{ri.CONDA_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    subprocess.run([str(ri.SIM), f"+ibex_tracer_file_base={run / 'trace'}", f"--meminit=ram,{elf}"],
                   cwd=run, env=env, capture_output=True, text=True, check=True)
    measured = int((run / "ibex_simple_system.log").read_text().split("IMG,")[1].split(",")[4])
    trace = next(run.glob("trace_*.log"))
    try:
        rows, pcs = [], set()
        with open(trace) as f:
            next(f)
            prev = None
            for line in f:
                p = line.split("\t")
                cyc, pc = int(p[1]), int(p[2], 16)
                rows.append((pc, p[4].strip(), p[3].strip(), cyc - prev if prev is not None else 0))
                prev = cyc
                pcs.add(pc)
    finally:
        trace.unlink()
    sym_addr, sym_name = symbols(elf)
    info = addr2line(elf, sorted(pcs))
    pool = pooling_lines()
    memo, cyc_cat, n_cat, cyc_mul = {}, collections.Counter(), collections.Counter(), collections.Counter()
    for pc, mnem, insn, d in rows:
        key = (pc, mnem)
        if key not in memo:
            memo[key] = classify(pc, mnem, insn, info, sym_addr, sym_name, pool)
        cat = memo[key]
        cyc_cat[cat] += d
        n_cat[cat] += 1
        if cat == "Multiplier routine":
            cyc_mul[insn_class(mnem, insn)] += d
    return task, cyc_cat, n_cat, cyc_mul, measured


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashion")
    p.add_argument("--models", nargs="+", default=["mlp", "lenet5"])
    p.add_argument("--mults", nargs="+", default=ri.MULTIPLIERS)
    p.add_argument("--jobs", type=int, default=16)
    args = p.parse_args()
    tasks = [(args.dataset, m, k) for m in args.models for k in args.mults]
    rows, mrows = [], []
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for (ds, model, mult), cyc, n, mul, measured in ex.map(profile_one, tasks):
            total = sum(cyc[c] for c in CATEGORIES)
            for c in CATEGORIES:
                rows.append({"dataset": ds, "model": model, "multiplier": mult, "category": c,
                             "cycles": cyc[c], "instructions": n[c], "share": cyc[c] / total,
                             "inference_cycles": total, "measured_mcycle": measured})
            for k, v in sorted(mul.items()):
                mrows.append({"dataset": ds, "model": model, "multiplier": mult, "insn_class": k,
                              "cycles": v, "share_of_routine": v / max(1, cyc["Multiplier routine"])})
            ri.log(f"{model:6s} {mult:12s} traced {total / 1e6:6.2f} M vs mcycle {measured / 1e6:6.2f} M | "
                   + " ".join(f"{c.split()[0]} {cyc[c] / total:5.1%}" for c in CATEGORIES)
                   + f" | excluded: print {cyc['Printing']}, startup {cyc['Startup']} ({time.time() - t0:.0f}s)")
    ri.write_csv(ri.OUT / "profile.csv", rows)
    ri.write_csv(ri.OUT / "profile_mult_insns.csv", mrows)


if __name__ == "__main__":
    main()
