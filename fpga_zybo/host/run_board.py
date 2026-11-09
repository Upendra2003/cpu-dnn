"""Run the prebuilt programs on the Zybo and save the results (run on the PC connected to the board).

    python3 host/run_board.py --port /dev/ttyUSB1 --selftest            # first: LOD instruction check
    python3 host/run_board.py --port /dev/ttyUSB1                       # everything: BRAM programs, then DDR
    python3 host/run_board.py --port /dev/ttyUSB1 --only toyadmos int8 dn1 --mem ddr --limit 1   # smoke test
    python3 host/run_board.py --backend tb --only toyadmos int8 dn1 --mem ddr --limit 1          # no board:
                                                         # same flow on the Verilator testbench (sim/build_tb.sh)

For every batch of programs/manifest.json and memory mode (bram: everything in the 192 KB block RAM, the setting
that matches the Verilator study; ddr: code in BRAM, all data in the board's DDR) it loads the ELF, runs it and
checks every output against the bit-exact Python model (logit hash and prediction). For BRAM programs it also
checks the cycle counts against the Verilator run of the identical program, which must be equal.

Results (appended as they come, so an interrupted run can be resumed with the same command):
    results/<dataset>/<model>_board_images.csv     one row per input
    results/<dataset>/<model>_board_runs.csv       one row per batch (PL cycles, wall time, DDR traffic)
then results/board_summary.csv (one row per task/precision/method/memory) when everything is done, or with
    python3 host/run_board.py --summary-only
"""
import argparse
import csv
import json
import pathlib
import struct
import subprocess
import sys
import time
import zlib

FZ = pathlib.Path(__file__).resolve().parents[1]
BRAM_ADDR, DDR_ADDR = 0x0010_0000, 0x8000_0000
CHUNK = 1024
P_LOD, P_NONE = 0.829, 0.793          # relative core power used in the study (replace with reports/ numbers)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def elf_segments(path):
    """[(address, bytes)] of the PT_LOAD segments with file contents."""
    b = pathlib.Path(path).read_bytes()
    assert b[:4] == b"\x7fELF" and b[4] == 1, f"{path}: not a 32-bit ELF"
    phoff, = struct.unpack_from("<I", b, 28)
    phentsize, phnum = struct.unpack_from("<HH", b, 42)
    segs = []
    for i in range(phnum):
        typ, off, _, paddr, filesz = struct.unpack_from("<IIIII", b, phoff + i * phentsize)
        if typ == 1 and filesz:
            segs.append((paddr, b[off:off + filesz]))
    return segs


# ---------------------------------------------------------------- backends

class Board:
    """The ARM program of vitis/src/main.c over the USB-UART."""

    def __init__(self, port, baud):
        import serial                                    # pip install pyserial
        self.s = serial.Serial(port, baud, timeout=1)
        self.s.reset_input_buffer()
        self.mirror = {BRAM_ADDR: bytearray(), DDR_ADDR: bytearray()}
        self.s.write(b"P")
        line = self.expect("#PONG", 5)
        self.info = dict(kv.split("=") for kv in line.split()[1:])
        log("board:", line)

    def readline(self, timeout):
        t0 = time.time()
        buf = b""
        while time.time() - t0 < timeout:
            c = self.s.readline()
            if c:
                buf += c
                if buf.endswith(b"\n"):
                    return buf.decode(errors="replace").strip()
        raise TimeoutError(f"no line from the board within {timeout}s (got {buf!r})")

    def expect(self, prefix, timeout):
        while True:
            line = self.readline(timeout)
            if line.startswith(prefix):
                return line
            if line.startswith("#ERR"):
                raise RuntimeError(line)

    def write_block(self, addr, data):
        self.s.write(b"W" + struct.pack("<II", addr, len(data)) + data + struct.pack("<I", zlib.crc32(data)))
        self.expect("#OK", 10)

    def load(self, segs):
        """Send only the 1 KB chunks that differ from what the board's staging copy already holds."""
        sent = 0
        for addr, data in segs:
            base = BRAM_ADDR if addr < DDR_ADDR else DDR_ADDR
            m = self.mirror[base]
            off = addr - base
            if len(m) < off + len(data):
                m.extend(b"\0" * (off + len(data) - len(m)))
            for i in range(0, len(data), CHUNK):
                piece = data[i:i + CHUNK]
                if m[off + i:off + i + len(piece)] != piece:
                    self.write_block(addr + i, bytes(piece))
                    m[off + i:off + i + len(piece)] = piece
                    sent += len(piece)
        return sent

    def run(self, timeout):
        self.s.write(b"R" + struct.pack("<I", int(timeout)))
        lines = []
        while True:
            line = self.readline(timeout + 30)
            if line.startswith("#DONE"):
                return lines, {k: int(v) for k, v in (kv.split("=") for kv in line.split()[1:])}
            if line.startswith("#ERR"):
                raise RuntimeError(line)
            lines.append(line)


class Testbench:
    """Same flow on the Verilator model of the SoC (sim/obj_dir/Vibex_zybo_soc), for testing without a board."""

    def __init__(self, ddr_latency):
        self.exe = FZ / "sim" / "obj_dir" / "Vibex_zybo_soc"
        assert self.exe.exists(), "build it first: sim/build_tb.sh"
        self.lat = ddr_latency
        self.info = {"fclk_hz": "0", "backend": "verilator"}
        self.elf = None

    def load(self, segs):
        return 0

    def run(self, timeout):
        out = subprocess.run([str(self.exe), str(self.elf), str(self.lat)], capture_output=True, text=True,
                             timeout=timeout + 600).stdout.splitlines()
        tb = next(l for l in out if l.startswith("TB: PL cycles"))
        f = tb.split()
        stats = {"pl_cycles": int(f[3].rstrip(",")), "arm_us": 0, "ddr_reads": int(f[6]), "ddr_writes": int(f[8]),
                 "ddr_wait": int(f[10]), "err": int(any("data bus error" in l for l in out)), "timeout": 0}
        return [l for l in out if not l.startswith("TB:")], stats


# ---------------------------------------------------------------- results

def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def done_keys(out_dir):
    keys = set()
    for p in out_dir.glob("*/*_board_runs.csv"):
        for r in csv.DictReader(open(p)):
            if r.get("ok") == "1":
                keys.add((r["dataset"], r["bits"], r["method"], r["mem"], r["first"]))
    return keys


def summarize(out_dir):
    rows = {}
    for p in out_dir.glob("*/*_board_images.csv"):
        for r in csv.DictReader(open(p)):
            k = (r["dataset"], r["model"], int(r["bits"]), r["method"], r["mem"])
            rows.setdefault(k, []).append(r)
    summary = []
    for (ds, model, bits, method, mem), rs in sorted(rows.items()):
        cyc = sum(int(r["cycles"]) for r in rs) / len(rs)
        sim = [int(r["sim_cycles"]) for r in rs if r["sim_cycles"]]
        summary.append({"dataset": ds, "model": model, "bits": bits, "method": method, "mem": mem,
                        "lod": int(method in ("dn1", "dn2", "sqapx")), "inputs": len(rs),
                        "logits_match_ref": sum(r["logits_match"] == "1" for r in rs) / len(rs),
                        "cycles_per_input": cyc,
                        "sim_cycles_per_input": sum(sim) / len(sim) if len(sim) == len(rs) else "",
                        "cycles_equal_sim": (sum(r["cycles_equal_sim"] == "1" for r in rs) / len(rs)
                                             if len(sim) == len(rs) else ""),
                        "us_per_input": (sum(float(r["us_per_input"]) for r in rs) / len(rs)
                                         if all(r["us_per_input"] for r in rs) else "")})
    lod_factor = None                                        # P(core with LOD) / P(core without), from Vivado
    cores = FZ / "reports" / "fpga_cores.csv"
    if cores.exists():
        by = {r["variant"]: r for r in csv.DictReader(open(cores))}
        try:
            lod_factor = float(by["lod"]["core_power_w"]) / float(by["nolod"]["core_power_w"])
        except (KeyError, ValueError, ZeroDivisionError):
            pass
    for s in summary:
        base = next((t for t in summary if t["dataset"] == s["dataset"] and t["bits"] == s["bits"]
                     and t["mem"] == s["mem"] and t["method"] == "exact"), None)
        s["speedup_vs_exact"] = base["cycles_per_input"] / s["cycles_per_input"] if base else ""
        s["energy_vs_exact"] = (s["cycles_per_input"] * (P_LOD if s["lod"] else P_NONE)
                                / (base["cycles_per_input"] * P_NONE)) if base else ""
        # same, with the board's own power of the two cores (exact runs on a core without the LOD)
        s["energy_vs_exact_fpga"] = (s["cycles_per_input"] * (lod_factor if s["lod"] else 1.0)
                                     / base["cycles_per_input"]) if base and lod_factor else ""
    if summary:
        path = out_dir / "board_summary.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        log(f"wrote {path} ({len(summary)} rows)")
        for s in summary:
            log(f"{s['dataset']:16s} int{s['bits']:<2} {s['method']:5s} {s['mem']:4s} "
                f"match {s['logits_match_ref']:.2f} | {s['cycles_per_input'] / 1e6:8.2f} M cycles"
                f" | {s['speedup_vs_exact'] or 0:.2f}x"
                + (f" | sim equal {s['cycles_equal_sim']:.2f}" if s["cycles_equal_sim"] != "" else ""))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB1", help="the Zybo's USB-UART (Windows: COM5 etc.)")
    p.add_argument("--baud", type=int, default=460800)
    p.add_argument("--backend", choices=["board", "tb"], default="board")
    p.add_argument("--tb-ddr-latency", type=int, default=20)
    p.add_argument("--manifest", default=str(FZ / "programs" / "manifest.json"))
    p.add_argument("--out", default=str(FZ / "results"))
    p.add_argument("--mem", nargs="+", default=["bram", "ddr"], choices=["bram", "ddr"])
    p.add_argument("--only", nargs="*", default=[], help="filters, e.g. toyadmos int8 dn1")
    p.add_argument("--limit", type=int, default=0, help="stop after this many batches")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per batch")
    p.add_argument("--redo", action="store_true", help="run batches that already have results")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--selftest", action="store_true", help="run programs/selftest/lod_test_bram.elf and stop")
    args = p.parse_args()
    out = pathlib.Path(args.out)
    if args.summary_only:
        summarize(out)
        return

    man = json.loads(pathlib.Path(args.manifest).read_text())
    dev = Board(args.port, args.baud) if args.backend == "board" else Testbench(args.tb_ddr_latency)
    if args.selftest:                                        # LOD instruction vs a software loop, on the board
        elf = FZ / "programs" / "selftest" / "lod_test_bram.elf"
        if isinstance(dev, Testbench):
            dev.elf = elf
        dev.load(elf_segments(elf))
        lines, st = dev.run(120)
        for l in lines:
            log("   |", l)
        log("#DONE", st)
        ok = any("LOD_TEST PASS" in l for l in lines)
        log("SELFTEST", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    fclk = int(dev.info.get("fclk_hz", 0) or 0)
    have = set() if args.redo else done_keys(out)
    jobs = []
    for mem in args.mem:                                     # all BRAM programs first: they match the study
        for e in man["programs"]:
            if mem not in e["elf"]:
                continue
            tags = {e["dataset"], e["model"], f"int{e['bits']}", e["method"], mem}
            if any(t not in tags for t in args.only):
                continue
            if (e["dataset"], str(e["bits"]), e["method"], mem, str(e["first"])) in have:
                continue
            jobs.append((mem, e))
    if args.limit:
        jobs = jobs[:args.limit]
    log(f"{len(jobs)} batches to run ({len(have)} already done)")
    bad = 0
    for n, (mem, e) in enumerate(jobs, 1):
        elf = FZ / e["elf"][mem]
        segs = elf_segments(elf)
        if isinstance(dev, Testbench):
            dev.elf = elf
        t0 = time.time()
        sent = dev.load(segs)
        lines, st = dev.run(args.timeout)
        secs = time.time() - t0
        img = [l for l in lines if l.startswith("IMG,")]
        ok = len(img) == len(e["images"]) and "DNN_DONE" in lines and not st["err"] and not st["timeout"]
        for l in img:
            f = l.split(",")
            k = e["images"].index(int(f[1]))
            cyc = int(f[5])
            sim = e.get("sim_cycles", [None] * len(e["images"]))[k]
            match = f[4].lower() == e["ref_hash"][k]
            ok &= match
            append_csv(out / e["dataset"] / f"{e['model']}_board_images.csv", {
                "dataset": e["dataset"], "model": e["model"], "bits": e["bits"], "method": e["method"], "mem": mem,
                "image": int(f[1]), "label": int(f[2]), "board_pred": int(f[3]), "ref_pred": e["ref_pred"][k],
                "logits_match": int(match), "cycles": cyc, "sim_cycles": sim if sim is not None else "",
                "cycles_equal_sim": int(cyc == sim) if sim is not None else "",
                "us_per_input": cyc / fclk * 1e6 if fclk else "",
                **{f"op{i}_cycles": int(v) for i, v in enumerate(f[6:])}})
        append_csv(out / e["dataset"] / f"{e['model']}_board_runs.csv", {
            "dataset": e["dataset"], "model": e["model"], "bits": e["bits"], "method": e["method"], "mem": mem,
            "first": e["first"], "ok": int(ok), "bytes_sent": sent, "seconds": round(secs, 2), **st,
            "fclk_hz": fclk, "elf": e["elf"][mem]})
        bad += not ok
        log(f"{n}/{len(jobs)} {e['dataset']} int{e['bits']} {e['method']} {mem} b{e['first']}: "
            f"{'OK' if ok else 'MISMATCH/ERROR'} | {st['pl_cycles'] / 1e6:.1f} M PL cycles, "
            f"{st['arm_us'] / 1e6:.2f} s on board, {sent / 1024:.0f} KB sent, {secs:.0f} s total")
        if not ok:
            for l in lines[-8:]:
                log("   |", l)
    summarize(out)
    if bad:
        log(f"{bad} batches had mismatches or errors: see the *_board_runs.csv rows with ok=0")
        sys.exit(1)


if __name__ == "__main__":
    main()
