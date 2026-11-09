"""Build the RISC-V programs the board runs (run this on the training server, where the checkpoints live).

    python3 fpga_zybo/host/make_programs.py                # all 3 tasks, seed 0, int4/8/16, every method

For every (task, precision, method) it writes the same 20 test inputs as the Verilator study, in batches of 5, as
    fpga_zybo/programs/<dataset>/<model>/int<bits>/<method>/b<first>_bram.elf   (only if it fits the 192 KB BRAM)
    fpga_zybo/programs/<dataset>/<model>/int<bits>/<method>/b<first>_ddr.elf    (always: data in the DDR window)
and fpga_zybo/programs/manifest.json with, per batch, the labels, the expected predictions and logit hashes from
the bit-exact Python model, and (for BRAM programs) the cycle counts of the identical program on Verilator.

The C sources are the study's own (hw/sw/tiny.c, tiny_kernels.h, csrc/roba*.{c,h}); only the linker script
differs (fpga_zybo/sw/link_zybo_{bram,ddr}.ld).
"""
import argparse
import csv
import hashlib
import json
import pathlib
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
FZ = ROOT / "fpga_zybo"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hw"))
import run_ibex as ri  # noqa: E402
import run_tiny as rt  # noqa: E402
from tiny import data as D  # noqa: E402
from tiny import qnet as Q  # noqa: E402

TASKS = [("cifar10", "resnet8"), ("speech_commands", "dscnn"), ("toyadmos", "fcae")]
OBJDUMP = ri.TOOLS / "riscv32-unknown-elf-objdump"


def text_digest(elf):
    """Digest of the code's instruction sequence (address + mnemonic per instruction, constants ignored).
    Two programs with equal digests execute the same instructions at the same addresses, so on the 1-cycle block
    RAM they take the same cycles; they may differ in constants such as the stack address."""
    dis = subprocess.run([str(OBJDUMP), "-d", "-j", ".vectors", "-j", ".text", str(elf)], check=True,
                         capture_output=True, text=True).stdout
    seq = [" ".join(l.split("\t")[0:1] + l.split("\t")[2:3]) for l in dis.splitlines()
           if l.startswith("  ") and l.count("\t") >= 2]
    return hashlib.sha256("\n".join(seq).encode()).hexdigest()


def sim_cycles(dataset, model, tag):
    """{(bits, method, image): (cycles, [op cycles])} from a Verilator results CSV, if present."""
    path = ROOT / "results" / "tiny" / dataset / f"{model}_ibex_images{tag}.csv"
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        ops = [int(float(r[k])) for k in sorted((k for k in r if k.startswith("op")), key=lambda k: int(k[2:-7]))]
        out[(int(r["bits"]), r["method"], int(r["image"]))] = (int(float(r["cycles"])), ops)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", nargs="+", default=[t[0] for t in TASKS])
    args = p.parse_args()
    torch.set_num_threads(16)
    manifest = {"images": args.images, "batch": args.batch, "seed": args.seed, "bram_bytes": 192 * 1024,
                "programs": []}
    old = FZ / "programs" / "manifest.json"
    if old.exists():                                   # keep the tasks not rebuilt this time
        manifest["programs"] = [e for e in json.loads(old.read_text())["programs"] if e["dataset"] not in args.tasks]
    for dataset, model in TASKS:
        if dataset not in args.tasks:
            continue
        data = D.load(dataset, "cpu", val=5000)
        x, y = data["test"]
        pick = torch.arange(args.images) * (len(x) // args.images if data["task"] == "ae" else 1)
        x, y = x[pick], y[pick]
        sim = sim_cycles(dataset, model, "_arena")
        ck = ROOT / "checkpoints" / "tiny" / dataset
        for bits in Q.BITS:
            for method in Q.METHODS:
                if not Q.feasible(bits, method):
                    continue
                prog = torch.load(ck / f"{model}_s{args.seed}_int{bits}_{Q.TRAINED_WITH[method]}.pt",
                                  map_location="cpu", weights_only=False)["prog"]
                z = Q.int_forward(prog, x, method)
                pdir = FZ / "programs" / dataset / model / f"int{bits}" / method
                gdir = FZ / "build" / "gen" / dataset / model / f"int{bits}" / method
                gdir.mkdir(parents=True, exist_ok=True)
                pdir.mkdir(parents=True, exist_ok=True)
                rt.write_net_h(gdir / "net.h", prog)
                xq = Q.quantize_input(x, bits, prog)
                for b0 in range(0, args.images, args.batch):
                    idx = list(range(b0, min(b0 + args.batch, args.images)))
                    bdir = gdir / f"b{b0:05d}"
                    bdir.mkdir(exist_ok=True)
                    rt.write_images_h(bdir / "images.h", xq[idx], y[idx].tolist(), idx)
                    entry = {"dataset": dataset, "model": model, "task": data["task"], "bits": bits,
                             "method": method, "lod": int(method in ("dn1", "dn2", "sqapx")), "first": b0,
                             "images": idx, "labels": y[idx].tolist(),
                             "ref_pred": z[idx].argmax(1).tolist(),
                             "ref_hash": [f"{ri.logit_hash(r):08x}" for r in z[idx]], "elf": {}}
                    for mem in ("bram", "ddr"):
                        elf = pdir / f"b{b0:05d}_{mem}.elf"
                        try:
                            ri.compile_elf(ROOT / "hw/sw/tiny.c", elf, include_dirs=[bdir, gdir],
                                           defines=rt.defines(method), linker=FZ / "sw" / f"link_zybo_{mem}.ld")
                        except subprocess.CalledProcessError:
                            if mem == "ddr":
                                raise
                            continue                       # does not fit the block RAM: DDR only
                        entry["elf"][mem] = str(elf.relative_to(FZ))
                    if "bram" in entry["elf"]:
                        # the Verilator study compiled the same C with the 1 MB script; the code is the same
                        # bytes when only data addresses differ, so its cycle counts are this program's
                        ref_elf = ROOT / "hw/gen/tiny" / dataset / model / f"s{args.seed}" / f"int{bits}" / method \
                            / f"b{b0:05d}" / "tiny.elf"
                        same = ref_elf.exists() and text_digest(ref_elf) == text_digest(FZ / entry["elf"]["bram"])
                        if same and all((bits, method, i) in sim for i in idx):
                            entry["sim_cycles"] = [sim[(bits, method, i)][0] for i in idx]
                            entry["sim_op_cycles"] = [sim[(bits, method, i)][1] for i in idx]
                    manifest["programs"].append(entry)
                    print(f"{dataset} int{bits} {method} b{b0:05d}: {', '.join(entry['elf'])}"
                          f"{' (sim cycles known)' if 'sim_cycles' in entry else ''}", flush=True)
    (FZ / "programs" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n = len(manifest["programs"])
    nb = sum("bram" in e["elf"] for e in manifest["programs"])
    print(f"wrote {n} batches ({nb} fit the BRAM) to fpga_zybo/programs/manifest.json")


if __name__ == "__main__":
    main()
