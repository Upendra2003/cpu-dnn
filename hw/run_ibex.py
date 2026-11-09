"""Run the int8 MNIST / Fashion-MNIST networks on the multiplierless Ibex (Verilator).

Stages
  mult      MULT kernel (10K multiplies per routine and width), LOD instruction vs software LOD
  dnn       integer-only inference on the first N test images for every (dataset, model) and every
            multiplier, fused into the MAC loop (hw/sw/dnn.c default), plus "sqtab": no multiplier,
            the exact polarization identity with a square table           -> dnn_{images,summary}.csv
  call      the same multipliers through the original wrapper, one routine call per product
                                                                          -> dnn_call_{images,summary}.csv
  zeroskip  ablation for exact, RoBA-Dn1 and sqtab: zero-skipping on/off x normalized/raw pixels,
            with the full-test-set accuracy of each from the Python reference -> zeroskip_{...}.csv
Every Ibex logit is checked against a bit-exact Python reference; cycles are recorded per image and
per layer. Each multiplier uses its own int8 fine-tuned checkpoint (sqtab uses the exact one).

A configuration is a multiplier name plus optional flags: "+call" (original wrapper), "+zs"
(ZERO_SKIP), "+raw" (input normalization folded into layer 1, so the network sees raw pixels).

Outputs go to results/ibex/. Example:
  python3 hw/run_ibex.py --stage all --images 100 --jobs 48
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roba.data import DATASETS, load_dataset  # noqa: E402
from roba.matmul import approx_conv2d, approx_linear  # noqa: E402
from roba.multipliers import MULTIPLIERS  # noqa: E402
from roba.nn import MODELS, configure  # noqa: E402

HW = ROOT / "hw"
SW = HW / "sw"
SIM = HW / "ibex/build/lowrisc_ibex_ibex_simple_system_0/sim-verilator/Vibex_simple_system"
IBEX_COMMON = HW / "ibex/examples/sw/simple_system/common"
TOOLS = pathlib.Path.home() / "tools/riscv/bin"
CONDA_LIB = pathlib.Path.home() / "miniconda3/envs/ibex/lib"
OUT = ROOT / "results" / "ibex"
SH = 24                      # fixed-point fraction bits of the per-output rescale
M_INSNS = re.compile(r"\t(mul|mulh|mulhsu|mulhu|div|divu|rem|remu)\t")
CFLAGS = ["-march=rv32ic", "-mabi=ilp32", "-static", "-mcmodel=medany", "-Wall", "-O3",
          "-fvisibility=hidden", "-nostdlib", "-nostartfiles", "-ffreestanding", "-DROBA_LOD_INSN"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log("wrote", path.relative_to(ROOT))


def compile_elf(prog_c, out_elf, include_dirs=(), defines=(), extra_flags=(), linker=None):
    out_elf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(TOOLS / "riscv32-unknown-elf-gcc"), *CFLAGS, *extra_flags, *[f"-D{d}" for d in defines],
           f"-I{IBEX_COMMON}", f"-I{ROOT / 'csrc'}", f"-I{SW}", *[f"-I{d}" for d in include_dirs],
           "-T", str(linker or SW / "link_1mb.ld"), str(prog_c), str(ROOT / "csrc/roba.c"),
           str(IBEX_COMMON / "simple_system_common.c"), str(IBEX_COMMON / "crt0.S"),
           "-o", str(out_elf), "-lgcc"]
    subprocess.run(cmd, check=True)
    dis = subprocess.run([str(TOOLS / "riscv32-unknown-elf-objdump"), "-d", str(out_elf)],
                         check=True, capture_output=True, text=True).stdout
    if M_INSNS.search(dis):
        raise RuntimeError(f"{out_elf} contains M-extension instructions")
    return dis


def run_sim(elf, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=f"{CONDA_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    t0 = time.time()
    # The Ibex tracer would write ~1 GB of instruction trace per image; keep it off.
    res = subprocess.run([str(SIM), "+ibex_tracer_enable=0", f"--meminit=ram,{elf}"], cwd=run_dir, env=env,
                         capture_output=True, text=True)
    (run_dir / "sim_stdout.log").write_text(res.stdout + res.stderr)
    speed = re.search(r"Simulation speed: ([\d.eE+]+) cycles/s", res.stdout)
    return ((run_dir / "ibex_simple_system.log").read_text(), time.time() - t0,
            float(speed.group(1)) if speed else 0.0)


# ---------------------------------------------------------------- stage: mult

def stage_mult(args):
    rows = []
    for lod in (1, 0):
        elf = SW / "build" / f"mult_bench_{'lod' if lod else 'sw'}.elf"
        subprocess.run(["make", "-s", "-C", str(SW), "PROG=mult_bench", f"LOD={lod}"], check=True)
        text, secs, speed = run_sim(elf, HW / "runs" / f"mult_{'lod' if lod else 'sw'}")
        for line in text.splitlines():
            if line.startswith("MULT,"):
                _, name, bits, cyc, ins, _ = line.split(",")
                rows.append({"lod": "instruction" if lod else "software", "multiplier": name,
                             "bits": int(bits), "cycles_per_mult": int(cyc) / 10000,
                             "instr_per_mult": int(ins) / 10000})
        log(f"mult lod={lod}: {secs:.0f}s at {speed / 1e3:.0f} kHz")
    write_csv(OUT / "mult_bench.csv", rows)


# ---------------------------------------------------------------- integer model export

LAYERS = {"mlp": ["net.1", "net.3", "net.5"],
          "lenet5": ["net.0", "net.3", "net.7", "net.9", "net.11"]}


def int_model(name, sd, raw=False, ds=None):
    """int8 weights + fixed-point rescale per layer, following roba.nn's int8 quantization.

    raw=True folds the input normalization (x - mean) / std into layer 1, so the network takes raw
    pixels x in [0, 1] (int8 0..127, background exactly 0): weights W / std, bias
    b - mean * sum(W / std) over the taps that see the image. For LeNet-5's zero-padded conv1 that
    sum depends on the output position, so its offset C0 becomes a (6, 28, 28) map."""
    keys = LAYERS[name]
    sx = [(sd[f"{k}.act_max"].float() / 127).clamp_min(1e-12) for k in keys]
    if raw:
        _, mean, std = DATASETS[ds]
        sx[0] = torch.tensor(1 / 127)
    layers = []
    for i, k in enumerate(keys):
        w = sd[f"{k}.weight"].float()
        b = sd[f"{k}.bias"].double()
        if raw and i == 0:
            w = w / std
        sw = (w.abs().flatten(1).amax(1).clamp_min(1e-12) / 127)
        shape = (-1, *([1] * (w.dim() - 1)))
        wq = torch.clamp(torch.round(w / sw.view(shape)), -127, 127)
        if raw and i == 0:
            wd = (wq * sw.view(shape)).double()             # the folded weights the integers stand for
            if w.dim() == 4:
                taps = F.conv2d(torch.ones(1, 1, 28, 28, dtype=torch.float64), wd, padding=2)[0]
                b = b.view(-1, 1, 1) - mean * taps
            else:
                b = b - mean * wd.sum(1)
        s_next = float(sx[i + 1]) if i + 1 < len(keys) else 1.0
        scale = float(sx[i]) * sw.double() / s_next
        m = torch.round(scale * 2 ** SH).long()
        c = torch.round(b / s_next * 2 ** SH).long()
        assert int(m.abs().max()) < 2 ** 31, f"{name} layer {i}: rescale does not fit int32"
        layers.append({"wq": wq.to(torch.int64), "m": m, "c": c})
    return layers, sx[0]


def quantize_images(x, sx0, raw=False, ds=None):
    """Normalized images -> int8 network input; raw=True gives round(127 * pixel) instead."""
    if raw:
        _, mean, std = DATASETS[ds]
        k = torch.round((x * std + mean) * 255).to(torch.int64)    # the original 8-bit pixel
        return (k * 254 + 255) // 510                               # round(k * 127 / 255)
    return torch.clamp(torch.round(x / sx0), -127, 127).to(torch.int64)


def ref_forward(name, layers, xq, mult):
    """Bit-exact Python version of hw/sw/dnn.c. xq: (N, 1, 28, 28) int64."""
    def rescale(acc, L):
        return (acc * L["m"] + L["c"] + (1 << (SH - 1))) >> SH

    def relu_q(v):
        return v.clamp(0, 127)

    def lin(x, L):
        acc = approx_linear(x.double(), L["wq"].double(), mult).round().to(torch.int64)
        return rescale(acc, L)

    def conv(x, L, pad):
        acc = approx_conv2d(x.double(), L["wq"].double(), mult, padding=pad).round().to(torch.int64)
        c = L["c"].view(1, -1, 1, 1) if L["c"].dim() == 1 else L["c"].unsqueeze(0)   # per position
        return rescale(acc, {"m": L["m"].view(1, -1, 1, 1), "c": c})

    if name == "mlp":
        h = relu_q(lin(xq.flatten(1), layers[0]))
        h = relu_q(lin(h, layers[1]))
        z = lin(h, layers[2])
    else:
        h = torch.nn.functional.max_pool2d(relu_q(conv(xq, layers[0], 2)).double(), 2).to(torch.int64)
        h = torch.nn.functional.max_pool2d(relu_q(conv(h, layers[1], 0)).double(), 2).to(torch.int64)
        h = relu_q(lin(h.flatten(1), layers[2]))
        h = relu_q(lin(h, layers[3]))
        z = lin(h, layers[4])
    return z


def logit_hash(z):
    h = 0x811C9DC5
    rot = lambda v: ((v << 5) | (v >> 27)) & 0xFFFFFFFF
    for v in z.tolist():
        u = v & 0xFFFFFFFFFFFFFFFF
        h = rot(h) ^ (u & 0xFFFFFFFF)
        h = rot(h) ^ (u >> 32)
    return h


def c_array(ctype, name, values, per_line=24):
    vals = [str(int(v)) + ("LL" if ctype == "int64_t" else "") for v in values]
    body = ",\n  ".join(", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line))
    return f"static const {ctype} {name}[{len(vals)}] = {{\n  {body}\n}};\n"


def write_model_h(path, name, layers, raw=False):
    parts = [f"/* generated by hw/run_ibex.py */\n#define MODEL_{name.upper()} 1\n#define SH {SH}\n"]
    if raw:
        parts.append("#define INPUT_NONNEG 1\n")
    if layers[0]["c"].dim() > 1:
        parts.append("#define C0_PER_POSITION 1\n")
    for i, L in enumerate(layers):
        parts += [c_array("int8_t", f"W{i}", L["wq"].flatten()), c_array("int32_t", f"M{i}", L["m"]),
                  c_array("int64_t", f"C{i}", L["c"].flatten())]
    path.write_text("".join(parts))


def write_images_h(path, xq, labels, indices):
    parts = [f"/* generated by hw/run_ibex.py */\n#define N_IMG {len(indices)}\n",
             f"static const int8_t IMG[N_IMG][784] = {{\n"]
    parts += ["  {" + ", ".join(str(int(v)) for v in img.flatten()) + "},\n" for img in xq]
    parts.append("};\n")
    parts.append(c_array("uint8_t", "LABEL", labels))
    parts.append(c_array("uint32_t", "IMG_INDEX", indices))
    path.write_text("".join(parts))


# ---------------------------------------------------------------- stages: dnn, call, zeroskip

CONFIG_FLAGS = ("call", "zs", "raw")
# stage -> (configurations, output prefix, baseline for speedups, full-test-set Python accuracy)
STAGES = {
    "dnn": ([*MULTIPLIERS, "sqtab"], "dnn", "exact", False),
    "call": ([f"{m}+call" for m in MULTIPLIERS], "dnn_call", "exact+call", False),
    "zeroskip": ([m + zs + raw for m in ("exact", "roba_dn1", "sqtab") for raw in ("", "+raw") for zs in ("", "+zs")],
                 "zeroskip", "exact", True),
}


def parse_config(config):
    """'roba_dn1' / 'exact+call' / 'sqtab+zs+raw' -> (multiplier, set of flags)."""
    mult, *flags = config.split("+")
    assert mult in (*MULTIPLIERS, "sqtab") and set(flags) <= set(CONFIG_FLAGS), config
    assert not (mult == "sqtab" and "call" in flags), config
    return mult, set(flags)


def mac_kind(config):
    mult, flags = parse_config(config)
    return "table" if mult == "sqtab" else ("call" if "call" in flags else "fused")


def c_routine(mult):
    """Python multiplier name -> unsigned routine in csrc/roba.c."""
    return f"{mult}_u" if mult.startswith("roba_") else f"roba_{mult}_u"


def config_defines(config):
    """Compile-time selection of the dot-product code in hw/sw/dnn.c (raw input is in model.h)."""
    mult, flags = parse_config(config)
    kind = mac_kind(config)
    d = (["SQTAB"] if kind == "table" else
         ["MAC_CALL", f"MULT_U={c_routine(mult)}"] if kind == "call" else
         [f"MULT_NZ={c_routine(mult)[:-2]}_nz"])                     # inline routine in roba_core.h
    return d + (["ZERO_SKIP"] if "zs" in flags else [])


def prepare_jobs(args, configs, full_test):
    jobs, refs, cache = [], {}, {}
    for ds in args.datasets:
        (_, _), (xte, yte) = load_dataset(ds, "cpu")
        x, y = xte[:args.images], yte[:args.images]
        for name in args.models:
            for config in configs:
                mult, flags = parse_config(config)
                ckpt = "exact" if mult == "sqtab" else mult       # sqtab computes exact products
                raw = "raw" in flags
                key = (ds, name, config)
                sd = torch.load(ROOT / "checkpoints" / ds / f"{name}_ft_int8_{ckpt}.pt", map_location="cpu")
                layers, sx0 = int_model(name, sd, raw, ds)
                xq = quantize_images(x, sx0, raw, ds)
                if (ds, name, ckpt, raw) not in cache:
                    z = ref_forward(name, layers, xq, ckpt)
                    # The PyTorch int8 model (as evaluated on the GPU, normalized input) on the same images.
                    model = MODELS[name]()
                    model.load_state_dict(sd)
                    configure(model, 8, ckpt)
                    model.eval()
                    with torch.no_grad():
                        torch_pred = model(x).argmax(1)
                    full = ""
                    if full_test:
                        full, n = 0, len(xte)
                        for i in range(0, n, 1000):
                            zz = ref_forward(name, layers, quantize_images(xte[i:i + 1000], sx0, raw, ds), ckpt)
                            full += int((zz.argmax(1) == yte[i:i + 1000]).sum())
                        full /= n
                    cache[(ds, name, ckpt, raw)] = {"pred": z.argmax(1), "hash": [logit_hash(r) for r in z],
                                                    "torch_pred": torch_pred, "label": y, "full_acc": full}
                refs[key] = cache[(ds, name, ckpt, raw)]
                gen = HW / "gen" / ds / name / config
                gen.mkdir(parents=True, exist_ok=True)
                write_model_h(gen / "model.h", name, layers, raw)
                for b0 in range(0, args.images, args.batch):
                    idx = list(range(b0, min(b0 + args.batch, args.images)))
                    bdir = gen / f"b{b0:05d}"
                    bdir.mkdir(exist_ok=True)
                    write_images_h(bdir / "images.h", xq[idx], y[idx], idx)
                    jobs.append({"key": key, "bdir": bdir, "elf": bdir / "dnn.elf", "first": b0})
                r = refs[key]
                log(f"prepared {ds}/{name}/{config}: python reference accuracy "
                    f"{(r['pred'] == y).float().mean().item():.4f} on {len(y)} images"
                    + (f", {r['full_acc']:.4f} on the full test set" if full_test else ""))
    return jobs, refs


def build_job(job):
    """Compile one batch; returns how many places call __mulsi3."""
    dis = compile_elf(SW / "dnn.c", job["elf"], include_dirs=[job["bdir"], job["bdir"].parent],
                      defines=config_defines(job["key"][2]))
    return len(re.findall(r"\tjal\t.*<__mulsi3>", dis))


def run_job(job):
    """Simulate one batch, or reuse a finished log of the very same program (resume after a stop)."""
    run_dir = HW / "runs" / "dnn" / "/".join(job["key"]) / job["bdir"].name
    digest = hashlib.sha256(job["elf"].read_bytes()).hexdigest()
    log_file, sha_file = run_dir / "ibex_simple_system.log", run_dir / "elf.sha256"
    if (log_file.exists() and "DNN_DONE" in log_file.read_text()
            and sha_file.exists() and sha_file.read_text().strip() == digest):
        text, secs, speed = log_file.read_text(), 0.0, 0.0
    else:
        text, secs, speed = run_sim(job["elf"], run_dir)
        if "DNN_DONE" not in text:
            raise RuntimeError(f"{job['bdir']}: simulation did not finish:\n{text[-500:]}")
        sha_file.write_text(digest)
    lines = [l for l in text.splitlines() if l.startswith("IMG,")]
    return job, lines, secs, speed


def stage_dnn(args, stage):
    configs, prefix, baseline, full_test = STAGES[stage]
    configs = [c for c in configs if parse_config(c)[0] in args.mults or parse_config(c)[0] == "sqtab"]
    jobs, refs = prepare_jobs(args, configs, full_test)
    log(f"compiling {len(jobs)} programs")
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        mul_calls = list(ex.map(build_job, jobs))
    for job, n in zip(jobs, mul_calls):
        job["mulsi3_call_sites"] = n
    log(f"running {len(jobs)} simulations on {args.jobs} workers")
    images, t0, done = [], time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for fut in cf.as_completed([ex.submit(run_job, j) for j in jobs]):
            job, lines, secs, speed = fut.result()
            ds, name, config = job["key"]
            ref = refs[job["key"]]
            for line in lines:
                f = line.split(",")
                i = int(f[1])
                row = {"dataset": ds, "model": name, "config": config, "image": i,
                       "label": int(f[2]), "ibex_pred": int(f[3]),
                       "ref_pred": int(ref["pred"][i]), "torch_pred": int(ref["torch_pred"][i]),
                       "logits_match": int(f[4], 16) == ref["hash"][i],
                       "cycles": int(f[5])}
                row.update({f"layer{l}_cycles": int(v) for l, v in enumerate(f[6:])})
                images.append(row)
            done += 1
            if speed or done == len(jobs):
                log(f"{done}/{len(jobs)} batches done ({time.time() - t0:.0f}s): {'/'.join(job['key'])}/{job['bdir'].name}"
                    + (f" simulated in {secs:.0f}s at {speed / 1e3:.0f} kHz" if speed else " (reused)"))
    images.sort(key=lambda r: (r["dataset"], r["model"], configs.index(r["config"]), r["image"]))
    # layer columns differ per model; pad so one CSV holds both
    cols = max((len([k for k in r if k.startswith("layer")]) for r in images))
    for r in images:
        for l in range(cols):
            r.setdefault(f"layer{l}_cycles", "")
    write_csv(OUT / f"{prefix}_images.csv", images)

    summary = []
    for ds in args.datasets:
        for name in args.models:
            per = {c: [r for r in images if r["dataset"] == ds and r["model"] == name and r["config"] == c]
                   for c in configs}
            cyc = {c: sum(r["cycles"] for r in rs) / len(rs) for c, rs in per.items()}
            for config, rs in per.items():
                mult, flags = parse_config(config)
                summary.append({
                    "dataset": ds, "model": name, "config": config, "multiplier": mult,
                    "mac": mac_kind(config), "zero_skip": int("zs" in flags),
                    "input": "raw" if "raw" in flags else "normalized", "images": len(rs),
                    "ibex_acc": sum(r["ibex_pred"] == r["label"] for r in rs) / len(rs),
                    "torch_int8_acc": sum(r["torch_pred"] == r["label"] for r in rs) / len(rs),
                    "pred_match_ref": sum(r["ibex_pred"] == r["ref_pred"] for r in rs) / len(rs),
                    "logits_match_ref": sum(r["logits_match"] for r in rs) / len(rs),
                    "pred_match_torch": sum(r["ibex_pred"] == r["torch_pred"] for r in rs) / len(rs),
                    "ref_acc_full_test": refs[(ds, name, config)]["full_acc"],
                    "cycles_per_image": cyc[config],
                    "speedup_vs_baseline": cyc[baseline] / cyc[config] if baseline in cyc else "",
                    **{f"layer{l}_cycles": sum(r[f"layer{l}_cycles"] or 0 for r in rs) / len(rs) for l in range(cols)},
                })
                s = summary[-1]
                log(f"{ds:7s} {name:6s} {config:18s} ibex acc {s['ibex_acc']:.3f} | logits match {s['logits_match_ref']:.3f} "
                    f"| {s['cycles_per_image'] / 1e6:6.2f} M cycles/image")
    write_csv(OUT / f"{prefix}_summary.csv", summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "mult", *STAGES])
    p.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    p.add_argument("--models", nargs="+", default=["mlp", "lenet5"])
    p.add_argument("--mults", nargs="+", default=MULTIPLIERS, help="restrict the multipliers (sqtab always runs)")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--jobs", type=int, default=48)
    args = p.parse_args()
    torch.set_num_threads(16)
    if args.stage in ("all", "mult"):
        stage_mult(args)
    for stage in STAGES:
        if args.stage in ("all", stage):
            stage_dnn(args, stage)


if __name__ == "__main__":
    main()
