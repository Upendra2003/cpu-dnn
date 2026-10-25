"""MNIST DNN with RoBA-family approximate multipliers.

Stages (run in order with --stage all):
  errors    error statistics of each multiplier on 1M random operand pairs
  train     float32 baseline training
  eval      post-training evaluation: domain {float, trunc, int8, int16} x multiplier
  finetune  approximation-aware fine-tuning (approximate forward, STE backward) at int8/int16
"""
import argparse
import csv
import json
import math
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roba import cost  # noqa: E402
from roba.data import load_mnist  # noqa: E402
from roba.multipliers import APPROX_MULTIPLIERS, MULTIPLIERS, approx_mul  # noqa: E402
from roba.nn import MODELS, approx_layers, calibrate, configure, count_macs  # noqa: E402

RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------- stage: errors

def stage_errors(args, device):
    g = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    for bits in (8, 16):
        hi = 2 ** bits - 1 if bits < 16 else 2 ** 15 - 1
        a = torch.randint(1, hi + 1, (1_000_000,), generator=g, device=device).double()
        b = torch.randint(1, hi + 1, (1_000_000,), generator=g, device=device).double()
        exact = a * b
        for mult in APPROX_MULTIPLIERS:
            rel = (approx_mul(a, b, mult) - exact) / exact
            rows.append({
                "bits": bits, "multiplier": mult,
                "mean_abs_rel_err_pct": 100 * rel.abs().mean().item(),
                "max_abs_rel_err_pct": 100 * rel.abs().max().item(),
                "mean_signed_rel_err_pct": 100 * rel.mean().item(),
                "frac_exact": (rel == 0).double().mean().item(),
            })
            log(f"errors {bits:2d}-bit {mult:12s} mean {rows[-1]['mean_abs_rel_err_pct']:.3f}% "
                f"max {rows[-1]['max_abs_rel_err_pct']:.3f}% bias {rows[-1]['mean_signed_rel_err_pct']:+.3f}%")
    write_csv(RESULTS / "multiplier_errors.csv", rows)


# ---------------------------------------------------------------- training / evaluation helpers

@torch.no_grad()
def evaluate(model, x, y, batch=1000):
    model.eval()
    loss, preds = 0.0, []
    for i in range(0, len(x), batch):
        out = model(x[i:i + batch])
        loss += F.cross_entropy(out, y[i:i + batch], reduction="sum").item()
        preds.append(out.argmax(1))
    preds = torch.cat(preds)
    loss /= len(x)
    return {"loss": loss if math.isfinite(loss) else float("nan"),
            "acc": (preds == y).float().mean().item()}, preds


def train_epochs(model, x, y, epochs, lr, batch, seed, cosine=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(x) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cosine else None
    g = torch.Generator(device=x.device).manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=g, device=x.device)
        tot, n = 0.0, 0
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        yield ep, tot / n


def load_float(name, device):
    model = MODELS[name]().to(device)
    model.load_state_dict(torch.load(CKPT / f"{name}_float.pt", map_location=device))
    return model


# ---------------------------------------------------------------- stage: train

def stage_train(args, device, data):
    (xtr, ytr), (xte, yte) = data
    CKPT.mkdir(exist_ok=True)
    for name in args.models:
        torch.manual_seed(args.seed)
        model = MODELS[name]().to(device)
        configure(model, None, "exact")
        t0 = time.time()
        for ep, tl in train_epochs(model, xtr, ytr, args.epochs, 1e-3, 128, args.seed):
            m, _ = evaluate(model, xte, yte)
            log(f"train {name} epoch {ep + 1}/{args.epochs} train_loss {tl:.4f} "
                f"test_loss {m['loss']:.4f} test_acc {m['acc']:.4f}")
        torch.save(model.state_dict(), CKPT / f"{name}_float.pt")
        log(f"train {name} done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- stage: eval

def _domain_bits(domain):
    return domain if isinstance(domain, int) else 32


def _row(name, domain, mult, metrics, model, macs, agree):
    acc_max = max((m.acc_max for m in approx_layers(model)), default=0.0)
    est = cost.estimate(macs, mult, _domain_bits(domain))
    return {"model": name, "domain": domain, "multiplier": mult,
            "test_acc": metrics["acc"], "test_loss": metrics["loss"],
            "agree_with_exact": agree,
            "acc_bits": math.ceil(math.log2(acc_max + 1)) + 1 if acc_max else "",
            "macs": macs, **est}


def stage_eval(args, device, data):
    (xtr, _), (xte, yte) = data
    for name in args.models:
        model = load_float(name, device)
        macs = sum(count_macs(model))
        calibrate(model, xtr[:5000])
        configure(model, None, "exact")
        base, _ = evaluate(model, xte, yte)
        rows = [{"model": name, "domain": "float32", "multiplier": "native",
                 "test_acc": base["acc"], "test_loss": base["loss"], "agree_with_exact": 1.0,
                 "acc_bits": "", "macs": macs, "mult_cycles": "", "energy_rel": ""}]
        log(f"eval {name} float32 native acc {base['acc']:.4f} loss {base['loss']:.4f}")
        for domain in args.domains:
            exact_preds = None
            for mult in MULTIPLIERS:
                configure(model, domain, mult)
                t0 = time.time()
                m, preds = evaluate(model, xte, yte)
                if mult == "exact":
                    exact_preds = preds
                agree = (preds == exact_preds).float().mean().item()
                rows.append(_row(name, domain, mult, m, model, macs, agree))
                log(f"eval {name} {str(domain):6s} {mult:12s} acc {m['acc']:.4f} "
                    f"loss {m['loss']:.4f} agree {agree:.4f} ({time.time() - t0:.1f}s)")
        write_csv(RESULTS / f"eval_{name}.csv", rows)


# ---------------------------------------------------------------- stage: finetune

def stage_finetune(args, device, data):
    (xtr, ytr), (xte, yte) = data
    for name in args.models:
        rows = []
        for bits in args.ft_bits:
            for mult in MULTIPLIERS:          # "exact" = plain quantization-aware training
                model = load_float(name, device)
                macs = sum(count_macs(model))
                calibrate(model, xtr[:5000])
                configure(model, bits, mult)
                before, _ = evaluate(model, xte, yte)
                t0 = time.time()
                for ep, tl in train_epochs(model, xtr, ytr, args.ft_epochs, args.ft_lr, 256,
                                           args.seed):
                    pass
                after, _ = evaluate(model, xte, yte)
                rows.append({"model": name, "bits": bits, "multiplier": mult,
                             "acc_before": before["acc"], "loss_before": before["loss"],
                             "acc_after": after["acc"], "loss_after": after["loss"],
                             "train_loss_last_epoch": tl, "seconds": round(time.time() - t0, 1),
                             **cost.estimate(macs, mult, bits)})
                log(f"finetune {name} int{bits} {mult:12s} acc {before['acc']:.4f} -> "
                    f"{after['acc']:.4f} loss {before['loss']:.4f} -> {after['loss']:.4f} "
                    f"({time.time() - t0:.0f}s)")
                torch.save(model.state_dict(), CKPT / f"{name}_ft_int{bits}_{mult}.pt")
        write_csv(RESULTS / f"finetune_{name}.csv", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "errors", "train", "eval", "finetune"])
    p.add_argument("--models", nargs="+", default=["mlp", "lenet5"])
    p.add_argument("--domains", nargs="+", default=["float", "trunc", "8", "16"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--ft-bits", nargs="+", type=int, default=[8, 16])
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.domains = [int(d) if d.isdigit() else d for d in args.domains]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "config.json").write_text(json.dumps(vars(args), indent=2))
    log("device", device, torch.cuda.get_device_name() if device == "cuda" else "")

    stages = ["errors", "train", "eval", "finetune"] if args.stage == "all" else [args.stage]
    data = load_mnist(device) if set(stages) - {"errors"} else None
    for s in stages:
        t0 = time.time()
        if s == "errors":
            stage_errors(args, device)
        elif s == "train":
            stage_train(args, device, data)
        elif s == "eval":
            stage_eval(args, device, data)
        else:
            stage_finetune(args, device, data)
        log(f"stage {s} finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
