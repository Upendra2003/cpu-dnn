"""GPU stages of the MLPerf Tiny study.

  train   float model with batch norm, one per seed                       -> checkpoints/tiny/<ds>/<model>_s<seed>_float.pt
  qat     per seed and precision: pick the activation calibration on the validation split (post-training
          quantization, exact), then quantization-aware fine-tuning with each trained method (exact, dn1, dn2)
                                                                           -> .../<model>_s<seed>_int<bits>_<method>.pt
  eval    full test set with the bit-exact integer reference, every feasible (precision, method)
                                                                           -> results/tiny/<ds>/<model>_accuracy.csv

  python3 tiny/run.py --dataset cifar10 --model resnet8 --seeds 0 1 2 --stage all
"""
import argparse
import csv
import math
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiny import data as D  # noqa: E402
from tiny import qnet as Q  # noqa: E402
from tiny.models import MODELS  # noqa: E402

CALIB_PCT = (99.9, 99.99, 99.999, 100.0)
QAT_EPOCHS = {4: 30, 8: 15, 16: 5}
QAT_LRS = (0.01, 0.002)          # each is tried; the best of these and plain quantization on validation is kept
TRAIN_METHODS = ("exact", "dn1", "dn2")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def ckpt_dir(args):
    d = ROOT / "checkpoints" / "tiny" / args.dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def batches(x, y, bs, gen, shuffle=True):
    idx = torch.randperm(len(x), device=x.device, generator=gen) if shuffle else torch.arange(len(x), device=x.device)
    for i in range(0, len(x), bs):
        j = idx[i:i + bs]
        yield x[j], y[j]


@torch.no_grad()
def accuracy(fn, x, y, bs=500):
    hit = 0
    for i in range(0, len(x), bs):
        hit += (fn(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return hit / len(x)


def auc(scores, labels):
    """Area under the ROC curve (probability that an anomaly scores higher than a normal clip; ties count half)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    gt = (pos[:, None] > neg[None]).double().mean()
    eq = (pos[:, None] == neg[None]).double().mean()
    return float(gt + 0.5 * eq)


@torch.no_grad()
def evaluate(data, fn, split, bs=None):
    """Classification: accuracy. Autoencoder: -(mean squared reconstruction error) on train-like splits, and on
    the test split the anomaly AUC (clip score = mean error over its vectors), averaged over machine IDs."""
    x, y = data[split]
    if data["task"] != "ae":
        return accuracy(fn, x, y, bs or 500)
    bs = bs or 20000
    err = torch.cat([((fn(x[i:i + bs]).float() - x[i:i + bs].float().flatten(1)) ** 2).mean(1)
                     for i in range(0, len(x), bs)])
    if split != "test":
        return -float(err.mean())
    info = data["test_info"]
    clip = torch.zeros(len(info["anomaly"]), dtype=torch.float64).index_add_(0, info["clip"], err.double().cpu())
    clip = clip / torch.bincount(info["clip"]).double()
    return sum(auc(clip[info["machine"] == m], info["anomaly"][info["machine"] == m])
               for m in info["machine"].unique().tolist()) / len(info["machine"].unique())


def int_fn(prog, method=None):
    """Integer program as a function: logits, or (autoencoder) the reconstruction in float units."""
    s = prog["ops"][-1].get("s_out", 1.0)
    return lambda x: Q.int_forward(prog, x, method).double() * s


def train_loop(model, params, data, epochs, lr, wd, seed, to_input, nesterov=True):
    if data["opt"] == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=nesterov)
    xtr, ytr = data["train"]
    bs = data["batch"]
    steps = epochs * math.ceil(len(xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15) if epochs > 1 else None
    gen = torch.Generator(device=xtr.device).manual_seed(seed)
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for xb, yb in batches(xtr, ytr, bs, gen):
            inp = to_input(data["augment"](xb, gen))
            out = model(inp)
            loss = F.mse_loss(out, inp.flatten(1)) if data["task"] == "ae" else F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        yield ep, tot / n


# ---------------------------------------------------------------- stage: train

def stage_train(args, data):
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = MODELS[args.model]().cuda()
        to_float = data["to_float"]
        t0 = time.time()
        lr, wd = (1e-3, 0.0) if data["opt"] == "adam" else (0.1, 5e-4)
        for ep, tl in train_loop(model, model.parameters(), data, args.epochs, lr, wd, seed, lambda x: x):
            if (ep + 1) % 20 == 0 or ep + 1 == args.epochs:
                model.eval()
                va = evaluate(data, lambda x: model(to_float(x)), "val")
                log(f"train {args.model} seed {seed} epoch {ep + 1}/{args.epochs} loss {tl:.4f} val {va:.4f}")
        model.eval()
        te = evaluate(data, lambda x: model(to_float(x)), "test")
        torch.save(model.state_dict(), ckpt_dir(args) / f"{args.model}_s{seed}_float.pt")
        log(f"train {args.model} seed {seed} done in {time.time() - t0:.0f}s, float test acc {te:.4f}")


# ---------------------------------------------------------------- stage: qat

def load_float(args, seed):
    model = MODELS[args.model]().cuda()
    model.load_state_dict(torch.load(ckpt_dir(args) / f"{args.model}_s{seed}_float.pt", map_location="cuda"))
    return model.eval()


def stage_qat(args, data):
    for seed in args.seeds:
        ops = Q.fold(load_float(args, seed))
        shapes = Q.buffer_shapes(ops, data["val"][0][:1])
        for bits in args.bits:
            # calibration: post-training quantization with exact products, best percentile on the validation split
            best = None
            calib = data["calib"]
            for pct in CALIB_PCT:
                sc = Q.calibrate(ops, calib, pct, to_float=lambda x: data["to_float"](x).double())
                s_in = None
                if data["signed"]:                     # signed input (MFCC): symmetric, same percentile of |x|
                    a = calib.abs().flatten().float()
                    a = a[torch.randperm(a.numel(), device=a.device)[:8_000_000]] if a.numel() > 8_000_000 else a
                    s_in = float(torch.quantile(a, pct / 100)) if pct < 100 else float(calib.abs().max())
                q = Q.QNet(ops, {k: v / Q.QA[bits] for k, v in sc.items()}, bits, "exact", s_in).cuda()
                prog = Q.export(q, shapes)
                va = evaluate(data, int_fn(prog), "val")
                log(f"calib seed {seed} int{bits} pct {pct}: ptq val {va:.4f}")
                if best is None or va > best[0]:
                    best = (va, pct, q.s, s_in)
            ptq_val, pct, scales, s_in = best
            for method in args.train_methods:
                q = Q.QNet(ops, {k: v for k, v in scales.items() if k != "in"}, bits, method, s_in).cuda()
                ptq = Q.export(q, shapes)
                va0 = evaluate(data, int_fn(ptq), "val")
                keep = {"prog": ptq, "state": q.state_dict(), "val": va0, "qat_lr": 0}
                tried = [f"ptq {va0:.4f}"]
                t0 = time.time()
                epochs = (max(1, round(QAT_EPOCHS[bits] * data.get("qat_scale", 1.0))) if args.qat_epochs is None
                          else args.qat_epochs)
                lrs = [lr / 10 for lr in QAT_LRS] if data["opt"] == "adam" else QAT_LRS
                for lr in (lrs if args.qat_lr is None else [args.qat_lr]):
                    torch.manual_seed(seed)
                    q = Q.QNet(ops, {k: v for k, v in scales.items() if k != "in"}, bits, method, s_in).cuda()
                    for ep, tl in train_loop(q, q.parameters(), data, epochs, lr, 0.0, seed, lambda x: x,
                                             nesterov=False):
                        pass
                    q.eval()
                    prog = Q.export(q, shapes)
                    va = evaluate(data, int_fn(prog), "val")
                    tried.append(f"lr {lr} {va:.4f}")
                    if va > keep["val"]:
                        keep = {"prog": prog, "state": q.state_dict(), "val": va, "qat_lr": lr}
                torch.save({**keep, "scales": q.s, "pct": pct, "ptq_val": va0},
                           ckpt_dir(args) / f"{args.model}_s{seed}_int{bits}_{method}.pt")
                log(f"qat seed {seed} int{bits} {method:5s} (calib {pct}): {' | '.join(tried)} -> kept "
                    f"{'ptq' if not keep['qat_lr'] else 'lr ' + str(keep['qat_lr'])} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- stage: eval

def stage_eval(args, data):
    xte, yte = data["test"]
    rows = []
    out = ROOT / "results" / "tiny" / args.dataset / f"{args.model}_accuracy.csv"
    for seed in args.seeds:
        model = load_float(args, seed)
        fl = evaluate(data, lambda x: model(data["to_float"](x)), "test")
        metric = "auc" if data["task"] == "ae" else "accuracy"
        rows.append({"seed": seed, "bits": "float32", "method": "float", "trained_with": "", "metric": metric,
                     "feasible": 1,
                     "calib_pct": "", "qat_lr": "", "ptq_val_acc": "", "val_acc": "", "test_acc": fl})
        log(f"eval seed {seed} float32: {fl:.4f}")
        for bits in args.bits:
            for method in Q.METHODS:
                base = {"seed": seed, "bits": bits, "method": method, "trained_with": Q.TRAINED_WITH[method],
                        "metric": metric}
                if not Q.feasible(bits, method):
                    rows.append({**base, "feasible": 0, "calib_pct": "", "qat_lr": "", "ptq_val_acc": "",
                                 "val_acc": "", "test_acc": ""})
                    continue
                f = ckpt_dir(args) / f"{args.model}_s{seed}_int{bits}_{Q.TRAINED_WITH[method]}.pt"
                if not f.exists():
                    continue
                ck = torch.load(f, map_location="cpu", weights_only=False)
                t0 = time.time()
                acc = evaluate(data, int_fn(ck["prog"], method), "test",
                               bs=(100 if method == "sqapx" else 500) if data["task"] != "ae" else
                               (2000 if method == "sqapx" else 20000))
                rows.append({**base, "feasible": 1, "calib_pct": ck["pct"], "qat_lr": ck["qat_lr"],
                             "ptq_val_acc": ck["ptq_val"],
                             "val_acc": ck["val"], "test_acc": acc})
                log(f"eval seed {seed} int{bits} {method:5s}: test {acc:.4f} ({time.time() - t0:.0f}s)")
    out.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if out.exists():                                   # keep rows of seeds not evaluated in this run
        with open(out) as fh:
            old = [r for r in csv.DictReader(fh) if int(r["seed"]) not in args.seeds]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(old + rows)
    log("wrote", out.relative_to(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="resnet8")
    p.add_argument("--stage", default="all", choices=["all", "train", "qat", "eval"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--bits", nargs="+", type=int, default=list(Q.BITS))
    p.add_argument("--train-methods", nargs="+", default=list(TRAIN_METHODS))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--qat-epochs", type=int, default=None)
    p.add_argument("--qat-lr", type=float, default=None, help="one learning rate instead of QAT_LRS")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = D.load(args.dataset, "cuda")
    for s in (["train", "qat", "eval"] if args.stage == "all" else [args.stage]):
        t0 = time.time()
        {"train": stage_train, "qat": stage_qat, "eval": stage_eval}[s](args, data)
        log(f"stage {s} finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
