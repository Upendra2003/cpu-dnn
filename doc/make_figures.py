"""Draws every plot in doc/roba_explained.tex from the CSV files in results/.

    python3 doc/make_figures.py      -> doc/figures/*.pdf
"""
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roba import reference  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "doc" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MULTS = ["exact", "mitchell", "drum6", "roba_dn1", "roba_updn", "roba_dn2", "roba_updndn", "roba_updn2"]
NAME = {"exact": "Exact", "mitchell": "Mitchell", "drum6": "DRUM6", "roba_dn1": "RoBA-Dn1",
        "roba_updn": "RoBA-UpDn", "roba_dn2": "RoBA-Dn2", "roba_updndn": "RoBA-UpDnDn",
        "roba_updn2": "RoBA-(UpDn)2", "sqtab": "Square table"}
COLOR = {"exact": "#3b4351", "mitchell": "#9aa3ad", "drum6": "#9aa3ad", "roba_dn1": "#2a78d6",
         "roba_updn": "#2a78d6", "roba_dn2": "#e5683f", "roba_updndn": "#e5683f",
         "roba_updn2": "#e5683f", "sqtab": "#7b52d6"}
DS = {"mnist": "MNIST", "fashion": "Fashion-MNIST"}
MODEL = {"mlp": "MLP", "lenet5": "LeNet-5"}
P_LOD, P_NONE = 0.829, 0.793          # core power relative to a core with a single-cycle multiplier

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "savefig.bbox": "tight", "font.family": "serif"})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", (FIG / f"{name}.pdf").relative_to(ROOT))


def label_bars(ax, bars, fmt="{:.1f}", dx=0.0):
    for b in bars:
        w = b.get_width()
        ax.text(w + dx, b.get_y() + b.get_height() / 2, " " + fmt.format(w), va="center", fontsize=7)


# ---------------------------------------------------------------- 1. error map of two multipliers

def fig_error_map():
    n = 64
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    for ax, m in zip(axes, ["roba_dn1", "roba_updn", "roba_updn2"]):
        err = np.array([[100 * (reference.mul(a, b, m) - a * b) / (a * b) for a in range(1, n)]
                        for b in range(1, n)])
        im = ax.imshow(err, origin="lower", cmap="RdBu", vmin=-25, vmax=25,
                       extent=[0.5, n - 0.5, 0.5, n - 0.5])
        ax.set_title(NAME[m])
        ax.set_xlabel("a")
        ax.grid(False)
        for p in (2, 4, 8, 16, 32):
            ax.axvline(p, color="k", lw=0.3, alpha=0.4)
            ax.axhline(p, color="k", lw=0.3, alpha=0.4)
    axes[0].set_ylabel("b")
    fig.colorbar(im, ax=axes, shrink=0.9, label="error of a×b (%)")
    save(fig, "error_map")


# ---------------------------------------------------------------- 2. error statistics

def fig_errors():
    rows = [r for r in read(RES / "mnist/multiplier_errors.csv") if r["bits"] == "8"]
    ms = [r["multiplier"] for r in rows]
    mean = [float(r["mean_abs_rel_err_pct"]) for r in rows]
    mx = [float(r["max_abs_rel_err_pct"]) for r in rows]
    bias = [float(r["mean_signed_rel_err_pct"]) for r in rows]
    y = np.arange(len(ms))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, constrained_layout=True)
    b1 = axes[0].barh(y + 0.2, mx, 0.38, color="#c9d3df", label="worst case")
    b2 = axes[0].barh(y - 0.2, mean, 0.38, color=[COLOR[m] for m in ms], label="average")
    label_bars(axes[0], b1)
    label_bars(axes[0], b2, "{:.2f}")
    axes[0].set_yticks(y, [NAME[m] for m in ms])
    axes[0].set_xlabel("size of the error (% of the true product)")
    axes[0].legend(loc="lower right", fontsize=7)
    axes[0].set_xlim(0, 30)
    b3 = axes[1].barh(y, bias, 0.6, color=[COLOR[m] for m in ms])
    for b, v in zip(b3, bias):
        axes[1].text(v + (0.3 if v >= 0 else -0.3), b.get_y() + b.get_height() / 2, f"{v:+.2f}",
                     va="center", ha="left" if v >= 0 else "right", fontsize=7)
    axes[1].axvline(0, color="k", lw=0.6)
    axes[1].set_xlim(-12, 3)
    axes[1].set_xlabel("average signed error (%): below 0 = too small")
    save(fig, "errors")


# ---------------------------------------------------------------- 3. GPU accuracy

def fig_gpu_acc():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
    for i, ds in enumerate(["mnist", "fashion"]):
        for j, model in enumerate(["mlp", "lenet5"]):
            ax = axes[i][j]
            rows = read(RES / ds / f"eval_{model}.csv")
            base = next(float(r["test_acc"]) for r in rows if r["domain"] == "float32")
            acc = {r["multiplier"]: 100 * float(r["test_acc"]) for r in rows if r["domain"] == "8"}
            ft = {r["multiplier"]: 100 * float(r["acc_after"])
                  for r in read(RES / ds / f"finetune_{model}.csv") if r["bits"] == "8"}
            x = np.arange(len(MULTS))
            ax.bar(x - 0.2, [acc[m] for m in MULTS], 0.38, color=[COLOR[m] for m in MULTS],
                   label="int8, no fine-tuning")
            ax.bar(x + 0.2, [ft[m] for m in MULTS], 0.38, color=[COLOR[m] for m in MULTS], alpha=0.45,
                   label="int8, after fine-tuning")
            ax.axhline(100 * base, color="k", lw=0.8, ls="--", label="float32 (normal PyTorch)")
            lo = min(min(acc.values()), min(ft.values()), 100 * base)
            hi = max(max(acc.values()), max(ft.values()), 100 * base)
            ax.set_ylim(lo - 0.6, hi + 0.4)
            ax.set_xticks(x, [NAME[m].replace("RoBA-", "") for m in MULTS], rotation=40, ha="right", fontsize=7)
            ax.set_title(f"{DS[ds]}, {MODEL[model]}")
            ax.set_ylabel("test accuracy (%)")
            if i == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="lower left")
    save(fig, "gpu_acc")


# ---------------------------------------------------------------- 4. number formats ("domains")

def fig_domains():
    fig, ax = plt.subplots(figsize=(6.0, 2.3), constrained_layout=True)
    labels, vals, cols = [], [], []
    for ds in ["mnist", "fashion"]:
        for model in ["mlp", "lenet5"]:
            rows = read(RES / ds / f"eval_{model}.csv")
            for dom, name, c in [("float32", "float32", "#3b4351"), ("trunc", "cut to int", "#e5683f"),
                                 ("8", "int8", "#2a78d6"), ("16", "int16", "#7fb0ea")]:
                r = next(r for r in rows if r["domain"] == dom and r["multiplier"] in ("exact", "native"))
                labels.append(f"{DS[ds]}\n{MODEL[model]}")
                vals.append(100 * float(r["test_acc"]))
                cols.append((name, c))
    x = np.arange(4)
    for k, (name, c) in enumerate(cols[:4]):
        v = vals[k::4]
        bars = ax.bar(x + (k - 1.5) * 0.2, v, 0.19, color=c, label=name)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                    ha="center", fontsize=6)
    ax.set_xticks(x, labels[::4])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "domains")


# ---------------------------------------------------------------- 5. cycles per multiply

def fig_mult_bench():
    rows = read(RES / "ibex/mult_bench.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, constrained_layout=True)
    for ax, lod, title in zip(axes, ["instruction", "software"],
                              ["with the LOD instruction", "LOD done in software (no new instruction)"]):
        for m in MULTS:
            pts = sorted((int(r["bits"]), float(r["cycles_per_mult"])) for r in rows
                         if r["lod"] == lod and r["multiplier"] == m)
            ls = "-" if m in ("exact", "roba_dn1") else "--"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o", ms=3,
                    color=COLOR[m] if m not in ("mitchell", "drum6") else ("#6c757d" if m == "drum6" else "#adb5bd"),
                    lw=1.6 if m in ("exact", "roba_dn1") else 1.0, label=NAME[m])
        ax.set_xticks([4, 8, 16, 31])
        ax.set_xlabel("operand size (bits)")
        ax.set_title(title)
    axes[0].set_ylabel("cycles per multiplication")
    axes[1].legend(fontsize=6.5, ncol=2, loc="upper left")
    save(fig, "mult_bench")


# ---------------------------------------------------------------- Ibex summaries

def ibex(name):
    return read(RES / f"ibex/{name}_summary.csv")


def cyc(r):
    return float(r["cycles_per_image"]) / 1e6


def fig_ibex_call():
    rows = ibex("dnn_call")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    for i, ds in enumerate(["mnist", "fashion"]):
        for j, model in enumerate(["mlp", "lenet5"]):
            ax = axes[i][j]
            sel = {r["multiplier"]: r for r in rows if r["dataset"] == ds and r["model"] == model}
            y = np.arange(len(MULTS))[::-1]
            bars = ax.barh(y, [cyc(sel[m]) for m in MULTS], color=[COLOR[m] for m in MULTS])
            label_bars(ax, bars)
            ax.set_yticks(y, [NAME[m] for m in MULTS], fontsize=7)
            ax.set_title(f"{DS[ds]}, {MODEL[model]}")
            ax.set_xlabel("million cycles per image")
            ax.set_xlim(0, max(cyc(r) for r in sel.values()) * 1.18)
    save(fig, "ibex_call")


def fig_profile():
    rows = read(RES / "ibex/profile.csv")
    cats = ["Multiplier routine", "Sign & zero handling", "Operand loads & stores",
            "Loop control & accumulate", "Rescale & ReLU", "Pooling", "Bookkeeping"]
    ccol = ["#2a78d6", "#9cc3f0", "#5b6573", "#aab2bd", "#e5683f", "#d6dbe1", "#eceff3"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ["mlp", "lenet5"]):
        share = {}
        for r in rows:
            if r["model"] == model:
                share.setdefault(r["multiplier"], {})[r["category"]] = 100 * float(r["share"])
        y = np.arange(len(MULTS))[::-1]
        left = np.zeros(len(MULTS))
        for c, col in zip(cats, ccol):
            v = np.array([share[m].get(c, 0) for m in MULTS])
            ax.barh(y, v, left=left, color=col, label=c)
            if c == "Multiplier routine":
                for yy, vv in zip(y, v):
                    ax.text(1, yy, f"{vv:.0f}%", va="center", color="white", fontsize=7)
            left += v
        ax.set_yticks(y, [NAME[m] for m in MULTS], fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of the inference time (%)")
        ax.set_title(f"Fashion-MNIST, {MODEL[model]}")
    axes[1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "profile")


def fig_fused():
    call = {(r["dataset"], r["model"], r["multiplier"]): cyc(r) for r in ibex("dnn_call")}
    fused = {(r["dataset"], r["model"], r["multiplier"]): cyc(r) for r in ibex("dnn")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ["mlp", "lenet5"]):
        y = np.arange(len(MULTS))[::-1]
        for yy, m in zip(y, MULTS):
            c, f = call[("fashion", model, m)], fused[("fashion", model, m)]
            ax.plot([f, c], [yy, yy], color="#c0c7d0", lw=2, zorder=1)
            ax.scatter([c], [yy], s=28, facecolor="white", edgecolor="#3b4351", zorder=2)
            ax.scatter([f], [yy], s=28, color=COLOR[m], zorder=3)
            ax.text(c, yy + 0.28, f"{c:.1f}", fontsize=6, ha="center", color="#6c757d")
            ax.text(f, yy - 0.42, f"{f:.1f}", fontsize=6, ha="center")
        ax.set_yticks(y, [NAME[m] for m in MULTS], fontsize=7)
        ax.set_xlabel("million cycles per image")
        ax.set_title(f"Fashion-MNIST, {MODEL[model]}")
        ax.set_xlim(0, None)
    axes[0].scatter([], [], s=28, facecolor="white", edgecolor="#3b4351", label="one call per multiply")
    axes[0].scatter([], [], s=28, color="#2a78d6", label="fused into the loop")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fused")


def fig_sqtab():
    rows = {(r["dataset"], r["model"], r["config"]): cyc(r) for r in ibex("dnn")}
    combos = [(ds, m) for ds in ["mnist", "fashion"] for m in ["mlp", "lenet5"]]
    fig, ax = plt.subplots(figsize=(7.0, 2.8), constrained_layout=True)
    x = np.arange(len(combos))
    for k, m in enumerate(["exact", "roba_dn1", "sqtab"]):
        v = [rows[(ds, mo, m)] for ds, mo in combos]
        bars = ax.bar(x + (k - 1) * 0.27, v, 0.26, color=COLOR[m], label=NAME[m] + (" (fused)" if m != "sqtab" else ""))
        for b, vv, (ds, mo) in zip(bars, v, combos):
            sp = rows[(ds, mo, "exact")] / vv
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{vv:.1f}\n{sp:.2f}×",
                    ha="center", fontsize=6)
    ax.set_xticks(x, [f"{DS[ds]}\n{MODEL[mo]}" for ds, mo in combos])
    ax.set_ylabel("million cycles per image")
    ax.set_ylim(0, 30)
    ax.legend(fontsize=7)
    save(fig, "sqtab")


def fig_zeroskip():
    rows = {(r["dataset"], r["model"], r["config"]): cyc(r) for r in ibex("zeroskip")}
    settings = [("", "normalized"), ("+zs", "normalized + skip"), ("+raw", "raw pixels"),
                ("+zs+raw", "raw + skip")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
    for i, ds in enumerate(["mnist", "fashion"]):
        for j, model in enumerate(["mlp", "lenet5"]):
            ax = axes[i][j]
            x = np.arange(3)
            base = rows[(ds, model, "exact")]
            for k, (suf, lab) in enumerate(settings):
                v = [rows[(ds, model, m + suf)] for m in ["exact", "roba_dn1", "sqtab"]]
                bars = ax.bar(x + (k - 1.5) * 0.2, v, 0.19, color=["#3b4351", "#2a78d6", "#7b52d6"],
                              alpha=[0.35, 0.55, 0.8, 1.0][k], label=lab)
                for b, vv in zip(bars, v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{base / vv:.1f}×",
                            ha="center", fontsize=5.5, rotation=90)
            ax.set_xticks(x, ["Exact", "RoBA-Dn1", "Square table"])
            ax.set_title(f"{DS[ds]}, {MODEL[model]}")
            ax.set_ylabel("million cycles per image")
            ax.set_ylim(0, base * 1.35)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#3b4351", alpha=a) for a in [0.35, 0.55, 0.8, 1.0]]
    axes[0][0].legend(handles, [s[1] for s in settings], fontsize=6.5, loc="upper right")
    save(fig, "zeroskip")


def fig_energy():
    rows = {(r["dataset"], r["model"], r["config"]): cyc(r) for r in ibex("dnn")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, constrained_layout=True)
    ms = MULTS + ["sqtab"]
    for ax, model in zip(axes, ["mlp", "lenet5"]):
        e0 = rows[("fashion", model, "exact")] * P_NONE
        y = np.arange(len(ms))[::-1]
        v = [rows[("fashion", model, m)] * (P_NONE if m in ("exact", "sqtab") else P_LOD) / e0 for m in ms]
        bars = ax.barh(y, v, color=[COLOR[m] for m in ms])
        label_bars(ax, bars, "{:.2f}")
        ax.axvline(1, color="k", lw=0.6, ls="--")
        ax.set_yticks(y, [NAME[m] for m in ms], fontsize=7)
        ax.set_xlabel("energy per image (exact = 1)")
        ax.set_title(f"Fashion-MNIST, {MODEL[model]} (fused)")
        ax.set_xlim(0, 1.45)
    save(fig, "energy")


def fig_layers():
    rows = {(r["dataset"], r["model"], r["config"]): r for r in ibex("dnn")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), constrained_layout=True)
    names = {"mlp": ["FC 784→256", "FC 256→128", "FC 128→10"],
             "lenet5": ["conv1 + pool", "conv2 + pool", "FC 400→120", "FC 120→84", "FC 84→10"]}
    cols = ["#2a78d6", "#7fb0ea", "#e5683f", "#f2a585", "#9aa3ad"]
    for ax, model in zip(axes, ["mlp", "lenet5"]):
        ms = ["exact", "roba_dn1", "sqtab"]
        y = np.arange(len(ms))[::-1]
        left = np.zeros(len(ms))
        for li, ln in enumerate(names[model]):
            v = np.array([float(rows[("fashion", model, m)][f"layer{li}_cycles"]) / 1e6 for m in ms])
            ax.barh(y, v, left=left, color=cols[li], label=ln)
            left += v
        ax.set_yticks(y, [NAME[m] for m in ms], fontsize=7)
        ax.set_xlabel("million cycles per image")
        ax.set_title(f"Fashion-MNIST, {MODEL[model]} (fused)")
        ax.legend(fontsize=6, loc="lower right")
    save(fig, "layers")



# ---------------------------------------------------------------- Phase 3: MLPerf Tiny (results/tiny/)

TINY = [("cifar10", "resnet8", "CIFAR-10, ResNet-8"), ("speech_commands", "dscnn", "Speech Commands, DS-CNN"),
        ("toyadmos", "fcae", "ToyADMOS, autoencoder")]
TM = {"exact": ("Exact", "#3b4351"), "sqtab": ("Square table", "#7b52d6"), "dn1": ("RoBA-Dn1", "#2a78d6"),
      "dn2": ("RoBA-Dn2", "#e5683f"), "sqapx": ("Approx. square", "#9aa3ad")}


def tiny_acc(ds, model):
    """{(bits, method): [test metric per seed, in %]}"""
    out = {}
    for r in read(RES / f"tiny/{ds}/{model}_accuracy.csv"):
        if not r["test_acc"]:
            continue
        out.setdefault((r["bits"], r["method"]), {})[r["seed"]] = 100 * float(r["test_acc"])
    return out


def tiny_ibex(ds, model):
    return {(r["bits"], r["method"]): r for r in read(RES / f"tiny/{ds}/{model}_ibex_summary.csv")}


def fig_tiny_acc():
    """Accuracy change against exact at the same precision, per seed then mean +- std."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True, constrained_layout=True)
    for ax, (ds, model, title) in zip(axes, TINY):
        a = tiny_acc(ds, model)
        x = np.arange(3)
        for k, m in enumerate(["dn1", "dn2"]):
            mu, sd = [], []
            for b in ["4", "8", "16"]:
                d = [a[(b, m)][s] - a[(b, "exact")][s] for s in a[(b, m)]]
                mu.append(np.mean(d)); sd.append(np.std(d, ddof=1))
            ax.bar(x + (k - 0.5) * 0.36, mu, 0.34, yerr=sd, capsize=2, color=TM[m][1], label=TM[m][0],
                   error_kw={"lw": 0.7})
            for xx, v, e in zip(x + (k - 0.5) * 0.36, mu, sd):
                ax.text(xx, v - e - 0.05 if v < 0 else v + e + 0.05, f"{v:+.2f}", ha="center",
                        va="top" if v < 0 else "bottom", fontsize=5.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, ["int4", "int8", "int16"])
        ax.set_title(title, fontsize=8)
        ax.set_ylim(-2.4, 1.1)
    axes[0].set_ylabel("change vs. exact (points)")
    axes[0].legend(fontsize=7, loc="lower left")
    save(fig, "tiny_acc")


def fig_tiny_speed():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True, constrained_layout=True)
    ms = ["sqtab", "dn1", "dn2"]
    for ax, (ds, model, title) in zip(axes, TINY):
        r = tiny_ibex(ds, model)
        x = np.arange(3)
        for k, m in enumerate(ms):
            for xi, b in enumerate(["4", "8", "16"]):
                xx = xi + (k - 1) * 0.27
                if (b, m) not in r:
                    ax.text(xx, 0.1, "not\nfeasible", ha="center", va="bottom", fontsize=5.5, rotation=90,
                            color=TM[m][1])
                    continue
                v = float(r[(b, m)]["speedup_vs_exact"])
                ax.bar(xx, v, 0.26, color=TM[m][1], label=TM[m][0] if xi == 1 else None)
                ax.text(xx, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)
        ax.axhline(1, color="k", lw=0.6, ls="--")
        ax.set_xticks(x, ["int4", "int8", "int16"])
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 3.8)
    axes[0].set_ylabel("speed-up vs. exact (×)")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "tiny_speed")


def fig_tiny_cycles():
    """Million cycles per input against precision: exact doubles, RoBA barely moves."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), constrained_layout=True)
    for ax, (ds, model, title) in zip(axes, TINY):
        r = tiny_ibex(ds, model)
        for m in ["exact", "sqtab", "dn1", "dn2"]:
            bs = [b for b in ["4", "8", "16"] if (b, m) in r]
            ax.plot([int(b) for b in bs], [float(r[(b, m)]["cycles_per_input"]) / 1e6 for b in bs], "o-",
                    color=TM[m][1], label=TM[m][0], ms=3.5, lw=1.4)
        ax.set_xticks([4, 8, 16], ["int4", "int8", "int16"])
        ax.set_ylim(0, None)
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("million cycles per input")
    axes[0].legend(fontsize=6.5)
    save(fig, "tiny_cycles")


def fig_tiny_tradeoff():
    """Every (dataset, precision, approximate method): speed-up against accuracy change."""
    fig, ax = plt.subplots(figsize=(6.2, 3.3), constrained_layout=True)
    marks = {"4": "s", "8": "o", "16": "^"}
    for ds, model, title in TINY:
        a, r = tiny_acc(ds, model), tiny_ibex(ds, model)
        for (b, m), row in r.items():
            if m in ("exact", "sqapx"):
                continue
            d = np.mean([a[(b, m)][s] - a[(b, "exact")][s] for s in a[(b, m)]])
            ax.scatter(float(row["speedup_vs_exact"]), d, marker=marks[b], color=TM[m][1], s=28,
                       edgecolor="white", lw=0.5)
    for m in ["sqtab", "dn1", "dn2"]:
        ax.scatter([], [], color=TM[m][1], marker="o", label=TM[m][0])
    for b in ["4", "8", "16"]:
        ax.scatter([], [], color="#777", marker=marks[b], label=f"int{b}")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(1, color="k", lw=0.6, ls="--")
    ax.set_xlabel("speed-up vs. exact at the same precision (×)")
    ax.set_ylabel("accuracy change (points)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    save(fig, "tiny_tradeoff")


def fig_tiny_layers():
    """CIFAR-10 int8: cycles per layer. Ops 0..10 are conv1 s1a s1b s2s s2a s2b s3s s3a s3b pool fc."""
    r = tiny_ibex("cifar10", "resnet8")
    names = ["conv1", "s1a", "s1b", "s2s", "s2a", "s2b", "s3s", "s3a", "s3b"]
    fig, ax = plt.subplots(figsize=(7.2, 2.5), constrained_layout=True)
    x = np.arange(len(names))
    for k, m in enumerate(["exact", "sqtab", "dn1", "dn2"]):
        v = [float(r[("8", m)][f"op{i}_cycles"]) / 1e6 for i in range(len(names))]
        ax.bar(x + (k - 1.5) * 0.2, v, 0.19, color=TM[m][1], label=TM[m][0])
    ax.set_xticks(x, names)
    ax.set_ylabel("million cycles per image")
    ax.set_title("CIFAR-10, ResNet-8, int8: cycles per layer", fontsize=8)
    ax.legend(fontsize=7, ncol=4)
    save(fig, "tiny_layers")



# ---------------------------------------------------------------- Phase 4: the Zybo FPGA board (fpga_zybo/results/)

BOARD = ROOT / "fpga_zybo" / "results"


def board_rows():
    return {(r["dataset"], r["bits"], r["method"], r["mem"]): r for r in read(BOARD / "board_summary.csv")}


def fig_board_mem():
    """Speech Commands: speed-up over exact with everything in block RAM vs data in DDR."""
    b = board_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True, constrained_layout=True)
    for ax, mem, title in zip(axes, ["bram", "ddr"], ["block RAM, 1-cycle", "data in DDR, ~15 cycles"]):
        x = np.arange(3)
        for k, m in enumerate(["sqtab", "dn1", "dn2"]):
            for xi, bits in enumerate(["4", "8", "16"]):
                xx = xi + (k - 1) * 0.27
                r = b.get(("speech_commands", bits, m, mem))
                if r is None:
                    ax.text(xx, 0.1, "not\nfeasible", ha="center", va="bottom", fontsize=5.5, rotation=90,
                            color=TM[m][1])
                    continue
                v = float(r["speedup_vs_exact"])
                ax.bar(xx, v, 0.26, color=TM[m][1], label=TM[m][0] if xi == 1 else None)
                ax.text(xx, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)
        ax.axhline(1, color="k", lw=0.6, ls="--")
        ax.set_xticks(x, ["int4", "int8", "int16"])
        ax.set_title(f"Speech Commands, board, {title}", fontsize=8)
        ax.set_ylim(0, 3.4)
    axes[0].set_ylabel("speed-up vs. exact (×)")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "board_mem")


def fig_board_ddr():
    """Every task with its data in DDR: speed-up over exact."""
    b = board_rows()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True, constrained_layout=True)
    for ax, (ds, model, title) in zip(axes, TINY):
        x = np.arange(3)
        for k, m in enumerate(["sqtab", "dn1", "dn2"]):
            for xi, bits in enumerate(["4", "8", "16"]):
                xx = xi + (k - 1) * 0.27
                r = b.get((ds, bits, m, "ddr"))
                if r is None:
                    ax.text(xx, 0.1, "not\nfeasible", ha="center", va="bottom", fontsize=5.5, rotation=90,
                            color=TM[m][1])
                    continue
                v = float(r["speedup_vs_exact"])
                ax.bar(xx, v, 0.26, color=TM[m][1], label=TM[m][0] if xi == 1 else None)
                ax.text(xx, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)
        ax.axhline(1, color="k", lw=0.6, ls="--")
        ax.set_xticks(x, ["int4", "int8", "int16"])
        ax.set_title(title + " (DDR)", fontsize=8)
        ax.set_ylim(0, 2.6)
    axes[0].set_ylabel("speed-up vs. exact (×)")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "board_ddr")


def fig_board_wait():
    """Speech Commands int8 with data in DDR: cycles per input split into work and waiting for DDR."""
    runs = read(BOARD / "speech_commands" / "dscnn_board_runs.csv")
    b = board_rows()
    ms = ["exact", "sqtab", "dn1", "dn2"]
    fig, ax = plt.subplots(figsize=(6.4, 2.3), constrained_layout=True)
    y = np.arange(len(ms))[::-1]
    for yi, m in zip(y, ms):
        rs = [r for r in runs if r["bits"] == "8" and r["method"] == m and r["mem"] == "ddr"]
        n = 5 * len(rs)
        wait = sum(int(r["ddr_wait"]) for r in rs) / n / 1e6
        total = float(b[("speech_commands", "8", m, "ddr")]["cycles_per_input"]) / 1e6
        onchip = float(b[("speech_commands", "8", m, "bram")]["cycles_per_input"]) / 1e6
        ax.barh(yi, total - wait, color=TM[m][1], label=None)
        ax.barh(yi, wait, left=total - wait, color=TM[m][1], alpha=0.35, hatch="///", edgecolor="white")
        ax.text(total + 2, yi, f"{total:.0f} M  (on-chip: {onchip:.0f} M)", va="center", fontsize=6.5)
    ax.set_yticks(y, [TM[m][0] for m in ms], fontsize=7)
    ax.set_xlabel("million cycles per input (solid: computing, hatched: waiting for DDR)")
    ax.set_xlim(0, 290)
    ax.set_title("Speech Commands, int8, data in DDR", fontsize=8)
    save(fig, "board_wait")


if __name__ == "__main__":
    fig_error_map()
    fig_errors()
    fig_gpu_acc()
    fig_domains()
    fig_mult_bench()
    fig_ibex_call()
    fig_profile()
    fig_fused()
    fig_sqtab()
    fig_zeroskip()
    fig_energy()
    fig_layers()
    fig_tiny_acc()
    fig_tiny_speed()
    fig_tiny_cycles()
    fig_tiny_tradeoff()
    fig_tiny_layers()
    fig_board_mem()
    fig_board_ddr()
    fig_board_wait()
