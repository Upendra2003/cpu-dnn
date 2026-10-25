"""Build index.html from site/template.html and the CSVs in results/<dataset>/."""
import csv
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roba import cost  # noqa: E402

RESULTS = ROOT / "results"
PAPER_MAX = {"roba_dn2": 6.24, "roba_updndn": 2.74, "roba_updn2": 0.83}  # % (paper text)
MACS = {"mlp": 234_752, "lenet5": 416_520}
DATASET_NAMES = {"mnist": "MNIST", "fashion": "Fashion-MNIST"}


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_dataset_results(folder):
    evals, finetune = [], []
    for model in MACS:
        evals += [{"model": r["model"], "domain": r["domain"], "multiplier": r["multiplier"],
                   "acc": float(r["test_acc"]), "loss": float(r["test_loss"]),
                   "agree": float(r["agree_with_exact"]), "acc_bits": r["acc_bits"]}
                  for r in rows(folder / f"eval_{model}.csv")]
        finetune += [{"model": r["model"], "bits": int(r["bits"]), "multiplier": r["multiplier"],
                      "acc_before": float(r["acc_before"]), "acc_after": float(r["acc_after"]),
                      "loss_before": float(r["loss_before"]), "loss_after": float(r["loss_after"])}
                     for r in rows(folder / f"finetune_{model}.csv")]
    config = json.loads((folder / "config.json").read_text())
    meta = {"name": DATASET_NAMES.get(folder.name, folder.name), "seed": config["seed"],
            "epochs": config["epochs"], "ft_epochs": config["ft_epochs"], "ft_lr": config["ft_lr"],
            "date": time.strftime("%Y-%m-%d", time.localtime((folder / "run.log").stat().st_mtime))}
    return {"eval": evals, "finetune": finetune, "meta": meta}


def main():
    datasets = {}
    for key in DATASET_NAMES:
        folder = RESULTS / key
        if (folder / "finetune_lenet5.csv").exists():
            datasets[key] = load_dataset_results(folder)
    # The multiplier error study does not depend on the dataset; take it from the first run.
    first = RESULTS / next(iter(datasets))
    errors = [{"bits": int(r["bits"]), "multiplier": r["multiplier"],
               "mean": float(r["mean_abs_rel_err_pct"]), "max": float(r["max_abs_rel_err_pct"]),
               "bias": float(r["mean_signed_rel_err_pct"]), "frac_exact": float(r["frac_exact"])}
              for r in rows(first / "multiplier_errors.csv")]
    data = {
        "errors": errors, "datasets": datasets,
        "cycles": {b: cost.CYCLES_PER_MULT[b] for b in (8, 16)},
        "power": {"none": round(cost.POWER_NONE, 4), "lod": round(cost.POWER_NONE_LOD, 4)},
        "macs": MACS, "paper_max": PAPER_MAX, "gpu": "NVIDIA L40",
    }
    template = (ROOT / "site" / "template.html").read_text()
    # "</" cannot appear inside the inline JSON script.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = template.replace("__DATA__", payload)
    # Standalone document so it renders in standards mode when served or opened directly.
    head, body = page.split("</style>", 1)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'{head}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n')
    (ROOT / "index.html").write_text(page)
    print("wrote index.html with datasets:", ", ".join(datasets))


if __name__ == "__main__":
    main()
