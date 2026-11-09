"""Collect the Vivado numbers of the two cores (with and without LOD) into one table.

    python3 host/collect_reports.py         -> reports/fpga_cores.csv, and prints the LOD power factor

Reads reports/<variant>/{utilization_hier,power_hier}.rpt written by vivado/build.tcl for variant lod and nolod.
The ratio P(core with LOD) / P(core without LOD) is the board's own version of the study's P_LOD / P_NONE
(0.829 / 0.793 = 1.045 from the paper); host/run_board.py uses it for energy when reports/fpga_cores.csv exists.
"""
import csv
import pathlib
import re

FZ = pathlib.Path(__file__).resolve().parents[1]
CELLS = {"soc": "u_soc", "ibex_top": "u_ibex", "core": "u_ibex_core", "alu": "alu_i", "decoder": "decoder_i",
         "regfile": "register_file_i", "bram": "bram"}


def table_rows(path):
    """Rows of the first '|'-separated table(s) of a Vivado report: [cells...] with the name cell stripped."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("|") and line.count("|") >= 3:
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def first_match(rows, name):
    for r in rows:
        if r and (r[0] == name or r[0].endswith("/" + name) or r[0].split()[0] == name):
            return r
    return None


def number(s):
    m = re.match(r"^[<]?\s*([0-9.]+)", s or "")
    return float(m.group(1)) if m else None


def main():
    out = []
    for variant in ("lod", "nolod"):
        d = FZ / "reports" / variant
        if not (d / "power_hier.rpt").exists():
            print(f"missing reports/{variant}: run  vivado -mode batch -source vivado/build.tcl -tclargs {variant}")
            continue
        util = table_rows(d / "utilization_hier.rpt")
        powr = table_rows(d / "power_hier.rpt")
        header = next((r for r in util if r and r[0] == "Instance"), None)
        row = {"variant": variant}
        for key, cell in CELLS.items():
            u = first_match(util, cell)
            p = first_match(powr, cell)
            if u and header:
                for col in ("Total LUTs", "FFs", "RAMB36", "RAMB18", "DSP Blocks"):
                    if col in header:
                        row[f"{key}_{col.replace(' ', '_').lower()}"] = u[header.index(col)]
            if p:
                row[f"{key}_power_w"] = number(p[1]) if len(p) > 1 else ""
        summ = FZ / "reports" / f"fpga_summary_{variant}.csv"
        if summ.exists():
            row.update(next(csv.DictReader(open(summ))))
        out.append(row)
    if not out:
        return
    keys = sorted({k for r in out for k in r}, key=lambda k: (k != "variant", k))
    path = FZ / "reports" / "fpga_cores.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print("wrote", path)
    by = {r["variant"]: r for r in out}
    if "lod" in by and "nolod" in by:
        for key in ("core", "ibex_top"):
            a, b = by["lod"].get(f"{key}_power_w"), by["nolod"].get(f"{key}_power_w")
            if a and b:
                print(f"{key}: {a:.4f} W with LOD, {b:.4f} W without -> LOD power factor {a / b:.4f} "
                      f"(paper: {0.829 / 0.793:.4f})")
        for key in ("core_total_luts", "core_ffs"):
            print(f"{key}: lod {by['lod'].get(key)}  nolod {by['nolod'].get(key)}")


if __name__ == "__main__":
    main()
