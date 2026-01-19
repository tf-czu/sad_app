import argparse
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def group_from_name(s):
    # Note: "aut2" contains "aut", so check aut2 first.
    s = str(s)
    if "aut2" in s:
        return "aut2"
    if "aut" in s:
        return "aut"
    if "sp" in s:
        return "sp"
    return None


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def load_groups(csv_path):
    # Reads the CSV produced by the matching script.
    # Uses the 1st column for image name and the 7th column for DICE (1-based).
    out = {"sp": [], "aut": [], "aut2": []}

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        assert header is not None, f"Empty CSV: {csv_path}"

        for row in r:
            if not row:
                continue
            assert len(row) >= 7, f"Expected >=7 columns, got {len(row)} in {csv_path}"
            img = row[0]
            g = group_from_name(img)
            if g is None:
                continue
            d = to_float(row[6])
            if d is None or math.isnan(d):
                continue
            out[g].append(d)

    # Convert to numpy arrays
    for k in out:
        out[k] = np.array(out[k], dtype=float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano", required=True, help="Nano CSV")
    ap.add_argument("--medium", required=True, help="Medium CSV")
    ap.add_argument("--out_plot", default="dice_boxplot.png", help="Output plot path")
    args = ap.parse_args()

    nano = load_groups(args.nano)
    medium = load_groups(args.medium)

    print("Mann–Whitney U test (two-sided):")
    for g in ["sp", "aut", "aut2"]:
        x = nano[g]
        y = medium[g]
        print(f"\nGroup {g}: n_nano={len(x)}, n_medium={len(y)}")
        if len(x) == 0 or len(y) == 0:
            print("  Skipped (missing data in one set).")
            continue
        res = mannwhitneyu(x, y, alternative="two-sided")
        print(f"  U = {res.statistic:.3f}, p = {res.pvalue:.6g}")

    # Build one boxplot: for each group, two boxes side-by-side (nano vs medium).
    groups = ["sp", "aut", "aut2"]
    data = []
    positions = []
    xticklabels = []

    pos = 1.0
    gap = 1.0
    off = 0.4

    for g, label in zip(groups, ["Spring", "Autumn", "Autumn2"]):
        data.append(nano[g])
        positions.append(pos - off)
        xticklabels.append(f"{label}\nnano")

        data.append(medium[g])
        positions.append(pos + off)
        xticklabels.append(f"{label}\nmedium")

        pos += 1.0 + gap

    plt.figure(figsize=(6, 4))
    bp = plt.boxplot(data, positions=positions, widths=0.35, showfliers=True)
    plt.setp(bp['boxes'], color="k")
    plt.setp(bp['whiskers'], color="k")
    plt.setp(bp['caps'], color="k")
    plt.setp(bp['medians'], color="k")

    plt.xticks(positions, xticklabels)
    plt.ylabel("DICE")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=1200)
    plt.close()

    print(f"\nSaved plot: {args.out_plot}")


if __name__ == "__main__":
    main()
