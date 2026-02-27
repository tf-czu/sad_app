import argparse
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import mannwhitneyu
from scipy import stats
import scikit_posthocs as sp
from itertools import combinations


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


def compare_groups_dunn(data, labels):
    h_stat, p_global = stats.kruskal(*data)

    print(f"{'GLOBAL TEST (K-W)':<25} {'STAT':<10} {'P-VALUE':<10}")
    print("-" * 50)
    print(f"{'Result':<25} {h_stat:<10.4f} {p_global:<10.4f}")
    print("\n")

    if p_global < 0.05:
        # Dunn test vrací matici p-hodnot
        # p_adjust může být 'holm', 'bonferroni', 'bh' (Benjamini-Hochberg) atd.
        p_matrix = sp.posthoc_dunn(data, p_adjust='holm')

        header = f"{'Comparison':<20} | {'Holm-Adj. p':<12} | {'Significant'}"
        print(header)
        print("-" * len(header))

        num_groups = len(labels)
        for i, j in combinations(range(num_groups), 2):
            # p_matrix je v podstatě numpy array (pokud není vstupem DataFrame)
            # indexujeme i+1 a j+1, protože scikit-posthocs indexuje od 1
            p_val = p_matrix.iloc[i, j]

            is_sig = "YES" if p_val < 0.05 else "no"
            pair_label = f"{labels[i]} vs {labels[j]}"
            print(f"{pair_label:<20} | {p_val:<12.4f} | {is_sig}")
    else:
        print("No significant differences found.")


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
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
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
        print(f"{label}, nano, DSC median: {np.median(nano[g])}")
        positions.append(pos - off)
        xticklabels.append(f"{label}\nnano")

        data.append(medium[g])
        print(f"{label}, medium, DSC median: {np.median(medium[g])}")
        positions.append(pos + off)
        xticklabels.append(f"{label}\nmedium")

        pos += 1.0 + gap

    compare_groups_dunn(data, ["sp_n", "sp_m", "aut_n", "aut_m", "aut2_n", "aut2_m"])

    plt.figure(figsize=(6, 4))
    bp = plt.boxplot(data, positions=positions, widths=0.35, showfliers=True)
    plt.setp(bp['boxes'], color="k")
    plt.setp(bp['whiskers'], color="k")
    plt.setp(bp['caps'], color="k")
    plt.setp(bp['medians'], color="k")

    plt.xticks(positions, xticklabels)
    plt.ylabel("DSC")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=1200)
    plt.close()

    print(f"\nSaved plot: {args.out_plot}")


if __name__ == "__main__":
    main()
