import argparse
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
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

    # Build one boxplot: for each group, two boxes side-by-side (nano vs medium).
    groups = ["sp", "aut", "aut2"]
    data = []
    positions = []
    xticklabels = []

    pos = 1.0
    gap = 1.0
    off = 0.4

    for g, label in zip(groups, ["Spring", "Autumn", "Autumn2"]):
        # Print means and number of zero items.
        sub_data_nano = nano[g]
        sub_data_nano_td = [num for num in sub_data_nano if  num !=0]
        sub_data_medium = medium[g]
        sub_data_medium_td = [num for num in sub_data_medium if num != 0]
        print(f"{label} nano, median: {np.median(sub_data_nano_td):.3f}, zeros: "
              f"{np.count_nonzero(sub_data_nano==0)}, detections: {len(sub_data_nano)}")
        print(f"{label} medium, median: {np.median(sub_data_medium_td):.3f}, zeros: "
              f"{np.count_nonzero(sub_data_medium == 0)}, detections: {len(sub_data_medium)}")

        data.append(sub_data_nano_td)
        positions.append(pos - off)
        xticklabels.append(f"{label}\nnano")

        data.append(sub_data_medium_td)
        positions.append(pos + off)
        xticklabels.append(f"{label}\nmedium")

        pos += 1.0 + gap

    # --- Statistical Analysis ---
    # Perform global Kruskal-Wallis test across all 6 datasets
    kw_stat, kw_p = stats.kruskal(*data)
    print(f"\nKruskal-Wallis test p-value: {kw_p:.4e}")

    # Perform Dunn's post-hoc test with Holm adjustment
    p_matrix = sp.posthoc_dunn(data, p_adjust="holm")
    print(f"P_matrix for Dunn's post-hoc test: {p_matrix}")

    # Generate Compact Letter Display (CLD)
    cld = sp.compact_letter_display(p_matrix, alpha=0.05)

    plt.figure(figsize=(6, 3))
    bp = plt.boxplot(data, positions=positions, widths=0.35, showfliers=True)
    plt.setp(bp['boxes'], color="k")
    plt.setp(bp['whiskers'], color="k")
    plt.setp(bp['caps'], color="k")
    plt.setp(bp['medians'], color="k")

    # Annotate CLD letters above the upper whiskers/fliers dynamically
    for ii, pos_x in enumerate(positions):
        max_y = np.max(data[ii]) if len(data[ii]) > 0 else 1.0
        plt.text(
                pos_x,
                max_y + 0.02,
                cld[ii + 1],  # scikit-posthocs uses 1-based indexing for groups
                ha = "center",
                va = "bottom",
                fontsize = 10,
                fontweight = "bold",
            )

    plt.xticks(positions, xticklabels)
    plt.ylabel("DSC", fontsize=12)

    # Adjust Y-limit slightly to leave room for the significance letters at the top
    all_max = max([np.max(d) for d in data if len(d) > 0])
    plt.ylim(top=all_max + 0.08)

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=1200)
    plt.close()

    print(f"\nSaved plot: {args.out_plot}")


if __name__ == "__main__":
    main()
