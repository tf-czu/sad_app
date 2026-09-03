"""
Compare segmentation algorithms by DSC across dataset groups.

Reads dice_{variant}_{alg}.csv files (one per algorithm x variant). Each row is
one detected tree with its DSC (column index 6). Detections are assigned to a
dataset group (sp / aut / aut2) by image name.

For every (variant, datagroup) pair (nano-sp, medium-aut, ...) the DSC
concentrations of all algorithms are compared with a global Kruskal-Wallis
test; if significant, Dunn's post-hoc (Holm adjustment) is run per pair.

Zeros (unmatched / false-positive detections, DSC==0) are included by default.

Usage:
    python compare_dice_groups.py --dir data/tmp_dice_res
    python compare_dice_groups.py --dir data/tmp_dice_res --exclude-zeros
"""

import argparse
import csv
import glob
import math
import os
from itertools import combinations

import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp

VARIANTS = ["nano", "medium"]
GROUPS = ["sp", "aut", "aut2"]
GROUP_LABELS = {"sp": "Spring", "aut": "Autumn", "aut2": "Autumn2"}


def group_from_name(s):
    # "aut2" contains "aut", so check aut2 first.
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


def load_csv(csv_path, exclude_zeros):
    """Return dict group -> list[float] of DSC for one CSV file."""
    out = {g: [] for g in GROUPS}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if not row or len(row) < 7:
                continue
            g = group_from_name(row[0])
            if g is None:
                continue
            d = to_float(row[6])
            if d is None or math.isnan(d):
                continue
            if exclude_zeros and d == 0.0:
                continue
            out[g].append(d)
    return out


def run_kruskal_dunn(data, labels):
    h, p = stats.kruskal(*data)
    print(f"  Kruskal-Wallis:  H = {h:.4f},  p = {p:.4g}")
    if p >= 0.05:
        print("  No significant difference across algorithms (p >= 0.05).")
        return
    p_matrix = sp.posthoc_dunn(data, p_adjust="holm")
    print("  Dunn post-hoc (Holm-adjusted p):")
    for i, j in combinations(range(len(labels)), 2):
        pv = p_matrix.iloc[i, j]
        sig = "YES" if pv < 0.05 else "no"
        print(f"    {labels[i]:<6} vs {labels[j]:<6} | Holm adj. p = {pv:.4g} | significant: {sig}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/tmp_dice_res",
                    help="Directory with dice_*.csv files")
    ap.add_argument("--exclude-zeros", action="store_true",
                    help="Drop DSC==0 (unmatched) detections before testing")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "dice_*.csv")))
    if not files:
        print(f"No dice_*.csv files found in {args.dir}")
        return

    by_name = {}
    for f in files:
        base = os.path.basename(f)
        stem = base[len("dice_"):-len(".csv")]
        variant, _, alg = stem.partition("_")
        if variant not in VARIANTS:
            continue
        by_name[(variant, alg)] = f

    algs = sorted({alg for (_, alg) in by_name})
    print(f"Algorithms found: {algs}")
    print(f"Variants: {VARIANTS}")
    print(f"Groups: {GROUPS}")
    print(f"Zeros: {'excluded' if args.exclude_zeros else 'included'} in analysis")
    print("=" * 72)

    for variant in VARIANTS:
        for g in GROUPS:
            label = f"{variant}-{g}"
            print(f"\n### Group: {label}  ({GROUP_LABELS[g]}, {variant} variant)")
            print("-" * 72)
            data, labels = [], []
            for alg in algs:
                key = (variant, alg)
                if key not in by_name:
                    print(f"  {alg}: missing file, skipped")
                    continue
                vals = np.array(load_csv(by_name[key], args.exclude_zeros)[g], dtype=float)
                nz = int(np.count_nonzero(vals == 0))
                stats_line = f"n={len(vals):<4} mean={np.mean(vals):.4f} " \
                             f"median={np.median(vals):.4f} zeros={nz}"
                print(f"  {alg:<6}: {stats_line}")
                data.append(vals)
                labels.append(alg)
            if len(labels) >= 2:
                run_kruskal_dunn(data, labels)


if __name__ == "__main__":
    main()
