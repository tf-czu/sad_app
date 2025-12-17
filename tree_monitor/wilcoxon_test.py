"""
    Load and test data via Wilcoxon two-sided test
"""

import csv
import sys

from scipy import stats

def wilcoxon_test(a, b, label):  # two-sided test
    w_stat, p_value_w = stats.wilcoxon(a, b, alternative='two-sided')
    print(f"\nWilcoxon test for: {label}")
    # print data:
    print(f"Data a: {a}")
    print(f"Data b: {b}")
    print(f"W = {w_stat:.4f}, p-value = {p_value_w:.6f}")

def load_data_from_csv(file_path):
    labels = None
    medium_data = [[], [], [], []]
    nano_data = [[], [], [], []]
    with open(file_path) as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)  # skip first line
        for row in reader:
            # print(row)
            assert len(row) == 10, row
            raw = row[2:]
            if labels is None:
                labels = []
                for item in raw:
                    labels.append(item)

            else:
                values = [float(num) for num in raw]
                for ii in range(4):
                    medium_data[ii].append(values[ii])
                    nano_data[ii].append(values[ii + 4])

    return medium_data, nano_data, labels


if __name__ == "__main__":
    assert len(sys.argv) == 2, sys.argv
    file_path = sys.argv[1]
    medium_data, nano_data, labels = load_data_from_csv(file_path)
    print("\n")
    for med, nano, lab in zip(medium_data, nano_data, labels):
        wilcoxon_test(med, nano, lab)
