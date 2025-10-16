"""
    Draw PR data
"""

import csv
import sys

from matplotlib import pyplot as plt

def load_data_from_csv(file_path):
    precision = []
    recall = []
    try:
        with open(file_path, encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                # print(row)
                if len(row) >= 2:
                    try:
                        p = float(row[0].strip())  # precision
                        r = float(row[1].strip())  # recall
                        precision.append(p)
                        recall.append(r)
                    except ValueError:
                        print(f"Warning: skipping line in file {file_path}: {row}")
                        continue
        return precision, recall

    except FileNotFoundError:
        return None


def plot_precision_recall_curves(csv_medium, csv_nano):
    precision_med, recall_med = load_data_from_csv(csv_medium)
    precision_nano, recall_nano = load_data_from_csv(csv_nano)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.plot(recall_nano, precision_nano, "-b", label="Nano model")
    ax.plot(recall_med, precision_med, "-r", label="Medium model")

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.legend(loc='lower left')

    # plt.show()
    plt.savefig("pr_plot_tmp", dpi=1200)

if __name__ == "__main__":
    assert len(sys.argv) == 3, sys.argv
    plot_precision_recall_curves(sys.argv[1], sys.argv[2])
