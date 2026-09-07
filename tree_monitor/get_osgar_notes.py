"""
Walk through all osgar .log files in a given directory and print a table:
    log name | contents of the "note" parameter

The "note" parameter is stored as the first record on stream 0
(written by osgar.logger.LogWriter when the log is created).

Usage:
    python get_osgar_notes.py /path/to/dir
    python get_osgar_notes.py /path/to/dir --pattern "*.log" --width 80
    python get_osgar_notes.py /path/to/dir --csv notes.csv
"""

import argparse
import glob
import csv
import os
import sys
from ast import literal_eval

try:
    from osgar.logger import LogReader
except ImportError:
    sys.exit("Error: could not import 'osgar'. Install it with:\n"
              "    pip install osgar")


def get_note(filename):
    """Return the note content (first record on stream 0), or an error/empty message."""
    try:
        with LogReader(filename, only_stream_id=0) as log:
            data = next(log)[-1]
        try:
            print(type(data.decode("utf-8")), data.decode("utf-8"))
            return literal_eval(data.decode("utf-8"))[-3]
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")
    except StopIteration:
        return "(stream 0 is empty)"
    except Exception as e:
        return f"(error: {e})"


def collect_logs(directory, pattern):
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        sys.exit(f"No files matching pattern '{pattern}' found in directory '{directory}'.")
    return paths


def print_table(rows, note_width):
    """rows: list of (log_name, note) - simple plain-text table printout."""
    name_col = max([len("log")] + [len(name) for name, _ in rows])

    def clip(text, width):
        text = text.replace("\n", " ").replace("\r", " ")
        if len(text) > width:
            return text[: width - 1] + "…"
        return text

    header = f"{'log'.ljust(name_col)} | {'note'}"
    print(header)
    print("-" * name_col + "-+-" + "-" * min(note_width, 60))

    for name, note in rows:
        print(f"{name.ljust(name_col)} | {clip(note, note_width)}")


def save_csv(rows, csv_path):
    """rows: list of (log_name, note) - save the full (unclipped) results to a CSV file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["log", "note"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to '{csv_path}'.")

def main():
    parser = argparse.ArgumentParser(description="List the 'note' parameter from osgar logs (stream 0).")
    parser.add_argument("directory", help="directory containing osgar logs")
    parser.add_argument("--pattern", default="*.log", help="filename pattern (default: *.log)")
    parser.add_argument("--width", type=int, default=100, help="max width of the 'note' column (default: 100)")
    parser.add_argument("--csv", default=None, help="if given, save results to this CSV file")
    args = parser.parse_args()

    log_paths = collect_logs(args.directory, args.pattern)

    rows = []
    for path in log_paths:
        note = get_note(path)
        rows.append((os.path.basename(path), note))

    print_table(rows, args.width)
    if args.csv:
        save_csv(rows, args.csv)


if __name__ == "__main__":
    main()
