"""
Read a CSV file (columns: log, note) produced by get_osgar_notes.py,
then for each log extract all "app.color" frames (jpeg-encoded) and
save them as numbered .jpg files into a folder named after the note.

Usage:
    python extract_images.py notes.csv --logdir /path/to/dir
    python extract_images.py notes.csv --logdir /path/to/dir --outdir extracted
"""
import argparse
import csv
import os
import re
import sys

try:
    from osgar.logger import LogReaderEx
except ImportError:
    sys.exit("Error: could not import 'osgar'. Install it with:\n"
              "    pip install osgar")

STREAM_NAME = "app.color"


def sanitize(name):
    """Turn a note string into a safe directory name."""
    name = name.strip()
    if not name or name.startswith("("):  # e.g. "(stream 0 is empty)", "(error: ...)"
        name = "unknown"
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


def extract_images(log_path, out_dir):
    """Extract all app.color frames from a single log into out_dir, numbered from 001."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with LogReaderEx(log_path, names=[STREAM_NAME]) as log:
        for timestamp, channel_name, data in log:
            count += 1
            img_path = os.path.join(out_dir, f"{count:03d}.jpg")
            with open(img_path, "wb") as f:
                f.write(data)
    return count


def read_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["log"], row["note"]))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract 'app.color' jpeg frames from osgar logs into folders named after their note."
    )
    parser.add_argument("csv_file", help="CSV file with columns: log, note")
    parser.add_argument("--logdir", default=".", help="directory where the log files are located (default: current dir)")
    parser.add_argument("--outdir", default=".", help="directory where the per-note folders will be created (default: current dir)")
    args = parser.parse_args()

    rows = read_csv(args.csv_file)

    for log_name, note in rows:
        log_path = os.path.join(args.logdir, log_name)
        if not os.path.isfile(log_path):
            print(f"Skipping '{log_name}': file not found in '{args.logdir}'.")
            continue

        folder_name = sanitize(note)
        out_dir = os.path.join(args.outdir, folder_name)

        try:
            count = extract_images(log_path, out_dir)
            print(f"{log_name}: saved {count} image(s) to '{out_dir}'.")
        except Exception as e:
            print(f"{log_name}: error while extracting images: {e}")


if __name__ == "__main__":
    main()
