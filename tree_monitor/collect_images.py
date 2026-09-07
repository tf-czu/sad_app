"""
Read a CSV file (columns: log, note) where "note" is the name of a folder
containing extracted images (e.g. produced by extract_images.py).
For each folder, copy one image with a given filename (e.g. "015.jpg")
into a single output folder, renumbered sequentially with a given prefix
(e.g. "2s_001.jpg", "2s_002.jpg", ...).

Usage:
    python collect_images.py notes.csv --imagedir extracted --image 015.jpg --outdir 2s --prefix 2s
"""
import argparse
import csv
import os
import shutil

def read_notes(csv_path):
    notes = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes.append(row["note"])
    return notes


def main():
    parser = argparse.ArgumentParser(
        description="Copy one named image from each note-folder into a single renumbered output folder."
    )
    parser.add_argument("csv_file", help="CSV file with columns: log, note")
    parser.add_argument("--imagedir", default=".", help="directory containing the per-note image folders (default: current dir)")
    parser.add_argument("--image", required=True, help="image filename to pick from each folder, e.g. 015.jpg")
    parser.add_argument("--outdir", required=True, help="output folder for the collected images")
    parser.add_argument("--prefix", default=None, help="prefix for output filenames (default: same as --outdir name)")
    args = parser.parse_args()

    prefix = args.prefix if args.prefix is not None else os.path.basename(os.path.normpath(args.outdir))
    ext = os.path.splitext(args.image)[1] or ".jpg"

    os.makedirs(args.outdir, exist_ok=True)

    notes = read_notes(args.csv_file)

    count = 0
    for note in notes:
        src = os.path.join(args.imagedir, note, args.image)
        if not os.path.isfile(src):
            print(f"Skipping '{note}': '{args.image}' not found.")
            continue

        count += 1
        dst_name = f"{prefix}_{count:03d}{ext}"
        dst = os.path.join(args.outdir, dst_name)
        shutil.copyfile(src, dst)
        print(f"{note}/{args.image} -> {dst}")

    print(f"Done: copied {count} image(s) to '{args.outdir}'.")


if __name__ == "__main__":
    main()
