"""
    extract images, rename and move
"""
import os
import shutil


def main(directory, prefix, out_dir):
    im_list = sorted(os.listdir(directory))
    jj = 0
    for ii, im in enumerate(im_list):
        if ii%50 == 0:
            shutil.move(os.path.join(directory, im), os.path.join(out_dir, f"{prefix}_{jj:03d}"))
            jj += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='path to directory')
    parser.add_argument('--prefix', help='image prefix', default="im")
    parser.add_argument('--out', help='out directory', default="logs/tmp_im")
    args = parser.parse_args()

    main(args.dir, args.prefix, args.out)
