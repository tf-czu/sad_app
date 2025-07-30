import os
import random
import shutil


def main(src_folder, num_img = 500):
    dst_folder = os.path.join(src_folder, "val")
    os.makedirs(dst_folder)
    all_images = [f for f in os.listdir(src_folder) if f.lower().endswith((".jpg", ".jpeg"))]

    if len(all_images) < num_img:
        raise ValueError(f"In the dir '{src_folder}' are just {len(all_images)} images!")

    selected = random.sample(all_images, num_img)
    for im in selected:
        shutil.copy(os.path.join(src_folder, im), os.path.join(dst_folder, im))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('im', help="Images dir")
    args = parser.parse_args()

    main(args.im)
