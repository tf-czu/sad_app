import os
import shutil


def main(im_path, train_path):
    os.makedirs(os.path.join(im_path, "tmp"))

    images = [f for f in os.listdir(im_path) if f.lower().endswith((".jpg", ".jpeg"))]
    train_images = [f for f in os.listdir(train_path) if f.lower().endswith((".jpg", ".jpeg"))]

    # Get file sizes
    sizes_train = {f: os.path.getsize(os.path.join(train_path, f)) for f in train_images}

    # Open the note file
    with open(os.path.join(im_path, "notes.txt"), "w") as notes:
        for img in images:
            path_img = os.path.join(im_path, img)
            im_size = os.path.getsize(path_img)

            for img_train, size_train in sizes_train.items():
                if im_size == size_train:
                    shutil.move(path_img, os.path.join(im_path, "tmp", img))
                    notes.write(f"{img} <-> {img_train}\n")
                    break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('im', help="Images dir")
    parser.add_argument('train', help='Train directory')
    args = parser.parse_args()

    main(args.im, args.train)
