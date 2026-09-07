"""
Simulate a darker/low-light version of a photo (steps 1-3, simplified):
    1. Undo gamma encoding with a simple power curve (approx. sRGB, gamma 2.2).
    2. Scale down linear brightness by an exposure factor
       (simulates less light hitting the sensor).
    3. Add a single Gaussian noise term whose strength grows as the
       exposure factor drops (approximates the combined effect of
       photon shot noise + sensor read noise, without modeling them
       separately).
    4. Re-apply gamma encoding and clip back to 0-255.

All images found in the input directory are processed and saved
(with the same filenames) into the output directory.

Usage:
    python simulate_low_light.py input_dir output_dir --exposure 0.5 --noise 4
"""
import argparse
import glob
import os

import cv2
import numpy as np

GAMMA = 2.2
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def srgb_to_linear(img):
    return (img.astype(np.float32) / 255.0) ** GAMMA


def linear_to_srgb(img):
    img = np.clip(img, 0, 1)
    return (img ** (1.0 / GAMMA) * 255.0).round().astype(np.uint8)


def simulate_low_light(img, exposure=0.2, noise_std=4.0):
    """
    img: uint8 BGR image (as read by cv2.imread)
    exposure: fraction of original linear brightness to keep (0-1, lower = darker)
    noise_std: base noise std-dev on a 0-255 scale, at exposure = 1.0;
               scaled up by 1/sqrt(exposure) as the scene gets darker
    """
    linear = srgb_to_linear(img)

    # 2) Reduce exposure
    linear = linear * exposure

    # 3) Combined shot+read noise, approximated as a single Gaussian term
    #    that grows as the exposure factor drops (less light -> noisier).
    effective_std = (noise_std / 255.0) / np.sqrt(exposure)
    linear_noisy = linear + np.random.normal(0, effective_std, linear.shape)

    return linear_to_srgb(linear_noisy)


def collect_images(input_dir):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Simulate a low-light version of all images in a directory.")
    parser.add_argument("input_dir", help="directory with input images")
    parser.add_argument("output_dir", help="directory where processed images will be saved")
    parser.add_argument("--exposure", type=float, default=0.3,
                         help="fraction of original brightness to keep, 0-1 (default: 0.3)")
    parser.add_argument("--noise", type=float, default=2.0,
                         help="base noise std-dev on a 0-255 scale, at exposure=1.0 (default: 2.0)")
    args = parser.parse_args()

    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in '{args.input_dir}'.")

    os.makedirs(args.output_dir, exist_ok=True)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping '{path}': could not read image.")
            continue

        result = simulate_low_light(img, exposure=args.exposure, noise_std=args.noise)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, result)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
