import cv2
import numpy as np
import argparse
import os

def main():
    """
    Loads a 16-bit depth PNG, normalizes it for visualization,
    and displays it as a colorized depth map.
    """
    parser = argparse.ArgumentParser(description="View a 16-bit depth PNG file.")
    parser.add_argument("filepath", help="Path to the 16-bit PNG depth image.")
    parser.add_argument("--max-dist", type=float, default=10.0,
                        help="Maximum distance in meters for colormap visualization. (default: 10.0)")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found at '{args.filepath}'")
        return

    # Load the 16-bit PNG image. The flag IMREAD_UNCHANGED is crucial.
    depth_mm = cv2.imread(args.filepath, cv2.IMREAD_UNCHANGED)

    if depth_mm is None:
        print(f"Error: Could not read the image file at '{args.filepath}'. It might be corrupted or not a valid image.")
        return

    if depth_mm.dtype != 'uint16':
        print(f"Warning: The image is not in 16-bit format (uint16). Its format is {depth_mm.dtype}.")
        # Attempt to continue, but the data might not represent millimeters.

    print(f"Loaded image with shape: {depth_mm.shape}, data type: {depth_mm.dtype}")
    min_val, max_val, _, _ = cv2.minMaxLoc(depth_mm)
    print(f"Depth values range from {min_val}mm to {max_val}mm.")

    # Normalize the depth image to 0-255 range for visualization
    max_dist_mm = args.max_dist * 1000
    depth_clipped = np.clip(depth_mm, 0, max_dist_mm)
    
    # Avoid division by zero if max distance is 0
    if max_dist_mm > 0:
        depth_display = cv2.convertScaleAbs(depth_clipped, alpha=(255.0 / max_dist_mm))
    else:
        depth_display = np.zeros(depth_mm.shape, dtype=np.uint8)

    # Apply a colormap for better visualization
    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

    # Make pixels with 0 distance (unknown) black
    depth_colormap[depth_mm == 0] = 0

    # Display the image
    window_name = f"Depth Preview - {os.path.basename(args.filepath)}"
    cv2.imshow(window_name, depth_colormap)
    print("\nPress any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
