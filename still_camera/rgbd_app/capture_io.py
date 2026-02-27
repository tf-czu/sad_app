import os
import json
from datetime import datetime
import cv2

def create_capture_directory(base_dir="captures", camera_name=""):
    """
    Creates a unique directory for a capture session.
    Format: base_dir/YYYYMMDD_HHMMSS_CAMERANAME/
    Args:
        base_dir (str): The base directory to create the capture folder in.
        camera_name (str): The name of the camera (e.g., 'D405').
    Returns:
        str: The path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{camera_name}"
    capture_path = os.path.join(base_dir, dir_name)
    os.makedirs(capture_path, exist_ok=True)
    return capture_path

def save_rgb(path, frame_index, image):
    """Saves a BGR numpy array as a PNG file."""
    filename = os.path.join(path, f"frame_{frame_index:04d}_rgb.png")
    cv2.imwrite(filename, image)

def save_depth_16bit(path, frame_index, image):
    """
    Saves a uint16 numpy array as a 16-bit PNG.

    Each pixel value in the saved PNG represents the distance to the object
    at that point, measured in millimeters. A value of 0 means the distance
    is unknown or outside the measurable range.
    """
    filename = os.path.join(path, f"frame_{frame_index:04d}_depth.png")
    cv2.imwrite(filename, image)

def save_depth_preview(path, frame_index, image):
    """Saves a colorized depth preview (BGR) as a PNG."""
    filename = os.path.join(path, f"frame_{frame_index:04d}_depth_preview.png")
    cv2.imwrite(filename, image)

def save_metadata(path, frame_index, data):
    """Saves a dictionary as a JSON file."""
    filename = os.path.join(path, f"frame_{frame_index:04d}_meta.json")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
