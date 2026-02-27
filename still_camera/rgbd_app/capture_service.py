import time
from datetime import datetime
import pyrealsense2 as rs
from . import capture_io
from .realsense_device import RealsenseDevice

def capture_one(device: object, camera_name: str):
    """
    Captures and stores a single snapshot from the device.
    Args:
        device (object): An initialized and streaming device (RealsenseDevice or OakDevice).
        camera_name (str): The name of the camera being used.
    Returns:
        str: The path to the capture directory, or None on failure.
    """
    # The 'profile' attribute is used as a flag to check if streaming is active.
    # It's set in both RealsenseDevice and OakDevice during start().
    if not getattr(device, 'profile', False):
        print("Error: Device is not streaming. Please start the stream first.")
        return None

    # Create directory for this capture
    capture_path = capture_io.create_capture_directory(camera_name=camera_name)
    print(f"Saving snapshot to: {capture_path}")

    # Get frames. Note: get_frames() might return None.
    color_img, depth_img, depth_colormap = device.get_frames()
    if color_img is None or depth_img is None:
        print("Error: Failed to capture frame.")
        return None

    # Prepare metadata
    metadata = {
        'timestamp_utc': datetime.utcnow().isoformat() + "Z", # ISO 8601 format
        'serial_number': device.serial_number,
        'intrinsics': device.get_intrinsics(),
        'exposure_rgb': device.get_option(rs.option.exposure, 'color'),
        'gain_rgb': device.get_option(rs.option.gain, 'color'),
        'exposure_depth': device.get_option(rs.option.exposure, 'depth'),
        'gain_depth': device.get_option(rs.option.gain, 'depth'),
        'frame_index': 0,
    }

    # Save all data
    capture_io.save_rgb(capture_path, 0, color_img)
    capture_io.save_depth_16bit(capture_path, 0, depth_img)
    capture_io.save_depth_preview(capture_path, 0, depth_colormap)
    capture_io.save_metadata(capture_path, 0, metadata)

    print("Snapshot saved successfully.")
    return capture_path

def capture_sequence(device: object, camera_name: str, n_frames: int, interval_ms: int, progress_callback=None):
    """
    Captures and stores a sequence of frames.
    Args:
        device (object): An initialized and streaming device (RealsenseDevice or OakDevice).
        camera_name (str): The name of the camera being used.
        n_frames (int): The number of frames to capture.
        interval_ms (int): The interval between captures in milliseconds.
        progress_callback (callable, optional): A function to call with progress updates.
                                                 It receives (current_frame_number, total_frames).
    Returns:
        str: The path to the capture directory, or None on failure.
    """
    # The 'profile' attribute is used as a flag to check if streaming is active.
    # It's set in both RealsenseDevice and OakDevice during start().
    if not getattr(device, 'profile', False):
        print("Error: Device is not streaming. Please start the stream first.")
        return None

    # Create directory for this capture sequence
    capture_path = capture_io.create_capture_directory(camera_name=camera_name)
    print(f"Saving sequence of {n_frames} frames to: {capture_path}")

    interval_sec = interval_ms / 1000.0

    for i in range(n_frames):
        start_time = time.monotonic()
        
        if progress_callback:
            progress_callback(i + 1, n_frames)
        else:
            print(f"Capturing frame {i+1}/{n_frames}...")

        # Get frames
        color_img, depth_img, depth_colormap = device.get_frames()
        if color_img is None or depth_img is None:
            print(f"Error: Failed to capture frame {i+1}. Skipping.")
            # We still need to wait for the interval
            elapsed_time = time.monotonic() - start_time
            wait_time = interval_sec - elapsed_time
            if wait_time > 0:
                time.sleep(wait_time)
            continue
        
        # Prepare metadata
        metadata = {
            'timestamp_utc': datetime.utcnow().isoformat() + "Z", # ISO 8601 format
            'serial_number': device.serial_number,
            'intrinsics': device.get_intrinsics(),
            'exposure_rgb': device.get_option(rs.option.exposure, 'color'),
            'gain_rgb': device.get_option(rs.option.gain, 'color'),
            'exposure_depth': device.get_option(rs.option.exposure, 'depth'),
            'gain_depth': device.get_option(rs.option.gain, 'depth'),
            'frame_index': i,
        }

        # Save all data
        capture_io.save_rgb(capture_path, i, color_img)
        capture_io.save_depth_16bit(capture_path, i, depth_img)
        capture_io.save_depth_preview(capture_path, i, depth_colormap)
        capture_io.save_metadata(capture_path, i, metadata)
        
        # Wait for the next interval, accounting for processing time
        elapsed_time = time.monotonic() - start_time
        wait_time = interval_sec - elapsed_time
        if wait_time > 0:
            time.sleep(wait_time)

    print("Sequence capture finished successfully.")
    return capture_path
