import sys
import os
import time
import json

# Add project root to path to allow importing rgbd_app
# This is a common pattern for scripts in a subdirectory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rgbd_app.realsense_device import RealsenseDevice, list_devices
from rgbd_app.capture_service import capture_one, capture_sequence

# The serial number is given in tasks.md for the target device
SERIAL = "218722270084"

def main():
    """
    Smoke test for the RealsenseDevice class.
    Connects to the device, grabs a few frames, prints intrinsics, and exits cleanly.
    """
    print("Listing available RealSense devices...")
    try:
        available_devices = list_devices()
        print("Available devices:", available_devices)
    except Exception as e:
        print(f"Error listing devices: {e}")
        print("Please ensure the Intel RealSense SDK is installed correctly.")
        return

    if SERIAL not in available_devices:
        print(f"\nError: Target device with serial number {SERIAL} not found.")
        print("Please connect the camera and ensure drivers are installed.")
        return

    device = None
    try:
        print(f"\nConnecting to device {SERIAL}...")
        device = RealsenseDevice(SERIAL)
        
        print("Starting stream...")
        device.start()

        # Allow some frames for auto-exposure to settle.
        print("Waiting 2 seconds for auto-exposure to settle...")
        time.sleep(2)

        print("\nGrabbing 30 frames...")
        for i in range(30):
            color, depth, _ = device.get_frames()
            if color is not None and depth is not None:
                # In a real app, you might display or process the frames here.
                # For the smoke test, we just confirm they are received.
                print(f"  Frame {i+1:2d}/30: Color={color.shape}, Depth={depth.shape}")
            else:
                print(f"  Frame {i+1:2d}/30: Failed to get frame.")
            time.sleep(0.01) # Small delay to avoid overwhelming the console
        
        print("\n--- Camera Intrinsics ---")
        intrinsics = device.get_intrinsics()
        if intrinsics:
            print(json.dumps(intrinsics, indent=2))
        else:
            print("Could not retrieve intrinsics.")
        print("-------------------------\n")

        print("\n--- Testing Single Capture ---")
        capture_one(device)
        print("----------------------------\n")

        print("\n--- Testing Sequence Capture (3 frames, 500ms interval) ---")
        capture_sequence(device, n_frames=3, interval_ms=500)
        print("-----------------------------------------------------------\n")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
    finally:
        if device:
            print("Stopping stream...")
            device.stop()
            print("Stream stopped.")
    
    print("\nSmoke test finished successfully.")

if __name__ == "__main__":
    main()
