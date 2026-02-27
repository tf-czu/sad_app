import depthai as dai
import inspect

def main():
    """
    Connects to the first available OAK device and lists the available
    resolutions for its color and mono sensors as defined in the depthai library.
    """
    # Connect to the device to check its features
    try:
        with dai.Device() as device:
            print(f"Connected to device: {device.getDeviceId()}")
            print("Device has the following camera sensors:")
            
            # Get a list of connected camera features
            features = device.getConnectedCameraFeatures()
            
            for camera_feature in features:
                print(f"- Socket: {camera_feature.socket}, Sensor name: {camera_feature.sensorName}, Supported types: {camera_feature.supportedTypes}")

    except RuntimeError as e:
        print(f"Could not connect to OAK device: {e}")
        print("Will still list available resolutions from the library.")


    print("\n--- Available Resolutions in DepthAI Library ---")

    # List resolutions for ColorCamera
    print("\nColor Camera Resolutions (dai.ColorCameraProperties.SensorResolution):")
    for name, member in inspect.getmembers(dai.ColorCameraProperties.SensorResolution):
        if not name.startswith('_') and isinstance(member, dai.ColorCameraProperties.SensorResolution):
            print(f"  - {name}")

    # List resolutions for MonoCamera
    print("\nMono Camera Resolutions (dai.MonoCameraProperties.SensorResolution):")
    for name, member in inspect.getmembers(dai.MonoCameraProperties.SensorResolution):
        if not name.startswith('_') and isinstance(member, dai.MonoCameraProperties.SensorResolution):
            print(f"  - {name}")

if __name__ == "__main__":
    main()
