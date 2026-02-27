import depthai as dai

def main():
    """Lists the MXIDs (serial numbers) of all connected OAK devices."""
    
    # Get the list of all available devices
    available_devices = dai.Device.getAllAvailableDevices()
    
    if not available_devices:
        print("No OAK devices found.")
        return
        
    print(f"Found {len(available_devices)} OAK device(s):")
    for device_info in available_devices:
        # To get the MXID, we need to briefly connect to the device.
        # This is the most reliable method across different depthai versions.
        try:
            with dai.Device(device_info) as device:
                print(f"  - MXID: {device.getDeviceId()}")
        except RuntimeError as e:
            print(f"  - Could not get MXID for a device. It might be in use. Error: {e}")

if __name__ == "__main__":
    main()
