import depthai as dai
import numpy as np

# get pipeline for the device
pipeline = dai.Pipeline()

# RGB camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Mono cameras for stereo
mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)


# 2. Device initialization and load of calibration
with dai.Device(pipeline) as device:
    calib = device.readCalibration()

    # TODO resolution
    rgb_size = (1920, 1080)
    depth_size = (640, 400)

    # 3. RGB intrinsics
    rgb_intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, *rgb_size)
    print("RGB intrinsics (", rgb_size, "):")
    print(np.array(rgb_intr))

    # 4. Depth intrinsics (right mono camera)
    depth_intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, *depth_size)
    print("\nDepth (mono right) intrinsics (", depth_size, "):")
    print(np.array(depth_intr))

    # 5. Extrinsics – transformation from RIGHT (depth) to RGB camera
    extrinsics = calib.getCameraExtrinsics(dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.RGB)
    R = np.array(extrinsics['rotation']).reshape(3, 3)
    t = np.array(extrinsics['translation']).reshape(3)

    print("\nExtrinsics (RIGHT → RGB):")
    print("Rotation matrix R:")
    print(R)
    print("Translation vector t (in meters):")
    print(t)
