import depthai as dai
import numpy as np
import cv2
import pyrealsense2 as rs # For option enums, to match RealsenseDevice interface

def list_devices():
    """Returns a list of MXIDs of connected OAK devices."""
    devices = []
    for device_info in dai.Device.getAllAvailableDevices():
        try:
            # We must connect to get the MXID (getDeviceId)
            with dai.Device(device_info) as device:
                devices.append(device.getDeviceId())
        except RuntimeError:
            # Device might be in use, skip.
            pass
    return devices

class OakDevice:
    """Manages a connection to a single OAK device."""

    def __init__(self, serial_number):
        self.serial_number = serial_number
        self.pipeline = None
        self.device = None
        self.color_q = None
        self.depth_q = None
        self.control_q = None
        self.device_info = None
        self.profile = None # For compatibility with capture_service

        # Store last manual values for exposure/gain for disabling AE
        self.last_exp_us = 10000  # 10ms default
        self.last_iso = 800       # ISO 800 default

    def connect(self):
        """Checks if the device with the specified serial number is connected."""
        try:
            self.device_info = dai.DeviceInfo(self.serial_number)
            print(f"Device {self.serial_number} found.")
        except RuntimeError as e:
            raise RuntimeError(f"Device with serial number {self.serial_number} not found or busy. Error: {e}")

    def start(self, width=1280, height=720, fps=15):
        """Configures and starts the camera streams using a DepthAI v2 compatible API."""
        if self.device_info is None:
            self.connect()

        self.pipeline = dai.Pipeline()

        # Nodes - Use the universal Camera node for OAK-D-SR
        # One node per physical camera sensor.
        left_cam = self.pipeline.create(dai.node.Camera)
        right_cam = self.pipeline.create(dai.node.Camera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        # Nodes for I/O
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        control_in = self.pipeline.create(dai.node.XLinkIn)

        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")
        control_in.setStreamName("control")

        # Properties for Left Camera (provides color and left mono)
        left_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B) # CAM_B is LEFT
        left_cam.setSize(1280, 720)
        left_cam.setFps(fps)
        # Configure the 'preview' output for color, which the ISP will generate.
        left_cam.setPreviewSize(width, height)

        # Properties for Right Camera (provides right mono)
        right_cam.setBoardSocket(dai.CameraBoardSocket.CAM_C) # CAM_C is RIGHT
        right_cam.setSize(1280, 720)
        right_cam.setFps(fps)

        # Properties for StereoDepth
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)  # Best for short-range cameras like SR
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        stereo.setOutputSize(width, height)

        # Linking
        left_cam.video.link(stereo.left)
        right_cam.video.link(stereo.right)
        left_cam.preview.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)
        control_in.out.link(left_cam.inputControl)

        # Connect to device and start the pipeline
        try:
            # The constructor taking a pipeline and device_info is a robust way to
            # connect that should avoid "device in use" errors with v2 API.
            self.device = dai.Device(self.pipeline, self.device_info)
        except RuntimeError as e:
            self.device = None  # Ensure device object is cleared on failure
            raise e

        self.color_q = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.depth_q = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.control_q = self.device.getInputQueue(name="control")
        self.profile = True  # Indicates streaming is active

        print(f"Streaming started for device {self.serial_number}.")

    def stop(self):
        """Stops the camera streams."""
        if self.device:
            self.device.close()
        self.device = None
        self.pipeline = None
        self.profile = None
        print(f"Streaming stopped for device {self.serial_number}.")

    def get_frames(self):
        """Waits for and returns a coherent pair of frames (color and depth)."""
        color_image = None
        depth_image = None
        depth_colormap = None

        # Use blocking .get() to ensure we wait for a frame from each queue.
        # This mirrors the behavior of RealSense's wait_for_frames() and prevents
        # the UI from freezing due to empty, non-blocking calls.
        in_rgb = self.color_q.get()
        in_depth = self.depth_q.get()

        if in_rgb is not None and in_depth is not None:
            color_image = in_rgb.getCvFrame()
            depth_image = in_depth.getFrame() # uint16, millimeters

            # Create a colorized depth map for visualization.
            depth_clipped = np.clip(depth_image, 0, 10000) # Clip at 10m
            depth_viz = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        return color_image, depth_image, depth_colormap

    def get_option_range(self, option, sensor_type='color'):
        """
        Gets the range of a sensor option. For OAK, these are hardcoded
        as they are not queryable from the device in the same way as RealSense.
        """
        if sensor_type != 'color':
            return None # Controls only for color sensor

        # Mock a range object similar to pyrealsense2's rs.option_range
        class OptionRange:
            def __init__(self, min_val, max_val, step, default):
                self.min = min_val
                self.max = max_val
                self.step = step
                self.default = default

        if option == rs.option.exposure:
            # Exposure time in microseconds
            return OptionRange(min_val=1, max_val=33000, step=1, default=10000)
        elif option == rs.option.gain: # Corresponds to ISO for OAK
            # ISO value
            return OptionRange(min_val=100, max_val=1600, step=1, default=800)
        
        return None

    def get_option(self, option, sensor_type='color'):
        """
        Gets the current value of a sensor option. For OAK, returns a default
        as current values aren't directly queryable.
        """
        if sensor_type != 'color':
            return None
            
        if option == rs.option.exposure:
            return self.last_exp_us
        elif option == rs.option.gain:
            return self.last_iso
        elif option == rs.option.enable_auto_exposure:
            # OAK camera starts with auto exposure on by default.
            return True

        return None

    def set_option(self, option, value, sensor_type='color'):
        """Sets the value of a sensor option."""
        if not self.control_q or sensor_type != 'color': # Only color sensor is controllable
            return
        
        ctrl = dai.CameraControl()
        if option == rs.option.enable_auto_exposure:
            if value == 1.0:
                ctrl.setAutoExposureEnable()
            else: # Disable and use last known manual values
                ctrl.setManualExposure(self.last_exp_us, self.last_iso)
        elif option == rs.option.exposure:
            self.last_exp_us = int(value)
            ctrl.setManualExposure(self.last_exp_us, self.last_iso)
        elif option == rs.option.gain:
            self.last_iso = int(value)
            ctrl.setManualExposure(self.last_exp_us, self.last_iso)
        
        try:
            self.control_q.send(ctrl)
        except Exception as e:
            print(f"Error setting option {option} to {value} for {sensor_type} sensor: {e}")

    def get_intrinsics(self):
        """Returns the intrinsics of the color and depth streams."""
        if not self.device:
            return None
        
        calib_data = self.device.readCalibration()
        
        # When aligned, depth intrinsics match color intrinsics.
        # The color stream comes from the LEFT camera (CAM_B).
        # We need the intrinsics for the final output size (1280x720).
        w, h = 1280, 720
        intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, w, h)
        coeffs = calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B)

        formatted = {
            'width': w,
            'height': h,
            'fx': intrinsics[0][0],
            'fy': intrinsics[1][1],
            'ppx': intrinsics[0][2],
            'ppy': intrinsics[1][2],
            'model': 'Brown-Conrady',
            'coeffs': list(coeffs),
        }
        
        return {'depth': formatted, 'color': formatted}
