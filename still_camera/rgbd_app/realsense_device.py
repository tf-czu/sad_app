import pyrealsense2 as rs
import numpy as np
import cv2

def list_devices():
    """Returns a list of serial numbers of connected RealSense devices."""
    ctx = rs.context()
    devices = ctx.query_devices()
    return [d.get_info(rs.camera_info.serial_number) for d in devices]

class RealsenseDevice:
    """Manages a connection to a single RealSense device."""

    def __init__(self, serial_number):
        """
        Initializes the device manager.
        Args:
            serial_number (str): The serial number of the device to connect to.
        """
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = None
        self.device_found = False
        self.color_sensor = None
        self.depth_sensor = None
        self.depth_scale = None

    def connect(self):
        """
        Checks if the device with the specified serial number is connected.
        Raises:
            RuntimeError: If the device is not found.
        """
        if self.serial_number not in list_devices():
            raise RuntimeError(f"Device with serial number {self.serial_number} not found.")
        self.config.enable_device(self.serial_number)
        self.device_found = True
        print(f"Device {self.serial_number} found.")

    def start(self, width=1280, height=720, fps=15):
        """
        Configures and starts the camera streams.
        Args:
            width (int): The width of the frames.
            height (int): The height of the frames.
            fps (int): The frames per second. D405 supports up to 90fps at lower res,
                     but high resolution support is more limited. 15fps is safe.
        """
        if not self.device_found:
            self.connect()

        # Enable depth and color streams. Using BGR8 format for color as it's common for OpenCV.
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Get the depth sensor first, it's essential for this application.
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        if not self.depth_sensor:
            raise RuntimeError("Could not find a depth sensor on the device.")

        # Setting the Visual Preset mode
        if self.depth_sensor.supports(rs.option.visual_preset):
            # For High Density, change to: rs.rs400_visual_preset.high_density
            preset = rs.rs400_visual_preset.high_accuracy
            try:
                self.depth_sensor.set_option(rs.option.visual_preset, int(preset))
                print("Visual preset set to High Accuracy.")
            except Exception as e:
                print(f"Error setting visual preset: {e}")
        
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Try to get a dedicated color sensor. This might fail on some devices (e.g., D405)
        # where the depth sensor also provides the color stream.
        try:
            self.color_sensor = self.profile.get_device().first_color_sensor()
        except RuntimeError:
            print("No dedicated color sensor found. Checking if depth sensor can provide color.")
            self.color_sensor = None

        # If no dedicated color sensor was found, check if the depth sensor can be used.
        # This is the case for cameras like the D405.
        if not self.color_sensor and self.depth_sensor.supports(rs.option.exposure):
            print("Using depth sensor for color controls.")
            self.color_sensor = self.depth_sensor
        
        if not self.color_sensor:
            raise RuntimeError("Could not find a sensor with color capabilities.")

        print(f"Streaming started for device {self.serial_number}.")

    def stop(self):
        """Stops the camera streams."""
        self.pipeline.stop()
        self.profile = None
        self.color_sensor = None
        self.depth_sensor = None
        print(f"Streaming stopped for device {self.serial_number}.")

    def get_frames(self):
        """
        Waits for and returns a coherent pair of frames (color and depth).
        Returns:
            A tuple (color_frame, depth_frame, depth_colormap) where:
            - color_frame is a numpy array (BGR).
            - depth_frame is a numpy array (uint16).
            - depth_colormap is a numpy array (BGR) for visualization.
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None

        # Convert images to numpy arrays
        depth_image_raw = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth from raw units to millimeters (as uint16) to match OAK camera output
        depth_image_mm = (depth_image_raw * self.depth_scale * 1000).astype(np.uint16)

        # Create a colorized depth map for visualization.
        # This preview is what gets saved to disk, not what is shown in the live preview.
        # We clip to 3 meters for a reasonable visualization range.
        depth_clipped = np.clip(depth_image_mm, 0, 3000)
        depth_display = cv2.convertScaleAbs(depth_clipped, alpha=255.0 / 3000.0)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        return color_image, depth_image_mm, depth_colormap

    def get_option_range(self, option, sensor_type='color'):
        """
        Gets the range of a sensor option.
        Args:
            option: The pyrealsense2.option to query.
            sensor_type (str): 'color' or 'depth'.
        Returns:
            rs.option_range object or None if not supported/streaming.
        """
        sensor = self.color_sensor if sensor_type == 'color' else self.depth_sensor
        if not sensor or not sensor.supports(option):
            return None
        return sensor.get_option_range(option)

    def get_option(self, option, sensor_type='color'):
        """
        Gets the current value of a sensor option.
        Args:
            option: The pyrealsense2.option to query.
            sensor_type (str): 'color' or 'depth'.
        Returns:
            The current value of the option, or None.
        """
        sensor = self.color_sensor if sensor_type == 'color' else self.depth_sensor
        if not sensor or not sensor.supports(option):
            return None
        return sensor.get_option(option)

    def set_option(self, option, value, sensor_type='color'):
        """
        Sets the value of a sensor option.
        Args:
            option: The pyrealsense2.option to set.
            value: The value to set.
            sensor_type (str): 'color' or 'depth'.
        """
        sensor = self.color_sensor if sensor_type == 'color' else self.depth_sensor
        if not sensor or not sensor.supports(option):
            print(f"Warning: Option {option} not supported for {sensor_type} sensor.")
            return
        try:
            sensor.set_option(option, value)
        except Exception as e:
            print(f"Error setting option {option} to {value} for {sensor_type} sensor: {e}")

    def get_intrinsics(self):
        """
        Returns the intrinsics of the color and depth streams.
        Returns:
            A dictionary containing intrinsics for 'color' and 'depth' streams,
            or None if the stream is not started.
        """
        if not self.profile:
            return None
        
        depth_stream = self.profile.get_stream(rs.stream.depth)
        color_stream = self.profile.get_stream(rs.stream.color)
        
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        return {
            'depth': {
                'width': depth_intrinsics.width,
                'height': depth_intrinsics.height,
                'fx': depth_intrinsics.fx,
                'fy': depth_intrinsics.fy,
                'ppx': depth_intrinsics.ppx,
                'ppy': depth_intrinsics.ppy,
                'model': str(depth_intrinsics.model),
                'coeffs': list(depth_intrinsics.coeffs),
            },
            'color': {
                'width': color_intrinsics.width,
                'height': color_intrinsics.height,
                'fx': color_intrinsics.fx,
                'fy': color_intrinsics.fy,
                'ppx': color_intrinsics.ppx,
                'ppy': color_intrinsics.ppy,
                'model': str(color_intrinsics.model),
                'coeffs': list(color_intrinsics.coeffs),
            }
        }
