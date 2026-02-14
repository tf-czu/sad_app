import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import json

from .realsense_device import RealsenseDevice, list_devices as list_realsense_devices
from .oak_device import OakDevice, list_devices as list_oak_devices
from . import capture_service

class App:
    """Main application GUI."""
    def __init__(self, master):
        self.master = master
        self.master.title("RGBD Snapshot App")

        self.device = None
        self.is_streaming = False
        self.preview_w = 960  # Target width, will be adjusted to fit screen
        self.preview_size_determined = False
        self.max_depth_m = tk.DoubleVar(value=3.0)
        self.camera_configs = []
        self.selected_camera_config = None

        # Camera control variables
        self.ae_var_rgb = tk.BooleanVar(value=True)
        self.exp_var_rgb = tk.DoubleVar()
        self.gain_var_rgb = tk.DoubleVar()

        self.ae_var_depth = tk.BooleanVar(value=True)
        self.exp_var_depth = tk.DoubleVar()
        self.gain_var_depth = tk.DoubleVar()

        # Create main frame
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.load_camera_configs()
        self.create_widgets(main_frame)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_camera_configs(self):
        """Loads camera configurations from cameras.json."""
        try:
            with open("cameras.json", "r") as f:
                self.camera_configs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showerror("Error", f"Could not load cameras.json: {e}")
            self.camera_configs = []

    def create_widgets(self, parent):
        """Create all the widgets for the application."""
        # --- Controls ---
        controls_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        controls_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        parent.columnconfigure(0, weight=1)

        ttk.Label(controls_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.camera_var = tk.StringVar()
        camera_names = [config['name'] for config in self.camera_configs]
        self.camera_selector = ttk.Combobox(
            controls_frame, textvariable=self.camera_var, values=camera_names, state="readonly", width=18
        )
        if camera_names:
            self.camera_selector.current(0)
        self.camera_selector.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        self.connect_button = ttk.Button(controls_frame, text="Connect", command=self.connect_device)
        self.connect_button.grid(row=0, column=2, padx=5, pady=5)

        self.disconnect_button = ttk.Button(controls_frame, text="Disconnect", command=self.disconnect_device, state=tk.DISABLED)
        self.disconnect_button.grid(row=0, column=3, padx=5, pady=5)

        controls_frame.columnconfigure(1, weight=1) # Allow slider to expand

        # Depth visualization controls
        ttk.Label(controls_frame, text="Max Depth (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.depth_limit_slider = ttk.Scale(
            controls_frame, orient=tk.HORIZONTAL, variable=self.max_depth_m,
            from_=0.5, to=10.0, command=self.on_depth_limit_change, state=tk.DISABLED
        )
        self.depth_limit_slider.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        self.depth_limit_label = ttk.Label(controls_frame, text="", width=6)
        self.depth_limit_label.grid(row=1, column=3, sticky=tk.W, padx=5)

        # --- Previews ---
        preview_frame = ttk.LabelFrame(parent, text="Live Preview", padding="10")
        preview_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        parent.rowconfigure(1, weight=1)

        # RGB Preview
        self.rgb_panel = ttk.Label(preview_frame, text="RGB Preview")
        self.rgb_panel.grid(row=0, column=0, padx=5, pady=5)

        # Depth Preview
        self.depth_panel = ttk.Label(preview_frame, text="Depth Preview")
        self.depth_panel.grid(row=0, column=1, padx=5, pady=5)
        preview_frame.columnconfigure(1, weight=1)

        # --- Camera Controls ---
        cam_controls_frame = ttk.LabelFrame(parent, text="Camera Controls", padding="10")
        cam_controls_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        cam_controls_frame.columnconfigure(0, weight=1)
        cam_controls_frame.columnconfigure(1, weight=1)

        # --- RGB Sensor Controls ---
        rgb_controls = ttk.LabelFrame(cam_controls_frame, text="RGB Sensor", padding="5")
        rgb_controls.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        rgb_controls.columnconfigure(1, weight=1)

        self.ae_check_rgb = ttk.Checkbutton(rgb_controls, text="Auto Exposure", variable=self.ae_var_rgb, command=lambda: self._on_toggle_ae('color'), state=tk.DISABLED)
        self.ae_check_rgb.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5)

        ttk.Label(rgb_controls, text="Exposure:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.exp_slider_rgb = ttk.Scale(rgb_controls, orient=tk.HORIZONTAL, variable=self.exp_var_rgb, command=lambda v: self._on_exposure_change(v, 'color'), state=tk.DISABLED)
        self.exp_slider_rgb.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.exp_label_rgb = ttk.Label(rgb_controls, text="", width=5)
        self.exp_label_rgb.grid(row=1, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(rgb_controls, text="Gain:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.gain_slider_rgb = ttk.Scale(rgb_controls, orient=tk.HORIZONTAL, variable=self.gain_var_rgb, command=lambda v: self._on_gain_change(v, 'color'), state=tk.DISABLED)
        self.gain_slider_rgb.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        self.gain_label_rgb = ttk.Label(rgb_controls, text="", width=5)
        self.gain_label_rgb.grid(row=2, column=2, sticky=tk.W, padx=5)

        # --- Depth Sensor Controls ---
        depth_controls = ttk.LabelFrame(cam_controls_frame, text="Depth Sensor", padding="5")
        depth_controls.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        depth_controls.columnconfigure(1, weight=1)

        self.ae_check_depth = ttk.Checkbutton(depth_controls, text="Auto Exposure", variable=self.ae_var_depth, command=lambda: self._on_toggle_ae('depth'), state=tk.DISABLED)
        self.ae_check_depth.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5)

        ttk.Label(depth_controls, text="Exposure:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.exp_slider_depth = ttk.Scale(depth_controls, orient=tk.HORIZONTAL, variable=self.exp_var_depth, command=lambda v: self._on_exposure_change(v, 'depth'), state=tk.DISABLED)
        self.exp_slider_depth.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.exp_label_depth = ttk.Label(depth_controls, text="", width=5)
        self.exp_label_depth.grid(row=1, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(depth_controls, text="Gain:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.gain_slider_depth = ttk.Scale(depth_controls, orient=tk.HORIZONTAL, variable=self.gain_var_depth, command=lambda v: self._on_gain_change(v, 'depth'), state=tk.DISABLED)
        self.gain_slider_depth.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        self.gain_label_depth = ttk.Label(depth_controls, text="", width=5)
        self.gain_label_depth.grid(row=2, column=2, sticky=tk.W, padx=5)

        # --- Capture ---
        capture_frame = ttk.LabelFrame(parent, text="Capture", padding="10")
        capture_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.snapshot_button = ttk.Button(capture_frame, text="Capture Snapshot", command=self.capture_snapshot, state=tk.DISABLED)
        self.snapshot_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        ttk.Label(capture_frame, text="N Frames:").grid(row=0, column=1, sticky=tk.W, padx=(20, 5), pady=2)
        self.n_frames_var = tk.StringVar(value="10")
        self.n_frames_entry = ttk.Entry(capture_frame, textvariable=self.n_frames_var, width=8)
        self.n_frames_entry.grid(row=0, column=2, sticky=tk.W, pady=2)

        ttk.Label(capture_frame, text="Interval (ms):").grid(row=1, column=1, sticky=tk.W, padx=(20, 5), pady=2)
        self.interval_var = tk.StringVar(value="500")
        self.interval_entry = ttk.Entry(capture_frame, textvariable=self.interval_var, width=8)
        self.interval_entry.grid(row=1, column=2, sticky=tk.W, pady=2)

        self.sequence_button = ttk.Button(capture_frame, text="Start Sequence", command=self.capture_sequence, state=tk.DISABLED)
        self.sequence_button.grid(row=0, column=3, rowspan=2, padx=5, pady=5, sticky=tk.E)

        # --- Log ---
        log_frame = ttk.LabelFrame(parent, text="Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=5, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        log_frame.columnconfigure(0, weight=1)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))

    def log(self, message):
        """Appends a message to the log text widget."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(message) # Also print to console

    def connect_device(self):
        """Connects to the RealSense device and starts streaming."""
        selected_name = self.camera_var.get()
        if not selected_name:
            messagebox.showerror("Error", "Please select a camera.")
            return

        self.selected_camera_config = next(
            (c for c in self.camera_configs if c['name'] == selected_name), None
        )
        if not self.selected_camera_config:
            messagebox.showerror("Error", "Selected camera configuration not found.")
            return

        serial = self.selected_camera_config['serial']
        self.log(f"Searching for device {serial} ({selected_name})...")
        self.status_var.set("Connecting...")
        self.master.update_idletasks()

        try:
            cam_type = self.selected_camera_config.get('type', 'realsense')
            
            if cam_type == 'realsense':
                available_devices = list_realsense_devices()
                if serial not in available_devices:
                    raise RuntimeError(f"Device {serial} not found. Available: {available_devices}")
                self.device = RealsenseDevice(serial_number=serial)
            elif cam_type == 'oak':
                # For OAK devices, we don't pre-check availability by connecting, as the
                # rapid connect/disconnect/connect sequence can cause "device busy" errors.
                # We will attempt to connect directly and let the device class handle errors.
                self.device = OakDevice(serial_number=serial)
            else:
                raise ValueError(f"Unknown camera type: {cam_type}")

            cam_config = self.selected_camera_config
            self.device.start(
                width=cam_config['width'],
                height=cam_config['height'],
                fps=cam_config['fps']
            )
            self.is_streaming = True

            self.setup_camera_controls()
            self.depth_limit_slider.config(state=tk.NORMAL)
            self.on_depth_limit_change(self.max_depth_m.get())

            self.snapshot_button.config(state=tk.NORMAL)
            self.sequence_button.config(state=tk.NORMAL)

            self.connect_button.config(state=tk.DISABLED)
            self.disconnect_button.config(state=tk.NORMAL)
            self.camera_selector.config(state=tk.DISABLED)

            self.log(f"Successfully connected to {serial}.")
            self.status_var.set("Streaming...")

            self.update_preview() # Start the preview loop
        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Connection Failed", str(e))
            self.status_var.set("Connection failed.")
            self.disconnect_device() # Safely clean up device and reset UI state

    def disconnect_device(self):
        """Stops the stream and disconnects from the device. Resets UI to disconnected state."""
        self.is_streaming = False  # Stop the preview loop first

        if self.device:
            self.device.stop()
            self.device = None
            self.log("Device disconnected.")

        self.preview_size_determined = False  # Reset for next connection

        self.disable_camera_controls()
        self.depth_limit_slider.config(state=tk.DISABLED)
        self.depth_limit_label.config(text="")
        self.snapshot_button.config(state=tk.DISABLED)
        self.sequence_button.config(state=tk.DISABLED)

        self.connect_button.config(state=tk.NORMAL)
        self.disconnect_button.config(state=tk.DISABLED)
        self.camera_selector.config(state="readonly")

        # Clear the preview panels
        self.rgb_panel.config(image='')
        self.depth_panel.config(image='')

        self.status_var.set("Ready.")

    def update_preview(self):
        """Fetches frames from the device and updates the GUI."""
        if not self.is_streaming:
            return

        try:
            color_img, depth_img, _ = self.device.get_frames()

            if color_img is not None and depth_img is not None:
                # Generate depth colormap with user-defined clipping
                max_depth_mm = self.max_depth_m.get() * 1000
                if max_depth_mm > 0:
                    # Clip values beyond max_depth to max_depth
                    depth_clipped = np.clip(depth_img, 0, max_depth_mm)
                    # Normalize and apply colormap
                    depth_display = cv2.convertScaleAbs(depth_clipped, alpha=255.0 / max_depth_mm)
                    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                else: # In case of zero max depth, show a black image
                    depth_colormap = np.zeros_like(color_img)

                # On first frame, determine the optimal preview size that fits the screen
                if not self.preview_size_determined:
                    screen_w = self.master.winfo_screenwidth()
                    screen_h = self.master.winfo_screenheight()

                    # Original frame dimensions and aspect ratio
                    h, w, _ = color_img.shape
                    aspect_ratio = h / w if w > 0 else 9/16

                    # Estimate window overhead (padding, controls, title bar)
                    width_overhead = 80  # Horizontal space for borders, padding
                    height_overhead = 350 # Vertical space for title bar, controls, status bar

                    # Calculate max preview width based on screen width (for two previews)
                    max_w_from_width = (screen_w - width_overhead) / 2
                    
                    # Calculate max preview width based on screen height
                    # preview_h = preview_w * aspect_ratio; preview_h < screen_h - overhead
                    max_w_from_height = (screen_h - height_overhead) / aspect_ratio if aspect_ratio > 0 else max_w_from_width

                    # Use the smaller of the two constraints, but not more than the target width
                    self.preview_w = int(min(self.preview_w, max_w_from_width, max_w_from_height))
                    self.preview_size_determined = True

                # Resize for display
                h, w, _ = color_img.shape
                preview_h = int(h * (self.preview_w / w)) if w > 0 else int(self.preview_w * 9/16)
                
                color_resized = cv2.resize(color_img, (self.preview_w, preview_h))
                depth_resized = cv2.resize(depth_colormap, (self.preview_w, preview_h))

                # Convert BGR (OpenCV) to RGB
                color_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)

                # Convert to PhotoImage
                self.rgb_photo = ImageTk.PhotoImage(image=Image.fromarray(color_rgb))
                self.depth_photo = ImageTk.PhotoImage(image=Image.fromarray(depth_resized))

                # Update panels
                self.rgb_panel.config(image=self.rgb_photo)
                self.depth_panel.config(image=self.depth_photo)
            
            # Schedule next update
            self.master.after(15, self.update_preview) # ~66 FPS refresh rate
        except Exception as e:
            self.log(f"Error during preview update: {e}")
            self.disconnect_device()

    def setup_camera_controls(self):
        """Queries device for control ranges and sets up the GUI elements."""
        self._setup_sensor_controls('color')
        self._setup_sensor_controls('depth')

    def _setup_sensor_controls(self, sensor_type):
        """Helper to set up controls for a specific sensor type."""
        if sensor_type == 'color':
            ae_check, ae_var = self.ae_check_rgb, self.ae_var_rgb
            exp_slider, exp_var, exp_label = self.exp_slider_rgb, self.exp_var_rgb, self.exp_label_rgb
            gain_slider, gain_var, gain_label = self.gain_slider_rgb, self.gain_var_rgb, self.gain_label_rgb
        else: # depth
            ae_check, ae_var = self.ae_check_depth, self.ae_var_depth
            exp_slider, exp_var, exp_label = self.exp_slider_depth, self.exp_var_depth, self.exp_label_depth
            gain_slider, gain_var, gain_label = self.gain_slider_depth, self.gain_var_depth, self.gain_label_depth

        # Auto Exposure
        ae_enabled = self.device.get_option(rs.option.enable_auto_exposure, sensor_type)
        if ae_enabled is not None:
            ae_var.set(bool(ae_enabled))
            ae_check.config(state=tk.NORMAL)
        else:
            ae_check.config(state=tk.DISABLED)

        # Exposure
        exp_range = self.device.get_option_range(rs.option.exposure, sensor_type)
        if exp_range:
            exp_slider.config(from_=exp_range.min, to=exp_range.max)
            current_exp = self.device.get_option(rs.option.exposure, sensor_type)
            if current_exp is not None:
                exp_var.set(current_exp)
                exp_label.config(text=f"{int(current_exp)}")
            else:
                # If current value can't be read, disable control
                exp_slider.config(state=tk.DISABLED)
                exp_label.config(text="N/A")
        else:
            exp_slider.config(state=tk.DISABLED)
            exp_label.config(text="N/A")

        # Gain
        gain_range = self.device.get_option_range(rs.option.gain, sensor_type)
        if gain_range:
            gain_slider.config(from_=gain_range.min, to=gain_range.max)
            current_gain = self.device.get_option(rs.option.gain, sensor_type)
            if current_gain is not None:
                gain_var.set(current_gain)
                gain_label.config(text=f"{int(current_gain)}")
            else:
                # If current value can't be read, disable control
                gain_slider.config(state=tk.DISABLED)
                gain_label.config(text="N/A")
        else:
            gain_slider.config(state=tk.DISABLED)
            gain_label.config(text="N/A")

        self._on_toggle_ae(sensor_type) # Set initial state of sliders based on AE

    def disable_camera_controls(self):
        """Disables all camera control widgets."""
        self.ae_check_rgb.config(state=tk.DISABLED)
        self.exp_slider_rgb.config(state=tk.DISABLED)
        self.gain_slider_rgb.config(state=tk.DISABLED)
        self.exp_label_rgb.config(text="")
        self.gain_label_rgb.config(text="")

        self.ae_check_depth.config(state=tk.DISABLED)
        self.exp_slider_depth.config(state=tk.DISABLED)
        self.gain_slider_depth.config(state=tk.DISABLED)
        self.exp_label_depth.config(text="")
        self.gain_label_depth.config(text="")
        
    def _on_toggle_ae(self, sensor_type):
        """Handles the Auto Exposure checkbox toggle for a given sensor."""
        if not self.device: return
        
        ae_var = self.ae_var_rgb if sensor_type == 'color' else self.ae_var_depth
        exp_slider = self.exp_slider_rgb if sensor_type == 'color' else self.exp_slider_depth
        gain_slider = self.gain_slider_rgb if sensor_type == 'color' else self.gain_slider_depth

        is_ae = ae_var.get()
        self.device.set_option(rs.option.enable_auto_exposure, 1.0 if is_ae else 0.0, sensor_type)
        self.log(f"{sensor_type.capitalize()} Auto Exposure {'Enabled' if is_ae else 'Disabled'}")
        
        # When AE is on, manual controls are disabled
        slider_state = tk.DISABLED if is_ae else tk.NORMAL
        
        # Only enable sliders if the option is supported (checked in setup)
        if self.device.get_option_range(rs.option.exposure, sensor_type):
            exp_slider.config(state=slider_state)
        if self.device.get_option_range(rs.option.gain, sensor_type):
            gain_slider.config(state=slider_state)

    def _on_exposure_change(self, value, sensor_type):
        """Handles changes from the exposure slider."""
        if not self.device: return
        
        label = self.exp_label_rgb if sensor_type == 'color' else self.exp_label_depth
        exp_val = int(float(value))
        label.config(text=f"{exp_val}")
        self.device.set_option(rs.option.exposure, float(exp_val), sensor_type)

    def _on_gain_change(self, value, sensor_type):
        """Handles changes from the gain slider."""
        if not self.device: return
        
        label = self.gain_label_rgb if sensor_type == 'color' else self.gain_label_depth
        gain_val = int(float(value))
        label.config(text=f"{gain_val}")
        self.device.set_option(rs.option.gain, float(gain_val), sensor_type)

    def on_depth_limit_change(self, value):
        """Handles changes from the depth limit slider."""
        max_depth = float(value)
        self.depth_limit_label.config(text=f"{max_depth:.1f}m")

    def capture_snapshot(self):
        """Captures a single snapshot in a background thread."""
        if not self.device:
            self.log("Error: Not connected to a device.")
            return

        # Disable buttons to prevent multiple captures
        self.snapshot_button.config(state=tk.DISABLED)
        self.sequence_button.config(state=tk.DISABLED)

        self.log("Starting single snapshot capture...")
        self.status_var.set("Capturing snapshot...")

        camera_name = self.selected_camera_config['name']
        thread = threading.Thread(target=self._capture_snapshot_worker, args=(camera_name,))
        thread.start()

    def _capture_snapshot_worker(self, camera_name):
        """Worker function for snapshot capture."""
        try:
            capture_path = capture_service.capture_one(self.device, camera_name)
            if capture_path:
                self.log(f"Snapshot saved to: {capture_path}")
            else:
                self.log("Snapshot capture failed.")
        except Exception as e:
            self.log(f"Error during snapshot capture: {e}")
        finally:
            # Re-enable buttons on the main thread if still streaming
            self.master.after(0, self.enable_capture_buttons)
            if self.is_streaming:
                self.status_var.set("Streaming...")

    def capture_sequence(self):
        """Captures a sequence of frames in a background thread."""
        if not self.device:
            self.log("Error: Not connected to a device.")
            return

        try:
            n_frames = int(self.n_frames_var.get())
            interval_ms = int(self.interval_var.get())
            if n_frames <= 0 or interval_ms <= 0:
                raise ValueError("Number of frames and interval must be positive.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numbers for frames and interval.\n{e}")
            return

        # Disable buttons
        self.snapshot_button.config(state=tk.DISABLED)
        self.sequence_button.config(state=tk.DISABLED)

        self.log(f"Starting sequence capture ({n_frames} frames, {interval_ms}ms interval)...")
        self.status_var.set("Capturing sequence...")

        camera_name = self.selected_camera_config['name']
        thread = threading.Thread(target=self._capture_sequence_worker, args=(camera_name, n_frames, interval_ms))
        thread.start()

    def _capture_sequence_worker(self, camera_name, n_frames, interval_ms):
        """Worker function for sequence capture."""
        try:
            def progress_update(current, total):
                # This is called from the worker thread, so we need to schedule
                # the GUI update on the main thread.
                self.master.after(0, lambda: self.status_var.set(f"Capturing sequence... {current}/{total}"))

            capture_path = capture_service.capture_sequence(
                self.device, camera_name, n_frames, interval_ms, progress_callback=progress_update
            )
            if capture_path:
                self.log(f"Sequence saved to: {capture_path}")
            else:
                self.log("Sequence capture failed.")
        except Exception as e:
            self.log(f"Error during sequence capture: {e}")
        finally:
            # Re-enable buttons on the main thread if still streaming
            self.master.after(0, self.enable_capture_buttons)
            if self.is_streaming:
                self.status_var.set("Streaming...")

    def enable_capture_buttons(self):
        """Enables capture buttons. To be called from main thread."""
        if self.is_streaming:
            self.snapshot_button.config(state=tk.NORMAL)
            self.sequence_button.config(state=tk.NORMAL)

    def on_closing(self):
        """Handles the window closing event."""
        self.disconnect_device()
        self.master.destroy()
