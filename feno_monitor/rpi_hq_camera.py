"""
  Osgar driver for Raspberry Pi HQ Camera (Sony IMX477) using picamera2.
  Captures a JPEG image and publishes it on stream 'color'.

  Supports two modes, chosen automatically based on config['sleep']:
    - fast mode (short sleep, e.g. streaming at ~5 fps): camera stays on
      continuously between captures to avoid start-up latency.
    - slow mode (long sleep, e.g. every 2 hours): camera is stopped between
      captures to reduce power draw and sensor heating.
"""

from threading import Thread
import sys
import io

try:
    from picamera2 import Picamera2
except ImportError as e:
    print("Required module missing: %s" % e, file=sys.stderr)
    raise


class RPiHQCamera:
    def __init__(self, config, bus):
        self.input_thread = Thread(target=self.run_input, daemon=True)
        self.bus = bus
        bus.register('color')

        self.sleep = config['sleep']  # seconds
        self.stop_camera_threshold = config.get('stop_camera_threshold', 30)
        self.camera_warmup = config.get('camera_warmup', 5)  # seconds
        assert self.sleep >= self.camera_warmup
        self.keep_camera_on = self.sleep < self.stop_camera_threshold
        width = config.get('width', 1920)
        height = config.get('height', 1080)

        self.cam = Picamera2()
        self.cam.configure(
            self.cam.create_still_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
        )
        if self.keep_camera_on:
            self.cam.start()
        else:
            self.sleep -= self.camera_warmup

    def start(self):
        self.input_thread.start()

    def join(self, timeout=None):
        self.input_thread.join(timeout=timeout)

    def run_input(self):
        stream = io.BytesIO()
        try:
            while self.bus.is_alive():
                stream.seek(0)
                stream.truncate()
                try:
                    if not self.keep_camera_on:
                        self.cam.start()
                        self.bus.sleep(self.camera_warmup)
                    self.cam.capture_file(stream, format='jpeg')
                    if not self.keep_camera_on:
                        self.cam.stop()
                    self.bus.publish('color', stream.getvalue())
                except Exception as e:
                    print(f"Capture failed: {e}", file=sys.stderr)

                self.bus.sleep(self.sleep)
        finally:
            self.cam.stop()

    def request_stop(self):
        self.bus.shutdown()
