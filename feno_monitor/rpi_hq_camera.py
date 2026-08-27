"""
  Osgar driver for Raspberry Pi HQ Camera (Sony IMX477) using picamera2.
  Captures a JPEG image and publishes it on stream 'color'.
  The capture frequency is very low, so sleeping is defined in hours.
"""

from threading import Thread
import sys

try:
    import cv2
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
        width = config.get('width', 1920)
        height = config.get('height', 1080)

        self.cam = Picamera2()
        self.cam.configure(
            self.cam.create_still_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
        )
        self.cam.start()

    def start(self):
        self.input_thread.start()

    def join(self, timeout=None):
        self.input_thread.join(timeout=timeout)

    def run_input(self):
        try:
            while self.bus.is_alive():
                image = self.cam.capture_array()
                retval, data = cv2.imencode('*.jpg', image)
                if retval and len(data) > 0:
                    self.bus.publish('color', data.tobytes())
                self.bus.sleep(self.sleep)
        finally:
            self.cam.stop()

    def request_stop(self):
        self.bus.shutdown()
