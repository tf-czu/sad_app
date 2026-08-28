"""
  Osgar node that receives JPEG images from osgar (one by one) and streams
  them over HTTP as MJPEG (multipart/x-mixed-replace) so they can be viewed
  in VLC:  vlc http://<host>:<port>/stream
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from osgar.node import Node


class JpegStreamer(Node):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 8080)
        self.stream_interval = config.get('stream_interval', 1.0)

        self._lock = threading.Lock()
        self._latest_jpeg = None

        self._httpd = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self._http_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _make_handler(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/stream':
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        with streamer._lock:
                            jpeg = streamer._latest_jpeg
                        if jpeg is not None:
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                            self.wfile.write(jpeg)
                            self.wfile.write(b'\r\n')
                            self.wfile.flush()
                        streamer.sleep(streamer.stream_interval)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                pass  # suppress default logging

        return Handler

    def on_color(self, data):
        with self._lock:
            self._latest_jpeg = data

    def start(self):
        self._http_thread.start()
        super().start()

    def request_stop(self):
        self._httpd.shutdown()
        super().request_stop()


# vim: expandtab sw=4 ts=4