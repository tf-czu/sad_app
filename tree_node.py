"""
    TODO
"""
import cv2
import numpy as np

from osgar.node import Node
from model.detector import Detector


def list2xy(data):
    x = [coord[0] for coord in data]
    y = [coord[1] for coord in data]
    return x, y

def draw_detections(img, detections):
    for (x1, y1, x2, y2), polygon, conf in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if polygon is not None:
            # cv2.polylines(image, [polygon], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.drawContours(img, [polygon], -1, color=(0, 0, 255), thickness=-1)
        label = f"{conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img


class TreeNode(Node):

    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('tree')  # register a stream to be published
        self.verbose = False

        self.tree_detector = Detector("model/my_models/run6_best_n.pt")
        self.id = 0
        self.debug_detections = []
        self.debug_images = []

    def on_pose3d(self, data):
        pass

    def on_color(self, data):
        im = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
        detections = self.tree_detector.detect(im)
        if self.verbose:
            self.debug_images.append(im)
            self.debug_detections.append(detections)

    def on_depth(self, data):
        pass


    def draw(self):
        # in verbose mode and with --draw parameter: draw a plot
        # import matplotlib.pyplot as plt
        # x, y = list2xy(self.debug_gps_orig)
        # plt.plot(x, y, "k+-", label="gps orig")

        # plt.legend()
        # plt.axis('equal')
        # plt.show()

        with open("tmp_video.mjpeg", "wb") as f:
            for im, detections in zip(self.debug_images, self.debug_detections):
                detect_im = draw_detections(im, detections)
                _, jpeg_data = cv2.imencode('.jpg', detect_im)
                f.write(jpeg_data.tobytes())



