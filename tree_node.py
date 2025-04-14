"""
    TODO
"""
from http.cookiejar import debug
from os import write

import cv2
import numpy as np

from osgar.node import Node
from tree_analyse import TreeAnalyse


def list2xy(data):
    x = [coord[0] for coord in data]
    y = [coord[1] for coord in data]
    return x, y


class TreeNode(Node):

    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('tree')  # register a stream to be published
        self.verbose = False

        self.tree_analyse = TreeAnalyse((1080, 1920))
        self.debug_images = []
        self.debug_trees = []

    def on_pose3d(self, data):
        pass

    def on_color(self, data):
        im = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
        assert im.shape == (1080, 1920, 3)
        tree_data, debug_img = self.tree_analyse.process(im)
        # print(tree_data)
        if self.verbose:
            self.debug_images.append(debug_img)
            self.debug_trees.append(tree_data)

    def on_depth(self, data):
        pass


    def draw(self):
        # in verbose mode and with --draw parameter
        writer = None
        for img, tree_data in zip(self.debug_images, self.debug_trees):
            if tree_data:
                for tree_bbox, canopy in tree_data:
                    # print(canopy)
                    x1, y1, x2, y2 = tree_bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.drawContours(img, canopy, -1, (0, 255, 0), 2)
            if writer is None:
                height, width = img.shape[:2]
                writer = cv2.VideoWriter("tmp_video.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), 5, (width, height))
            writer.write(img)
        writer.release()
