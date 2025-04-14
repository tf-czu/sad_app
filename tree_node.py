"""
    TODO
"""
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

        self.last_color_seq = None
        self.last_color = None

        self.is_synced_depth_color = None

        self.tree_analyse = TreeAnalyse((1080, 1920))
        self.debug_images = []
        self.debug_trees = []

    def on_color_seq(self, data):
        self.last_color_seq = data

    def on_depth_seq(self, last_depth_seq):
        # Data are received in order color_seq, color, depth_seq, depth
        # Check if frames id are the same
        if last_depth_seq[0] == self.last_color_seq[0]:
            self.is_synced_depth_color = True
            # compare internal camera time in us
            assert abs(last_depth_seq[1] - self.last_color_seq[1]) < 35_000, (self.last_depth_seq, self.last_color_seq)
        else:
            self.is_synced_depth_color = False

    def on_pose3d(self, data):
        pass

    def on_color(self, data):
        self.last_color = data

    def on_depth(self, depth_data):
        # The spring data do not contain seq streams
       if self.is_synced_depth_color is not False:
           self.process_data(self.last_color, depth_data)

    def process_data(self, color_data, depth_data):
        im = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint8), 1)
        assert im.shape == (1080, 1920, 3)
        tree_data, debug_img = self.tree_analyse.process(im)
        self.publish("tree", tree_data)
        if self.verbose:
            self.debug_images.append(debug_img)
            self.debug_trees.append(tree_data)


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
