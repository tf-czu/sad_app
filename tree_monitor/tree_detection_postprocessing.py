"""
    TODO
"""
import cv2
import numpy as np

from tree_analyse import TreeAnalyse

class TreeDetection:
    def __init__(self, images, model_path):
        self.tree_analyse = TreeAnalyse((1080, 1920))

    def process_data(self, color_data):
        im = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint8), 1)
        assert im.shape == (1080, 1920, 3)
        tree_data, debug_img = self.tree_analyse.process(im)

    def run_detection(self):
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', help='path to image dir')
    parser.add_argument('--models', help='Path to models', default="TODO")
    args = parser.parse_args()

    detect = TreeDetection(args.images, args.models)
    detect.run_detection()