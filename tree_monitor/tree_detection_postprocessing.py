"""
    TODO
"""
import os

import cv2

from tree_analyse import TreeAnalyse


class TreeDetection:
    def __init__(self, images_path, model_path):
        self.images_path = images_path
        self.tree_analyse = TreeAnalyse((1080, 1920), model_path)
        self.result_dir = os.path.join(self.images_path, "results")
        os.makedirs(self.result_dir)

    def process_data(self, im_name):
        im = cv2.imread(os.path.join(self.images_path, im_name))
        assert im.shape == (1080, 1920, 3)
        tree_data, debug_img = self.tree_analyse.process(im)
        self.draw(debug_img, tree_data, im_name)


    def draw(self, img, tree_data, im_name):
        if tree_data:
            for tree_bbox, canopy in tree_data:
                # print(canopy)
                x1, y1, x2, y2 = tree_bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.drawContours(img, canopy, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.result_dir, im_name), img)

    def run_detection(self):
        for im_name in sorted(os.listdir(self.images_path)):
            if im_name.lower().endswith((".jpg", ".jpeg")):
                self.process_data(im_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', help='path to image dir')
    parser.add_argument('--models', help='Path to models', default="tree_monitor/model/my_models/medium/")
    args = parser.parse_args()

    detect = TreeDetection(args.images, args.models)
    detect.run_detection()
