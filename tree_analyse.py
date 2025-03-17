"""
    Tree analise, detect and filter
"""

import cv2
import numpy as np

from model.detector import Detector


def draw_detections(img, detections, color):
    for (x1, y1, x2, y2), polygon, conf in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if polygon is not None:
            # cv2.polylines(image, [polygon], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.drawContours(img, [polygon], -1, color=(0, 0, 255), thickness=2)
        label = f"{conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def bbox_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 -y1)


class TreeAnalyse:
    def __init__(self, im_shape):
        # original image is rotated
        im_width, im_height = im_shape
        self.background = np.zeros((im_width, im_height), dtype=np.uint8)
        self.y1_min = im_width * 0.05
        self.y2_max = im_width * 0.95
        self.min_tree_spacing = 350
        self.tree_detector = Detector("model/my_models/run6_best_n.pt")
        self.canopy_detector = Detector("model/my_models/run4_best_seg_n.pt")
        # self.tree_detector = Detector("model/my_models/run4_best_m.pt")
        # self.canopy_detector = Detector("model/my_models/run3_best_seg_m.pt")


    def process(self, img):
        assert img.shape[:2] == self.background.shape
        tree_detections = self.tree_detector.detect(img)
        canopy_detections = self.canopy_detector.detect(img)

        trees = self.filter_and_assign_canopy(tree_detections, canopy_detections)

        return trees, draw_detections(draw_detections(img, canopy_detections, (0,0,255)),
                                     tree_detections, (255, 0,0))

    def filter_and_assign_canopy(self, tree_detections, canopy_detections):
        # filter tree bboxes
        bboxes = [bbox for bbox, __, __ in tree_detections]
        tree_bboxes = self.filter_tree_bboxes(bboxes)
        trees = []
        for t_bbox in tree_bboxes:
            tree = self.assign_canopy(t_bbox, canopy_detections)
            trees.append(tree)

        return trees

    def filter_tree_bboxes(self, tree_bboxes):
        tree_bboxes2 = []
        for (x1, y1, x2, y2) in tree_bboxes:
            if y1 >= self.y1_min and y2 < self.y2_max:
                tree_bboxes2.append([x1, y1, x2, y2])

        if len(tree_bboxes2) < 2:
            return tree_bboxes2

        filtered_trees = []
        tree_bboxes2 = sorted(tree_bboxes2, key= lambda x: x[1])
        while len(tree_bboxes2) > 1:
            x1_1, y1_1, x2_1, y2_1 = tree_bboxes2[0]
            for x1_2, y1_2, x2_2, y2_2 in tree_bboxes2[1:].copy():
                # test distance between centroids (only for y)
                if (y1_2 - y1_1 + y2_2 - y2_1)/2 > self.min_tree_spacing:
                    filtered_trees.append([x1_1, y1_1, x2_1, y2_1])
                    tree_bboxes2.pop(0)  # delete the first element, already added to result
                    if len(tree_bboxes2) == 1:  # only last tree in detection, add it as well
                        filtered_trees.append([x1_2, y1_2, x2_2, y2_2])
                    break  # let's continue with next tree

                # if the first tree is bigger, remove the second one.
                elif bbox_area(x1_1, y1_1, x2_1, y2_1) > bbox_area(x1_2, y1_2, x2_2, y2_2):
                    tree_bboxes2.pop(1)

                else:
                    tree_bboxes2.pop(0)
                    if len(tree_bboxes2) == 1:
                        # only the last tree remains and it is bigger than the current tree. Add it.
                        filtered_trees.append([x1_2, y1_2, x2_2, y2_2])
                    break  # continue with the next tree

            else:
                # finally add the tree and remove it from the list
                tree_bboxes2.pop(0)
                filtered_trees.append([x1_1, y1_1, x2_1, y2_1])

        return filtered_trees

    def assign_canopy(self, tree_bbox, canopy_detections):
        background = self.background.copy()
        mask = np.ones(background.shape, dtype=np.uint8)
        x1, y1, x2, y2 = tree_bbox
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        for __, polygon, __ in canopy_detections:
            cv2.drawContours(background, [polygon], -1, color=255, thickness=-1)
        background[mask==1] = 0
        contours, __ = cv2.findContours(background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return tree_bbox, contours
