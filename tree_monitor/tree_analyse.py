"""
    Tree analise, detect and filter
"""
import os.path

import cv2
import numpy as np

from tree_monitor.model.detector import Detector


def draw_detections(img, detections, color):
    for (x1, y1, x2, y2), __, conf in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # label = f"{conf:.2f}"
        # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def draw_polygon_detection(img, detections, color):
    for (x1, y1, x2, y2), polygon, conf in detections:
        if polygon is not None:
            contours = unite_contours(img, [polygon])
            cv2.drawContours(img, contours, -1, color=color, thickness=2)
        # label = f"{conf:.2f}"
        # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def unite_contours(img, contours):
    background = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.drawContours(background, contours, -1, color=255, thickness=-1)
    k = np.ones((3, 3), np.uint8)  # 3x3 kernel
    background = cv2.morphologyEx(background, cv2.MORPH_OPEN, k)
    contours, __ = cv2.findContours(background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def is_one_tree_only(bbox, canopy_bin_im):
    x1, y1, x2, y2 = bbox
    sub_im = canopy_bin_im[y1:y2, x1:x2]
    sub_im[sub_im != 0] = 1
    # For each line calculate centroid (medium order)  # yt = sum(y * px_value) / sum(px_values)
    indices = np.arange(sub_im.shape[0]).reshape((sub_im.shape[0], 1))  # to number lines and set correct mat shape
    weighted = indices * sub_im  # Order of non zero elements > y
    sums = weighted.sum(axis=0)  # sum(y * px_value)
    counts = sub_im.sum(axis=0)  # sum(px_values) in columns
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore divide by zero warning
        centroids = np.divide(sums, counts)
    centroids[np.isnan(centroids)] = 0
    centroids = np.round(centroids).astype(np.int32)
    # get number of centroid which are in a non-zero area.
    num_nz = np.sum(sub_im[centroids, np.arange(sub_im.shape[1])])
    if num_nz/sub_im.shape[1] > 0.8:
        # print(num_nz/sub_im.shape[1])
        return True
    else:
        # print(num_nz / sub_im.shape[1])
        return False


def bbox_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 -y1)


class TreeAnalyse:
    def __init__(self, im_shape, models_path):
        # original image is rotated
        im_width, im_height = im_shape
        self.background = np.zeros((im_width, im_height), dtype=np.uint8)
        self.y1_min = im_width * 0.01
        self.y2_max = im_width * 0.99
        self.min_tree_spacing = 300
        self.tree_detector = Detector(os.path.join(models_path, "best.pt"))
        self.canopy_detector = Detector(os.path.join(models_path, "best_seg.pt"))


    def process(self, img):
        assert img.shape[:2] == self.background.shape
        debug_img = img.copy()
        tree_detections = self.tree_detector.detect(img)
        debug_img = draw_detections(debug_img, tree_detections, (255, 255, 0))
        canopy_detections = self.canopy_detector.detect(img)
        # debug_img = draw_polygon_detection(debug_img, canopy_detections, (0, 255, 0))

        trees = self.filter_and_assign_canopy(tree_detections, canopy_detections)
        if trees:
            for tree_bbox, canopy in trees:
                x1, y1, x2, y2 = tree_bbox
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.drawContours(debug_img, canopy, -1, (0, 0, 255), 2)

        return trees, debug_img


    def filter_and_assign_canopy(self, tree_detections, canopy_detections):
        # returns list of tree bboxes and assigned canopy [[tree_bbox, canopy], ... ]
        # filter tree bboxes
        bboxes = [bbox for bbox, __, __ in tree_detections]
        trees = []
        for t_bbox in bboxes:
            tree = self.assign_canopy(t_bbox, canopy_detections)
            if tree is not None:
                trees.append(tree)

        filtered_trees = self.filter_tree_bboxes(trees)

        return filtered_trees

    def filter_tree_bboxes(self, trees):
        trees2 = []
        for (x1, y1, x2, y2), canopy in trees:
            if y1 >= self.y1_min and y2 < self.y2_max:
                trees2.append([(x1, y1, x2, y2), canopy])

        if len(trees2) <= 1:
            return trees2

        filtered_trees = []
        trees2 = sorted(trees2, key= lambda x: x[0][1])
        while len(trees2) > 1:
            (x1_1, y1_1, x2_1, y2_1), canopy_1 = trees2[0]
            for (x1_2, y1_2, x2_2, y2_2), canopy_2 in trees2[1:].copy():
                # test distance between centroids (only for y)
                if (y1_2 - y1_1 + y2_2 - y2_1)/2 > self.min_tree_spacing:
                    filtered_trees.append([(x1_1, y1_1, x2_1, y2_1), canopy_1])
                    trees2.pop(0)  # delete the first element, already added to result
                    if len(trees2) == 1:  # only last tree in detection, add it as well
                        filtered_trees.append([(x1_2, y1_2, x2_2, y2_2), canopy_2])
                    break  # let's continue with next tree

                # if the first tree is bigger, remove the second one.
                elif bbox_area(x1_1, y1_1, x2_1, y2_1) > bbox_area(x1_2, y1_2, x2_2, y2_2):
                    trees2.pop(1)

                else:
                    trees2.pop(0)
                    if len(trees2) == 1:
                        # only the last tree remains and it is bigger than the current tree. Add it.
                        filtered_trees.append([(x1_2, y1_2, x2_2, y2_2), canopy_2])
                    break  # continue with the next tree

            else:
                # finally add the tree and remove it from the list
                trees2.pop(0)
                filtered_trees.append([(x1_1, y1_1, x2_1, y2_1), canopy_1])

        return filtered_trees

    def assign_canopy(self, tree_bbox, canopy_detections):
        background = self.background.copy()
        mask = np.ones(background.shape, dtype=np.uint8)
        x1, y1, x2, y2 = tree_bbox
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        for __, polygon, __ in canopy_detections:
            cv2.drawContours(background, [polygon], -1, color=255, thickness=-1)
        background[mask==1] = 0

        k = np.ones((3, 3), np.uint8)  # 3x3 kernel
        background = cv2.morphologyEx(background, cv2.MORPH_OPEN, k)

        contours, __ = cv2.findContours(background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            contours_area = sum([cv2.contourArea(cnt) for cnt in contours])
            if contours_area/((x2 - x1)*(y2-y1)) > 0.1:  # Require a minimum content of canopy in the tree.
                # debug_ratio = contours_area/((x2 - x1)*(y2-y1))
                return tree_bbox, contours
