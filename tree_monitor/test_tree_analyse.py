import unittest

import numpy as np
import cv2

from tree_analyse import TreeAnalyse, is_one_tree_only

def show_im(im):
    cv2.namedWindow("im", cv2.WINDOW_NORMAL)
    cv2.imshow("im", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class TestTreeAnalysis(unittest.TestCase):
    def test_filter_tree_bboxes(self):
        c = TreeAnalyse(im_shape=(1080, 1920))
        tree_bboxes = [(100, 0, 1000, 200), (40, 180, 1000, 300), (100, 800, 1200, 1079)]
        expected_res = [[40, 180, 1000, 300]]
        filtered = c.filter_tree_bboxes(tree_bboxes)
        print(filtered)

        tree_bboxes2 = [(100, 100, 1000, 200), (700, 520, 900, 550), (100, 540, 1000, 650), (100, 640, 1000, 800)]
        expected_res2 = [[100, 100, 1000, 200], [100, 640, 1000, 800]]
        filtered2 = c.filter_tree_bboxes(tree_bboxes2)
        print(filtered2)

        tree_bboxes3 = [(100,110,1000,210), (500,100,1000,220), (500,500,1000,600), (100,520,1000,680)]
        expected_res3 = [[100,110,1000,210], [100,520,1000,680]]
        filtered3 = c.filter_tree_bboxes(tree_bboxes3)
        print(filtered3)

        tree_bboxes4 = [(300,100,800,200), (200,180,900,300), (100,280,1000,400)]
        expected_res4 = [[100,280,1000,400]]
        filtered4 = c.filter_tree_bboxes(tree_bboxes4)
        print(filtered4)

        self.assertEqual(filtered, expected_res)
        self.assertEqual(filtered2, expected_res2)
        self.assertEqual(filtered3, expected_res3)
        self.assertEqual(filtered4, expected_res4)

    def test_assign_canopy(self):
        tree_bbox = (400, 100, 1000, 500)
        canopy = [[None,
                  ( np.array([ [[450, 100]], [[460, 1000]], [[470, 1000]], [[460, 100]] ], dtype=np.int32) ),
                  None]]
        c = TreeAnalyse(im_shape=(1080, 1920))
        tree = c.assign_canopy(tree_bbox, canopy)
        print(tree)
        bbox, contours, debug_ratio, background = tree  # TODO remove debug things
        self.assertEqual(tree_bbox, bbox)
        print(np.max(background))
        # show_im(background)

    def test_is_one_tree_only(self):
        canopy_im = np.zeros((10, 16), dtype=np.uint8)
        cv2.rectangle(canopy_im, (2, 4), (10, 6), 255, -1)
        show_im(canopy_im)
        tree_bbox = (2, 1, 10, 8)
        self.assertTrue(is_one_tree_only(tree_bbox, canopy_im))

        canopy_im = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.rectangle(canopy_im, (100, 450), (1000, 470), 255, -1)
        show_im(canopy_im)
        tree_bbox = (100, 400, 1000, 500)
        self.assertTrue(is_one_tree_only(tree_bbox, canopy_im))

        canopy_im = np.zeros((10, 16), dtype=np.uint8)
        cv2.rectangle(canopy_im, (2, 3), (10, 4), 255, -1)
        cv2.rectangle(canopy_im, (2, 6), (10, 8), 255, -1)
        show_im(canopy_im)
        tree_bbox = (2, 1, 10, 8)
        self.assertFalse(is_one_tree_only(tree_bbox, canopy_im))

if __name__ == '__main__':
    unittest.main()
