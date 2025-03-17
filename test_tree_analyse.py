import unittest

from tree_analyse import TreeAnalyse


class TestTreeAnalysis(unittest.TestCase):
    def test_filter_tree_bboxes(self):
        c = TreeAnalyse(im_width=1080)
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

if __name__ == '__main__':
    unittest.main()
