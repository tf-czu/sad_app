import unittest

from get_dice import match_by_y


class TestMatchByY(unittest.TestCase):

    def test_empty_detections(self):
        self.assertEqual(match_by_y([], [100, 200]), [])

    def test_empty_annotations(self):
        self.assertEqual(match_by_y([100, 200], []), [None, None])

    def test_single_pair(self):
        self.assertEqual(match_by_y([100], [100]), [0])

    def test_well_separated(self):
        self.assertEqual(match_by_y([100, 200], [100, 200]), [0, 1])

    def test_multiple_detections_same_object(self):
        # 3 detections of the same object (y~110-130), 2 annotations (y=100, y=500)
        # Only the closest detection should be matched to ann0; the rest stay unmatched.
        self.assertEqual(match_by_y([110, 120, 130], [100, 500]), [0, None, None])

    def test_equal_counts_clustered(self):
        # Same number of detections and annotations, but both detections belong to ann0.
        # The second detection must NOT be assigned to ann1.
        self.assertEqual(match_by_y([110, 120], [100, 500]), [0, None])

    def test_mixed_clusters(self):
        # det0, det1 -> ann0; det2 -> ann1
        self.assertEqual(match_by_y([110, 120, 190], [100, 200]), [0, None, 1])

    def test_more_annotations(self):
        self.assertEqual(match_by_y([110], [100, 200]), [0])

    def test_tie_breaking(self):
        # Equal distance to both annotations -> pick the smaller annotation index
        self.assertEqual(match_by_y([150], [100, 200]), [0])

    def test_reverse_order(self):
        # Detections in reverse order should still match correctly
        self.assertEqual(match_by_y([200, 100], [100, 200]), [1, 0])

    def test_annotation_at_most_one(self):
        # 4 detections of the same object, only 1 annotation
        self.assertEqual(match_by_y([110, 120, 130, 140], [100]), [0, None, None, None])

    def test_three_annotations_two_clusters(self):
        # det0, det1 -> ann0; det2 -> ann1; det3 -> ann2
        self.assertEqual(match_by_y([110, 120, 210, 310], [100, 200, 300]), [0, None, 1, 2])

    def test_detection_between_two_annotations(self):
        # det0 is closer to ann0, det1 is closer to ann1
        self.assertEqual(match_by_y([105, 195], [100, 200]), [0, 1])


if __name__ == "__main__":
    unittest.main()