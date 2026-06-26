import unittest
from unittest import mock

import numpy as np

from src.core.detector.data_types import TextLine
from src.core.large_image_detection import LargeImageDetectorWrapper


def make_line(x1, y1, x2, y2, confidence=0.9):
    return TextLine(
        np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32),
        confidence=confidence,
    )


class LargeImageDetectionTests(unittest.TestCase):
    def _detect_from_transformed_lines(self, transformed_lines):
        fake_detector = mock.Mock()
        fake_detector.requires_merge = False
        fake_detector.detector_id = "default"
        fake_detector._detect_raw.side_effect = [
            ([make_line(0, 0, 20, 20)], None) for _ in transformed_lines
        ]

        wrapper = LargeImageDetectorWrapper(fake_detector)
        patches = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in transformed_lines]
        context = mock.Mock(is_rearranged=True)
        image = np.zeros((2600, 600, 3), dtype=np.uint8)

        with (
            mock.patch(
                "src.core.large_image_detection.slice_image_for_detection",
                return_value=(patches, context),
            ),
            mock.patch(
                "src.core.large_image_detection.transform_textlines_to_original",
                side_effect=transformed_lines,
            ),
        ):
            return wrapper._detect_with_slicing(
                image,
                im_w=600,
                im_h=2600,
                merge_lines=False,
                enable_aux_yolo_detection=False,
                edge_ratio_threshold=0.0,
                expand_ratio=0,
                expand_top=0,
                expand_bottom=0,
                expand_left=0,
                expand_right=0,
            )

    def test_deduplicates_same_textline_from_overlapping_slices(self):
        first = make_line(100, 1000, 220, 1030, confidence=0.6)
        second = make_line(102, 998, 222, 1028, confidence=0.9)

        result = self._detect_from_transformed_lines([[first], [second]])

        self.assertEqual(1, len(result.raw_lines))
        self.assertIs(result.raw_lines[0], second)

    def test_keeps_distinct_neighboring_textlines(self):
        first = make_line(100, 1000, 220, 1030, confidence=0.9)
        second = make_line(100, 1034, 220, 1064, confidence=0.9)

        result = self._detect_from_transformed_lines([[first], [second]])

        self.assertEqual(2, len(result.raw_lines))


if __name__ == "__main__":
    unittest.main()
