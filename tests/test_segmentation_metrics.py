import unittest

import numpy as np

from src.utils.metrics import (
    PREC,
    SENS,
    SPEC,
    calculate_metrics,
    precision,
    sentitivity,
    specificity,
)


class BinaryMetricEdgeCaseTests(unittest.TestCase):
    def test_undefined_rates_are_nan_only_when_their_denominator_is_empty(self):
        self.assertTrue(np.isnan(sentitivity(0.0, 0.0)))
        self.assertTrue(np.isnan(specificity(0.0, 0.0)))
        self.assertTrue(np.isnan(precision(0.0, 0.0)))

    def test_missed_or_spurious_positives_score_zero_instead_of_nan(self):
        self.assertEqual(sentitivity(0.0, 4.0), 0.0)
        self.assertEqual(precision(0.0, 4.0), 0.0)
        self.assertEqual(specificity(0.0, 4.0), 0.0)

    def test_all_foreground_mask_does_not_crash(self):
        ground_truth = np.ones((1, 1, 4, 4), dtype=np.uint8)
        segmentation = np.ones_like(ground_truth)

        metrics = calculate_metrics(ground_truth, segmentation, "all-foreground")

        self.assertEqual(metrics[SENS], 1.0)
        self.assertEqual(metrics[PREC], 1.0)
        self.assertTrue(np.isnan(metrics[SPEC]))


if __name__ == "__main__":
    unittest.main()
