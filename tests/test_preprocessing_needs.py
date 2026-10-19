"""E4 C3: ``scripts/preprocessing_needs`` reads mapping.csv statistics and never writes data."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocessing_needs import collect, lesion_stats, main
from src.dataset import paths


def _mapping(pixels, size=128, boxes=None):
    """One row per mask; ``None`` pixels is a row without a mask (ISIC classification)."""
    rows = []
    for index, value in enumerate(pixels):
        side = boxes[index] if boxes else (0 if not value else 4)
        rows.append({"img_path": f"img{index}.png", "dim1": size, "dim2": size,
                     "tumor_pixels": np.nan if value is None else value,
                     "y_size": np.nan if value is None else side,
                     "x_size": np.nan if value is None else side + 1})
    return pd.DataFrame(rows)


class LesionStatsTests(unittest.TestCase):
    def test_counts_empty_masks_and_small_lesions_over_mask_rows_only(self):
        stats = lesion_stats(_mapping([0, 0, 20, 100, 400, None, None]), small_px=50)
        self.assertEqual(stats["rows"], 7)
        self.assertEqual(stats["masks"], 5)
        self.assertEqual(stats["empty_share"], 0.4)
        self.assertEqual(stats["positives"], 3)
        self.assertEqual(stats["area_median_px"], 100.0)
        self.assertEqual(stats["small"], 1)
        self.assertEqual(stats["small_share"], round(1 / 3, 3))
        self.assertEqual(stats["size"], "128")

    def test_share_of_frame_and_shorter_box_side(self):
        stats = lesion_stats(_mapping([256, 1024], size=64, boxes=[3, 7]))
        self.assertEqual(stats["area_median_pct"], round(np.median([256 / 4096, 1024 / 4096]) * 100, 2))
        self.assertEqual(stats["min_side_p5"], round(float(np.percentile([3, 7], 5)), 1))

    def test_a_mapping_without_positive_masks_has_no_lesion_numbers(self):
        stats = lesion_stats(_mapping([0, 0]))
        self.assertEqual(stats["positives"], 0)
        self.assertEqual(stats["empty_share"], 1.0)
        self.assertTrue(np.isnan(stats["area_median_px"]))


class CollectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        mapping = paths.mapping_file({"root": str(self.root), "dataset": "TCGA_LGG",
                                      "variant": "processed_128"})
        mapping.parent.mkdir(parents=True)
        _mapping([0, 30, 300]).to_csv(mapping, index=False)

    def _listing(self):
        return sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))

    def test_reads_present_variants_reports_missing_ones_and_writes_nothing(self):
        before = self._listing()
        table = collect(str(self.root), ["TCGA_LGG"], ["processed_128", "processed_256"])
        self.assertEqual(table["variant"].tolist(), ["processed_128", "processed_256"])
        self.assertEqual(table.iloc[0]["positives"], 2)
        self.assertEqual(table.iloc[1]["size"], "missing")
        self.assertEqual(self._listing(), before)

    def test_cli_prints_a_markdown_table_and_optional_csv(self):
        out_csv = self.root.parent / f"{self.root.name}_needs.csv"
        self.addCleanup(lambda: out_csv.unlink(missing_ok=True))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            status = main(["--root", str(self.root), "--datasets", "TCGA_LGG",
                           "--variants", "processed_128", "--csv", str(out_csv)])
        self.assertEqual(status, 0)
        self.assertIn("| TCGA_LGG | processed_128 | 128 | 3 | 3 |", out.getvalue())
        self.assertEqual(pd.read_csv(out_csv)["positives"].tolist(), [2])


if __name__ == "__main__":
    unittest.main()
