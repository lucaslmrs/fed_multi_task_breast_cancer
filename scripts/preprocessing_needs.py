"""Measure what each dataset variant asks of the preprocessing: lesion sizes and empty masks.

Read-only. It reads only the ``mapping.csv`` of each requested variant, located through
``src/dataset/paths.py``, and never opens an image: the mapping already records, per row, the
mask statistics of the image actually written (``tumor_pixels`` and the bounding box). A variant
that does not exist is reported as missing, never built.

Per dataset and variant it reports, over the rows that carry a mask (ISIC classification rows
have none and are skipped):

* ``empty_share`` -- fraction of masks with no positive pixel (valid negative targets);
* the positive lesions' area in pixels (median, p5) and as a share of the frame (median);
* ``min_side_p5`` -- 5th percentile of the shorter bounding-box side, in pixels;
* ``small`` -- positive lesions below ``--small-px`` pixels, and their share.

Example (E4):
    python -m scripts.preprocessing_needs --variants processed_128 processed_256
    python -m scripts.preprocessing_needs --datasets TCGA_LGG SIIM_ACR --csv needs.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.dataset import paths

DEFAULT_DATASETS = ("Curated_BUSI", "ISIC_2018", "TCGA_LGG", "SIIM_ACR")
DEFAULT_VARIANTS = ("processed_128", "processed_256")
DEFAULT_SMALL_PX = 50
FIELDS = ["dataset", "variant", "size", "rows", "masks", "empty_share", "positives",
          "area_median_px", "area_p5_px", "area_median_pct", "min_side_p5", "small", "small_share"]


def lesion_stats(mapping: pd.DataFrame, small_px: int = DEFAULT_SMALL_PX) -> dict:
    """Mask statistics of one variant's mapping; rows without a mask (NaN pixels) are skipped."""
    pixels = pd.to_numeric(mapping["tumor_pixels"], errors="coerce")
    with_mask = mapping[pixels.notna()]
    pixels = pixels[pixels.notna()]
    positive = with_mask[pixels > 0]
    area = pixels[pixels > 0].to_numpy(dtype=float)
    frame = (pd.to_numeric(positive["dim1"]) * pd.to_numeric(positive["dim2"])).to_numpy(float)
    min_side = np.minimum(pd.to_numeric(positive["y_size"]).to_numpy(float),
                          pd.to_numeric(positive["x_size"]).to_numpy(float))
    sizes = sorted({int(value) for value in pd.concat([mapping["dim1"], mapping["dim2"]]).dropna()})

    def stat(values: np.ndarray, fn, digits: int):
        return round(float(fn(values)), digits) if len(values) else float("nan")

    small = int((area < small_px).sum())
    return {
        "size": "x".join(str(value) for value in sizes) if sizes else "",
        "rows": int(len(mapping)),
        "masks": int(len(with_mask)),
        "empty_share": round(float((pixels == 0).mean()), 3) if len(pixels) else float("nan"),
        "positives": int(len(area)),
        "area_median_px": stat(area, np.median, 1),
        "area_p5_px": stat(area, lambda v: np.percentile(v, 5), 1),
        "area_median_pct": stat(area / frame * 100 if len(area) else area, np.median, 2),
        "min_side_p5": stat(min_side, lambda v: np.percentile(v, 5), 1),
        "small": small,
        "small_share": round(small / len(area), 3) if len(area) else float("nan"),
    }


def collect(root: str, datasets: Iterable[str], variants: Iterable[str],
            small_px: int = DEFAULT_SMALL_PX) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for variant in variants:
            mapping_path = paths.mapping_file({"root": root, "dataset": dataset, "variant": variant})
            if not mapping_path.exists():
                rows.append({"dataset": dataset, "variant": variant, "size": "missing"})
                continue
            stats = lesion_stats(pd.read_csv(mapping_path), small_px)
            rows.append({"dataset": dataset, "variant": variant, **stats})
    # object dtype keeps counts as integers next to the rows of a missing variant.
    return pd.DataFrame(rows, columns=FIELDS, dtype=object)


def to_markdown(table: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, float) and np.isnan(value):
            return ""
        return str(value)

    lines = ["| " + " | ".join(table.columns) + " |",
             "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |"
              for row in table.itertuples(index=False)]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data", help="data.root (default: data)")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--small-px", type=int, default=DEFAULT_SMALL_PX,
                        help=f"area below which a lesion counts as small (default {DEFAULT_SMALL_PX})")
    parser.add_argument("--csv", help="also write the table to this CSV")
    args = parser.parse_args(argv)
    if args.small_px < 1:
        parser.error("--small-px must be positive")

    table = collect(args.root, args.datasets, args.variants, args.small_px)
    print(to_markdown(table))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwritten {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
