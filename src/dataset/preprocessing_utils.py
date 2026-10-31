"""Helpers shared by the per-dataset preprocessing scripts.

Every dataset needs its own script (``Curated_BUSI_preprocessing``, ``ISIC_2018_preprocessing``)
because the raw layouts have nothing in common, but they all must produce the same contract:

    data/<dataset>/<variant>/{images/, masks/, mapping.csv}

What is genuinely common -- mask statistics, the mapping metadata columns, and the guards that stop
a script from writing into the wrong folder -- lives here. See ``src/dataset/paths.py`` for how the
paths themselves are resolved from config.

The scripts that accept a command line (``TCGA_LGG_preprocessing``, ``SIIM_ACR_preprocessing``) read
their settings through :func:`preprocessing_data_config`, which can write a NEW variant at another
resolution without editing ``src/config.yaml``:

    python -m src.dataset.TCGA_LGG_preprocessing --variant processed_256 --image-size 256
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from src.dataset import paths

# Columns of mapping.csv derived from a segmentation mask. They are NaN for rows that have no mask
# (e.g. ISIC's classification-only images) -- NaN meaning "not applicable", as opposed to 0, which
# legitimately means "empty mask" (BUSI's `normal` class).
MASK_COLUMNS = ["tumor_pixels", "y_max", "y_min", "x_max", "x_min", "y_size", "x_size"]


def count_pixels(segmentation: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(segmentation, return_counts=True)
    return dict(zip(unique, counts))


def size_tumor(seg: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    y_indexes, x_indexes = np.nonzero(seg != 0)
    if len(y_indexes) == 0 or len(x_indexes) == 0:
        return 0, 0, 0, 0, 0, 0
    ymin, xmin = [max(0, int(np.min(idx))) for idx in (y_indexes, x_indexes)]
    ymax, xmax = [int(np.max(arr) + 1) for arr in (y_indexes, x_indexes)]
    return ymax, ymin, xmax, xmin, ymax - ymin, xmax - xmin


def mask_metadata(mask_path) -> Optional[Dict[str, int]]:
    """Mask-derived columns for one row, or None when the row has no mask."""
    if mask_path is None or (isinstance(mask_path, float) and np.isnan(mask_path)) or mask_path == "":
        return None

    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        raise FileNotFoundError(f"Mask '{mask_path}' referenced by the mapping could not be read")

    ymax, ymin, xmax, xmin, y_size, x_size = size_tumor(mask)
    return {"tumor_pixels": count_pixels(mask).get(255, 0),
            "y_max": ymax, "y_min": ymin, "x_max": xmax, "x_min": xmin,
            "y_size": y_size, "x_size": x_size}


def add_image_metadata(df_mapping: pd.DataFrame, sort_by: Sequence[str] = ("class", "id")) -> pd.DataFrame:
    """Fill dim1/dim2 for every row and the MASK_COLUMNS for the rows that have a mask.

    Reads back what was actually written to disk, so the numbers describe the output rather than
    the intent.
    """
    dims1, dims2, mask_rows = [], [], []

    for img_path, mask_path in zip(df_mapping["img_path"], df_mapping["mask_path"]):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image '{img_path}' referenced by the mapping could not be read")
        dims1.append(img.shape[0])
        dims2.append(img.shape[1])
        mask_rows.append(mask_metadata(mask_path))

    df_mapping["dim1"] = dims1
    df_mapping["dim2"] = dims2
    for col in MASK_COLUMNS:
        df_mapping[col] = [row[col] if row is not None else np.nan for row in mask_rows]

    return df_mapping.sort_values(by=list(sort_by))


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Centre the image on a black square canvas, so a later resize preserves the aspect ratio."""
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=0)


def resize_image(img: np.ndarray, size: int, interpolation: int, preserve_aspect: bool = False) -> np.ndarray:
    """Resize to ``size`` x ``size``.

    ``interpolation`` is explicit rather than chosen here on purpose: INTER_AREA is the right
    choice when downscaling a photograph, but Curated BUSI was built with INTER_NEAREST and
    switching it would change the curated images, invalidating the frozen federated partition and
    every result produced from it.
    """
    if preserve_aspect:
        img = pad_to_square(img)
    return cv2.resize(img, (size, size), interpolation=interpolation)


def resize_mask(mask: np.ndarray, size: int, preserve_aspect: bool = False) -> np.ndarray:
    """Resize a binary mask: nearest-neighbour, then re-binarize to {0, 255}.

    The re-binarization is defensive -- nearest-neighbour should not introduce intermediate values,
    but any other interpolation would, and a mask that silently stops being binary is hard to spot
    downstream.
    """
    resized = resize_image(mask, size, cv2.INTER_NEAREST, preserve_aspect)
    return ((resized > 127) * 255).astype(np.uint8)


def assert_target_dataset(config_data: dict, expected: str) -> None:
    """Refuse to run when config.yaml points at a different dataset.

    Without this, running the ISIC script while `data.dataset: Curated_BUSI` would resolve the
    output to data/Curated_BUSI/<variant>/ and overwrite the curated BUSI images.
    """
    actual = config_data["dataset"]
    if actual != expected:
        raise SystemExit(
            f"[ABORT] This script preprocesses '{expected}', but config.yaml has "
            f"data.dataset: {actual}.\nSet `data.dataset: {expected}` in src/config.yaml "
            f"(and a matching `data.variant`/`data.classes`) before running it.")


def assert_variant_resolution(mapping_path: Path, image_size: int) -> None:
    """Refuse to mix resolutions inside one variant folder.

    A variant is defined by its resolution; writing 224px images into an existing processed_128/
    would leave a mapping.csv whose rows disagree about the image size.
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        return

    existing = pd.read_csv(mapping_path)
    if existing.empty or "dim1" not in existing.columns:
        return

    sizes = set(existing["dim1"].unique()) | set(existing["dim2"].unique())
    if sizes != {image_size}:
        raise SystemExit(
            f"[ABORT] '{mapping_path.parent}' already holds images of size {sorted(sizes)}, but "
            f"data.image_size is {image_size}.\nUse a different `data.variant` for a different "
            f"resolution, or delete that folder to rebuild it.")


def assert_new_variant(variant_dir: Path) -> None:
    """Refuse to write a variant requested on the command line into a folder that holds anything.

    A variant named on the command line is a new artefact: other studies may already read an
    existing one, and a partial earlier run would leave files the new mapping does not describe.
    An existing EMPTY folder is accepted; anything else must be removed by hand, never resumed.
    """
    variant_dir = Path(variant_dir)
    if not variant_dir.exists():
        return
    if not variant_dir.is_dir():
        raise SystemExit(f"[ABORT] Variant path '{variant_dir}' already exists and is not a folder.")
    has_mapping = (variant_dir / paths.MAPPING_FILENAME).exists()
    if has_mapping or any(variant_dir.iterdir()):
        detail = f"it has a {paths.MAPPING_FILENAME}" if has_mapping else "it is not empty"
        raise SystemExit(
            f"[ABORT] Variant folder '{variant_dir}' already exists ({detail}).\nA variant is "
            f"written once: choose another --variant, or delete that folder by hand to rebuild it "
            f"(a partial run is never resumed).")


_VARIANT_NAME = re.compile(r"^processed_(\d+)$")


def preprocessing_data_config(dataset: str, argv: Optional[Sequence[str]] = None,
                              config_file: str = "./src/config.yaml") -> dict:
    """The ``data`` settings a preprocessing script runs with.

    Without flags this is exactly the ``data`` block of ``config_file``, so the historical
    workflow (edit ``data.dataset``/``data.variant`` first) is unchanged and still guarded by
    :func:`assert_target_dataset`.

    ``--variant processed_<N> --image-size N`` (always together) selects ``dataset`` and a NEW
    variant instead: ``classes`` then come from the ``datasets.<dataset>`` registry entry, the
    variant name must spell the size, and :func:`assert_new_variant` refuses an existing folder.
    There is deliberately no ``--dataset`` flag: each script knows the one dataset it handles.
    """
    parser = argparse.ArgumentParser(
        description=f"Preprocess {dataset} into {paths.MAPPING_FILENAME} + images/ + masks/ of one "
                    f"variant. Without flags, src/config.yaml's data block is used as is.")
    parser.add_argument("--variant", help="new variant folder, processed_<N> (needs --image-size)")
    parser.add_argument("--image-size", type=int,
                        help="side N of the square output (needs --variant processed_<N>)")
    args = parser.parse_args(argv)

    with open(config_file, encoding="utf-8") as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
    config_data = dict(config["data"])
    if args.variant is None and args.image_size is None:
        return config_data

    if args.variant is None or args.image_size is None:
        parser.error("--variant and --image-size must be given together")
    if args.image_size < 1:
        parser.error(f"--image-size must be a positive integer, got {args.image_size}")
    match = _VARIANT_NAME.match(args.variant)
    if match is None or int(match.group(1)) != args.image_size:
        parser.error(f"--variant must be processed_{args.image_size} for --image-size "
                     f"{args.image_size}, got '{args.variant}'")

    entry = (config.get("datasets") or {}).get(dataset) or {}
    classes = entry.get("classes")
    if not classes:
        raise SystemExit(f"[ABORT] '{config_file}' has no datasets.{dataset}.classes; the command-line "
                         f"variant takes the class list from the datasets registry.")

    config_data.update(dataset=dataset, variant=args.variant, image_size=args.image_size,
                       classes=list(classes))
    assert_new_variant(paths.processed_dir(config_data))
    return config_data
