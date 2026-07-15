#!/usr/bin/env python
# coding: utf-8

"""
Script to preprocess the ISIC 2018 skin-lesion dataset into the repo's standard variant layout:
- Ingest BOTH challenge tasks: Task 1 (segmentation) and Task 3 / HAM10000 (classification)
- Resize images (RGB) and masks (binary) to `data.image_size`
- Generate a mapping CSV tagged with `task`, `official_split` and `lesion_id`

WHY THIS IS NOT SHAPED LIKE THE BUSI SCRIPT
-------------------------------------------
ISIC 2018's two tasks are DISJOINT image sets -- verified, not assumed (see
data/ISIC_2018/PAPER_NOTES.md):

    Task 1 (masks, no labels) : ISIC_0000000 .. ISIC_0016072   (3694 images)
    Task 3 (labels, no masks) : ISIC_0024306 .. ISIC_0034320   (11720 images)
    intersection: 0

So no image carries both supervisions, and the BUSI assumption that every mapping row has a mask
AND a class does not hold. Instead every row is tagged `task` = seg | cls, and `mask_path`/`class`
are left empty where the release provides nothing. Numeric ids stay globally unique because the two
ranges do not overlap.

Run with:  python -m src.dataset.ISIC_2018_preprocessing
"""

from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd
import yaml

from src.dataset import paths
from src.dataset.preprocessing_utils import (add_image_metadata, assert_target_dataset,
                                             assert_variant_resolution, resize_image, resize_mask)

# =======================
# User-Configurable Settings
# =======================
CONFIG_FILE = "./src/config.yaml"
DATASET = "ISIC_2018"

# Dermoscopic photos are downscaled hard (up to 3072x2304 -> 128), where INTER_AREA is the correct
# filter. Masks ignore this and always use nearest-neighbour (see resize_mask).
INTERPOLATION = cv2.INTER_AREA

# False squashes straight to a square, matching Curated BUSI. Task 3 is uniformly 600x450 so its
# distortion is at least constant, but Task 1 has 76+ distinct sizes so its distortion is not.
# Asymmetry is a diagnostic criterion for skin lesions ("A" of ABCD), so set this True to pad to a
# square first and keep the aspect ratio instead.
PRESERVE_ASPECT = False

PROGRESS_EVERY = 1000

# Raw layout. Each official archive extracts to <name>/<name>/, and every folder also carries
# ATTRIBUTION.txt / LICENSE.txt (plus Zone.Identifier residue on Windows), so we glob by explicit
# extension rather than listing the directory.
SEG_SOURCES = [
    ("train", "ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth"),
    ("val", "ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth"),
    ("test", "ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth"),
]
CLS_SOURCES = [
    ("train", "ISIC2018_Task3_Training_Input", "ISIC2018_Task3_Training_GroundTruth"),
    ("val", "ISIC2018_Task3_Validation_Input", "ISIC2018_Task3_Validation_GroundTruth"),
    ("test", "ISIC2018_Task3_Test_Input", "ISIC2018_Task3_Test_GroundTruth"),
]

# Maps image -> lesion_id. Multiple images of the SAME physical lesion must not be split across
# train/test. Published for Task 3 training only, so val/test rows get an empty lesion_id.
LESION_GROUPINGS = "ISIC2018_Task3_Training_LesionGroupings.csv"

RAW_MASK_SUFFIX = "_segmentation"
MAPPING_COLUMNS = ["img_path", "mask_path", "class", "id", "task", "official_split", "lesion_id"]


# =======================
# Raw dataset readers
# =======================
def isic_id(stem: str) -> int:
    """'ISIC_0024306' -> 24306. Unique across the dataset: the two tasks' ranges are disjoint."""
    return int(stem.split("_")[-1])


def find_dir(raw: Path, name: str) -> Path:
    """The official archives extract to <name>/<name>/; tolerate a flattened layout too."""
    outer = raw / name
    if not outer.is_dir():
        raise SystemExit(f"[ABORT] Expected raw folder '{outer}' does not exist. "
                         f"Extract the ISIC 2018 archives into {raw}/ first.")
    inner = outer / name
    return inner if inner.is_dir() else outer


def image_index(folder: Path) -> Dict[str, Path]:
    return {p.stem: p for p in sorted(folder.glob("*.jpg"))}


def mask_index(folder: Path) -> Dict[str, Path]:
    """Keyed by the image stem: 'ISIC_0000000_segmentation.png' -> 'ISIC_0000000'."""
    index = {}
    for p in sorted(folder.glob("*.png")):
        stem = p.stem[: -len(RAW_MASK_SUFFIX)] if p.stem.endswith(RAW_MASK_SUFFIX) else p.stem
        index[stem] = p
    return index


def label_table(folder: Path) -> pd.DataFrame:
    """Read a Task 3 ground-truth CSV (one-hot over 7 classes) into (image, class)."""
    csvs = sorted(folder.glob("*.csv"))
    if len(csvs) != 1:
        raise SystemExit(f"[ABORT] Expected exactly one ground-truth CSV in '{folder}', found {len(csvs)}")

    df = pd.read_csv(csvs[0])
    one_hot = df.drop(columns=["image"])
    if not (one_hot.sum(axis=1) == 1.0).all():
        raise SystemExit(f"[ABORT] '{csvs[0]}' has rows that are not one-hot; cannot derive a class")

    return pd.DataFrame({"image": df["image"], "class": one_hot.idxmax(axis=1)})


def collect_seg_rows(raw: Path) -> List[dict]:
    """Task 1: images paired with a segmentation mask, no diagnosis label."""
    rows = []
    for split, img_folder, mask_folder in SEG_SOURCES:
        images = image_index(find_dir(raw, img_folder))
        masks = mask_index(find_dir(raw, mask_folder))

        unpaired = set(images) ^ set(masks)
        if unpaired:
            raise SystemExit(f"[ABORT] {img_folder}: {len(unpaired)} image(s) without a matching "
                             f"mask (or vice versa), e.g. {sorted(unpaired)[:3]}")

        for stem, img_src in images.items():
            rows.append({"stem": stem, "img_src": img_src, "mask_src": masks[stem],
                         "class": None, "id": isic_id(stem), "task": "seg",
                         "official_split": split, "lesion_id": None})
        print(f"[INFO] seg/{split:<5} {len(images)} images + masks")
    return rows


def collect_cls_rows(raw: Path) -> List[dict]:
    """Task 3 / HAM10000: images with a diagnosis label, no mask."""
    groupings = {}
    lesion_file = raw / LESION_GROUPINGS
    if lesion_file.exists():
        g = pd.read_csv(lesion_file)
        groupings = dict(zip(g["image"], g["lesion_id"]))
    else:
        print(f"[WARN] {LESION_GROUPINGS} not found; lesion_id will be empty for every row")

    rows = []
    for split, img_folder, gt_folder in CLS_SOURCES:
        images = image_index(find_dir(raw, img_folder))
        labels = label_table(find_dir(raw, gt_folder))

        missing = set(labels["image"]) - set(images)
        if missing:
            raise SystemExit(f"[ABORT] {gt_folder}: {len(missing)} labelled image(s) absent from "
                             f"{img_folder}, e.g. {sorted(missing)[:3]}")

        grouped = 0
        for stem, cls in zip(labels["image"], labels["class"]):
            lesion_id = groupings.get(stem)
            grouped += lesion_id is not None
            rows.append({"stem": stem, "img_src": images[stem], "mask_src": None,
                         "class": cls, "id": isic_id(stem), "task": "cls",
                         "official_split": split, "lesion_id": lesion_id})
        print(f"[INFO] cls/{split:<5} {len(labels)} images + labels "
              f"({grouped} with a lesion_id)")
    return rows


# =======================
# Writing the variant
# =======================
def write_variant(rows: List[dict], output_path: Path, image_size: int) -> None:
    """Resize and write every image (and mask, where there is one), recording the output paths."""
    img_dir, mask_dir = output_path / "images", output_path / "masks"
    total = len(rows)

    for n, row in enumerate(rows, start=1):
        img = cv2.imread(str(row["img_src"]), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"[ABORT] Could not read image '{row['img_src']}'")
        img_out = img_dir / f"{row['stem']}.png"
        cv2.imwrite(str(img_out), resize_image(img, image_size, INTERPOLATION, PRESERVE_ASPECT))
        row["img_path"] = img_out.as_posix()

        if row["mask_src"] is None:
            row["mask_path"] = None
        else:
            mask = cv2.imread(str(row["mask_src"]), 0)
            if mask is None:
                raise SystemExit(f"[ABORT] Could not read mask '{row['mask_src']}'")
            mask_out = mask_dir / f"{row['stem']}_mask.png"
            cv2.imwrite(str(mask_out), resize_mask(mask, image_size, PRESERVE_ASPECT))
            row["mask_path"] = mask_out.as_posix()

        if n % PROGRESS_EVERY == 0 or n == total:
            print(f"[INFO]   {n}/{total} written")


def build_mapping(rows: List[dict]) -> pd.DataFrame:
    """Columns come from the source metadata, not from parsing filenames (the BUSI script parses
    filenames because its raw layout encodes the class in them; ISIC's does not)."""
    df = pd.DataFrame(rows)[MAPPING_COLUMNS]
    return add_image_metadata(df, sort_by=("task", "official_split", "id"))


def summarize(df: pd.DataFrame, config_data: dict) -> None:
    print(f"\n[INFO] Total rows: {len(df)}")
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task]
        by_split = sub["official_split"].value_counts().reindex(["train", "val", "test"]).dropna()
        print(f"[INFO]   task={task}: {len(sub)}  ({by_split.to_dict()})")

    cls_rows = df[df["task"] == "cls"]
    if not cls_rows.empty:
        found = sorted(cls_rows["class"].dropna().unique())
        print(f"\n[INFO] Classification classes found ({len(found)}): {found}")
        print("[INFO] Distribution over cls rows (train split):")
        for name, count in cls_rows[cls_rows["official_split"] == "train"]["class"].value_counts().items():
            print(f"[INFO]     {name:<7} {count:>6}")
        if list(config_data["classes"]) != found:
            print(f"\n[WARN] config.yaml has data.classes: {config_data['classes']}")
            print(f"[WARN] To train on this dataset set:  classes: {found}")
            print(f"[WARN]                                 seg_exclude_classes: []")

    no_lesion = int(df["lesion_id"].isna().sum())
    print(f"\n[INFO] Rows without a lesion_id: {no_lesion} "
          f"(all seg rows + Task 3 val/test -- the groupings file covers training only)")


# =======================
# Main
# =======================
def main():
    with open(CONFIG_FILE) as cf:
        config_data = yaml.load(cf, Loader=yaml.FullLoader)["data"]

    assert_target_dataset(config_data, DATASET)
    assert_variant_resolution(paths.mapping_file(config_data), config_data["image_size"])

    image_size = config_data["image_size"]
    raw_path = paths.raw_dir(config_data)
    output_path = paths.processed_dir(config_data)

    print(f"[INFO] Starting '{config_data['dataset']}' dataset preprocessing")
    print(f"[INFO] Reading raw images from: {raw_path}")
    print(f"[INFO] Writing variant '{config_data['variant']}' to: {output_path}")
    print(f"[INFO] Target resize dimensions: ({image_size}, {image_size})")
    print(f"[INFO] Preserve aspect ratio: {PRESERVE_ASPECT}\n")

    assert raw_path.exists(), f"Raw dataset folder '{raw_path}' does not exist"
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "masks").mkdir(parents=True, exist_ok=True)

    rows = collect_seg_rows(raw_path) + collect_cls_rows(raw_path)

    duplicated = pd.Series([r["id"] for r in rows]).duplicated()
    if duplicated.any():
        raise SystemExit(f"[ABORT] {int(duplicated.sum())} duplicate numeric id(s); the mapping "
                         f"'id' column would not be unique")

    print(f"\n[INFO] Writing {len(rows)} images to {output_path} ...")
    write_variant(rows, output_path, image_size)

    df_mapping = build_mapping(rows)
    df_mapping.to_csv(str(paths.mapping_file(config_data)), index=False)

    summarize(df_mapping, config_data)
    print(f"\n[INFO] Preprocessing completed. Mapping saved at {paths.mapping_file(config_data)}")


if __name__ == "__main__":
    main()
