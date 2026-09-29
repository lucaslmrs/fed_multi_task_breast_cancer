#!/usr/bin/env python
# coding: utf-8

"""
Script to preprocess the SIIM-ACR Pneumothorax chest radiographs into the repo's standard variant
layout:
- Ingest the stage-1 DICOMs (train + test) and the official stage-2 training annotations
- Decode the run-length-encoded masks, merging the instances of one image into a single mask
- Resize images (1-channel) and masks (binary) to `data.image_size`
- Generate a mapping CSV that also records the official stage-1 split

PROVENANCE
----------
SIIM-ACR Pneumothorax Segmentation, Kaggle competition (2019):
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation. The competition served the DICOMs
through the Google Cloud Healthcare API, which is no longer reachable, so the DICOMs come from a
community mirror of the stage-1 release (`siim-acr-pneumothorax-segmentation-data`), while the
annotations come from the OFFICIAL `stage_2_train.csv`. That file labels the stage-1 train AND
stage-1 test images (12,047); the mirror's own `train-rle.csv` only covers stage-1 train and is
byte-for-byte a subset of it. The stage-2 test images carry no public labels and are ignored.

WHY THIS IS SHAPED LIKE THE TCGA-LGG SCRIPT
-------------------------------------------
Every labelled image carries BOTH supervisions, so there is no `task` column, and -- as in TCGA-LGG
-- `class` IS DERIVED FROM THE MASK: `pneumothorax` iff the mask has a positive pixel. The release
has no independent diagnosis label, so the classification target is a deterministic function of
the segmentation target. Anything reported from this dataset must say so.

NO PATIENT GROUPING IS POSSIBLE
-------------------------------
The anonymisation gave every DICOM its own PatientID (12,089 ids for 12,089 files), so a patient
imaged more than once cannot be identified and splits are necessarily image-level. The mapping
therefore has no `lesion_id`. Exact pixel duplicates are the one leak that CAN be closed: they are
resolved here by keeping a single copy (see `resolve_pixel_duplicates`).

RLE FORMAT
----------
The competition's encoding is RELATIVE (each start is an offset from the end of the previous run)
and COLUMN-MAJOR (the official `mask2rle` iterates x in the outer loop), hence the transpose in
`decode_rle`. The orientation was checked on the whole release: 86% of the lesion centroids fall
in the upper half of the radiograph, as expected of apical pneumothorax.

Run with:  python -m src.dataset.SIIM_ACR_preprocessing
"""

import hashlib
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import pydicom
import yaml

from src.dataset import paths
from src.dataset.preprocessing_utils import (add_image_metadata, assert_target_dataset,
                                             assert_variant_resolution, resize_image, resize_mask)

# =======================
# User-Configurable Settings
# =======================
CONFIG_FILE = "./src/config.yaml"
DATASET = "SIIM_ACR"

# 1024 -> 128 is a downscale, where INTER_AREA is the correct filter. Masks ignore this and always
# use nearest-neighbour (see resize_mask).
INTERPOLATION = cv2.INTER_AREA

# Every radiograph is already square (1024x1024).
PRESERVE_ASPECT = False

PROGRESS_EVERY = 1000

# Raw layout, relative to data/SIIM_ACR/raw/. The mirror also nests an identical copy of both
# folders under `pneumothorax/`; only the top-level ones are read.
MIRROR_DIR = "siim-acr-pneumothorax-segmentation-data"
DICOM_SPLITS = {"train": "dicom-images-train", "test": "dicom-images-test"}
LABELS_FILE = "siim-acr-pneumothorax-segmentation/stage_2_train.csv"
NO_MASK = "-1"

# The release provides no diagnosis label; `class` is computed from the mask. See module docstring.
DERIVE_CLASS_FROM_MASK = True
CLASS_POSITIVE = "pneumothorax"
CLASS_NEGATIVE = "no_pneumothorax"

RAW_SIZE = 1024

MAPPING_COLUMNS = ["img_path", "mask_path", "class", "id", "official_split"]


# =======================
# Raw dataset readers
# =======================
def index_dicoms(raw: Path) -> Dict[str, dict]:
    """ImageId (the DICOM SOPInstanceUID, which is also the file stem) -> file and official split."""
    mirror = raw / MIRROR_DIR
    found: Dict[str, dict] = {}
    for split, folder in DICOM_SPLITS.items():
        root = mirror / folder
        if not root.is_dir():
            raise SystemExit(f"[ABORT] '{root}' does not exist. Extract the "
                             f"'{MIRROR_DIR}' archive under '{raw}' first.")
        for dcm in root.rglob("*.dcm"):
            if dcm.stem in found:
                raise SystemExit(f"[ABORT] ImageId '{dcm.stem}' appears in more than one DICOM file")
            found[dcm.stem] = {"src": dcm, "official_split": split}
    print(f"[INFO] Found {len(found)} DICOMs "
          + ", ".join(f"{s}: {sum(r['official_split'] == s for r in found.values())}"
                      for s in DICOM_SPLITS))
    return found


def load_annotations(raw: Path) -> Dict[str, List[str]]:
    """ImageId -> list of RLE strings (empty list for a negative image)."""
    labels = pd.read_csv(raw / LABELS_FILE, usecols=["ImageId", "EncodedPixels"], dtype=str)
    labels["EncodedPixels"] = labels["EncodedPixels"].str.strip()

    by_image = labels.groupby("ImageId")["EncodedPixels"].apply(list)
    mixed = [uid for uid, rles in by_image.items() if NO_MASK in rles and len(rles) > 1]
    if mixed:
        raise SystemExit(f"[ABORT] {len(mixed)} image(s) list '{NO_MASK}' next to a mask, e.g. "
                         f"{mixed[:3]}; the annotation file is not the one this script expects")
    return {uid: [r for r in rles if r != NO_MASK] for uid, rles in by_image.items()}


def decode_rle(rles: List[str], size: int = RAW_SIZE) -> np.ndarray:
    """Union of relative, column-major RLE instances -> {0, 255} mask in row-major orientation."""
    flat = np.zeros(size * size, np.uint8)
    for rle in rles:
        values = np.asarray(rle.split(), dtype=np.int64)
        if len(values) % 2:
            raise ValueError(f"Odd-length RLE: '{rle[:60]}...'")
        position = 0
        for start, length in zip(values[0::2], values[1::2]):
            position += start
            if position + length > flat.size:
                raise ValueError("RLE run exceeds the image bounds")
            flat[position:position + length] = 255
            position += length
    return flat.reshape(size, size).T.copy()


def read_dicom(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    if ds.get("PhotometricInterpretation") != "MONOCHROME2":
        raise SystemExit(f"[ABORT] '{path}' is {ds.get('PhotometricInterpretation')}; only "
                         f"MONOCHROME2 is handled (the whole release is)")
    img = ds.pixel_array
    if img.shape != (RAW_SIZE, RAW_SIZE) or img.dtype != np.uint8:
        raise SystemExit(f"[ABORT] '{path}' is {img.shape} {img.dtype}; the release is uniformly "
                         f"{RAW_SIZE}x{RAW_SIZE} uint8")
    return img


def collect_rows(raw: Path) -> List[dict]:
    """One row per labelled DICOM. Unlabelled DICOMs are reported, never silently dropped."""
    dicoms = index_dicoms(raw)
    annotations = load_annotations(raw)

    missing_image = sorted(set(annotations) - set(dicoms))
    if missing_image:
        raise SystemExit(f"[ABORT] {len(missing_image)} annotated image(s) have no DICOM, e.g. "
                         f"{missing_image[:3]}; the download is incomplete")

    unlabelled = sorted(set(dicoms) - set(annotations))
    by_split = pd.Series([dicoms[u]["official_split"] for u in unlabelled]).value_counts().to_dict()
    print(f"[INFO] {len(unlabelled)} DICOM(s) have no annotation and are excluded {by_split}")

    rows = [{"stem": uid, "img_src": dicoms[uid]["src"],
             "official_split": dicoms[uid]["official_split"], "rles": annotations[uid]}
            for uid in sorted(annotations)]
    print(f"[INFO] {len(rows)} labelled images, "
          f"{sum(bool(r['rles']) for r in rows)} with a pneumothorax mask")
    return rows


def resolve_pixel_duplicates(rows: List[dict]) -> List[dict]:
    """Keep one copy of every pixel-identical group.

    Splits are image-level, so two copies of one radiograph could land on opposite sides of the
    train/test split. The copy kept is the lexicographically smallest ImageId, which makes the
    choice reproducible. A group whose copies disagree on the mask has no trustworthy target and is
    dropped whole.
    """
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(row["pixel_md5"], []).append(row)

    kept, dropped = [], []
    for members in groups.values():
        members = sorted(members, key=lambda r: r["stem"])
        if len(members) == 1:
            kept.append(members[0])
            continue
        masks = {decode_rle(m["rles"]).tobytes() for m in members}
        if len(masks) == 1:
            kept.append(members[0])
            dropped.extend(members[1:])
            print(f"[INFO] Pixel duplicates {[m['stem'] for m in members]}: kept {members[0]['stem']}")
        else:
            dropped.extend(members)
            print(f"[WARN] Pixel duplicates with DIFFERENT masks {[m['stem'] for m in members]}: "
                  f"all dropped")
    print(f"[INFO] Pixel-duplicate resolution removed {len(dropped)} image(s)")
    return sorted(kept, key=lambda r: r["stem"])


# =======================
# Writing the variant
# =======================
def write_variant(rows: List[dict], output_path: Path, image_size: int) -> None:
    """Resize and write every image and its mask, recording the output paths and derived class."""
    img_dir, mask_dir = output_path / "images", output_path / "masks"
    total = len(rows)

    for n, row in enumerate(rows, start=1):
        img_out = img_dir / f"{row['stem']}.png"
        cv2.imwrite(str(img_out), resize_image(read_dicom(row["img_src"]), image_size,
                                               INTERPOLATION, PRESERVE_ASPECT))
        row["img_path"] = img_out.as_posix()

        mask = decode_rle(row["rles"])
        resized_mask = resize_mask(mask, image_size, PRESERVE_ASPECT)
        mask_out = mask_dir / f"{row['stem']}_mask.png"
        cv2.imwrite(str(mask_out), resized_mask)
        row["mask_path"] = mask_out.as_posix()

        # Derived from the RESIZED mask on purpose, as in TCGA-LGG: a lesion that vanished at the
        # target size would otherwise be labelled positive while carrying an empty target.
        row["class"] = CLASS_POSITIVE if np.any(resized_mask) else CLASS_NEGATIVE
        row["raw_positive"] = bool(np.any(mask))

        if n % PROGRESS_EVERY == 0 or n == total:
            print(f"[INFO]   {n}/{total} written")


def build_mapping(rows: List[dict]) -> pd.DataFrame:
    """`id` is assigned over the sorted ImageIds, so it is stable across reruns."""
    for new_id, row in enumerate(sorted(rows, key=lambda r: r["stem"])):
        row["id"] = new_id

    df = pd.DataFrame(rows)[MAPPING_COLUMNS]
    return add_image_metadata(df, sort_by=("id",))


def summarize(df: pd.DataFrame, rows: List[dict], config_data: dict) -> None:
    print(f"\n[INFO] Total rows: {len(df)}")

    print("\n[INFO] Class distribution (derived from the resized mask):")
    for name, count in df["class"].value_counts().items():
        print(f"[INFO]     {name:<16} {count:>6}  ({count / len(df):.1%})")

    print("\n[INFO] Official stage-1 split (metadata only; partitions do not use it):")
    for split, sub in df.groupby("official_split"):
        positive = (sub["class"] == CLASS_POSITIVE).mean()
        print(f"[INFO]     {split:<6} {len(sub):>6}  positive {positive:.1%}")

    vanished = sum(1 for r in rows if r["raw_positive"] and r["class"] == CLASS_NEGATIVE)
    if vanished:
        print(f"\n[WARN] {vanished} image(s) had a non-empty mask at {RAW_SIZE}px that became empty "
              f"at {config_data['image_size']}px; they are labelled '{CLASS_NEGATIVE}' to stay "
              f"consistent with the target actually supervised.")

    positive = df[df["class"] == CLASS_POSITIVE]
    if not positive.empty:
        print(f"\n[INFO] Lesion pixels on positive images: min {int(positive['tumor_pixels'].min())}, "
              f"median {int(positive['tumor_pixels'].median())}, "
              f"max {int(positive['tumor_pixels'].max())}")

    found = sorted(df["class"].unique())
    if list(config_data["classes"]) != found:
        print(f"\n[WARN] config.yaml has data.classes: {config_data['classes']}")
        print(f"[WARN] To train on this dataset set:  classes: {found}")
        print(f"[WARN]                                 seg_exclude_classes: []")


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
    print(f"[INFO] Class labels derived from the mask: {DERIVE_CLASS_FROM_MASK}\n")

    assert raw_path.exists(), f"Raw dataset folder '{raw_path}' does not exist"
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "masks").mkdir(parents=True, exist_ok=True)

    rows = collect_rows(raw_path)
    # Hashed in a first pass rather than kept: 12k full-resolution radiographs would need ~12 GB.
    print(f"\n[INFO] Hashing {len(rows)} DICOMs for pixel duplicates ...")
    for n, row in enumerate(rows, start=1):
        row["pixel_md5"] = hashlib.md5(read_dicom(row["img_src"]).tobytes()).hexdigest()
        if n % PROGRESS_EVERY == 0 or n == len(rows):
            print(f"[INFO]   {n}/{len(rows)} hashed")
    rows = resolve_pixel_duplicates(rows)

    print(f"\n[INFO] Writing {len(rows)} images to {output_path} ...")
    write_variant(rows, output_path, image_size)

    df_mapping = build_mapping(rows)
    if not df_mapping["id"].is_unique:
        raise SystemExit("[ABORT] The mapping 'id' column is not unique")
    df_mapping.to_csv(str(paths.mapping_file(config_data)), index=False)

    summarize(df_mapping, rows, config_data)
    print(f"\n[INFO] Preprocessing completed. Mapping saved at {paths.mapping_file(config_data)}")


if __name__ == "__main__":
    main()
