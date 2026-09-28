#!/usr/bin/env python
# coding: utf-8

"""
Script to preprocess the TCGA-LGG brain-MRI dataset into the repo's standard variant layout:
- Ingest the Buda et al. (2019) release: one folder per patient, one TIFF pair per axial slice
- Resize images (3-channel) and masks (binary) to `data.image_size`
- Generate a mapping CSV whose `lesion_id` carries the patient id

PROVENANCE
----------
Images come from the TCGA-LGG collection on The Cancer Imaging Archive; the tumour masks were
annotated and published by Buda, M.; Saha, A.; Mazurowski, M.A. (2019), "Association of genomic
subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning
algorithm", Computers in Biology and Medicine 109:218-225. Kaggle only mirrors that release.

WHY THIS IS SHAPED LIKE THE BUSI SCRIPT, NOT THE ISIC ONE
---------------------------------------------------------
Every slice carries BOTH supervisions, so there is no `task` column: each mapping row has a mask
and a class, which is what `fold_strategy: stratified` and `client_topology: multi_task` require.

TWO PROPERTIES OF THIS DATASET THAT THE MAPPING HAS TO CARRY EXPLICITLY
-----------------------------------------------------------------------
1. `class` IS DERIVED, NOT ANNOTATED. The release ships no per-slice diagnosis label; the two
   classes are computed here as `tumor` iff the slice's mask has any positive pixel. So the
   classification target is a deterministic function of the segmentation target, and the two
   tasks are far less independent than in BUSI or ISIC. Anything reported from this dataset must
   say so. `DERIVE_CLASS_FROM_MASK` documents the choice; there is no alternative source.

2. ONE PATIENT CONTRIBUTES ~36 SLICES. 110 patients produce ~3,929 slices, and neighbouring
   slices of one patient are near-duplicates. The effective sample size is the patient count, not
   the row count, so `lesion_id` holds the patient id and splits MUST group on it. (The
   `stratified` partition path currently splits at image level -- see the note in PAPER_NOTES.md.)
   Consequently `no_tumor` means "this slice does not intersect the tumour", NOT "this patient has
   no tumour": every patient in the collection has a glioma.

CHANNELS
--------
Each TIFF holds three co-registered MRI sequences (pre-contrast, FLAIR, post-contrast), not
colour. They are kept as three channels -- collapsing to grayscale would discard two sequences.
OpenCV reads the file's channel order reversed and `imwrite` reverses it again, so the round-trip
preserves the original per-channel semantics.

Run with:  python -m src.dataset.TCGA_LGG_preprocessing
"""

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import yaml

from src.dataset import paths
from src.dataset.preprocessing_utils import (add_image_metadata, assert_target_dataset,
                                             assert_variant_resolution, resize_image, resize_mask)

# =======================
# User-Configurable Settings
# =======================
CONFIG_FILE = "./src/config.yaml"
DATASET = "TCGA_LGG"

# 256 -> 128 is a downscale, where INTER_AREA is the correct filter. Masks ignore this and always
# use nearest-neighbour (see resize_mask).
INTERPOLATION = cv2.INTER_AREA

# Every slice is already square (256x256), so padding would be a no-op; kept explicit for symmetry
# with the other scripts.
PRESERVE_ASPECT = False

PROGRESS_EVERY = 500

# Raw layout: the Kaggle archive extracts to kaggle_3m/<patient>/<patient>_<slice>.tif plus a
# matching <patient>_<slice>_mask.tif. Tolerates a flattened extraction too (see find_root).
RAW_ROOT_CANDIDATES = ["kaggle_3m", "lgg-mri-segmentation/kaggle_3m"]
RAW_MASK_SUFFIX = "_mask"

# The release provides no per-slice label; `class` is computed from the mask. See module docstring.
DERIVE_CLASS_FROM_MASK = True
CLASS_TUMOR = "tumor"
CLASS_NO_TUMOR = "no_tumor"

EXPECTED_RAW_SIZE = (256, 256)

MAPPING_COLUMNS = ["img_path", "mask_path", "class", "id", "lesion_id"]


# =======================
# Raw dataset readers
# =======================
def find_root(raw: Path) -> Path:
    """Locate the folder that holds the per-patient directories."""
    for candidate in RAW_ROOT_CANDIDATES:
        path = raw / candidate
        if path.is_dir():
            return path
    if any(p.is_dir() and p.name.startswith("TCGA_") for p in raw.iterdir()):
        return raw
    raise SystemExit(f"[ABORT] Could not find the per-patient folders under '{raw}'. Extract the "
                     f"lgg-mri-segmentation archive there first (expected '{raw}/kaggle_3m/').")


def slice_number(stem: str) -> int:
    """'TCGA_CS_4941_19960909_15' -> 15. Only orders slices within a patient; not a global id."""
    return int(stem.rsplit("_", 1)[-1])


def collect_rows(raw: Path) -> List[dict]:
    """One row per slice: image, its mask, and the patient it belongs to.

    Pairing is asserted in both directions. A slice whose mask is missing (or a mask with no
    slice) aborts the run rather than being dropped: a silent discard would not appear in any
    artefact, and the published counts would stop matching the release.
    """
    root = find_root(raw)
    patients = sorted(p for p in root.iterdir() if p.is_dir())
    if not patients:
        raise SystemExit(f"[ABORT] No patient folders found under '{root}'")

    rows: List[dict] = []
    for patient in patients:
        images: Dict[str, Path] = {}
        masks: Dict[str, Path] = {}
        for tif in sorted(patient.glob("*.tif")):
            if tif.stem.endswith(RAW_MASK_SUFFIX):
                masks[tif.stem[: -len(RAW_MASK_SUFFIX)]] = tif
            else:
                images[tif.stem] = tif

        orphan_images = sorted(set(images) - set(masks))
        orphan_masks = sorted(set(masks) - set(images))
        if orphan_images or orphan_masks:
            raise SystemExit(
                f"[ABORT] Patient '{patient.name}' has unpaired files: "
                f"{len(orphan_images)} image(s) without a mask {orphan_images[:3]}, "
                f"{len(orphan_masks)} mask(s) without an image {orphan_masks[:3]}. "
                f"Every row of the mapping must carry both supervisions; refusing to drop them "
                f"silently.")

        for stem in sorted(images, key=slice_number):
            rows.append({"stem": stem,
                         "img_src": images[stem],
                         "mask_src": masks[stem],
                         "lesion_id": patient.name,
                         "slice": slice_number(stem)})

    print(f"[INFO] Found {len(rows)} paired slices across {len(patients)} patients")
    return rows


# =======================
# Writing the variant
# =======================
def write_variant(rows: List[dict], output_path: Path, image_size: int) -> None:
    """Resize and write every slice and its mask, recording the output paths and derived class."""
    img_dir, mask_dir = output_path / "images", output_path / "masks"
    total = len(rows)

    for n, row in enumerate(rows, start=1):
        img = cv2.imread(str(row["img_src"]), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"[ABORT] Could not read image '{row['img_src']}'")
        if img.shape[:2] != EXPECTED_RAW_SIZE:
            raise SystemExit(f"[ABORT] '{row['img_src']}' is {img.shape[:2]}, expected "
                             f"{EXPECTED_RAW_SIZE}; the release is uniformly 256x256")
        img_out = img_dir / f"{row['stem']}.png"
        cv2.imwrite(str(img_out), resize_image(img, image_size, INTERPOLATION, PRESERVE_ASPECT))
        row["img_path"] = img_out.as_posix()

        mask = cv2.imread(str(row["mask_src"]), 0)
        if mask is None:
            raise SystemExit(f"[ABORT] Could not read mask '{row['mask_src']}'")
        resized_mask = resize_mask(mask, image_size, PRESERVE_ASPECT)
        mask_out = mask_dir / f"{row['stem']}_mask.png"
        cv2.imwrite(str(mask_out), resized_mask)
        row["mask_path"] = mask_out.as_posix()

        # Derived from the RESIZED mask on purpose: a tumour thin enough to vanish at 128px would
        # otherwise be labelled `tumor` while carrying an empty target, contradicting the mask the
        # model is actually trained on.
        row["class"] = CLASS_TUMOR if np.any(resized_mask) else CLASS_NO_TUMOR
        row["raw_tumor"] = bool(np.any(mask))

        if n % PROGRESS_EVERY == 0 or n == total:
            print(f"[INFO]   {n}/{total} written")


def build_mapping(rows: List[dict]) -> pd.DataFrame:
    """`id` is assigned over the patient/slice ordering, so it is stable across reruns."""
    for new_id, row in enumerate(sorted(rows, key=lambda r: (r["lesion_id"], r["slice"]))):
        row["id"] = new_id

    df = pd.DataFrame(rows)[MAPPING_COLUMNS]
    return add_image_metadata(df, sort_by=("lesion_id", "id"))


def summarize(df: pd.DataFrame, rows: List[dict], config_data: dict) -> None:
    print(f"\n[INFO] Total rows: {len(df)}")
    print(f"[INFO] Patients (lesion_id): {df['lesion_id'].nunique()}")

    slices_per_patient = df.groupby("lesion_id").size()
    print(f"[INFO] Slices per patient: min {slices_per_patient.min()}, "
          f"median {int(slices_per_patient.median())}, max {slices_per_patient.max()}")

    print("\n[INFO] Class distribution (derived from the resized mask):")
    for name, count in df["class"].value_counts().items():
        print(f"[INFO]     {name:<9} {count:>6}  ({count / len(df):.1%})")

    vanished = sum(1 for r in rows if r["raw_tumor"] and r["class"] == CLASS_NO_TUMOR)
    if vanished:
        print(f"\n[WARN] {vanished} slice(s) had a non-empty mask at 256px that became empty at "
              f"{config_data['image_size']}px; they are labelled '{CLASS_NO_TUMOR}' to stay "
              f"consistent with the target actually supervised.")

    with_tumor = df[df["class"] == CLASS_TUMOR]
    if not with_tumor.empty:
        print(f"\n[INFO] Tumour pixels on positive slices: min {int(with_tumor['tumor_pixels'].min())}, "
              f"median {int(with_tumor['tumor_pixels'].median())}, "
              f"max {int(with_tumor['tumor_pixels'].max())}")
    patients_all_negative = int((df.groupby("lesion_id")["class"]
                                 .apply(lambda s: (s == CLASS_NO_TUMOR).all())).sum())
    print(f"[INFO] Patients whose every slice is '{CLASS_NO_TUMOR}': {patients_all_negative}")

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

    duplicated = pd.Series([r["stem"] for r in rows]).duplicated()
    if duplicated.any():
        raise SystemExit(f"[ABORT] {int(duplicated.sum())} duplicate slice stem(s); output "
                         f"filenames would collide")

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
