#!/usr/bin/env python
# coding: utf-8

"""
Script to preprocess the BUSI Breast Ultrasound Images dataset:
- Resize images and masks
- Combine multiple masks into one
- Optionally filter using a Curated mapping CSV
- Generate mapping CSV with image info, dimensions, tumor pixels, and tumor sizes
- Logs number of masks per image for each class
"""

import os
from pathlib import Path
from typing import List
import pandas as pd
import yaml
import cv2

from src.dataset import paths
from src.dataset.preprocessing_utils import (add_image_metadata, assert_target_dataset,
                                             assert_variant_resolution, resize_image)

# =======================
# User-Configurable Settings
# =======================
CONFIG_FILE = "./src/config.yaml"
DATASET = "Curated_BUSI"

# Curated BUSI was built with nearest-neighbour resizing. It is not the best choice for the images
# (INTER_AREA would be), but changing it now would alter the curated dataset and invalidate the
# frozen federated partition and every result derived from it.
INTERPOLATION = cv2.INTER_NEAREST

# Set True to keep only the images listed in the dataset's curation_list.csv (duplicates removed
# via SSIM). With False every raw image is kept -- point `data.variant` at another folder first,
# otherwise the curated variant gets overwritten.
CURATED = True
CURATION_LIST_FILENAME = "curation_list.csv"


# =======================
# Utility Functions
# =======================
def load_class_dataframe(class_path: Path, class_name: str) -> pd.DataFrame:
    files = [f for f in sorted(os.listdir(class_path)) if f.endswith(".png")]
    ids = [f.replace(".png", "").split(" ")[-1].split("_")[0].replace("(", "").replace(")", "") for f in files]
    types = ["mask" if "mask" in f else "img" for f in files]
    print(f"[INFO] Loaded {len(ids)} files for class '{class_name}'")
    return pd.DataFrame({"class": [class_name] * len(ids), "ids": ids, "type": types})


def get_mask_counts(df: pd.DataFrame, class_name: str) -> List[int]:
    counts = df.groupby("ids").apply(lambda x: sum(x['type'] == 'mask')).to_dict()
    return [int(k) for k, v in counts.items() if v > 1]


def combine_and_resize_images(class_name: str, class_ids: List[str], multi_mask_ids: List[int],
                              path: Path, output_path: Path, image_size: int,
                              curated_ids: List[int] = None) -> None:
    for j in set(class_ids):
        j_int = int(j)
        if curated_ids is not None and j_int not in curated_ids:
            continue

        img_path = path / class_name / f"{class_name} ({j}).png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), 0)
        mask_files = [f"{class_name} ({j})_mask.png"]
        if j_int in multi_mask_ids:
            mask_files.append(f"{class_name} ({j})_mask_1.png")
        total_mask = sum(cv2.imread(str(path / class_name / f), 0) for f in mask_files)

        img = resize_image(img, image_size, INTERPOLATION)
        total_mask = resize_image(total_mask, image_size, INTERPOLATION)

        cv2.imwrite(str(output_path / "images" / f"{class_name}_id_{j}.png"), img)
        cv2.imwrite(str(output_path / "masks" / f"{class_name}_id_{j}_mask.png"), total_mask)


def process_all_classes(df_classes, class_names, input_path, output_path, image_size, curated_ids):
    multi_masks_per_class = []
    for i, cls in enumerate(class_names):
        multi_mask_ids = get_mask_counts(df_classes[i], cls)
        multi_masks_per_class.append(multi_mask_ids)
        combine_and_resize_images(cls, df_classes[i]['ids'], multi_mask_ids, input_path, output_path,
                                  image_size, curated_ids.get(cls))
    return multi_masks_per_class


def create_mapping_csv(output_path: Path) -> pd.DataFrame:
    img_paths = sorted((output_path / "images").glob("*.png"))
    # Build the mask path from the image's stem rather than substituting "images" -> "masks" in
    # the string: the dataset folder itself may contain either word.
    mask_paths = [output_path / "masks" / f"{p.stem}_mask.png" for p in img_paths]
    df_mapping = pd.DataFrame({"img_path": [p.as_posix() for p in img_paths],
                               "mask_path": [p.as_posix() for p in mask_paths]})
    df_mapping['class'] = df_mapping['img_path'].apply(lambda x: Path(x).stem.split('_')[0])
    df_mapping['id'] = df_mapping['img_path'].apply(lambda x: int(Path(x).stem.split('_')[-1]))
    return df_mapping


# =======================
# Main
# =======================
def main():
    with open(CONFIG_FILE) as cf:
        config_data = yaml.load(cf, Loader=yaml.FullLoader)["data"]

    assert_target_dataset(config_data, DATASET)
    assert_variant_resolution(paths.mapping_file(config_data), config_data["image_size"])

    class_names = config_data["classes"]
    image_size = config_data["image_size"]
    input_path = paths.raw_dir(config_data)
    output_path = paths.processed_dir(config_data)

    print(f"[INFO] Starting '{config_data['dataset']}' dataset preprocessing")
    print(f"[INFO] Reading raw images from: {input_path}")
    print(f"[INFO] Writing variant '{config_data['variant']}' to: {output_path}")
    print(f"[INFO] Target resize dimensions: ({image_size}, {image_size})")
    print(f"[INFO] Curated mode: {CURATED}")

    assert input_path.exists(), f"Raw dataset folder '{input_path}' does not exist"
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "masks").mkdir(parents=True, exist_ok=True)

    curated_ids_dict = {}
    if CURATED:
        curation_list = paths.dataset_dir(config_data) / CURATION_LIST_FILENAME
        assert curation_list.exists(), f"Curation list '{curation_list}' does not exist"
        curated_mapping = pd.read_csv(curation_list, sep=';')
        for cls in class_names:
            curated_ids_dict[cls] = curated_mapping[curated_mapping['class'] == cls]['id'].astype(int).tolist()
    else:
        curated_ids_dict = {cls: None for cls in class_names}

    df_classes = [load_class_dataframe(input_path / cls, cls) for cls in class_names]
    process_all_classes(df_classes, class_names, input_path, output_path, image_size, curated_ids_dict)

    df_mapping = create_mapping_csv(output_path)
    df_mapping = add_image_metadata(df_mapping)
    df_mapping.to_csv(str(paths.mapping_file(config_data)), index=False)

    # Summary
    total_images = len(df_mapping)
    print(f"[INFO] Preprocessing completed. Processed dataset saved at {output_path}")
    print(f"[INFO] Total images processed: {total_images}")
    for cls in class_names:
        count_cls = len(df_mapping[df_mapping['class'] == cls])
        print(f"[INFO] {cls.capitalize()} images: {count_cls}")


if __name__ == "__main__":
    main()
