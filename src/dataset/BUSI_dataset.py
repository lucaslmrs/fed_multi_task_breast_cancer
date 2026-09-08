from __future__ import division, print_function

import random
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, rotate, vflip

from src.utils.custom_transforms import apply_SOBEL_filter
from src.utils.images import min_max_scaler


DEFAULT_CLASSES = ("benign", "malignant", "normal")
# The old semantic-segmentation branch used a different numeric label order.  It is retained only
# when callers omit ``classes``; all new/multi-dataset callers derive labels from config order.
LEGACY_SEMANTIC_CLASSES = ("normal", "benign", "malignant")


def _missing(value) -> bool:
    """Return whether a scalar mapping value represents missing supervision/path metadata."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class BUSI(Dataset):
    """Channel-aware image dataset used by both BUSI and ISIC.

    The historical class name is kept for API compatibility.  Images and masks are loaded lazily
    in :meth:`__getitem__`; constructing an ISIC client therefore keeps only its mapping metadata
    in memory.  Missing task-specific supervision is represented by safe tensors: label ``-1`` for
    segmentation-only rows and a one-channel zero mask for classification-only rows.
    """

    def __init__(
            self,
            mapping_file: pd.DataFrame,
            transforms=None,
            augmentations=None,
            normalization=None,
            semantic_segmentation: bool = False,
            channels: int = 1,
            classes: Optional[Sequence[str]] = None,
            dataset: Optional[str] = None,
    ):
        super().__init__()

        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 (grayscale) or 3 (RGB), got {channels!r}")

        configured_classes = classes
        if configured_classes is None:
            configured_classes = LEGACY_SEMANTIC_CLASSES if semantic_segmentation else DEFAULT_CLASSES
        configured_classes = list(configured_classes)
        if not configured_classes:
            raise ValueError("classes must contain at least one class name")
        if len(configured_classes) != len(set(configured_classes)):
            raise ValueError(f"classes contains duplicate names: {configured_classes}")

        augmentations = augmentations or {}
        active_augmentations = [name for name, enabled in augmentations.items() if bool(enabled)]
        if channels == 3 and active_augmentations:
            raise ValueError(
                "Legacy CLAHE/SOBEL/brightness/contrast augmentations append grayscale feature "
                f"channels and are not supported for RGB input; disable {active_augmentations}"
            )

        # Keep the public attributes used by older notebooks, but store metadata only (no decoded
        # images/masks).  ``records`` normalises pandas scalar access and makes lazy reads cheap.
        self.mapping_file = mapping_file.copy().reset_index(drop=True)
        self.data = self.mapping_file.to_dict(orient="records")
        self.transforms = transforms
        self.semantic_segmentation = semantic_segmentation
        self.channels = channels
        self.classes = configured_classes
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.dataset = dataset or ""
        self.transforms_applied = {}
        self.augmentations = bool(active_augmentations)
        self.CLAHE = bool(augmentations.get("CLAHE", False))
        self.SOBEL = bool(augmentations.get("SOBEL", False))
        self.brightness_brighter = bool(augmentations.get("brightness_brighter", False))
        self.brightness_darker = bool(augmentations.get("brightness_darker", False))
        self.contrast_high = bool(augmentations.get("contrast_high", False))
        self.contrast_low = bool(augmentations.get("contrast_low", False))
        self.normalization = normalization

    def __len__(self):
        return len(self.data)

    def _load_image(self, row) -> tuple[np.ndarray, torch.Tensor]:
        path = row.get("img_path")
        if _missing(path):
            raise ValueError("Mapping row has no img_path")

        flag = cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR
        decoded = cv2.imread(str(Path(path)), flag)
        if decoded is None:
            raise FileNotFoundError(f"Image '{path}' referenced by the mapping could not be read")

        if self.channels == 1:
            image = torch.as_tensor(decoded, dtype=torch.float32).unsqueeze(0)
        else:
            # OpenCV decodes colour images as BGR; models and visualisation expect RGB.
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            image = torch.as_tensor(np.ascontiguousarray(decoded.transpose(2, 0, 1)), dtype=torch.float32)
        return decoded, image

    def _load_mask(self, row, height: int, width: int) -> torch.Tensor:
        path = row.get("mask_path")
        if _missing(path):
            return torch.zeros((1, height, width), dtype=torch.float32)

        flag = cv2.IMREAD_COLOR if self.semantic_segmentation else cv2.IMREAD_GRAYSCALE
        decoded = cv2.imread(str(Path(path)), flag)
        if decoded is None:
            raise FileNotFoundError(f"Mask '{path}' referenced by the mapping could not be read")

        if self.semantic_segmentation:
            return torch.as_tensor(np.ascontiguousarray(decoded.transpose(2, 0, 1)), dtype=torch.float32)

        decoded = decoded.copy()
        decoded[decoded == 255] = 1
        return torch.as_tensor(decoded, dtype=torch.float32).unsqueeze(0)

    def _label(self, row) -> tuple[torch.Tensor, str]:
        class_name = row.get("class")
        if _missing(class_name):
            return torch.full((1,), -1.0, dtype=torch.float32), ""
        if class_name not in self.class_to_index:
            raise ValueError(
                f"Unknown class {class_name!r}; configured class order is {self.classes}"
            )
        return torch.tensor([self.class_to_index[class_name]], dtype=torch.float32), str(class_name)

    def _legacy_augmented_channels(self, decoded_image: np.ndarray) -> list[torch.Tensor]:
        """Build the historical engineered grayscale channels (valid only for 1-channel input)."""
        augmented = []
        if self.CLAHE:
            clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(4, 4))
            augmented.append(torch.as_tensor(clahe.apply(decoded_image), dtype=torch.float32).unsqueeze(0))
        if self.SOBEL:
            augmented.append(torch.as_tensor(apply_SOBEL_filter(decoded_image), dtype=torch.float32).unsqueeze(0))
        if self.brightness_brighter:
            matrix = np.ones(decoded_image.shape, dtype="uint8") * 80
            augmented.append(torch.as_tensor(cv2.add(decoded_image, matrix), dtype=torch.float32).unsqueeze(0))
        if self.brightness_darker:
            matrix = np.ones(decoded_image.shape, dtype="uint8") * 80
            augmented.append(torch.as_tensor(cv2.subtract(decoded_image, matrix), dtype=torch.float32).unsqueeze(0))
        if self.contrast_low:
            matrix = np.ones(decoded_image.shape) * .02
            low = np.uint8(cv2.multiply(np.float64(decoded_image), matrix))
            augmented.append(torch.as_tensor(low, dtype=torch.float32).unsqueeze(0))
        if self.contrast_high:
            matrix = np.ones(decoded_image.shape) * 1.5
            high = np.uint8(np.clip(cv2.multiply(np.float64(decoded_image), matrix), 0, 255))
            augmented.append(torch.as_tensor(high, dtype=torch.float32).unsqueeze(0))
        return augmented

    def __getitem__(self, idx):
        patient_info = self.data[idx]
        decoded_image, image = self._load_image(patient_info)
        mask = self._load_mask(patient_info, image.shape[-2], image.shape[-1])
        label, class_name = self._label(patient_info)

        if self.normalization is not None:
            image = min_max_scaler(image)

        augmented = []
        if self.augmentations and not self.semantic_segmentation:
            augmented = self._legacy_augmented_channels(decoded_image)

        # Concatenating supervision and every input channel makes torchvision's random geometric
        # transform sample once and apply the identical geometry to mask, RGB and engineered
        # channels.  Split by the actual channel counts rather than assuming a grayscale image.
        if self.transforms is not None:
            mask_channels = mask.shape[0]
            joined = torch.cat([mask, image] + augmented, dim=0)
            joined = self.transforms(joined)
            mask = joined[:mask_channels]
            image = joined[mask_channels:]
        elif augmented:
            image = torch.cat([image] + augmented, dim=0)

        return {
            "patient_id": patient_info.get("id", -1),
            "label": label,
            "class": class_name,
            # Per-sample supervision flags.  A multi-task client owns rows whose mask or label is
            # absent, and its loss must skip the corresponding term instead of training against the
            # safe placeholders (an all-zero mask is a legitimate target only for BUSI's `normal`).
            "has_mask": torch.tensor(not _missing(patient_info.get("mask_path"))),
            "has_label": torch.tensor(not _missing(patient_info.get("class"))),
            "image": image,
            "mask": mask,
            "dim1": patient_info.get("dim1", image.shape[-2]),
            "dim2": patient_info.get("dim2", image.shape[-1]),
            "tumor_pixels": patient_info.get("tumor_pixels", float("nan")),
            "dataset": patient_info.get("dataset", self.dataset) or self.dataset,
        }


def testing_apply_transformations(image, transforms_sequential):
    """Legacy helper retained for notebooks/tests that inspect sampled transforms."""
    transforms_applied = {"horizontal_flip": False, "vertical_flip": False, "rotation": 0}

    if random.random() < transforms_sequential.get("horizontal_flip") != .0:
        transforms_applied["horizontal_flip"] = True
        image = hflip(image)

    if random.random() < transforms_sequential.get("vertical_flip") != .0:
        transforms_applied["vertical_flip"] = True
        image = vflip(image)

    if random.random() < transforms_sequential.get("rotation"):
        angle = int(np.random.choice(range(0, 360)))
        transforms_applied["rotation"] = angle
        image = rotate(image, angle)

    return image, transforms_applied
