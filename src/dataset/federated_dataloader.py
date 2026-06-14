"""
Per-client dataloaders for the federated setup.

The master partition CSV (produced by ``federated_partition.py``) already says, for every
(fold, client_id, split), which images a client owns. This module simply filters that CSV and
feeds the resulting sub-DataFrame to the existing ``BUSI`` dataset -- no change to the dataset
itself. Per-task oversampling / augmentation knobs are applied here.
"""

import logging

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset.BUSI_dataset import BUSI
from src.dataset.BUSI_dataloader import deterministic_oversampling


def list_clients(partition_file: str) -> pd.DataFrame:
    """Return the unique (client_id, task) pairs described by the partition file."""
    df = pd.read_csv(partition_file)
    return df[["client_id", "task"]].drop_duplicates().reset_index(drop=True)


def load_client_mapping(partition_file: str, fold: int, client_id: str, split: str) -> pd.DataFrame:
    """Filter the master CSV down to one (fold, client_id, split) slice."""
    df = pd.read_csv(partition_file)
    return df[(df["fold"] == fold) & (df["client_id"] == client_id) & (df["split"] == split)].copy()


def build_client_loader(
    partition_file: str,
    fold: int,
    client_id: str,
    split: str,
    batch_size: int,
    transforms=None,
    augmentations=None,
    oversampling: bool = False,
) -> DataLoader:
    """Build the DataLoader for a single client/split.

    - ``train``: shuffled, batch_size, optional deterministic oversampling, transforms applied.
    - ``val``  : not shuffled, batch_size, no oversampling, no geometric transforms.
    - ``test`` : not shuffled, batch_size 1, no oversampling, no geometric transforms.
    """
    mapping = load_client_mapping(partition_file, fold, client_id, split)

    is_train = split == "train"
    if is_train and oversampling and len(mapping) > 0:
        mapping = deterministic_oversampling(mapping)

    dataset = BUSI(
        mapping_file=mapping,
        transforms=transforms if is_train else None,
        augmentations=augmentations,
        normalization=None,
        semantic_segmentation=False,
    )

    logging.info(f"[{client_id}] fold {fold} {split}: {len(dataset)} samples")

    return DataLoader(
        dataset,
        batch_size=batch_size if split != "test" else 1,
        shuffle=is_train,
        drop_last=False,
    )
