"""Per-client data access and class weighting for the federated setup.

The master partition CSV is the sole source of split ownership.  This module never re-splits it:
it filters one client slice, optionally oversamples that slice, and feeds metadata to the lazy,
channel-aware dataset.  Class weights are deliberately resolved from the unmodified ``train``
rows of the master CSV, so validation/test data and oversampled replicas cannot leak into them.
"""

import hashlib
import logging
from typing import Iterator, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from src.dataset.BUSI_dataset import BUSI
from src.dataset.BUSI_dataloader import deterministic_oversampling


class RotatingBatchSampler(Sampler[list[int]]):
    """Deterministic, round-addressable batches over an infinite permutation stream.

    Each cycle visits every dataset index exactly once in a seeded random order. Consecutive
    cycles are concatenated before batches are formed, so a batch crossing a cycle boundary is
    still full. A round is addressed by its absolute offset in that stream; no in-memory cursor
    is required, which makes the sequence stable when Flower recreates a Ray client worker.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        steps_per_round: int,
        seed: int,
        server_round: int = 1,
    ):
        for name, value in (
            ("dataset_size", dataset_size),
            ("batch_size", batch_size),
            ("steps_per_round", steps_per_round),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.steps_per_round = steps_per_round
        self.seed = seed
        self.server_round = 1
        self.set_round(server_round)

    def set_round(self, server_round: int) -> None:
        """Select a one-based federated round without consuming earlier batches."""
        if isinstance(server_round, bool) or not isinstance(server_round, int) or server_round < 1:
            raise ValueError(f"server_round must be a positive integer, got {server_round!r}")
        self.server_round = server_round

    def __len__(self) -> int:
        return self.steps_per_round

    def _permutation(self, cycle: int) -> list[int]:
        payload = f"{self.seed}|{cycle}".encode("utf-8")
        cycle_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        generator = torch.Generator()
        generator.manual_seed(cycle_seed)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        examples_per_round = self.steps_per_round * self.batch_size
        stream_position = (self.server_round - 1) * examples_per_round
        cycle, offset = divmod(stream_position, self.dataset_size)
        permutation = self._permutation(cycle)

        for _ in range(self.steps_per_round):
            batch = []
            while len(batch) < self.batch_size:
                take = min(self.batch_size - len(batch), self.dataset_size - offset)
                batch.extend(permutation[offset:offset + take])
                offset += take
                if offset == self.dataset_size:
                    cycle += 1
                    offset = 0
                    permutation = self._permutation(cycle)
            yield batch


def list_clients(partition_file: str) -> pd.DataFrame:
    """Return the unique roster, including dataset when the master CSV provides it.

    Legacy single-dataset partitions retain their historical ``(client_id, task)`` shape.
    Multi-dataset partitions return ``(client_id, dataset, task)`` in roster tuple order.
    """
    df = pd.read_csv(partition_file, low_memory=False)
    columns = ["client_id", "task"]
    if "dataset" in df.columns:
        columns = ["client_id", "dataset", "task"]
    return df[columns].drop_duplicates().reset_index(drop=True)


def load_client_mapping(
    partition_file: str,
    fold: int,
    client_id: str,
    split: str,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """Filter the master CSV down to one ``(dataset, fold, client, split)`` slice."""
    df = pd.read_csv(partition_file, low_memory=False)
    keep = (df["fold"] == fold) & (df["client_id"] == client_id) & (df["split"] == split)
    # ``dataset`` is optional to keep frozen BUSI partition files usable.
    if dataset is not None and "dataset" in df.columns:
        keep &= df["dataset"] == dataset
    return df[keep].copy()


def _training_class_rows(
    partition_file: str,
    fold: int,
    dataset: Optional[str] = None,
    client_id: Optional[str] = None,
) -> pd.DataFrame:
    """Read only raw classification-training ownership rows from the master partition."""
    mapping = pd.read_csv(partition_file, low_memory=False)
    required = {"fold", "split", "class"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"Partition file is missing required columns for class weighting: {missing}")

    keep = (mapping["fold"] == fold) & (mapping["split"] == "train")
    if "task" in mapping.columns:
        keep &= mapping["task"] == "cls"
    if dataset is not None:
        if "dataset" not in mapping.columns:
            logging.debug("Ignoring dataset=%s for legacy partition without a dataset column", dataset)
        else:
            keep &= mapping["dataset"] == dataset
    if client_id is not None:
        if "client_id" not in mapping.columns:
            raise ValueError("Partition file has no client_id column required by balanced_local")
        keep &= mapping["client_id"] == client_id
    return mapping[keep].copy()


def _balanced_weights(mapping: pd.DataFrame, classes: Sequence[str]) -> list[float]:
    """Compute ``N / (K * n_c)`` in the exact configured class order.

    A class absent from a local client's training slice receives zero.  Its theoretical inverse
    frequency is undefined, but the weight is never selected because that client owns no target of
    the class; zero avoids infinities while preserving the formula for every observed class.
    """
    classes = list(classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError(f"classes must be a non-empty unique ordered sequence, got {classes}")

    labels = mapping["class"].dropna()
    unknown = sorted(set(labels.unique()) - set(classes))
    if unknown:
        raise ValueError(f"Training mapping contains classes absent from config order: {unknown}")
    if labels.empty:
        raise ValueError("No classification train rows remain after applying class-weight filters")

    counts = labels.value_counts().reindex(classes, fill_value=0)
    n_samples = int(counts.sum())
    n_classes = len(classes)
    weights = [n_samples / (n_classes * int(count)) if count else 0.0 for count in counts]
    if (counts == 0).any():
        absent = counts.index[counts == 0].tolist()
        logging.warning("No local training examples for classes %s; assigning weight 0", absent)
    return [float(weight) for weight in weights]


def resolve_class_weights(
    partition_file: str,
    fold: int,
    client_id: Optional[str],
    dataset: Optional[str],
    classes: Sequence[str],
    mode: Optional[str],
) -> Optional[list[float]]:
    """Resolve configured classification weights without inspecting validation/test rows.

    ``balanced_fold`` pools all classification clients of the dataset in the fold and therefore
    returns one common vector for federated and local-only clients.  ``balanced_local`` restricts
    the same calculation to ``client_id``.  Null/``none`` modes disable weighting.
    """
    normalised_mode = "none" if mode is None else str(mode).lower()
    if normalised_mode in {"none", "null", "false", "off"}:
        return None
    if normalised_mode not in {"balanced_fold", "balanced_local"}:
        raise ValueError(
            f"Unknown class_weighting mode {mode!r}; expected 'balanced_fold', "
            "'balanced_local', or null"
        )
    if normalised_mode == "balanced_local" and client_id is None:
        raise ValueError("balanced_local requires client_id")

    rows = _training_class_rows(
        partition_file=partition_file,
        fold=fold,
        dataset=dataset,
        client_id=client_id if normalised_mode == "balanced_local" else None,
    )
    weights = _balanced_weights(rows, classes)
    logging.info(
        "Class weights (%s, dataset=%s, fold=%s%s): %s",
        normalised_mode,
        dataset or "legacy",
        fold,
        f", client={client_id}" if normalised_mode == "balanced_local" else "",
        dict(zip(classes, weights)),
    )
    return weights


def build_client_loader(
    partition_file: str,
    fold: int,
    client_id: str,
    split: str,
    batch_size: int,
    transforms=None,
    augmentations=None,
    oversampling: bool = False,
    dataset: Optional[str] = None,
    channels: int = 1,
    classes: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    local_training_mode: str = "epochs",
    steps_per_round: int = 10,
    sampling_seed: int = 0,
) -> DataLoader:
    """Build the DataLoader for a single client/split.

    - ``train``: shuffled, batch_size, optional deterministic oversampling, transforms applied.
    - ``val``  : not shuffled, batch_size, no oversampling, no geometric transforms.
    - ``test`` : not shuffled, batch_size 1, no oversampling, no geometric transforms.

    ``max_samples`` is a deterministic smoke-test budget.  It is applied after optional train
    oversampling so it is a hard cap on loader work; class weights remain based on the complete,
    unmodified master partition via :func:`resolve_class_weights`.

    In ``steps`` mode, the train loader yields exactly ``steps_per_round`` full batches from a
    deterministic rotating stream. Call ``loader.batch_sampler.set_round(round)`` before use.
    """
    mapping = load_client_mapping(partition_file, fold, client_id, split, dataset=dataset)
    raw_num_samples = len(mapping)

    local_training_mode = str(local_training_mode).lower()
    if local_training_mode not in {"epochs", "steps"}:
        raise ValueError("local_training_mode must be 'epochs' or 'steps'")

    is_train = split == "train"
    if is_train and oversampling and len(mapping) > 0:
        mapping = deterministic_oversampling(mapping)
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError(f"max_samples must be a positive integer or None, got {max_samples!r}")
        if len(mapping) > max_samples:
            mapping = mapping.sample(n=max_samples, random_state=int(fold)).reset_index(drop=True)

    client_dataset = BUSI(
        mapping_file=mapping,
        transforms=transforms if is_train else None,
        augmentations=augmentations,
        normalization=None,
        semantic_segmentation=False,
        channels=channels,
        classes=classes,
        dataset=dataset,
    )

    logging.info(
        "[%s] dataset=%s fold %s %s: %s samples%s",
        client_id, dataset or "legacy", fold, split, len(client_dataset),
        f" (smoke cap={max_samples})" if max_samples is not None else "",
    )

    if is_train and local_training_mode == "steps" and len(client_dataset) > 0:
        loader = DataLoader(
            client_dataset,
            batch_sampler=RotatingBatchSampler(
                dataset_size=len(client_dataset),
                batch_size=batch_size,
                steps_per_round=steps_per_round,
                seed=sampling_seed,
            ),
        )
    else:
        loader = DataLoader(
            client_dataset,
            batch_size=batch_size if split != "test" else 1,
            # RandomSampler rejects an empty dataset. Empty slices remain iterable and surface as
            # zero batches, which lets the caller issue the domain-specific validation message.
            shuffle=is_train and len(client_dataset) > 0,
            drop_last=False,
        )

    # Explicit loader metadata lets clients report raw ownership, effective oversampled size and
    # actual exposure without re-reading the partition or leaking validation/test data.
    loader.raw_num_samples = raw_num_samples
    loader.effective_num_samples = len(client_dataset)
    loader.local_training_mode = local_training_mode
    return loader
