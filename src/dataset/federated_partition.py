"""
Federated data partitioning for the FedPer multi-task setup.

This script turns the centralized Curated BUSI ``mapping.csv`` into a single MASTER csv
describing, for every (fold, client, split) combination, which images each client owns.

Design (approved):
    1. Split at the IMAGE level with StratifiedKFold -> train_pool / test_pool per fold.
       An image that lands in train_pool can never appear in test_pool of the same fold.
    2. Each client owns exactly ONE task (``seg`` or ``cls``). The same image may therefore
       be used by a seg client AND a cls client (different tasks), but never crosses the
       train/test boundary within a fold.
    3. ``normal`` cases are removed from the seg task (empty masks add nothing) and kept for cls.
    4. TRAIN images are spread across the clients of a task with a label-distribution
       Dirichlet(alpha) partition (high alpha ~ IID). A validation slice is then carved from
       each client's train pool for server-side early stopping.
    5. TEST images are spread stratified-uniformly across the clients of a task (no Dirichlet).

Output columns: original mapping columns + ``fold``, ``client_id``, ``task``, ``split``.
Run with:  python -m src.dataset.federated_partition
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

CLASS_COL = "class"


def dirichlet_partition(df: pd.DataFrame, n_clients: int, alpha: float, seed: int) -> list:
    """Partition the rows of ``df`` across ``n_clients`` using a label-distribution
    Dirichlet(alpha). For every class, the proportion assigned to each client is sampled
    from Dirichlet(alpha * 1_n). A high alpha yields an (approximately) IID partition,
    while a low alpha concentrates classes on few clients (non-IID).
    """
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]

    for cls in sorted(df[CLASS_COL].unique()):
        cls_idx = df.index[df[CLASS_COL] == cls].to_numpy()
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        # cut points along the shuffled class indices
        cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
        for c, chunk in enumerate(np.split(cls_idx, cuts)):
            client_idx[c].extend(chunk.tolist())

    return [df.loc[idx].copy() for idx in client_idx]


def stratified_uniform_partition(df: pd.DataFrame, n_clients: int, seed: int) -> list:
    """Spread the rows of ``df`` as evenly as possible across ``n_clients`` while keeping
    each class roughly balanced. Used for the held-out test pool so every client gets a
    representative, comparable test set."""
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]

    for cls in sorted(df[CLASS_COL].unique()):
        cls_idx = df.index[df[CLASS_COL] == cls].to_numpy()
        rng.shuffle(cls_idx)
        for c, chunk in enumerate(np.array_split(cls_idx, n_clients)):
            client_idx[c].extend(chunk.tolist())

    return [df.loc[idx].copy() for idx in client_idx]


def carve_validation(df: pd.DataFrame, val_size: float, seed: int):
    """Split a client's train pool into (train, val). Falls back to a non-stratified split
    when any class is too small to stratify (common with Dirichlet + few samples)."""
    if len(df) < 2 or val_size <= 0:
        return df, df.iloc[0:0]

    can_stratify = df[CLASS_COL].nunique() >= 2 and df[CLASS_COL].value_counts().min() >= 2
    stratify = df[CLASS_COL] if can_stratify else None
    try:
        return train_test_split(df, test_size=val_size, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(df, test_size=val_size, random_state=seed)


def _tag(df: pd.DataFrame, fold: int, client_id: str, task: str, split: str) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = fold
    out["client_id"] = client_id
    out["task"] = task
    out["split"] = split
    return out


def build_federated_partition(
    mapping_path: str,
    output_path: str,
    n_folds: int,
    seed: int,
    n_clients_seg: int,
    n_clients_cls: int,
    dirichlet_alpha: float,
    val_size: float = 0.2,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError(f"This partitioning needs CV >= 2 folds, got {n_folds}. "
                         f"Set 'training.CV' to 2 or more in config.yaml.")

    mapping_path = Path(mapping_path).resolve()
    assert mapping_path.exists(), f"Mapping file '{mapping_path}' does not exist"
    mapping = pd.read_csv(mapping_path).reset_index(drop=True)
    logging.info(f"Loaded {len(mapping)} images from {mapping_path}")

    tasks = [("seg", n_clients_seg), ("cls", n_clients_cls)]
    rows = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_ix, test_ix) in enumerate(skf.split(mapping, mapping[CLASS_COL])):
        train_pool = mapping.iloc[train_ix]
        test_pool = mapping.iloc[test_ix]
        fold_seed = seed + fold  # vary the partition per fold but stay reproducible

        for task, n_clients in tasks:
            train_task = train_pool.copy()
            test_task = test_pool.copy()
            if task == "seg":  # normal cases have empty masks -> not useful for segmentation
                train_task = train_task[train_task[CLASS_COL] != "normal"]
                test_task = test_task[test_task[CLASS_COL] != "normal"]

            client_train = dirichlet_partition(train_task.reset_index(drop=True),
                                               n_clients, dirichlet_alpha, fold_seed)
            client_test = stratified_uniform_partition(test_task.reset_index(drop=True),
                                                       n_clients, fold_seed)

            for c in range(n_clients):
                client_id = f"{task}_{c}"
                train_df, val_df = carve_validation(client_train[c], val_size, fold_seed)
                rows.append(_tag(train_df, fold, client_id, task, "train"))
                rows.append(_tag(val_df, fold, client_id, task, "val"))
                rows.append(_tag(client_test[c], fold, client_id, task, "test"))

    master = pd.concat(rows, ignore_index=True)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    logging.info(f"Wrote federated partition ({len(master)} rows) to {output_path}")
    _log_summary(master)

    return master


def _log_summary(master: pd.DataFrame) -> None:
    """Log the per-client class distribution for fold 0 as a sanity check."""
    fold0 = master[master["fold"] == 0]
    logging.info("Federated partition summary (fold 0):")
    for (client_id, split), grp in fold0.groupby(["client_id", "split"]):
        dist = grp[CLASS_COL].value_counts().to_dict()
        logging.info(f"  {client_id:<8} {split:<6} n={len(grp):<4} {dist}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("./src/config.yaml") as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
    data_cfg, train_cfg, fed_cfg = config["data"], config["training"], config["federated"]

    build_federated_partition(
        mapping_path=f"{data_cfg['input_img']}/mapping.csv",
        output_path=fed_cfg["partition_file"],
        n_folds=train_cfg["CV"],
        seed=train_cfg["seed"],
        n_clients_seg=fed_cfg["n_clients_seg"],
        n_clients_cls=fed_cfg["n_clients_cls"],
        dirichlet_alpha=fed_cfg["dirichlet_alpha"],
        val_size=fed_cfg["val_size"],
    )
