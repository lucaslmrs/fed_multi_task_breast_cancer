"""Deterministic federated partitions for single- and multi-dataset experiments.

The legacy :func:`build_federated_partition` API is intentionally preserved for the frozen
Curated BUSI experiment.  The multi-dataset entry point is
:func:`build_multi_dataset_partition`; it applies the scientifically appropriate outer split to
each pool and writes a separate master CSV (``data/federated_multi/federated_mapping.csv`` by
default).

Multi-dataset rules:

* Curated BUSI uses one image-level stratified outer split shared by segmentation/classification.
* ISIC segmentation uses all 3,694 Task 1 rows with a shuffled outer split.
* ISIC classification uses only the official train rows and a grouped stratified outer split keyed by
  ``lesion_id``.  Lesions remain indivisible when assigning both train/test pools to clients and
  when carving validation from each client's training pool.

Run ``python -m src.dataset.federated_partition`` to build the multi-dataset master.  Pass
``--legacy`` only when deliberately regenerating the old single-dataset partition.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.dataset import paths
from src.dataset.splitting import (
    DEFAULT_HOLDOUT_TEST_SIZE,
    evaluation_settings,
    outer_split_indices,
)

CLASS_COL = "class"
GROUP_COL = "lesion_id"
TASKS = ("seg", "cls")


def _validate_partition_args(n_clients: int, alpha: float = None) -> None:
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if alpha is not None and alpha <= 0:
        raise ValueError(f"dirichlet_alpha must be > 0, got {alpha}")


def _label_values(series: pd.Series) -> list:
    """Stable class order which also supports the unlabeled ISIC segmentation pool."""
    values = sorted(series.dropna().unique().tolist(), key=str)
    if series.isna().any():
        values.append(None)
    return values


def _label_mask(series: pd.Series, label) -> pd.Series:
    return series.isna() if label is None else series == label


def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        raise ValueError(f"Grouped partition requires column '{group_col}'")
    if df[group_col].isna().any():
        raise ValueError(f"Grouped partition requires non-null '{group_col}' values")

    conflicts = df.groupby(group_col, sort=False)[CLASS_COL].nunique(dropna=False)
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.tolist()[:5]
        raise ValueError(f"Groups span multiple classes in '{group_col}': {bad}")

    summary = (
        df.groupby(group_col, sort=False, as_index=False)
        .agg(**{CLASS_COL: (CLASS_COL, "first"), "_size": (group_col, "size")})
    )
    return summary


def _grouped_dirichlet_partition(
    df: pd.DataFrame,
    n_clients: int,
    alpha: float,
    seed: int,
    group_col: str,
) -> list:
    """Label-Dirichlet partition in which every group is allocated as one atomic unit."""
    rng = np.random.default_rng(seed)
    summary = _group_summary(df, group_col)
    assignments = {c: [] for c in range(n_clients)}

    for cls in _label_values(summary[CLASS_COL]):
        class_groups = summary[_label_mask(summary[CLASS_COL], cls)].copy()
        class_groups = class_groups.iloc[rng.permutation(len(class_groups))]
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        targets = proportions * class_groups["_size"].sum()
        allocated = np.zeros(n_clients, dtype=float)

        # Greedily fill the sampled image-count targets, but never split a lesion.
        for group, size in class_groups[[group_col, "_size"]].itertuples(index=False, name=None):
            deficits = targets - allocated
            client = int(np.argmax(deficits)) if deficits.max() > 0 else int(np.argmin(allocated))
            assignments[client].append(group)
            allocated[client] += size

    return [df[df[group_col].isin(assignments[c])].copy() for c in range(n_clients)]


def dirichlet_partition(
    df: pd.DataFrame,
    n_clients: int,
    alpha: float,
    seed: int,
    group_col: str = None,
) -> list:
    """Partition rows by label-distribution Dirichlet.

    ``group_col`` is optional so the original row-level API and numerical BUSI behavior remain
    unchanged.  When provided, each group is assigned wholly to one client.
    """
    _validate_partition_args(n_clients, alpha)
    if group_col is not None:
        return _grouped_dirichlet_partition(df, n_clients, alpha, seed, group_col)

    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]
    for cls in _label_values(df[CLASS_COL]):
        cls_idx = df.index[_label_mask(df[CLASS_COL], cls)].to_numpy()
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
        for client, chunk in enumerate(np.split(cls_idx, cuts)):
            client_idx[client].extend(chunk.tolist())
    return [df.loc[idx].copy() for idx in client_idx]


def _grouped_uniform_partition(
    df: pd.DataFrame,
    n_clients: int,
    seed: int,
    group_col: str,
) -> list:
    rng = np.random.default_rng(seed)
    summary = _group_summary(df, group_col)
    assignments = {c: [] for c in range(n_clients)}

    for cls in _label_values(summary[CLASS_COL]):
        class_groups = summary[_label_mask(summary[CLASS_COL], cls)].copy()
        class_groups = class_groups.iloc[rng.permutation(len(class_groups))]
        allocated = np.zeros(n_clients, dtype=int)
        for group, size in class_groups[[group_col, "_size"]].itertuples(index=False, name=None):
            client = int(np.argmin(allocated))
            assignments[client].append(group)
            allocated[client] += size

    return [df[df[group_col].isin(assignments[c])].copy() for c in range(n_clients)]


def stratified_uniform_partition(
    df: pd.DataFrame,
    n_clients: int,
    seed: int,
    group_col: str = None,
) -> list:
    """Spread each class as evenly as possible, optionally keeping groups atomic."""
    _validate_partition_args(n_clients)
    if group_col is not None:
        return _grouped_uniform_partition(df, n_clients, seed, group_col)

    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]
    for cls in _label_values(df[CLASS_COL]):
        cls_idx = df.index[_label_mask(df[CLASS_COL], cls)].to_numpy()
        rng.shuffle(cls_idx)
        for client, chunk in enumerate(np.array_split(cls_idx, n_clients)):
            client_idx[client].extend(chunk.tolist())
    return [df.loc[idx].copy() for idx in client_idx]


def carve_validation(
    df: pd.DataFrame,
    val_size: float,
    seed: int,
    group_col: str = None,
):
    """Split a client pool into train/validation without ever splitting an optional group."""
    if not 0 <= val_size < 1:
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")
    if len(df) < 2 or val_size <= 0:
        return df, df.iloc[0:0]

    if group_col is not None:
        groups = _group_summary(df, group_col)
        if len(groups) < 2:
            return df, df.iloc[0:0]
        counts = groups[CLASS_COL].value_counts(dropna=False)
        can_stratify = groups[CLASS_COL].nunique(dropna=False) >= 2 and counts.min() >= 2
        stratify = groups[CLASS_COL] if can_stratify else None
        try:
            train_groups, val_groups = train_test_split(
                groups[group_col], test_size=val_size, random_state=seed, stratify=stratify
            )
        except ValueError:
            train_groups, val_groups = train_test_split(
                groups[group_col], test_size=val_size, random_state=seed
            )
        return (
            df[df[group_col].isin(train_groups)].copy(),
            df[df[group_col].isin(val_groups)].copy(),
        )

    can_stratify = df[CLASS_COL].nunique() >= 2 and df[CLASS_COL].value_counts().min() >= 2
    stratify = df[CLASS_COL] if can_stratify else None
    try:
        return train_test_split(df, test_size=val_size, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(df, test_size=val_size, random_state=seed)


def _tag(
    df: pd.DataFrame,
    fold: int,
    client_id: str,
    task: str,
    split: str,
    dataset: str = None,
) -> pd.DataFrame:
    out = df.copy()
    if dataset is not None:
        out["dataset"] = dataset
    out["fold"] = fold
    out["client_id"] = client_id
    out["task"] = task
    out["split"] = split
    return out


def _partition_task_pool(
    train_pool: pd.DataFrame,
    test_pool: pd.DataFrame,
    dataset: str,
    task: str,
    fold: int,
    n_clients: int,
    dirichlet_alpha: float,
    val_size: float,
    seed: int,
    group_col: str = None,
) -> list:
    client_train = dirichlet_partition(
        train_pool.reset_index(drop=True), n_clients, dirichlet_alpha, seed, group_col=group_col
    )
    client_test = stratified_uniform_partition(
        test_pool.reset_index(drop=True), n_clients, seed, group_col=group_col
    )

    rows = []
    for client in range(n_clients):
        client_id = f"{dataset}_{task}_{client}"
        train_df, val_df = carve_validation(
            client_train[client], val_size, seed, group_col=group_col
        )
        rows.extend(
            [
                _tag(train_df, fold, client_id, task, "train", dataset),
                _tag(val_df, fold, client_id, task, "val", dataset),
                _tag(client_test[client], fold, client_id, task, "test", dataset),
            ]
        )
    return rows


def build_federated_partition(
    mapping_path: str,
    output_path: str,
    n_folds: int,
    seed: int,
    n_clients_seg: int,
    n_clients_cls: int,
    dirichlet_alpha: float,
    val_size: float = 0.2,
    seg_exclude_classes: list = None,
    holdout_test_size: float = DEFAULT_HOLDOUT_TEST_SIZE,
) -> pd.DataFrame:
    """Build the original single-dataset partition (backward-compatible API)."""
    seg_exclude_classes = seg_exclude_classes or []
    evaluation_settings({"CV": n_folds, "holdout_test_size": holdout_test_size})

    mapping_path = Path(mapping_path).resolve()
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file '{mapping_path}' does not exist")
    mapping = pd.read_csv(mapping_path).reset_index(drop=True)
    logging.info(f"Loaded {len(mapping)} images from {mapping_path}")

    rows = []
    outer_splits = outer_split_indices(
        mapping,
        n_splits=n_folds,
        seed=seed,
        strategy="stratified",
        holdout_test_size=holdout_test_size,
        label_col=CLASS_COL,
    )
    for fold, (train_ix, test_ix) in enumerate(outer_splits):
        train_pool = mapping.iloc[train_ix]
        test_pool = mapping.iloc[test_ix]
        fold_seed = seed + fold
        for task, n_clients in (("seg", n_clients_seg), ("cls", n_clients_cls)):
            train_task, test_task = train_pool.copy(), test_pool.copy()
            if task == "seg":
                train_task = train_task[~train_task[CLASS_COL].isin(seg_exclude_classes)]
                test_task = test_task[~test_task[CLASS_COL].isin(seg_exclude_classes)]

            client_train = dirichlet_partition(
                train_task.reset_index(drop=True), n_clients, dirichlet_alpha, fold_seed
            )
            client_test = stratified_uniform_partition(
                test_task.reset_index(drop=True), n_clients, fold_seed
            )
            for client in range(n_clients):
                client_id = f"{task}_{client}"
                train_df, val_df = carve_validation(client_train[client], val_size, fold_seed)
                rows.extend(
                    [
                        _tag(train_df, fold, client_id, task, "train"),
                        _tag(val_df, fold, client_id, task, "val"),
                        _tag(client_test[client], fold, client_id, task, "test"),
                    ]
                )

    master = pd.concat(rows, ignore_index=True)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    logging.info(f"Wrote federated partition ({len(master)} rows) to {output_path}")
    _log_summary(master)
    return master


def _build_shared_pool_dataset(
    mapping: pd.DataFrame,
    dataset: str,
    dataset_cfg: dict,
    n_clients: dict,
    n_folds: int,
    seed: int,
    dirichlet_alpha: float,
    val_size: float,
    holdout_test_size: float,
) -> pd.DataFrame:
    rows = []
    outer_splits = outer_split_indices(
        mapping,
        n_splits=n_folds,
        seed=seed,
        strategy="stratified",
        holdout_test_size=holdout_test_size,
        label_col=CLASS_COL,
    )
    excluded = dataset_cfg.get("seg_exclude_classes", [])
    for fold, (train_ix, test_ix) in enumerate(outer_splits):
        train_pool, test_pool = mapping.iloc[train_ix], mapping.iloc[test_ix]
        for task in TASKS:
            train_task, test_task = train_pool.copy(), test_pool.copy()
            if task == "seg":
                train_task = train_task[~train_task[CLASS_COL].isin(excluded)]
                test_task = test_task[~test_task[CLASS_COL].isin(excluded)]
            rows.extend(
                _partition_task_pool(
                    train_task,
                    test_task,
                    dataset,
                    task,
                    fold,
                    n_clients[task],
                    dirichlet_alpha,
                    val_size,
                    seed + fold,
                )
            )
    return pd.concat(rows, ignore_index=True)


def _build_multitask_dataset(
    mapping: pd.DataFrame,
    dataset: str,
    dataset_cfg: dict,
    n_clients: dict,
    n_folds: int,
    seed: int,
    dirichlet_alpha: float,
    val_size: float,
    holdout_test_size: float,
) -> pd.DataFrame:
    """Build a topology in which one client owns SEVERAL tasks over the SAME images.

    The fold pool is partitioned exactly ONCE, so every image belongs to a single client.  That is
    the whole point of the topology: the single-task layout allocates each task independently, so
    the same image ends up owned by a segmentation client and by a *different* classification
    client -- duplicating one silo's data across two silos.

    The master schema is unchanged: each client emits one row per ``(image, task)`` it is
    supervised for, and ``seg_exclude_classes`` simply drops those images from the ``seg`` rows.
    """
    counts = {n_clients[task] for task in TASKS}
    if len(counts) != 1:
        raise ValueError(
            f"client_topology 'multi_task' needs one client count for '{dataset}', but "
            f"n_clients is {dict(n_clients)}; set the same value for every task"
        )
    n_multitask_clients = counts.pop()
    _validate_partition_args(n_multitask_clients, dirichlet_alpha)
    excluded = dataset_cfg.get("seg_exclude_classes", [])

    rows = []
    outer_splits = outer_split_indices(
        mapping,
        n_splits=n_folds,
        seed=seed,
        strategy="stratified",
        holdout_test_size=holdout_test_size,
        label_col=CLASS_COL,
    )
    for fold, (train_ix, test_ix) in enumerate(outer_splits):
        train_pool, test_pool = mapping.iloc[train_ix], mapping.iloc[test_ix]
        fold_seed = seed + fold
        client_train = dirichlet_partition(
            train_pool.reset_index(drop=True), n_multitask_clients, dirichlet_alpha, fold_seed
        )
        client_test = stratified_uniform_partition(
            test_pool.reset_index(drop=True), n_multitask_clients, fold_seed
        )
        for client in range(n_multitask_clients):
            client_id = f"{dataset}_mt_{client}"
            train_df, val_df = carve_validation(client_train[client], val_size, fold_seed)
            for task in TASKS:
                for split, frame in (
                    ("train", train_df), ("val", val_df), ("test", client_test[client]),
                ):
                    if task == "seg":
                        frame = frame[~frame[CLASS_COL].isin(excluded)]
                    rows.append(_tag(frame, fold, client_id, task, split, dataset))
    return pd.concat(rows, ignore_index=True)


def _build_per_task_dataset(
    mapping: pd.DataFrame,
    dataset: str,
    dataset_cfg: dict,
    n_clients: dict,
    n_folds: int,
    seed: int,
    dirichlet_alpha: float,
    val_size: float,
    holdout_test_size: float,
) -> pd.DataFrame:
    required = {"task", "official_split", GROUP_COL}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"Dataset '{dataset}' mapping is missing columns: {sorted(missing)}")

    rows = []
    seg_pool = mapping[mapping["task"] == "seg"].reset_index(drop=True)
    if seg_pool.empty:
        raise ValueError(f"Dataset '{dataset}' has no segmentation rows")
    seg_splits = outer_split_indices(
        seg_pool,
        n_splits=n_folds,
        seed=seed,
        strategy="random",
        holdout_test_size=holdout_test_size,
    )
    for fold, (train_ix, test_ix) in enumerate(seg_splits):
        rows.extend(
            _partition_task_pool(
                seg_pool.iloc[train_ix],
                seg_pool.iloc[test_ix],
                dataset,
                "seg",
                fold,
                n_clients["seg"],
                dirichlet_alpha,
                val_size,
                seed + fold,
            )
        )

    source_split = dataset_cfg.get("cls_source_split", "train")
    cls_pool = mapping[
        (mapping["task"] == "cls") & (mapping["official_split"] == source_split)
    ].reset_index(drop=True)
    if cls_pool.empty:
        raise ValueError(
            f"Dataset '{dataset}' has no classification rows in official split '{source_split}'"
        )
    _group_summary(cls_pool, GROUP_COL)  # fail before constructing any partial partition
    cls_splits = outer_split_indices(
        cls_pool,
        n_splits=n_folds,
        seed=seed,
        strategy="stratified_group",
        holdout_test_size=holdout_test_size,
        label_col=CLASS_COL,
        group_col=GROUP_COL,
    )
    for fold, (train_ix, test_ix) in enumerate(cls_splits):
        rows.extend(
            _partition_task_pool(
                cls_pool.iloc[train_ix],
                cls_pool.iloc[test_ix],
                dataset,
                "cls",
                fold,
                n_clients["cls"],
                dirichlet_alpha,
                val_size,
                seed + fold,
                group_col=GROUP_COL,
            )
        )

    return pd.concat(rows, ignore_index=True)


def _dataset_mapping_path(config: dict, dataset: str, dataset_cfg: dict) -> Path:
    root = dataset_cfg.get("root", config["data"]["root"])
    variant = dataset_cfg.get("variant", config["data"].get("variant", "processed_128"))
    return Path(root) / dataset / variant / paths.MAPPING_FILENAME


def _multi_output_path(config: dict, output_path: str = None) -> Path:
    if output_path is None:
        try:
            output_path = config["federated"]["partition_file"]
        except KeyError as exc:
            raise ValueError("federated.partition_file is required for multi-dataset mode") from exc
    candidate = Path(output_path).resolve()

    # Protect every dataset's historical frozen master, even if a config is accidentally changed.
    root = config["data"]["root"]
    frozen = {
        (Path(root) / name / paths.FEDERATED_DIRNAME / paths.PARTITION_FILENAME).resolve()
        for name in config.get("datasets", {})
    }
    if candidate in frozen:
        raise ValueError(
            f"Refusing to overwrite frozen single-dataset partition '{candidate}'. "
            "Use a separate federated.partition_file for the multi-dataset master."
        )
    return candidate


def _validate_multi_master(master: pd.DataFrame) -> None:
    required = {"dataset", "fold", "client_id", "task", "split", "img_path"}
    missing = required.difference(master.columns)
    if missing:
        raise AssertionError(f"Multi-dataset master is missing columns: {sorted(missing)}")

    duplicate_image = master.duplicated(["dataset", "fold", "task", "img_path"])
    if duplicate_image.any():
        raise AssertionError("An image was assigned more than once within a dataset/fold/task")

    # A client owning more than one task marks the multi-task topology, whose defining property is
    # that an image lives in exactly one silo. The single-task topology deliberately allows an
    # image in a seg client and in a different cls client, so the check is scoped, not global.
    tasks_per_client = master.groupby(["dataset", "client_id"])["task"].nunique()
    multitask_datasets = sorted(
        {dataset for (dataset, _), n_tasks in tasks_per_client.items() if n_tasks > 1}
    )
    for dataset in multitask_datasets:
        rows = master[master["dataset"] == dataset]
        ownership = rows.groupby(["fold", "img_path"])["client_id"].nunique()
        shared = int((ownership > 1).sum())
        if shared:
            raise AssertionError(
                f"{shared} images of '{dataset}' belong to more than one client within a fold; "
                "a multi-task topology must not duplicate an image across silos"
            )

    if GROUP_COL in master.columns:
        grouped = master[
            (master["task"] == "cls") & master[GROUP_COL].notna()
        ].groupby(["dataset", "fold", GROUP_COL])
        violations = grouped.agg(client_ids=("client_id", "nunique"), splits=("split", "nunique"))
        violations = violations[(violations["client_ids"] > 1) | (violations["splits"] > 1)]
        if not violations.empty:
            raise AssertionError(
                f"{len(violations)} lesion groups cross clients or splits in the master partition"
            )


def build_multi_dataset_partition(
    config: dict,
    output_path: str = None,
) -> pd.DataFrame:
    """Build and persist the configured multi-dataset federated master.

    Public configuration interface:
    ``datasets`` is the registry; ``federated.datasets`` selects active entries;
    ``federated.n_clients[dataset][task]`` defines the roster; and
    ``federated.partition_file`` is the independent output path.
    """
    if "datasets" not in config:
        raise ValueError("Top-level 'datasets' registry is required for multi-dataset mode")
    fed_cfg = config["federated"]
    active = fed_cfg.get("datasets")
    if not active:
        raise ValueError("federated.datasets must select at least one registered dataset")
    n_folds = config["training"]["CV"]
    _, _, holdout_test_size = evaluation_settings(config["training"])
    split_test_size = holdout_test_size or DEFAULT_HOLDOUT_TEST_SIZE

    output = _multi_output_path(config, output_path)
    frames = []
    for dataset in active:
        if dataset not in config["datasets"]:
            raise ValueError(f"Dataset '{dataset}' is not present in the datasets registry")
        dataset_cfg = config["datasets"][dataset]
        client_counts = fed_cfg.get("n_clients", {}).get(dataset)
        if not client_counts or any(task not in client_counts for task in TASKS):
            raise ValueError(f"federated.n_clients.{dataset} must define seg and cls")
        for task in TASKS:
            _validate_partition_args(client_counts[task], fed_cfg["dirichlet_alpha"])

        mapping_path = _dataset_mapping_path(config, dataset, dataset_cfg).resolve()
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file '{mapping_path}' does not exist")
        mapping = pd.read_csv(mapping_path).reset_index(drop=True)
        logging.info(f"Loaded {len(mapping)} rows for {dataset} from {mapping_path}")

        common = dict(
            mapping=mapping,
            dataset=dataset,
            dataset_cfg=dataset_cfg,
            n_clients=client_counts,
            n_folds=n_folds,
            seed=config["training"]["seed"],
            dirichlet_alpha=fed_cfg["dirichlet_alpha"],
            val_size=fed_cfg.get("val_size", 0.2),
            holdout_test_size=split_test_size,
        )
        strategy = dataset_cfg.get("fold_strategy", "stratified")
        topology = str(dataset_cfg.get("client_topology", "single_task")).lower()
        if topology not in {"single_task", "multi_task"}:
            raise ValueError(
                f"datasets.{dataset}.client_topology must be 'single_task' or 'multi_task', "
                f"got {topology!r}"
            )
        if topology == "multi_task":
            if strategy != "stratified":
                raise ValueError(
                    f"Dataset '{dataset}' uses fold_strategy '{strategy}', whose tasks are "
                    "disjoint image sets; a multi-task client requires images supervised for "
                    "every task, so only 'stratified' can host client_topology 'multi_task'"
                )
            frames.append(_build_multitask_dataset(**common))
        elif strategy == "stratified":
            frames.append(_build_shared_pool_dataset(**common))
        elif strategy == "per_task":
            frames.append(_build_per_task_dataset(**common))
        else:
            raise ValueError(f"Unsupported fold_strategy '{strategy}' for dataset '{dataset}'")

    master = pd.concat(frames, ignore_index=True, sort=False)
    metadata = ["dataset", "fold", "client_id", "task", "split"]
    source_columns = [column for column in master.columns if column not in metadata]
    master = master[source_columns + metadata]
    _validate_multi_master(master)

    output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output, index=False)
    logging.info(f"Wrote multi-dataset partition ({len(master)} rows) to {output}")
    _log_summary(master)
    return master


# Explicit alias for callers which prefer the longer federated name.
build_federated_multi_partition = build_multi_dataset_partition


def _log_summary(master: pd.DataFrame) -> None:
    """Log fold-zero per-client distributions without assuming labels are present."""
    fold0 = master[master["fold"] == 0]
    prefix = ["dataset"] if "dataset" in fold0.columns else []
    logging.info("Federated partition summary (fold 0):")
    for keys, group in fold0.groupby(prefix + ["client_id", "split"], dropna=False):
        distribution = group[CLASS_COL].value_counts(dropna=False).to_dict()
        logging.info(f"  {keys} n={len(group):<5} {distribution}")


def _load_config(config_path: str) -> dict:
    with open(config_path) as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="./src/config.yaml")
    parser.add_argument("--output", default=None, help="Override the configured output path")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Regenerate the legacy data.dataset partition instead of the multi-dataset master",
    )
    args = parser.parse_args()
    config = _load_config(args.config)

    if args.legacy:
        data_cfg, train_cfg, fed_cfg = config["data"], config["training"], config["federated"]
        build_federated_partition(
            mapping_path=paths.require_mapping_file(data_cfg),
            output_path=args.output or paths.partition_file(data_cfg),
            n_folds=train_cfg["CV"],
            seed=train_cfg["seed"],
            n_clients_seg=fed_cfg["n_clients_seg"],
            n_clients_cls=fed_cfg["n_clients_cls"],
            dirichlet_alpha=fed_cfg["dirichlet_alpha"],
            val_size=fed_cfg["val_size"],
            seg_exclude_classes=data_cfg.get("seg_exclude_classes", []),
            holdout_test_size=train_cfg.get(
                "holdout_test_size", DEFAULT_HOLDOUT_TEST_SIZE
            ),
        )
    else:
        build_multi_dataset_partition(config, output_path=args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
