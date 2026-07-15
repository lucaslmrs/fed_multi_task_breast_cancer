"""Configuration helpers shared by the multi-dataset federated pipeline."""

from pathlib import Path

from src.dataset import paths


def active_datasets(config: dict) -> list:
    """Datasets participating in this federation, preserving configured order."""
    configured = config["federated"].get("datasets")
    return list(configured) if configured else [config["data"]["dataset"]]


def dataset_config(config: dict, dataset: str) -> dict:
    """Resolve a registry entry over the legacy ``data`` defaults.

    The legacy single-dataset pipeline remains valid when no top-level ``datasets`` registry is
    present. Multi-dataset-only settings live in the registry and global loader settings continue
    to come from ``data`` unless explicitly overridden.
    """
    base = dict(config["data"])
    registry = config.get("datasets", {})
    if dataset in registry:
        base.update(registry[dataset])
    elif dataset != base.get("dataset"):
        raise KeyError(f"Dataset '{dataset}' has no entry in config.datasets")

    base["dataset"] = dataset
    base.setdefault("channels", config["model"].get("sequences", 1))
    base.setdefault("classes", config["data"]["classes"])
    base.setdefault("seg_exclude_classes", [])
    base.setdefault("class_weighting", "none")
    base.setdefault("augmentation", config["data"].get("augmentation", {}))
    base.setdefault("transforms", config["data"].get("transforms", {}))
    base.setdefault("batch_size", config["data"].get("batch_size", 32))
    base.setdefault("oversampling", config["federated"].get("oversampling", {}))
    return base


def aggregation_config(config: dict) -> dict:
    fed = config["federated"]
    aggregation = dict(fed.get("aggregation", {}))
    aggregation.setdefault("mode", "flat")
    aggregation.setdefault("task_weights", fed.get("task_weights", {"seg": 1.0, "cls": 1.0}))
    aggregation.setdefault("dataset_weights", fed.get("dataset_weights", {}))
    return aggregation


def partition_file(config: dict) -> Path:
    configured = config["federated"].get("partition_file")
    if configured:
        path = Path(configured)
        if not path.exists():
            raise FileNotFoundError(
                f"Federated partition '{path}' not found. Generate it with "
                "`python -m src.dataset.federated_partition`."
            )
        return path
    return paths.require_partition_file(config["data"])


def validate_federated_config(config: dict) -> None:
    datasets = active_datasets(config)
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("federated.datasets must contain unique dataset names")

    resolved = {name: dataset_config(config, name) for name in datasets}
    for name, cfg in resolved.items():
        if cfg["channels"] not in {1, 3}:
            raise ValueError(f"datasets.{name}.channels must be 1 or 3")
        classes = list(cfg["classes"])
        if not classes or len(classes) != len(set(classes)):
            raise ValueError(f"datasets.{name}.classes must be a non-empty unique list")
        if cfg["class_weighting"] not in {"none", "balanced_fold", "balanced_local"}:
            raise ValueError(
                f"datasets.{name}.class_weighting must be none, balanced_fold, or balanced_local"
            )
        if cfg["channels"] == 3 and any(bool(v) for v in cfg.get("augmentation", {}).values()):
            raise ValueError(
                f"Legacy channel-stacking augmentations must be disabled for RGB dataset '{name}'"
            )

    share_stem = config["federated"].get("share_stem", True)
    channel_counts = {cfg["channels"] for cfg in resolved.values()}
    if share_stem and len(channel_counts) > 1:
        raise ValueError(
            "share_stem=true is incompatible with active datasets that have different channels"
        )

    aggregation = aggregation_config(config)
    if aggregation["mode"] not in {"flat", "hierarchical"}:
        raise ValueError("federated.aggregation.mode must be flat or hierarchical")
    for field in ("task_weights", "dataset_weights"):
        invalid = {key: value for key, value in aggregation[field].items() if float(value) <= 0}
        if invalid:
            raise ValueError(f"federated.aggregation.{field} must be positive: {invalid}")
    missing = set(datasets) - set(aggregation["dataset_weights"])
    if aggregation["mode"] == "hierarchical" and missing:
        raise ValueError(f"Hierarchical aggregation needs dataset weights for: {sorted(missing)}")
    extra = set(aggregation["dataset_weights"]) - set(datasets)
    if aggregation["mode"] == "hierarchical" and extra:
        raise ValueError(f"Hierarchical aggregation has weights for inactive datasets: {sorted(extra)}")
