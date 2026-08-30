"""Configuration helpers shared by the multi-dataset federated pipeline."""

import math
from pathlib import Path

from src.dataset import paths
from src.dataset.splitting import evaluation_settings
from src.utils.training_runtime import validate_runtime_config


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
    loss_cfg = config.get("loss", {})
    base.setdefault("classification_criterion", loss_cfg.get("classification_criterion", "CE"))
    base.setdefault("focal_gamma", loss_cfg.get("focal_gamma", 2.0))
    return base


def aggregation_config(config: dict) -> dict:
    fed = config["federated"]
    aggregation = dict(fed.get("aggregation", {}))
    aggregation.setdefault("mode", "flat")
    aggregation.setdefault("client_weighting", "num_examples")
    aggregation.setdefault("task_weights", fed.get("task_weights", {"seg": 1.0, "cls": 1.0}))
    aggregation.setdefault("dataset_weights", fed.get("dataset_weights", {}))
    return aggregation


def local_training_config(config: dict) -> dict:
    fed = config["federated"]
    local_training = dict(fed.get("local_training", {}))
    local_training.setdefault("mode", "epochs")
    local_training.setdefault("steps_per_round", 10)
    local_training.setdefault("local_epochs", fed.get("local_epochs", 1))
    return local_training


def training_telemetry_config(config: dict) -> dict:
    """Resolve observational training-curve settings without affecting experiment design."""
    telemetry = dict(config["federated"].get("training_telemetry", {}))
    telemetry.setdefault("enabled", True)
    telemetry.setdefault("granularity", "epoch_and_round")
    telemetry.setdefault("formats", ["csv", "html", "png"])
    telemetry["formats"] = list(telemetry["formats"])
    return telemetry


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
    validate_runtime_config(config)
    if "training" in config:
        evaluation_settings(config["training"])
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
        if cfg["classification_criterion"] not in {"CE", "Focal"}:
            raise ValueError(f"datasets.{name}.classification_criterion must be CE or Focal")
        focal_gamma = float(cfg["focal_gamma"])
        if not math.isfinite(focal_gamma) or focal_gamma <= 0:
            raise ValueError(f"datasets.{name}.focal_gamma must be finite and positive")
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

    local_training = local_training_config(config)
    if local_training["mode"] not in {"epochs", "steps"}:
        raise ValueError("federated.local_training.mode must be epochs or steps")
    for field in ("steps_per_round", "local_epochs"):
        value = local_training[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"federated.local_training.{field} must be a positive integer")

    telemetry = training_telemetry_config(config)
    if not isinstance(telemetry["enabled"], bool):
        raise ValueError("federated.training_telemetry.enabled must be a boolean")
    if telemetry["granularity"] not in {"epoch_and_round", "round_only"}:
        raise ValueError(
            "federated.training_telemetry.granularity must be epoch_and_round or round_only"
        )
    allowed_formats = {"csv", "html", "png"}
    formats = telemetry["formats"]
    if not formats or len(formats) != len(set(formats)) or set(formats) - allowed_formats:
        raise ValueError(
            "federated.training_telemetry.formats must be a non-empty unique subset of "
            "csv, html, png"
        )

    aggregation = aggregation_config(config)
    if aggregation["mode"] not in {"flat", "hierarchical"}:
        raise ValueError("federated.aggregation.mode must be flat or hierarchical")
    if aggregation["client_weighting"] not in {"uniform", "num_examples"}:
        raise ValueError("federated.aggregation.client_weighting must be uniform or num_examples")
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
