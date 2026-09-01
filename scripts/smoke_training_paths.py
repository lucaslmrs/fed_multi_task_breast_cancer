"""One-epoch smoke for classic and centralized training entrypoints.

The smoke uses real preprocessed data and existing frozen partitions. Outputs are placed in a
temporary directory, checked for performance telemetry, and removed on completion.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.dataset import paths as dataset_paths
from src.training_centralized import run as run_centralized
from src.training_classification import run as run_classification
from src.training_multitask import run as run_multitask
from src.training_segmentation import run as run_segmentation


RUNNERS = {
    "multitask": (run_multitask, "MTnnUNet"),
    "segmentation": (run_segmentation, "nnUNet"),
    "classification": (run_classification, "nnUNetClassifier"),
    "centralized": (run_centralized, "MTnnUNet"),
}


def _smoke_config(base: dict, name: str, architecture: str, cpu: bool) -> dict:
    config = copy.deepcopy(base)
    config["model"]["architecture"] = architecture
    config["training"].update(
        epochs=1,
        max_patience=1,
        precision="fp32" if cpu else "bf16",
        cuda_benchmark=not cpu,
    )
    config.setdefault("runtime", {})["inference_batch_size"] = 32
    config["runtime"]["dataloader"] = {
        "num_workers": 0,
        "federated_num_workers": 0,
        "pin_memory": not cpu,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    config["runtime"]["telemetry"] = {
        "enabled": not cpu,
        "gpu_interval_seconds": 1.0,
    }
    if name == "centralized":
        master = dataset_paths.require_partition_file(config["data"])
        frame = pd.read_csv(master, usecols=["fold"])
        config["training"]["CV"] = int(frame["fold"].nunique())
    else:
        config["training"]["CV"] = 1
        config["training"]["holdout_test_size"] = 0.30
    return config


def _verify(run_path: Path, cpu: bool) -> dict:
    events_path = run_path / "runtime_events.csv"
    if not events_path.is_file():
        raise RuntimeError(f"Missing runtime telemetry: {events_path}")
    events = pd.read_csv(events_path)
    phases = set(events["phase"])
    missing = {"train_epoch", "validation_epoch", "test"}.difference(phases)
    if missing:
        raise RuntimeError(f"{run_path} is missing runtime phases {sorted(missing)}")
    if (events["duration_seconds"] <= 0).any():
        raise RuntimeError(f"{events_path} contains a non-positive duration")
    if not cpu:
        gpu_path = run_path / "gpu_telemetry.csv"
        if not gpu_path.is_file() or pd.read_csv(gpu_path).empty:
            raise RuntimeError(f"Missing or empty GPU telemetry: {gpu_path}")
    return {
        "events": len(events),
        "phases": sorted(phases),
        "peak_allocated_mb": float(events["cuda_peak_allocated_mb"].max()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--cpu", action="store_true", help="use explicit FP32 CPU mode")
    parser.add_argument(
        "--paths", nargs="+", choices=tuple(RUNNERS), default=list(RUNNERS)
    )
    args = parser.parse_args(argv)
    if not args.cpu and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise SystemExit("BF16 smoke requires CUDA with native BF16 support; use --cpu for FP32")

    with open(args.config, encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    with tempfile.TemporaryDirectory(prefix="training_paths_smoke_", dir="/tmp") as root:
        root_path = Path(root)
        for name in args.paths:
            runner, architecture = RUNNERS[name]
            config = _smoke_config(base, name, architecture, args.cpu)
            config_path = root_path / f"{name}.yaml"
            run_path = root_path / f"run_{name}"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            runner(config_path, run_path=run_path)
            summary = _verify(run_path, args.cpu)
            print(f"PATH_SMOKE_OK path={name} summary={summary}")


if __name__ == "__main__":
    main()
