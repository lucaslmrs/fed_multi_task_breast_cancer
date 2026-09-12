"""One-epoch smoke for classic and centralized training entrypoints.

The smoke uses real preprocessed data and existing frozen partitions. Outputs are placed in a
temporary directory, checked for performance telemetry, and removed on completion.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import torch
import yaml
from unittest.mock import patch
from torch.utils.data import DataLoader
from src.dataset.classic_dataloader import classic_data_config, class_weights_from_training
from src.dataset.BUSI_dataloader import load_datasets
from src.dataset.BUSI_dataset import BUSI

from src.dataset import paths as dataset_paths
from src.training_centralized import run as run_centralized, _centralized_loaders
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
    config["data"]["batch_size"] = 2
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


def _bounded_loaders(*args, samples_per_class=2, **kwargs):
    groups = load_datasets(*args, **kwargs)
    result = [[], [], []]
    for split, loaders in enumerate(groups):
        for loader in loaders:
            result[split].append(_limit_loader(loader, split, samples_per_class))
    return tuple(result)


def _limit_loader(loader, split, samples_per_class):
    source = loader.dataset
    frame = source.mapping_file.copy()
    frame['_supervision'] = frame['class'].fillna('__mask_only__')
    frame = frame.groupby('_supervision', sort=False).head(samples_per_class).copy()
    dataset = BUSI(frame, transforms=source.transforms, augmentations=None,
                   channels=source.channels, classes=source.classes, dataset=source.dataset)
    dataset.class_weights = class_weights_from_training(frame, source.classes)
    return DataLoader(dataset, batch_size=2, shuffle=split == 0)


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
    for csv in run_path.glob('fold_*/results_segmentation.csv'):
        frame = pd.read_csv(csv)
        required = {'dice_positive', 'iou_positive', 'empty_fp_image_rate',
                    'empty_predicted_area_fraction', 'n_positive', 'n_empty'}
        if not required.issubset(frame.columns):
            raise RuntimeError(f'{csv} lacks stratified segmentation metrics')
        if int(frame.n_positive.sum() + frame.n_empty.sum()) != len(frame):
            raise RuntimeError(f'{csv} has inconsistent supervision counts')
    return {
        "events": len(events),
        "phases": sorted(phases),
        "peak_allocated_mb": float(events["cuda_peak_allocated_mb"].max()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--dataset", choices=("Curated_BUSI", "ISIC_2018"), default="Curated_BUSI")
    parser.add_argument("--samples-per-class", type=int, default=2)
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
    if args.samples_per_class < 1:
        parser.error('--samples-per-class must be positive')
    if args.dataset == 'ISIC_2018' and 'centralized' in args.paths:
        parser.error('The historical centralized benchmark remains BUSI-only')
    base['data']['dataset'] = args.dataset
    base['data'] = classic_data_config(base)
    base['model']['sequences'] = base['data'].get('channels', 1)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="training_paths_smoke_", dir="/tmp") as root:
        root_path = Path(root)
        for name in args.paths:
            runner, architecture = RUNNERS[name]
            config = _smoke_config(base, name, architecture, args.cpu)
            config_path = root_path / f"{name}.yaml"
            run_path = root_path / f"run_{name}"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with ExitStack() as stack:
                if args.cpu:
                    stack.enter_context(patch(f'src.training_{name}.device_setup', return_value='cpu'))
                if name == 'centralized':
                    def limited_centralized(*pos, **kw):
                        return tuple(_limit_loader(loader, i, args.samples_per_class)
                                     for i, loader in enumerate(_centralized_loaders(*pos, **kw)))
                    stack.enter_context(patch('src.training_centralized._centralized_loaders',
                                              side_effect=limited_centralized))
                else:
                    def limited(*pos, **kw):
                        return _bounded_loaders(*pos, samples_per_class=args.samples_per_class, **kw)
                    stack.enter_context(patch(f'src.training_{name}.load_datasets', side_effect=limited))
                runner(config_path, run_path=run_path)
            summary = _verify(run_path, args.cpu)
            print(f"PATH_SMOKE_OK path={name} summary={summary}")


if __name__ == "__main__":
    main()
