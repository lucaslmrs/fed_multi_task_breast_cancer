"""
Non-federated, per-dataset baselines on the frozen federated master (plano.md, stage B0).

One MTnnUNet is trained per (dataset, mode) on the UNION of every client slice of that dataset in
the same frozen master the federated arms read, then scored with ``unified_eval`` on the SAME
per-client test slices. The numbers are therefore directly comparable with the federated and
local-only result tables:

    segmentation    seg rows of every client of the dataset; seg loss only
    classification  cls rows of every client of the dataset; cls loss only
    multitask       both, one row per image; ``L = sum_t lambda_t L_t`` over the samples each task
                    supervises (has_mask/has_label), with the federated task weights. ISIC's two
                    pools are disjoint image sets, so its multitask baseline is partially supervised.

Training is a normal centralized loop: ``training.epochs`` epochs over the pooled train rows, model
selection and early stopping on the pooled validation loss (``training.max_patience``), and the
configured LR scheduler. This differs from the federated fixed-round budget on purpose: the
baseline is "what an ordinary training on this dataset reaches".

    python -m src.training_baseline --config <federated config.yaml> --mode all --dataset all
    python -m src.training_baseline --config <...> --mode multitask --dataset ISIC_2018

Outputs: ``runs/baseline/<mode>/<dataset>_<timestamp>/`` with ``config.yaml``, ``execution.log``,
``history.csv``, ``fold_<k>/best.pt``, ``baseline_test_results.csv`` and
``baseline_cls_predictions.csv``.
"""

import argparse
import copy
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset.BUSI_dataset import BUSI
from src.dataset.BUSI_dataloader import deterministic_oversampling
from src.dataset.federated_dataloader import (list_clients, load_client_mapping,
                                              pivot_multitask_mapping, resolve_class_weights)
from src.dataset.splitting import evaluation_metadata
from src.experiments.training_diagnostics import run_training_diagnostics
from src.federated import local_trainer, unified_eval
from src.federated.client import _default_transforms, client_task_lambdas, resolve_device
from src.federated.config import (active_datasets, dataset_config,
                                  partition_file, validate_federated_config)
from src.utils.experiment_init import (device_setup, init_criterion_classification,
                                       init_criterion_segmentation_from_config, init_lr_scheduler,
                                       init_multitask_model, init_optimizer)
from src.utils.miscellany import init_log, seed_everything
from src.utils.supervision import attach_task_uncertainty, task_weighting_mode
from src.utils.training_runtime import dataloader_kwargs, precision_policy, runtime_config

MODES = {"segmentation": ("seg",), "classification": ("cls",), "multitask": ("seg", "cls")}
SETUP = "baseline"
RESULTS_FILE = "baseline_test_results.csv"
PREDICTIONS_FILE = "baseline_cls_predictions.csv"


def dataset_clients(master_file, dataset, tasks):
    """Clients of ``dataset`` that own at least one of ``tasks``, in roster order."""
    roster = list_clients(str(master_file))
    if "dataset" in roster.columns:
        roster = roster[roster["dataset"] == dataset]
    roster = roster[roster["task"].isin(tasks)]
    return list(dict.fromkeys(roster["client_id"]))


def pooled_mapping(master_file, fold, dataset, tasks, split):
    """Union of every client slice of ``dataset`` for ``split`` restricted to ``tasks``.

    Each image belongs to exactly one client per task in the frozen master, so the union has no
    duplicate (image, task) row. With several tasks the rows collapse to one row per image that
    carries only the supervision the partition granted (``pivot_multitask_mapping``).
    """
    frames = [load_client_mapping(master_file, fold, client_id, split, dataset=dataset, tasks=tasks)
              for client_id in dataset_clients(master_file, dataset, tasks)]
    mapping = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if mapping.empty:
        return mapping
    duplicated = mapping.duplicated(subset=["img_path", "task"]) if "task" in mapping else \
        mapping.duplicated(subset=["img_path"])
    if duplicated.any():
        raise ValueError(f"{dataset}/{split}: {int(duplicated.sum())} (image, task) rows appear in "
                         "more than one client; the pooled baseline would double-count them")
    if len(tasks) > 1:
        mapping = pivot_multitask_mapping(mapping)
    return mapping.reset_index(drop=True)


def _loader(mapping, data_cfg, dataset, split, batch_size, loader_options, tasks):
    is_train = split == "train"
    oversampling = data_cfg.get("oversampling")
    if isinstance(oversampling, dict):
        oversampling = any(bool(oversampling.get(task, False)) for task in tasks)
    if is_train and oversampling:
        if len(tasks) > 1:
            raise ValueError("Oversampling would duplicate segmentation targets on a multitask "
                             "baseline; use datasets.<name>.class_weighting instead")
        mapping = deterministic_oversampling(mapping)
    data = BUSI(
        mapping_file=mapping,
        transforms=_default_transforms(data_cfg.get("transforms", {})) if is_train else None,
        augmentations=data_cfg.get("augmentation"), normalization=None,
        semantic_segmentation=False, channels=data_cfg["channels"],
        classes=data_cfg["classes"], dataset=dataset,
    )
    return DataLoader(data, batch_size=batch_size, shuffle=is_train and len(data) > 0,
                      drop_last=False, **loader_options)


def _build_model(config, data_cfg, device):
    n_aug = sum(bool(value) for value in data_cfg.get("augmentation", {}).values())
    return init_multitask_model(
        architecture=config["model"]["architecture"], sequences=data_cfg["channels"] + n_aug,
        regions=1, n_classes=len(data_cfg["classes"]), width=config["model"]["width"],
        deep_supervision=config["model"]["deep_supervision"], save_folder=None,
    ).to(device)


def _limit(mapping, max_samples, fold):
    if max_samples is None or len(mapping) <= max_samples:
        return mapping
    return mapping.sample(n=max_samples, random_state=int(fold)).reset_index(drop=True)


def train_and_evaluate(config, dataset, mode, run_path, device, max_epochs=None, max_samples=None):
    """Train one baseline per fold and return its per-client test rows and predictions."""
    tasks = MODES[mode]
    master = partition_file(config)
    data_cfg = dataset_config(config, dataset)
    classes = list(data_cfg["classes"])
    precision = precision_policy(config, device)
    training = config["training"]
    epochs = int(max_epochs or training["epochs"])
    loader_options = dataloader_kwargs(config)
    task_lambdas = client_task_lambdas(config, tasks)

    rows, predictions, history = [], [], []
    for fold in range(int(training["CV"])):
        seed_everything(int(training["seed"]), cuda_benchmark=training.get("cuda_benchmark", False))
        train_map = _limit(pooled_mapping(master, fold, dataset, tasks, "train"), max_samples, fold)
        val_map = _limit(pooled_mapping(master, fold, dataset, tasks, "val"), max_samples, fold)
        if train_map.empty or val_map.empty:
            raise ValueError(f"{dataset}/{mode} fold {fold}: empty pooled train or val slice")
        logging.info("[%s/%s fold %s] pooled train=%s val=%s rows from clients %s", dataset, mode,
                     fold, len(train_map), len(val_map), dataset_clients(master, dataset, tasks))
        train_loader = _loader(train_map, data_cfg, dataset, "train", data_cfg["batch_size"],
                               loader_options, tasks)
        val_loader = _loader(val_map, data_cfg, dataset, "val", data_cfg["batch_size"],
                             loader_options, tasks)

        model = _build_model(config, data_cfg, device)
        if task_weighting_mode(config) == "uncertainty":
            attach_task_uncertainty(model, tasks)
        optimizer = init_optimizer(model, config["optimizer"]["opt"], config["optimizer"]["lr"])
        opt_cfg = config["optimizer"]
        scheduler = init_lr_scheduler(
            optimizer, scheduler=opt_cfg["scheduler"], t_max=opt_cfg.get("t_max", 40),
            patience=opt_cfg.get("patience", 20), min_lr=float(opt_cfg.get("min_lr", 1e-6)),
            factor=opt_cfg.get("decrease_factor", 0.5),
        )
        seg_criterion = init_criterion_segmentation_from_config(config["loss"])
        class_weights = resolve_class_weights(
            master, fold, None, dataset, classes, data_cfg.get("class_weighting", "none"),
        ) if "cls" in tasks else None
        cls_criterion = init_criterion_classification(
            n_classes=len(classes), classes_weighted=data_cfg.get("classes_weighted"),
            class_weights=class_weights,
            classification_criterion=data_cfg.get("classification_criterion", "CE"),
            device=device, focal_gamma=float(data_cfg.get("focal_gamma", 2.0)),
        )
        common = dict(task="+".join(tasks), device=device, num_classes=len(classes),
                      seg_criterion=seg_criterion, cls_criterion=cls_criterion,
                      inversely_weighted=config["loss"]["inversely_weighted"], tasks=tasks,
                      task_lambdas=task_lambdas, precision=precision)
        grad_norm = runtime_config(config)["diagnostics"]["grad_norm"]["enabled"]

        best_path = Path(run_path) / f"fold_{fold}" / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_val, best_epoch, patience = float("inf"), -1, 0
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train = local_trainer.train_local(
                model, train_loader, optimizer, local_epochs=1, training_mode="epochs",
                grad_norm=grad_norm, **common)
            val = local_trainer.evaluate_local(model, val_loader, **common)
            val_loss = float(val["loss"])
            if opt_cfg["scheduler"] == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)
            improved = val_loss < best_val
            if improved:
                best_val, best_epoch, patience = val_loss, epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "val_loss": val_loss}, best_path)
            else:
                patience += 1
            record = {"fold": fold, "epoch": epoch, "train_loss": float(train["loss"]),
                      "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"],
                      "seconds": time.perf_counter() - started, "best": improved}
            if grad_norm:
                record.update(train_grad_norm_mean=train["grad_norm_mean"],
                              train_grad_norm_max=train["grad_norm_max"])
            for task_name, metrics in val.get("metrics", {}).items():
                for name, value in metrics.items():
                    record[f"val_{name}_{task_name}"] = value
            history.append(record)
            logging.info("[%s/%s fold %s] epoch %s train %.4f val %.4f%s", dataset, mode, fold,
                         epoch, record["train_loss"], val_loss, " *" if improved else "")
            pd.DataFrame(history).to_csv(Path(run_path) / "history.csv", index=False)
            if patience > int(training["max_patience"]):
                logging.info("[%s/%s fold %s] early stop at epoch %s (best epoch %s)", dataset,
                             mode, fold, epoch, best_epoch)
                break

        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        evaluation = {**evaluation_metadata(training), **config.get("evaluation", {})}
        for task in tasks:
            for client_id in dataset_clients(master, dataset, (task,)):
                mapping = load_client_mapping(master, fold, client_id, "test", dataset=dataset,
                                              tasks=[task])
                if mapping.empty:
                    continue
                loader = _loader(_limit(mapping, max_samples, fold), data_cfg, dataset, "test",
                                 runtime_config(config)["inference_batch_size"], loader_options,
                                 (task,))
                metrics, preds = unified_eval.evaluate(
                    model, loader, task, len(classes), device, class_names=classes,
                    precision=precision)
                rows.append({**evaluation, "setup": SETUP, "baseline_mode": mode, "fold": fold,
                             "client_id": client_id, "dataset": dataset, "task": task,
                             "best_epoch": best_epoch, "best_val_loss": best_val, **metrics})
                if preds is not None:
                    for key, value in {"setup": SETUP, "fold": fold, "client_id": client_id,
                                       "dataset": dataset}.items():
                        preds.insert(0, key, value)
                    predictions.append(preds)
    return rows, predictions


def run(config_path, mode, dataset, out_root="runs/baseline", max_epochs=None, max_samples=None,
        device=None):
    """Train and score one (mode, dataset) baseline; return its run directory."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["federated"]["standalone"] = False
    validate_federated_config(config)
    if dataset not in active_datasets(config):
        raise ValueError(f"{dataset} is not an active dataset of {config_path}")
    requested = device or config["federated"].get("device", "auto")
    device = resolve_device(device_setup() if requested == "auto" else requested)
    precision_policy(config, device)  # fail early on unsupported BF16

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(out_root) / mode / f"{dataset}_{timestamp}"
    run_path.mkdir(parents=True, exist_ok=False)
    init_log(log_name=str(run_path / "execution.log"))
    config.setdefault("experiment", {}).update(
        {"baseline_mode": mode, "baseline_dataset": dataset, "source_config": str(config_path)})
    (run_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False),
                                          encoding="utf-8")
    started = time.perf_counter()
    rows, predictions = train_and_evaluate(config, dataset, mode, run_path, device,
                                           max_epochs=max_epochs, max_samples=max_samples)
    results = pd.DataFrame(rows)
    results.to_csv(run_path / RESULTS_FILE, index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(run_path / PREDICTIONS_FILE, index=False)
    primary = {"seg": config.get("evaluation", {}).get("segmentation_primary_metric", "dice"),
               "cls": "balanced_acc"}
    for task, group in results.groupby("task"):
        column = primary[task] if primary[task] in group else group.columns[-1]
        logging.info("[baseline %s/%s] task=%s: %s %.4f (n=%s client test slices)", mode, dataset,
                     task, column, group[column].mean(), len(group))
    logging.info("Saved %s | total time %.1fs", run_path / RESULTS_FILE,
                 time.perf_counter() - started)
    run_training_diagnostics(config, run_path)
    return run_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True,
                        help="federated config pinning the protocol (e.g. a run's config.yaml)")
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--dataset", default="all", help="dataset name or 'all'")
    parser.add_argument("--out-root", default="runs/baseline")
    parser.add_argument("--device", default=None, help="override federated.device (cpu/cuda)")
    parser.add_argument("--max-epochs", type=int, default=None, help="smoke-test cap")
    parser.add_argument("--max-samples", type=int, default=None, help="smoke-test cap per split")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    datasets = active_datasets(config) if args.dataset == "all" else [args.dataset]
    modes = list(MODES) if args.mode == "all" else [args.mode]
    for mode in modes:
        for dataset in datasets:
            path = run(args.config, mode, dataset, args.out_root, args.max_epochs,
                       args.max_samples, args.device)
            print(f"BASELINE_OK mode={mode} dataset={dataset} run_path={path}", flush=True)


if __name__ == "__main__":
    main()
