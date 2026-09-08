"""
Centralized MTL baseline for the comparison (the "upper bound").

A single multi-task MTnnUNet is trained per fold on ALL the data the federation collectively sees,
read from the SAME frozen master partition CSV (so folds are identical and there is no leakage),
then evaluated on the SAME per-client test splits via the shared `unified_eval`. No prediction-
refining is applied (unified_eval ignores it) and `normal` is already absent from seg test slices,
so the numbers are directly comparable with the federated and local-only setups.

Run with:  python -m src.training_centralized   (requires the master CSV to already exist)
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import paths
from src.dataset.BUSI_dataset import BUSI
from src.dataset.BUSI_dataloader import deterministic_oversampling
from src.dataset.federated_dataloader import build_client_loader, list_clients
from src.dataset.splitting import evaluation_metadata, evaluation_settings, validate_master_splits
from src.federated import unified_eval
from src.federated.client import _default_transforms, resolve_device
from src.utils.criterions import apply_criterion_multitask_segmentation_classification
from src.utils.experiment_init import (device_setup, init_multitask_model,
                                       load_multitask_experiment_artefacts)
from src.utils.miscellany import init_log, seed_everything
from src.utils.training_runtime import (
    RuntimeEvents,
    dataloader_kwargs,
    precision_policy,
    runtime_config,
    start_gpu_telemetry,
)


def _centralized_loaders(config, partition_file, fold):
    """Train/val loaders for the centralized model: the fold's train pool (unique images, all the
    federation sees) minus a fresh stratified val slice for early stopping. Test images (split==test
    in the master CSV) are globally held out, so this is leak-free."""
    master = pd.read_csv(partition_file)
    pool = master[(master["fold"] == fold) & (master["split"] != "test")].drop_duplicates("img_path")
    train_df, val_df = train_test_split(pool, test_size=config["federated"]["val_size"],
                                        random_state=config["training"]["seed"], stratify=pool["class"])
    if config["data"]["oversampling"]:
        train_df = deterministic_oversampling(train_df)

    aug, bs = config["data"]["augmentation"], config["data"]["batch_size"]
    train_ds = BUSI(mapping_file=train_df, transforms=_default_transforms(), augmentations=aug)
    val_ds = BUSI(mapping_file=val_df, transforms=None, augmentations=aug)
    loader_options = dataloader_kwargs(config)
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_options),
            DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_options))


def _multitask_loss(config, seg_crit, cls_crit, masks, outputs, label, logits):
    seg_loss, cls_loss = apply_criterion_multitask_segmentation_classification(
        seg_crit, masks, outputs, cls_crit, label, logits, config["loss"]["inversely_weighted"])
    alpha = config["training"]["alpha"]
    return alpha * seg_loss + (1 - alpha) * cls_loss


def _train_centralized(config, train_loader, val_loader, device, run_path, fold, runtime_events):
    model, optimizer, seg_crit, cls_crit, scheduler = load_multitask_experiment_artefacts(
        config["data"], config["model"], config["optimizer"], config["loss"], 0, None)
    model = model.to(device)
    precision = precision_policy(config, device)
    n_classes = len(config["data"]["classes"])

    best_val, patience = float("inf"), 0
    best_path = Path(run_path) / f"fold_{fold}" / "centralized_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        with runtime_events.measure(
            "train_epoch", setup="centralized", fold=fold, epoch=epoch
        ) as event:
            model.train()
            examples, batches = 0, 0
            for data in train_loader:
                inp = precision.move(data["image"])
                masks = precision.move(data["mask"])
                label = precision.move(data["label"])
                if n_classes > 2:
                    label = F.one_hot(label.flatten().to(torch.int64), n_classes).to(torch.float)
                optimizer.zero_grad(set_to_none=True)
                with precision.autocast():
                    logits, outputs = model(inp)
                    loss = _multitask_loss(
                        config, seg_crit, cls_crit, masks, outputs, label, logits
                    )
                precision.ensure_finite(loss, f"centralized fold={fold} epoch={epoch}")
                loss.backward()
                optimizer.step()
                examples += int(inp.shape[0])
                batches += 1
            event.update(examples=examples, batches=batches)

        with runtime_events.measure(
            "validation_epoch", setup="centralized", fold=fold, epoch=epoch
        ) as event:
            model.eval()
            val_loss, nb, val_examples = 0.0, 0, 0
            with torch.inference_mode():
                for data in val_loader:
                    inp = precision.move(data["image"])
                    masks = precision.move(data["mask"])
                    label = precision.move(data["label"])
                    if n_classes > 2:
                        label = F.one_hot(label.flatten().to(torch.int64), n_classes).to(torch.float)
                    with precision.autocast():
                        logits, outputs = model(inp)
                        loss = _multitask_loss(
                            config, seg_crit, cls_crit, masks, outputs, label, logits
                        )
                    precision.ensure_finite(loss, f"centralized validation fold={fold}")
                    val_loss += loss.item()
                    nb += 1
                    val_examples += int(inp.shape[0])
            event.update(examples=val_examples, batches=nb)
        val_loss /= max(nb, 1)
        scheduler.step() if config["optimizer"]["scheduler"] == "cosine" else scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save({"model_state": model.state_dict(), "val_loss": best_val}, best_path)
        else:
            patience += 1
        if patience > config["training"]["max_patience"]:
            logging.info(f"[centralized fold {fold}] early stop at epoch {epoch} (val {best_val:.4f})")
            break

    logging.info(f"[centralized fold {fold}] best val loss {best_val:.4f}")
    return best_path


def _build_model(config, device):
    n_aug = sum(v for v in config["data"]["augmentation"].values())
    return init_multitask_model(
        architecture=config["model"]["architecture"], sequences=config["model"]["sequences"] + n_aug,
        regions=1, n_classes=len(config["data"]["classes"]), width=config["model"]["width"],
        deep_supervision=config["model"]["deep_supervision"], save_folder=None).to(device)


def run(config_path="./src/config.yaml", run_path=None):
    init_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup = "centralized"

    with open(config_path) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
    fed, data_cfg, train_cfg = config["federated"], config["data"], config["training"]
    evaluation_settings(train_cfg)
    evaluation = evaluation_metadata(train_cfg)

    seed_everything(train_cfg["seed"], cuda_benchmark=train_cfg["cuda_benchmark"])
    dev_str = fed.get("device", "auto")
    device = resolve_device(device_setup() if dev_str == "auto" else dev_str)
    precision = precision_policy(config, device)

    if run_path is None:
        run_path = f"runs/{timestamp}_CENTRALIZED_{config['model']['architecture']}"
    Path(run_path).mkdir(parents=True, exist_ok=True)
    init_log(log_name=str(Path(run_path) / "execution.log"))
    with open(f"{run_path}/config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    gpu_telemetry = start_gpu_telemetry(config, run_path)
    runtime_events = RuntimeEvents(Path(run_path) / "runtime_events.csv", device)

    partition_file = paths.require_partition_file(data_cfg)
    validate_master_splits(pd.read_csv(partition_file, usecols=["fold", "split"]), train_cfg)

    roster = [(r.client_id, r.task) for r in list_clients(partition_file).itertuples()]
    n_classes = len(data_cfg["classes"])

    test_rows, pred_frames = [], []
    for fold in range(train_cfg["CV"]):
        logging.info(f"\n\n*************  FOLD {fold}  (centralized)  *************\n")
        train_loader, val_loader = _centralized_loaders(config, partition_file, fold)
        best_path = _train_centralized(
            config, train_loader, val_loader, device, run_path, fold, runtime_events
        )

        model = _build_model(config, device)
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        for client_id, task in roster:
            test_loader = build_client_loader(
                partition_file, fold, client_id, "test",
                batch_size=runtime_config(config)["inference_batch_size"],
                augmentations=data_cfg["augmentation"],
                loader_options=dataloader_kwargs(config),
            )
            with runtime_events.measure(
                "test", setup=setup, fold=fold, client_id=client_id
            ) as event:
                metrics, preds = unified_eval.evaluate(
                    model, test_loader, task, n_classes, device, precision=precision
                )
                event.update(examples=len(test_loader.dataset), batches=len(test_loader))
            test_rows.append({
                **evaluation,
                "setup": setup,
                "fold": fold,
                "client_id": client_id,
                "task": task,
                **metrics,
            })
            if preds is not None:
                for key, value in reversed(list(evaluation.items())):
                    preds.insert(0, key, value)
                preds.insert(0, "client_id", client_id)
                preds.insert(0, "fold", fold)
                preds.insert(0, "setup", setup)
                pred_frames.append(preds)

    df = pd.DataFrame(test_rows)
    df.to_csv(f"{run_path}/centralized_test_results.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(f"{run_path}/centralized_cls_predictions.csv", index=False)
    for task in df["task"].unique():
        sub, col = df[df["task"] == task], {"seg": "dice", "cls": "acc"}.get(task, "dice")
        mean_value, std_value = sub[col].mean(), sub[col].std()
        estimate = f"{mean_value:.4f}" if pd.isna(std_value) else f"{mean_value:.4f} ± {std_value:.4f}"
        logging.info(f"[centralized] task={task}: {col} {estimate} (n={len(sub)})")
    logging.info(f"Saved {run_path}/centralized_test_results.csv | total time {time.perf_counter() - init_time:.2f}s")
    runtime_events.write()
    gpu_telemetry.stop()


if __name__ == "__main__":
    run()
