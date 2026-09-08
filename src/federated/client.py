"""Flower client for task- and dataset-personalized FedPer training.

A client owns one or more tasks. The task set is read from the master partition -- never inferred
from ``client_id`` -- so the partition stays the single source of truth for ownership.
"""

import hashlib
import csv
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from flwr.client import NumPyClient
from flwr.common import Context

from src.dataset.federated_dataloader import (
    build_client_loader,
    client_tasks,
    resolve_class_weights,
)
from src.federated import local_trainer
from src.federated.config import (
    aggregation_config,
    dataset_config,
    local_training_config,
    training_telemetry_config,
)
from src.federated.model_split import (
    get_personalized_state,
    get_shared_state,
    set_personalized_state,
    set_shared_state,
)
from src.utils.experiment_init import (
    init_criterion_classification,
    init_criterion_segmentation,
    init_multitask_model,
    init_optimizer,
)
from src.utils.miscellany import seed_everything
from src.utils.training_runtime import (
    configure_worker_threads,
    dataloader_kwargs,
    precision_policy,
)


def stable_client_seed(base_seed, fold, client_id, phase="initialization", server_round=0):
    """Derive a process-independent seed shared by federated and local-only runs.

    Python's built-in ``hash`` is deliberately randomized between processes, so use a stable
    digest.  The setup name is intentionally absent: paired federated/local-only clients must
    start identically and sample the same shuffles/geometric transforms in each round.
    """
    payload = f"{int(base_seed)}|{int(fold)}|{client_id}|{phase}|{int(server_round)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def resolve_device(requested):
    """Resolve the device inside the Ray worker, where GPU visibility can differ."""
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        logging.warning(
            "device='cuda' requested but no GPU is visible to this worker; falling back to CPU"
        )
    return "cpu"


def _preserve_rng_state(function):
    """Run observational validation without perturbing later stochastic training."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return function()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class FederatedClient(NumPyClient):
    def __init__(self, client_id, dataset, task, fold, config, device, partition_file, run_dir,
                 standalone=False):
        self.client_id = client_id
        self.dataset = dataset
        # ``task`` is the roster hint; the partition decides the real ownership. A multi-task
        # client reports the joined name so telemetry keeps one group per task topology.
        self.tasks = client_tasks(partition_file, fold, client_id, dataset=dataset) or (task,)
        self.task = "+".join(self.tasks) if len(self.tasks) > 1 else self.tasks[0]
        self.multitask = len(self.tasks) > 1
        self.fold = fold
        self.standalone = standalone
        self.setup = "standalone" if standalone else "federated"
        self.device = resolve_device(device)
        self.config = config
        configure_worker_threads(config)
        self.precision = precision_policy(config, self.device)
        self.data_cfg = dataset_config(config, dataset)
        self.num_classes = len(self.data_cfg["classes"])
        self.share_stem = config["federated"].get("share_stem", True)
        self.inversely_weighted = config["loss"]["inversely_weighted"]
        local_training = local_training_config(config)
        self.local_training_mode = str(local_training.get("mode", "epochs")).lower()
        self.local_epochs = int(
            local_training["local_epochs"]
        )
        self.steps_per_round = int(local_training.get("steps_per_round", 10))
        self.telemetry = training_telemetry_config(config)
        self.base_seed = int(config["training"]["seed"])
        self.data_order_seed = stable_client_seed(
            self.base_seed, fold, client_id, phase="data_order"
        )
        self.cuda_benchmark = bool(config["training"].get("cuda_benchmark", False))
        # Ray workers are separate processes.  Seed before constructing the model so local stems
        # and heads are exactly paired between the federated and local-only arms.
        self.initial_seed = stable_client_seed(self.base_seed, fold, client_id)
        seed_everything(self.initial_seed, cuda_benchmark=self.cuda_benchmark)

        augmentations = self.data_cfg.get("augmentation", {})
        n_aug = sum(bool(value) for value in augmentations.values())
        self.model = init_multitask_model(
            architecture=config["model"]["architecture"],
            sequences=self.data_cfg["channels"] + n_aug,
            regions=1,
            n_classes=self.num_classes,
            width=config["model"]["width"],
            deep_supervision=config["model"]["deep_supervision"],
            save_folder=None,
        ).to(self.device)
        self.optimizer = init_optimizer(
            self.model, config["optimizer"]["opt"], config["optimizer"]["lr"]
        )
        self.seg_criterion = init_criterion_segmentation(config["loss"]["function"])

        weighting_mode = self.data_cfg.get("class_weighting", "none")
        self.class_weights = resolve_class_weights(
            partition_file=partition_file,
            fold=fold,
            client_id=client_id,
            dataset=dataset,
            classes=self.data_cfg["classes"],
            mode=weighting_mode,
        ) if "cls" in self.tasks else None
        self.classification_criterion_name = self.data_cfg.get(
            "classification_criterion", config["loss"]["classification_criterion"]
        )
        self.focal_gamma = float(
            self.data_cfg.get("focal_gamma", config["loss"].get("focal_gamma", 2.0))
        )
        self.cls_criterion = init_criterion_classification(
            n_classes=self.num_classes,
            classes_weighted=self.data_cfg.get("classes_weighted"),
            class_weights=self.class_weights,
            classification_criterion=self.classification_criterion_name,
            device=self.device,
            focal_gamma=self.focal_gamma,
        )

        oversampling_cfg = self.data_cfg.get("oversampling")
        if not isinstance(oversampling_cfg, dict):
            # In the legacy data block this key is a single bool; federated per-task overrides
            # remain authoritative when present.
            oversampling_cfg = config["federated"].get(
                "oversampling", {"seg": bool(oversampling_cfg), "cls": bool(oversampling_cfg)}
            )
        oversampling = any(bool(oversampling_cfg.get(name, False)) for name in self.tasks)
        # Task weights live in TWO places on purpose: as lambda_t inside the local objective, where
        # the gradients are separable and the weighting is real, and as a mass factor on the server
        # (see FedPerStrategy._base_weight). A multi-task client's trunk update is one entangled
        # object, so it cannot be decomposed per task at aggregation time.
        aggregation = aggregation_config(config)
        self.task_lambdas = _task_lambdas(aggregation["task_weights"], self.tasks)
        common = dict(
            partition_file=partition_file,
            fold=fold,
            client_id=client_id,
            dataset=dataset,
            channels=self.data_cfg["channels"],
            classes=self.data_cfg["classes"],
            batch_size=self.data_cfg["batch_size"],
            max_samples=config["federated"].get("max_samples_per_split"),
            loader_options=dataloader_kwargs(config, federated=True),
        )
        self.train_loader = build_client_loader(
            split="train",
            transforms=_default_transforms(self.data_cfg.get("transforms", {})),
            augmentations=augmentations,
            oversampling=oversampling,
            local_training_mode=self.local_training_mode,
            steps_per_round=self.steps_per_round,
            sampling_seed=self.data_order_seed,
            tasks=self.tasks,
            **common,
        )
        self.val_loader = build_client_loader(
            split="val", augmentations=None, oversampling=False, tasks=self.tasks, **common
        )
        self.n_train = self.train_loader.effective_num_samples
        self.n_train_raw = self.train_loader.raw_num_samples
        self.task_mass = dict(getattr(self.train_loader, "task_mass", {}) or {})

        self.state_dir = Path(run_dir) / f"fold_{fold}" / f"client_{client_id}"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.pt"
        self.history_path = self.state_dir / "training_history.csv"
        self._write_metadata(weighting_mode)

    def _write_metadata(self, weighting_mode):
        weights = self.class_weights
        if hasattr(weights, "detach"):
            weights = weights.detach().cpu().tolist()
        elif hasattr(weights, "tolist"):
            weights = weights.tolist()
        metadata = {
            "client_id": self.client_id,
            "dataset": self.dataset,
            "task": self.task,
            "tasks": list(self.tasks),
            "client_topology": "multi_task" if self.multitask else "single_task",
            "task_lambdas": {name: float(v) for name, v in self.task_lambdas.items()},
            "task_mass": {name: int(v) for name, v in self.task_mass.items()},
            "class_names": list(self.data_cfg["classes"]),
            "class_weighting": weighting_mode,
            "class_weights": weights,
            "classification_criterion": self.classification_criterion_name,
            "focal_gamma": self.focal_gamma,
            "channels": self.data_cfg["channels"],
            "share_stem": self.share_stem,
            "initial_seed": self.initial_seed,
            "data_order_seed": self.data_order_seed,
            "local_training_mode": self.local_training_mode,
            "local_epochs": self.local_epochs,
            "steps_per_round": self.steps_per_round,
            "raw_train_examples": self.n_train_raw,
            "effective_train_examples": self.n_train,
            "round_seed_policy": "sha256(training.seed, fold, client_id, phase, round)",
            "final_checkpoint_policy": (
                "last_round_local_full_model"
                if self.standalone
                else "last_round_global_shared_plus_latest_personalized"
            ),
            "best_checkpoint_scope": "disabled_post_aggregation_validation_only",
            "training_telemetry": self.telemetry,
        }
        with (self.state_dir / "metadata.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)

    def _decorate_history(self, rows, server_round, phase, split):
        decorated = []
        for row in rows:
            decorated.append({
                "setup": self.setup,
                "fold": int(self.fold),
                "round": int(server_round),
                "dataset": self.dataset,
                "client_id": self.client_id,
                "task": row["task"],
                "phase": phase,
                "split": split,
                "unit_type": row["unit_type"],
                "unit_index": int(row["unit_index"]),
                "metric_name": row["metric_name"],
                "value": float(row["value"]),
                "n_samples": int(row.get("n_samples", 0)),
                "optimizer_steps": int(row.get("optimizer_steps", 0)),
                "examples_processed": int(row.get("examples_processed", 0)),
                "duration_seconds": float(row.get("duration_seconds", 0.0)),
                "examples_per_second": float(row.get("examples_per_second", 0.0)),
                "cuda_peak_allocated_mb": float(row.get("cuda_peak_allocated_mb", 0.0)),
                "cuda_peak_reserved_mb": float(row.get("cuda_peak_reserved_mb", 0.0)),
            })
        return decorated

    def _evaluation_history(self, result, unit_type, unit_index):
        rows = []
        task_losses = result.get("task_losses", {})
        if self.multitask:
            rows.append({
                "task": "combined", "metric_name": "loss", "value": result["loss"],
                "n_samples": result.get("n", 0),
            })
        for task in self.tasks:
            if task in task_losses:
                rows.append({
                    "task": task, "metric_name": "loss", "value": task_losses[task],
                    "n_samples": result.get("n", 0),
                })
            for metric_name, value in result.get("metrics", {}).get(task, {}).items():
                rows.append({
                    "task": task, "metric_name": metric_name, "value": value,
                    "n_samples": result.get("n", 0),
                })
        for row in rows:
            row.update({
                "unit_type": unit_type,
                "unit_index": int(unit_index),
                "optimizer_steps": 0,
                "examples_processed": 0,
            })
        return rows

    def _training_round_history(self, result, server_round, performance=None):
        rows = []
        performance = dict(performance or {})
        if self.multitask:
            rows.append({
                "task": "combined", "metric_name": "loss", "value": result["loss"],
                "n_samples": self.n_train,
            })
        for task in self.tasks:
            rows.append({
                "task": task, "metric_name": "loss",
                "value": result["task_losses"][task], "n_samples": self.task_mass.get(task, 0),
            })
            for metric_name, value in result["task_metrics"].get(task, {}).items():
                rows.append({
                    "task": task, "metric_name": metric_name, "value": value,
                    "n_samples": self.task_mass.get(task, 0),
                })
        for row in rows:
            row.update({
                "unit_type": "round", "unit_index": int(server_round),
                "optimizer_steps": int(result["optimizer_steps"]),
                "examples_processed": int(result["examples_processed"]),
                **performance,
            })
        return rows

    def _append_history(self, rows):
        """Atomically replace duplicate measurement keys for this client's durable history."""
        if not self.telemetry["enabled"] or not rows:
            return
        fieldnames = [
            "setup", "fold", "round", "dataset", "client_id", "task", "phase", "split",
            "unit_type", "unit_index", "metric_name", "value", "n_samples",
            "optimizer_steps", "examples_processed",
            "duration_seconds", "examples_per_second",
            "cuda_peak_allocated_mb", "cuda_peak_reserved_mb",
        ]
        existing = []
        if self.history_path.exists():
            with self.history_path.open(newline="", encoding="utf-8") as stream:
                existing = list(csv.DictReader(stream))
        key_fields = (
            "fold", "round", "dataset", "client_id", "task", "phase", "split",
            "unit_type", "unit_index", "metric_name",
        )
        replacement_keys = {
            tuple(str(row[field]) for field in key_fields) for row in rows
        }
        kept = [
            row for row in existing
            if tuple(str(row[field]) for field in key_fields) not in replacement_keys
        ]
        combined = kept + rows
        combined.sort(key=lambda row: (
            int(row["fold"]), int(row["round"]), str(row["phase"]), str(row["split"]),
            int(row["unit_index"]), str(row["task"]), str(row["metric_name"]),
        ))
        temporary = self.history_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)
        temporary.replace(self.history_path)

    @staticmethod
    def _scalar_metrics(prefix, result):
        metrics = {f"{prefix}_loss": float(result["loss"])}
        for task, value in result.get("task_losses", {}).items():
            metrics[f"{prefix}_loss_{task}"] = float(value)
        for task, task_metrics in result.get("metrics", {}).items():
            for metric_name, value in task_metrics.items():
                metrics[f"{prefix}_{metric_name}_{task}"] = float(value)
        return metrics

    # ---- local state persistence ----------------------------------------------------------
    def _load_state(self):
        if self.state_path.exists():
            state = torch.load(self.state_path, map_location=self.device)
            if self.standalone:
                self.model.load_state_dict(state["model"])
            else:
                set_personalized_state(
                    self.model, state["personalized"], share_stem=self.share_stem
                )
            self.optimizer.load_state_dict(state["optimizer"])
        return self.state_path.exists()

    def _save_state(self, server_round):
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "round": int(server_round),
        }
        if self.standalone:
            payload["model"] = self.model.state_dict()
        else:
            payload["personalized"] = get_personalized_state(
                self.model, share_stem=self.share_stem
            )
        torch.save(payload, self.state_path)

    # ---- Flower API -----------------------------------------------------------------------
    def get_parameters(self, config):
        return get_shared_state(self.model, share_stem=self.share_stem)

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))
        seed_everything(
            stable_client_seed(
                self.base_seed, self.fold, self.client_id, phase="fit", server_round=server_round
            ),
            cuda_benchmark=self.cuda_benchmark,
        )

        has_local_state = self._load_state()
        # Both arms receive the exact same initial shared trunk.  On later rounds the standalone
        # arm resumes its own full model, whereas federated clients accept the new global trunk.
        if not self.standalone or not has_local_state:
            set_shared_state(self.model, parameters, share_stem=self.share_stem)

        if self.local_training_mode == "steps":
            batch_sampler = self.train_loader.batch_sampler
            if not hasattr(batch_sampler, "set_round"):
                raise RuntimeError("steps mode requires a round-addressable train batch sampler")
            batch_sampler.set_round(max(server_round, 1))

        record_detail = (
            self.telemetry["enabled"]
            and self.telemetry["granularity"] == "epoch_and_round"
        )

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        fit_started = time.perf_counter()
        train_result = local_trainer.train_local(
            self.model, self.train_loader, self.optimizer, self.task, self.device,
            self.local_epochs, self.num_classes, self.seg_criterion, self.cls_criterion,
            self.inversely_weighted, training_mode=self.local_training_mode,
            steps_per_round=self.steps_per_round,
            tasks=self.tasks, task_lambdas=self.task_lambdas,
            epoch_end_callback=None,
            record_step_history=record_detail and self.local_training_mode == "steps",
            precision=self.precision,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - fit_started
        logging.info(
            "[%s] round=%s local_training=%s optimizer_steps=%s examples_processed=%s",
            self.client_id, server_round, self.local_training_mode,
            train_result["optimizer_steps"], train_result["examples_processed"],
        )
        self._save_state(server_round)
        history_rows = []
        if record_detail:
            phase = "local_epoch" if self.local_training_mode == "epochs" else "local_step"
            history_rows.extend(self._decorate_history(
                train_result.get("history", []), server_round, phase, "train"
            ))
        history_rows.extend(self._decorate_history(
            self._training_round_history(train_result, server_round, {
                "duration_seconds": fit_seconds,
                "examples_per_second": (
                    train_result["examples_processed"] / fit_seconds if fit_seconds else 0.0
                ),
                "cuda_peak_allocated_mb": (
                    torch.cuda.max_memory_allocated() / 2**20 if self.device == "cuda" else 0.0
                ),
                "cuda_peak_reserved_mb": (
                    torch.cuda.max_memory_reserved() / 2**20 if self.device == "cuda" else 0.0
                ),
            }),
            server_round,
            "post_local_round",
            "train",
        ))
        self._append_history(history_rows)
        metrics = {
            "dataset": self.dataset,
            "task": self.task,
            "client_id": self.client_id,
            "local_training_mode": self.local_training_mode,
            "train_loss": float(train_result["loss"]),
            "optimizer_steps": int(train_result["optimizer_steps"]),
            "examples_processed": int(train_result["examples_processed"]),
            "raw_num_examples": int(self.n_train_raw),
            "effective_num_examples": int(self.n_train),
            "fit_seconds": float(fit_seconds),
            "examples_per_second": float(
                train_result["examples_processed"] / fit_seconds if fit_seconds else 0.0
            ),
            "cuda_peak_allocated_mb": float(
                torch.cuda.max_memory_allocated() / 2**20 if self.device == "cuda" else 0.0
            ),
            "cuda_peak_reserved_mb": float(
                torch.cuda.max_memory_reserved() / 2**20 if self.device == "cuda" else 0.0
            ),
        }
        metrics.update(self._scalar_metrics("train", {
            "loss": train_result["loss"],
            "task_losses": train_result["task_losses"],
            "metrics": train_result["task_metrics"],
        }))
        # Supervision mass per task is what lets the server weight a client by what it actually
        # contributes. A single-task client reports its whole slice under its own task and zero on
        # the other, which reproduces the historical `num_examples * task_weight[task]` exactly.
        metrics.update({f"task_mass_{name}": float(value) for name, value in self.task_mass.items()})
        metrics.update({
            f"task_batches_{name}": float(value)
            for name, value in train_result.get("task_batches", {}).items()
        })
        return get_shared_state(self.model, self.share_stem), self.n_train, metrics

    def evaluate(self, parameters, config):
        self._load_state()
        if not self.standalone:
            set_shared_state(self.model, parameters, share_stem=self.share_stem)
        val = local_trainer.evaluate_local(
            self.model, self.val_loader, self.task, self.device, self.num_classes,
            self.seg_criterion, self.cls_criterion, self.inversely_weighted,
            tasks=self.tasks, task_lambdas=self.task_lambdas,
            precision=self.precision,
        )
        server_round = int(config.get("server_round", 0))
        self._append_history(self._decorate_history(
            self._evaluation_history(val, "round", server_round),
            server_round,
            "post_aggregation_round",
            "val",
        ))
        metrics = {
            "dataset": self.dataset,
            "task": self.task,
            "client_id": self.client_id,
            "val_metric": float(val["metric"]),
        }
        metrics.update(self._scalar_metrics("val", val))
        metrics.update({
            f"val_metric_{name}": float(val[f"metric_{name}"])
            for name in self.tasks
            if f"metric_{name}" in val
        })
        metrics.update({f"task_mass_{name}": float(value) for name, value in self.task_mass.items()})
        return float(val["loss"]), val["n"], metrics


def _task_lambdas(task_weights, tasks):
    """Normalise the configured task weights over the tasks this client actually owns."""
    raw = {name: float(task_weights.get(name, 1.0)) for name in tasks}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(f"Task weights for {list(tasks)} must sum to a positive value")
    return {name: value / total for name, value in raw.items()}


def _default_transforms(settings=None):
    from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomVerticalFlip

    settings = settings or {}
    return torch.nn.Sequential(
        RandomHorizontalFlip(p=float(settings.get("horizontal_flip", 0.5))),
        RandomVerticalFlip(p=float(settings.get("vertical_flip", 0.5))),
        RandomRotation(degrees=360),
    )


def build_client_fn(config, device, partition_file, run_dir, roster, fold, standalone=False):
    """Return a Flower ``client_fn`` mapping partition id to (client, dataset, task)."""
    def client_fn(context: Context):
        cid = int(context.node_config.get("partition-id", context.node_id))
        # Roster entries carry the client's task tuple; the client re-reads ownership from the
        # partition, so the first task is only a hint for legacy single-task rosters.
        client_id, dataset, tasks = roster[cid]
        hint = tasks[0] if isinstance(tasks, (tuple, list)) else tasks
        client = FederatedClient(
            client_id, dataset, hint, fold, config, device, partition_file, run_dir, standalone
        )
        return client.to_client()

    return client_fn
