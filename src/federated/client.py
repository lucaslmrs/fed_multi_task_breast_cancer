"""Flower client for task- and dataset-personalized FedPer training."""

import hashlib
import logging
from pathlib import Path

import torch
import yaml
from flwr.client import NumPyClient
from flwr.common import Context

from src.dataset.federated_dataloader import build_client_loader, resolve_class_weights
from src.federated import local_trainer
from src.federated.config import dataset_config, local_training_config
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


class FederatedClient(NumPyClient):
    def __init__(self, client_id, dataset, task, fold, config, device, partition_file, run_dir,
                 standalone=False):
        self.client_id = client_id
        self.dataset = dataset
        self.task = task
        self.fold = fold
        self.standalone = standalone
        self.device = resolve_device(device)
        self.config = config
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
        ) if task == "cls" else None
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
        oversampling = bool(oversampling_cfg.get(task, False))
        common = dict(
            partition_file=partition_file,
            fold=fold,
            client_id=client_id,
            dataset=dataset,
            channels=self.data_cfg["channels"],
            classes=self.data_cfg["classes"],
            batch_size=self.data_cfg["batch_size"],
            max_samples=config["federated"].get("max_samples_per_split"),
        )
        self.train_loader = build_client_loader(
            split="train",
            transforms=_default_transforms(self.data_cfg.get("transforms", {})),
            augmentations=augmentations,
            oversampling=oversampling,
            local_training_mode=self.local_training_mode,
            steps_per_round=self.steps_per_round,
            sampling_seed=self.data_order_seed,
            **common,
        )
        self.val_loader = build_client_loader(
            split="val", augmentations=None, oversampling=False, **common
        )
        self.n_train = self.train_loader.effective_num_samples
        self.n_train_raw = self.train_loader.raw_num_samples

        self.state_dir = Path(run_dir) / f"fold_{fold}" / f"client_{client_id}"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.pt"
        self.best_path = self.state_dir / "best.pt"
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
            "best_checkpoint_scope": "local_post_fit_diagnostic_not_final",
        }
        with (self.state_dir / "metadata.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)

    # ---- local state persistence ----------------------------------------------------------
    def _load_state(self):
        best_val = float("inf")
        if self.state_path.exists():
            state = torch.load(self.state_path, map_location=self.device)
            if self.standalone:
                self.model.load_state_dict(state["model"])
            else:
                set_personalized_state(
                    self.model, state["personalized"], share_stem=self.share_stem
                )
            self.optimizer.load_state_dict(state["optimizer"])
            best_val = state["best_val"]
        return best_val

    def _save_state(self, best_val, server_round):
        payload = {
            "optimizer": self.optimizer.state_dict(),
            "best_val": best_val,
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

        has_local_state = self.state_path.exists()
        best_val = self._load_state()
        # Both arms receive the exact same initial shared trunk.  On later rounds the standalone
        # arm resumes its own full model, whereas federated clients accept the new global trunk.
        if not self.standalone or not has_local_state:
            set_shared_state(self.model, parameters, share_stem=self.share_stem)

        if self.local_training_mode == "steps":
            batch_sampler = self.train_loader.batch_sampler
            if not hasattr(batch_sampler, "set_round"):
                raise RuntimeError("steps mode requires a round-addressable train batch sampler")
            batch_sampler.set_round(max(server_round, 1))

        train_result = local_trainer.train_local(
            self.model, self.train_loader, self.optimizer, self.task, self.device,
            self.local_epochs, self.num_classes, self.seg_criterion, self.cls_criterion,
            self.inversely_weighted, training_mode=self.local_training_mode,
            steps_per_round=self.steps_per_round,
        )
        logging.info(
            "[%s] round=%s local_training=%s optimizer_steps=%s examples_processed=%s",
            self.client_id, server_round, self.local_training_mode,
            train_result["optimizer_steps"], train_result["examples_processed"],
        )
        val = local_trainer.evaluate_local(
            self.model, self.val_loader, self.task, self.device, self.num_classes,
            self.seg_criterion, self.cls_criterion, self.inversely_weighted,
        )

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "val_loss": best_val,
                    "round": server_round,
                    "scope": "local_post_fit_diagnostic_not_final",
                },
                self.best_path,
            )

        self._save_state(best_val, server_round)
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
            "val_loss": float(val["loss"]),
            "val_metric": float(val["metric"]),
        }
        return get_shared_state(self.model, self.share_stem), self.n_train, metrics

    def evaluate(self, parameters, config):
        self._load_state()
        if not self.standalone:
            set_shared_state(self.model, parameters, share_stem=self.share_stem)
        val = local_trainer.evaluate_local(
            self.model, self.val_loader, self.task, self.device, self.num_classes,
            self.seg_criterion, self.cls_criterion, self.inversely_weighted,
        )
        metrics = {
            "dataset": self.dataset,
            "task": self.task,
            "client_id": self.client_id,
            "val_metric": float(val["metric"]),
        }
        return float(val["loss"]), val["n"], metrics


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
        client_id, dataset, task = roster[cid]
        client = FederatedClient(
            client_id, dataset, task, fold, config, device, partition_file, run_dir, standalone
        )
        return client.to_client()

    return client_fn
