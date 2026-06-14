"""
Flower client for the FedPer multi-task setup.

Each client owns one task. Only the shared encoder travels to/from the server; the personalized
head/decoder and the optimizer state are persisted to disk per client (Flower simulation clients
are ephemeral, so we cannot keep them in memory between rounds). The client also keeps a ``best.pt``
snapshot (full model) of the round with the lowest validation loss -- this is what the final test
phase loads, giving us early-stopping semantics without terminating the Flower loop.
"""

import logging
from pathlib import Path

import torch
from flwr.client import NumPyClient
from flwr.common import Context

from src.dataset.federated_dataloader import build_client_loader
from src.federated import local_trainer
from src.federated.model_split import (get_personalized_state, get_shared_state,
                                       set_personalized_state, set_shared_state)
from src.utils.experiment_init import (init_criterion_classification, init_criterion_segmentation,
                                       init_multitask_model, init_optimizer)


def resolve_device(requested):
    """Resolve the device INSIDE the worker. Ray hides the GPU from an actor that was given
    ``num_gpus=0`` (CUDA_VISIBLE_DEVICES=""), so a fixed 'cuda' from the orchestrator would
    crash here. We re-check availability in this process and fall back to CPU with a warning."""
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        logging.warning("device='cuda' requested but no GPU is visible to this worker "
                        "(set federated.client_resources.num_gpus > 0); falling back to CPU")
    return "cpu"


class FederatedClient(NumPyClient):
    def __init__(self, client_id, task, fold, config, device, partition_file, run_dir):
        self.client_id = client_id
        self.task = task
        self.fold = fold
        self.device = resolve_device(device)
        self.config = config
        self.num_classes = len(config["data"]["classes"])
        self.inversely_weighted = config["loss"]["inversely_weighted"]
        self.local_epochs = config["federated"]["local_epochs"]

        n_aug = sum(v for v in config["data"]["augmentation"].values())
        self.model = init_multitask_model(
            architecture=config["model"]["architecture"],
            sequences=config["model"]["sequences"] + n_aug,
            regions=1,
            n_classes=self.num_classes,
            width=config["model"]["width"],
            deep_supervision=config["model"]["deep_supervision"],
            save_folder=None,
        ).to(device)
        self.optimizer = init_optimizer(self.model, config["optimizer"]["opt"], config["optimizer"]["lr"])
        self.seg_criterion = init_criterion_segmentation(config["loss"]["function"])
        self.cls_criterion = init_criterion_classification(
            n_classes=self.num_classes,
            classes_weighted=config["data"]["classes_weighted"],
            classification_criterion=config["loss"]["classification_criterion"],
        )

        oversampling = config["federated"]["oversampling"][task]
        transforms = _default_transforms()
        common = dict(partition_file=partition_file, fold=fold, client_id=client_id,
                      batch_size=config["data"]["batch_size"], augmentations=config["data"]["augmentation"])
        self.train_loader = build_client_loader(split="train", transforms=transforms,
                                                oversampling=oversampling, **common)
        self.val_loader = build_client_loader(split="val", **common)
        self.n_train = len(self.train_loader.dataset)

        self.state_dir = Path(run_dir) / f"fold_{fold}" / f"client_{client_id}"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.pt"
        self.best_path = self.state_dir / "best.pt"

    # ---- personalized state persistence -------------------------------------------------
    def _load_personalized(self):
        best_val = float("inf")
        if self.state_path.exists():
            st = torch.load(self.state_path, map_location=self.device)
            set_personalized_state(self.model, st["personalized"])
            self.optimizer.load_state_dict(st["optimizer"])
            best_val = st["best_val"]
        return best_val

    def _save_personalized(self, best_val):
        torch.save({"personalized": get_personalized_state(self.model),
                    "optimizer": self.optimizer.state_dict(),
                    "best_val": best_val}, self.state_path)

    # ---- Flower API ---------------------------------------------------------------------
    def get_parameters(self, config):
        return get_shared_state(self.model)

    def fit(self, parameters, config):
        set_shared_state(self.model, parameters)
        best_val = self._load_personalized()

        local_trainer.train_local(
            self.model, self.train_loader, self.optimizer, self.task, self.device,
            self.local_epochs, self.num_classes, self.seg_criterion, self.cls_criterion,
            self.inversely_weighted)
        val = local_trainer.evaluate_local(
            self.model, self.val_loader, self.task, self.device, self.num_classes,
            self.seg_criterion, self.cls_criterion, self.inversely_weighted)

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({"model_state": self.model.state_dict(), "val_loss": best_val}, self.best_path)

        self._save_personalized(best_val)
        metrics = {"task": self.task, "client_id": self.client_id,
                   "val_loss": float(val["loss"]), "val_metric": float(val["metric"])}
        return get_shared_state(self.model), self.n_train, metrics

    def evaluate(self, parameters, config):
        set_shared_state(self.model, parameters)
        self._load_personalized()
        val = local_trainer.evaluate_local(
            self.model, self.val_loader, self.task, self.device, self.num_classes,
            self.seg_criterion, self.cls_criterion, self.inversely_weighted)
        metrics = {"task": self.task, "client_id": self.client_id, "val_metric": float(val["metric"])}
        return float(val["loss"]), val["n"], metrics


def _default_transforms():
    from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
    return torch.nn.Sequential(RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomRotation(degrees=360))


def build_client_fn(config, device, partition_file, run_dir, roster, fold):
    """Return a Flower ``client_fn(context)`` mapping the partition id to a roster entry."""
    def client_fn(context: Context):
        cid = int(context.node_config.get("partition-id", context.node_id))
        client_id, task = roster[cid]
        client = FederatedClient(client_id, task, fold, config, device, partition_file, run_dir)
        return client.to_client()
    return client_fn
