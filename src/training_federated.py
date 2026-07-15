"""
Federated (FedPer) multi-task training orchestrator.

For each cross-validation fold it spins up a Flower simulation in which every client owns a single
task (seg or cls), trains the full MTnnUNet locally, and shares only its encoder. The custom
FedPerStrategy averages the encoders (sample x task weighted); personalized heads stay on disk.
After the rounds finish, each client is evaluated on its own held-out test split using its best
snapshot, and metrics are aggregated per task / per fold / overall.

Run with:  python -m src.training_federated
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

from src.dataset import paths
from src.dataset.federated_dataloader import build_client_loader, list_clients
from src.federated import unified_eval
from src.federated.client import build_client_fn
from src.federated.server import FedPerStrategy
from src.utils.experiment_init import device_setup, init_multitask_model
from src.utils.miscellany import init_log, seed_everything


class _DropFlwrDeprecation(logging.Filter):
    """Drop flwr's DEPRECATED-FEATURE log spam (e.g. start_simulation) while keeping real
    warnings. Migrating to the new run_simulation/ClientApp API would remove the source, but
    that is a larger refactor; the classic simulation API still works in flwr 1.30."""

    def filter(self, record):
        return "DEPRECATED FEATURE" not in record.getMessage()


def _build_model(config, device):
    n_aug = sum(v for v in config["data"]["augmentation"].values())
    return init_multitask_model(
        architecture=config["model"]["architecture"],
        sequences=config["model"]["sequences"] + n_aug,
        regions=1,
        n_classes=len(config["data"]["classes"]),
        width=config["model"]["width"],
        deep_supervision=config["model"]["deep_supervision"],
        save_folder=None,
    ).to(device)


def _test_client(config, device, partition_file, run_dir, fold, client_id, task, setup):
    """Evaluate one client's best snapshot on its held-out test split via the shared unified_eval.
    Returns (metrics_row, cls_predictions_df_or_None)."""
    model = _build_model(config, device)
    best_path = Path(run_dir) / f"fold_{fold}" / f"client_{client_id}" / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    else:
        logging.warning(f"No best.pt for {client_id} (fold {fold}); using initial weights")

    num_classes = len(config["data"]["classes"])
    test_loader = build_client_loader(partition_file, fold, client_id, "test",
                                      batch_size=1, augmentations=config["data"]["augmentation"])
    metrics, preds = unified_eval.evaluate(model, test_loader, task, num_classes, device)
    row = {"setup": setup, "fold": fold, "client_id": client_id, "task": task, **metrics}
    if preds is not None:
        preds.insert(0, "client_id", client_id)
        preds.insert(0, "fold", fold)
        preds.insert(0, "setup", setup)
    return row, preds


def run(config_path="./src/config.yaml"):
    init_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(config_path) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
    fed, data_cfg, train_cfg = config["federated"], config["data"], config["training"]

    seed_everything(train_cfg["seed"], cuda_benchmark=train_cfg["cuda_benchmark"])
    dev_str = fed.get("device", "auto")
    device = device_setup() if dev_str == "auto" else dev_str

    standalone = fed.get("standalone", False)
    setup = "standalone" if standalone else "federated"
    run_path = f"runs/{timestamp}_{setup.upper()}_{config['model']['architecture']}_" \
               f"{fed['n_clients_seg']}seg_{fed['n_clients_cls']}cls"
    Path(run_path).mkdir(parents=True, exist_ok=True)
    init_log(log_name=f"./{run_path}/execution.log")
    logging.getLogger("flwr").addFilter(_DropFlwrDeprecation())
    with open(f"{run_path}/config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # The master partition CSV must be generated ONCE and frozen so every setup shares identical
    # splits. Fail loudly instead of silently regenerating (which would invalidate comparisons).
    partition_file = paths.require_partition_file(data_cfg)

    roster = [(r.client_id, r.task) for r in list_clients(partition_file).itertuples()]
    num_clients = len(roster)
    client_resources = fed.get("client_resources", {"num_cpus": 1, "num_gpus": 0.0})

    test_rows, pred_frames = [], []
    for fold in range(train_cfg["CV"]):
        logging.info(f"\n\n*************  FOLD {fold}  ({setup})  *************\n")
        Path(f"{run_path}/fold_{fold}").mkdir(parents=True, exist_ok=True)

        init_params = ndarrays_to_parameters(
            [p.detach().cpu().numpy() for n, p in _build_model(config, device).state_dict().items()
             if n.startswith(("encoder1", "encoder2", "encoder3", "encoder4", "encoder5", "bottleneck"))])

        strategy = FedPerStrategy(
            task_weights=fed["task_weights"], initial_parameters=init_params,
            fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=num_clients,
            min_evaluate_clients=num_clients, min_available_clients=num_clients)

        start_simulation(
            client_fn=build_client_fn(config, device, partition_file, run_path, roster, fold, standalone),
            num_clients=num_clients, config=ServerConfig(num_rounds=fed["rounds"]),
            strategy=strategy, client_resources=client_resources,
            ray_init_args={"include_dashboard": False, "ignore_reinit_error": True,
                           "num_cpus": fed.get("ray_num_cpus", 2)})

        logging.info(f"[fold {fold}] best round = {strategy.best_round} "
                     f"(mean val loss {strategy.best_mean_val:.4f}); running test phase")
        for client_id, task in roster:
            row, preds = _test_client(config, device, partition_file, run_path, fold, client_id, task, setup)
            test_rows.append(row)
            if preds is not None:
                pred_frames.append(preds)

    _save_results(pd.DataFrame(test_rows), pred_frames, run_path, setup)
    logging.info(f"Total {setup} time: {time.perf_counter() - init_time:.2f}s")


def _save_results(df, pred_frames, run_path, setup):
    """One results CSV (+ cls predictions) per setup. Cross-setup aggregation lives in
    src/experiments/analyze.py; here we only log a quick per-task glance."""
    results_csv = f"{run_path}/{setup}_test_results.csv"
    df.to_csv(results_csv, index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(f"{run_path}/{setup}_cls_predictions.csv", index=False)

    glance_col = {"seg": "dice", "cls": "acc"}
    for task in df["task"].unique():
        sub, col = df[df["task"] == task], glance_col.get(task, "dice")
        logging.info(f"[{setup}] task={task}: {col} {sub[col].mean():.4f} ± {sub[col].std():.4f} (n={len(sub)})")
    logging.info(f"Saved {results_csv}")


if __name__ == "__main__":
    run()
