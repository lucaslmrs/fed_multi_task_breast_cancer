"""Multi-dataset FedPer training orchestrator.

Every client owns one dataset and one task. Dataset-specific stems and all decoders/heads remain
local; only the configured shared trunk travels through Flower. The same runner implements the
local-only baseline by setting ``federated.standalone``.
"""

import argparse
import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

from src.dataset.federated_dataloader import build_client_loader, list_clients
from src.dataset.splitting import evaluation_metadata, validate_master_splits
from src.federated import unified_eval
from src.federated.client import build_client_fn, stable_client_seed
from src.federated.config import (
    active_datasets,
    aggregation_config,
    dataset_config,
    partition_file,
    validate_federated_config,
)
from src.federated.model_split import (
    get_shared_state,
    set_personalized_state,
    set_shared_state,
    shared_keys,
)
from src.federated.server import FedPerStrategy
from src.utils.experiment_init import device_setup, init_multitask_model
from src.utils.miscellany import init_log, seed_everything


class _DropFlwrDeprecation(logging.Filter):
    def filter(self, record):
        return "DEPRECATED FEATURE" not in record.getMessage()


def _resume_config_signature(config):
    signature = copy.deepcopy(config)
    if signature.get("training", {}).get("CV", 0) > 1:
        signature["training"].pop("holdout_test_size", None)
    return signature


def _resolve_orchestrator_device(requested):
    if requested == "auto":
        return device_setup()
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable to the orchestrator; using CPU")
        return "cpu"
    return requested


def _build_model(config, dataset, device):
    data_cfg = dataset_config(config, dataset)
    n_aug = sum(bool(value) for value in data_cfg.get("augmentation", {}).values())
    return init_multitask_model(
        architecture=config["model"]["architecture"],
        sequences=data_cfg["channels"] + n_aug,
        regions=1,
        n_classes=len(data_cfg["classes"]),
        width=config["model"]["width"],
        deep_supervision=config["model"]["deep_supervision"],
        save_folder=None,
    ).to(device)


def _initial_shared_parameters(config, datasets, device, fold):
    """Build one model per dataset and fail before Flower if shared contracts diverge."""
    # Reset independently for every fold and arm. Orchestrator-side evaluation/model construction
    # after an earlier fold must not perturb the paired initial trunk of the next fold.
    seed_everything(
        stable_client_seed(
            config["training"]["seed"], fold, "__global__", phase="global_initialization"
        ),
        cuda_benchmark=config["training"].get("cuda_benchmark", False),
    )
    share_stem = config["federated"].get("share_stem", True)
    reference_model = _build_model(config, datasets[0], device)
    reference_keys = shared_keys(reference_model, share_stem)
    reference_arrays = get_shared_state(reference_model, share_stem)
    reference_shapes = [array.shape for array in reference_arrays]

    for dataset in datasets[1:]:
        candidate = _build_model(config, dataset, device)
        candidate_keys = shared_keys(candidate, share_stem)
        candidate_shapes = [array.shape for array in get_shared_state(candidate, share_stem)]
        if candidate_keys != reference_keys or candidate_shapes != reference_shapes:
            raise ValueError(
                f"Dataset '{dataset}' has an incompatible shared state. "
                "Use share_stem=false for heterogeneous channel counts."
            )
        del candidate
    return ndarrays_to_parameters(reference_arrays)


def _round_config(server_round):
    """Propagate the round number so every Ray worker can use paired deterministic RNG."""
    return {"server_round": int(server_round)}


def _test_client(
    config,
    device,
    master_file,
    run_dir,
    fold,
    client_id,
    dataset,
    task,
    setup,
    shared_arrays=None,
):
    model = _build_model(config, dataset, device)
    state_path = Path(run_dir) / f"fold_{fold}" / f"client_{client_id}" / "state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"No state.pt for {client_id} (fold {fold}); refusing to score initial weights"
        )
    state = torch.load(state_path, map_location=device)
    expected_round = int(config["federated"]["rounds"])
    if state.get("round") != expected_round:
        raise ValueError(
            f"Client {client_id} state is from round {state.get('round')}, "
            f"expected final round {expected_round}"
        )
    if setup == "federated":
        if shared_arrays is None:
            raise ValueError("Federated final evaluation requires the final global shared trunk")
        set_personalized_state(
            model,
            state["personalized"],
            share_stem=config["federated"].get("share_stem", True),
        )
        set_shared_state(
            model,
            shared_arrays,
            share_stem=config["federated"].get("share_stem", True),
        )
    else:
        model.load_state_dict(state["model"])

    data_cfg = dataset_config(config, dataset)
    test_loader = build_client_loader(
        partition_file=master_file,
        fold=fold,
        client_id=client_id,
        dataset=dataset,
        split="test",
        batch_size=1,
        channels=data_cfg["channels"],
        classes=data_cfg["classes"],
        augmentations=None,
        max_samples=config["federated"].get("max_samples_per_split"),
    )
    metrics, preds = unified_eval.evaluate(
        model, test_loader, task, len(data_cfg["classes"]), device,
        class_names=data_cfg["classes"],
    )
    experiment = config.get("experiment", {})
    evaluation = evaluation_metadata(config["training"])
    identifiers = {
        key: experiment[key]
        for key in ("study_id", "arm_id", "method_id", "seed")
        if key in experiment
    }
    row = {
        **identifiers,
        **evaluation,
        "setup": setup,
        "fold": fold,
        "client_id": client_id,
        "dataset": dataset,
        "task": task,
        **metrics,
    }
    if preds is not None:
        for key, value in reversed(list(evaluation.items())):
            preds.insert(0, key, value)
        for key, value in reversed(list(identifiers.items())):
            preds.insert(0, key, value)
        preds.insert(0, "dataset", dataset)
        preds.insert(0, "client_id", client_id)
        preds.insert(0, "fold", fold)
        preds.insert(0, "setup", setup)
    return row, preds


def _state_is_final(state_path, expected_round):
    if not Path(state_path).exists():
        return False
    try:
        state = torch.load(state_path, map_location="cpu")
    except Exception:
        return False
    return int(state.get("round", -1)) == int(expected_round)


def _completed_fold_state(run_path, fold, roster, standalone, expected_round):
    """Return whether a fold has enough durable state to skip its simulation."""
    fold_dir = Path(run_path) / f"fold_{fold}"
    if not (fold_dir / "aggregation_history.json").exists():
        return False
    if not all(
        _state_is_final(fold_dir / f"client_{client_id}" / "state.pt", expected_round)
        for client_id, _, _ in roster
    ):
        return False
    if standalone:
        return True
    global_path = fold_dir / "global_shared.pt"
    if not global_path.exists():
        return False
    try:
        global_state = torch.load(global_path, map_location="cpu")
    except Exception:
        return False
    return int(global_state.get("round", -1)) == int(expected_round)


def _load_final_shared(run_path, fold, expected_round):
    state = torch.load(
        Path(run_path) / f"fold_{fold}" / "global_shared.pt", map_location="cpu"
    )
    if int(state.get("round", -1)) != int(expected_round):
        raise ValueError(
            f"Fold {fold} global state is from round {state.get('round')}, "
            f"expected {expected_round}"
        )
    return [tensor.detach().cpu().numpy() for tensor in state["arrays"]]


def _preserve_partial_fold(fold_dir):
    fold_dir = Path(fold_dir)
    if not fold_dir.exists() or not any(fold_dir.iterdir()):
        return None
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    preserved = fold_dir.with_name(f"{fold_dir.name}.incomplete_{suffix}")
    if preserved.exists():
        raise RuntimeError(f"Cannot preserve partial fold; target exists: {preserved}")
    fold_dir.rename(preserved)
    return preserved


def run(config_path="./src/config.yaml", *, run_path=None, resume=False):
    init_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    validate_federated_config(config)
    fed, train_cfg = config["federated"], config["training"]
    datasets = active_datasets(config)
    aggregation = aggregation_config(config)

    seed_everything(train_cfg["seed"], cuda_benchmark=train_cfg["cuda_benchmark"])
    device = _resolve_orchestrator_device(fed.get("device", "auto"))
    standalone = fed.get("standalone", False)
    setup = "standalone" if standalone else "federated"

    dataset_tag = "-".join(datasets)
    if run_path is None:
        run_path = Path("runs") / (
            f"{timestamp}_{setup.upper()}_{config['model']['architecture']}_{dataset_tag}"
        )
    else:
        run_path = Path(run_path)
    if run_path.exists() and not resume:
        raise FileExistsError(
            f"Run directory '{run_path}' already exists; pass resume=True to reuse durable folds"
        )
    run_path.mkdir(parents=True, exist_ok=True)
    config_output = run_path / "config.yaml"
    if config_output.exists():
        existing = yaml.safe_load(config_output.read_text(encoding="utf-8"))
        if _resume_config_signature(existing) != _resume_config_signature(config):
            raise ValueError(
                f"Refusing to resume '{run_path}' with a different resolved configuration"
            )
    init_log(log_name=str(run_path / "execution.log"))
    logging.getLogger("flwr").addFilter(_DropFlwrDeprecation())
    with config_output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    master_file = partition_file(config)
    master_frame = pd.read_csv(master_file, usecols=lambda column: column in {"fold", "split"})
    validate_master_splits(master_frame, train_cfg)
    roster_df = list_clients(str(master_file))
    if "dataset" not in roster_df.columns:
        roster_df["dataset"] = config["data"]["dataset"]
    roster_df = roster_df[roster_df["dataset"].isin(datasets)].reset_index(drop=True)
    configured_clients = fed.get("n_clients", {})
    for dataset in datasets:
        for task in ("seg", "cls"):
            expected = configured_clients.get(dataset, {}).get(task)
            if expected is None:
                continue
            actual = roster_df[
                (roster_df["dataset"] == dataset) & (roster_df["task"] == task)
            ]["client_id"].nunique()
            if actual != int(expected):
                raise ValueError(
                    f"Partition roster has {actual} clients for {dataset}/{task}, but config "
                    f"expects {expected}; regenerate the frozen master after changing n_clients"
                )
    client_cap = fed.get("max_clients_per_dataset_task")
    if client_cap is not None:
        if isinstance(client_cap, bool) or int(client_cap) < 1:
            raise ValueError("federated.max_clients_per_dataset_task must be a positive integer")
        roster_df = (
            roster_df.groupby(["dataset", "task"], sort=False, group_keys=False)
            .head(int(client_cap))
            .reset_index(drop=True)
        )
    roster = [
        (row.client_id, row.dataset, row.task) for row in roster_df.itertuples(index=False)
    ]
    if not roster:
        raise ValueError(f"Partition '{master_file}' contains no configured clients")
    num_clients = len(roster)

    client_resources = dict(fed.get("client_resources", {"num_cpus": 1, "num_gpus": 0.0}))
    if device == "cpu" and client_resources.get("num_gpus", 0) > 0:
        logging.warning("GPU resources requested for a CPU run; setting client num_gpus=0")
        client_resources["num_gpus"] = 0.0

    fold_count = train_cfg["CV"]
    if fed.get("max_folds") is not None:
        if isinstance(fed["max_folds"], bool) or int(fed["max_folds"]) < 1:
            raise ValueError("federated.max_folds must be a positive integer")
        fold_count = min(fold_count, int(fed["max_folds"]))

    test_rows, pred_frames = [], []
    for fold in range(fold_count):
        logging.info(f"\n************* FOLD {fold} ({setup}) *************")
        fold_dir = run_path / f"fold_{fold}"
        fold_results_path = fold_dir / f"{setup}_test_results.csv"
        fold_predictions_path = fold_dir / f"{setup}_cls_predictions.csv"
        if resume and fold_results_path.exists():
            logging.info("[fold %s] reusing durable test results", fold)
            reused_results = pd.read_csv(fold_results_path)
            for key, value in evaluation_metadata(train_cfg).items():
                reused_results[key] = value
            test_rows.extend(reused_results.to_dict("records"))
            if fold_predictions_path.exists():
                reused_predictions = pd.read_csv(fold_predictions_path)
                for key, value in evaluation_metadata(train_cfg).items():
                    reused_predictions[key] = value
                pred_frames.append(reused_predictions)
            continue

        expected_round = int(fed["rounds"])
        simulation_complete = resume and _completed_fold_state(
            run_path, fold, roster, standalone, expected_round
        )
        if not simulation_complete:
            if resume:
                preserved = _preserve_partial_fold(fold_dir)
                if preserved is not None:
                    logging.warning("Preserved partial fold at %s", preserved)
            fold_dir.mkdir(parents=True, exist_ok=True)

        final_shared = None
        if simulation_complete:
            logging.info("[fold %s] reusing final-round client and global state", fold)
            if not standalone:
                final_shared = _load_final_shared(run_path, fold, expected_round)
        else:
            strategy = FedPerStrategy(
                task_weights=aggregation["task_weights"],
                dataset_weights=aggregation["dataset_weights"],
                aggregation_mode=aggregation["mode"],
                client_weighting=aggregation["client_weighting"],
                initial_parameters=_initial_shared_parameters(config, datasets, device, fold),
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
                accept_failures=False,
                on_fit_config_fn=_round_config,
                on_evaluate_config_fn=_round_config,
            )

            start_simulation(
                client_fn=build_client_fn(
                    config, device, str(master_file), str(run_path), roster, fold, standalone
                ),
                num_clients=num_clients,
                config=ServerConfig(num_rounds=fed["rounds"]),
                strategy=strategy,
                client_resources=client_resources,
                ray_init_args={
                    "include_dashboard": False,
                    "ignore_reinit_error": True,
                    "num_cpus": fed.get("ray_num_cpus", 2),
                },
            )

            with (fold_dir / "aggregation_history.json").open("w") as stream:
                json.dump(strategy.aggregation_history, stream, indent=2)

            if not standalone:
                if strategy.latest_parameters is None:
                    raise RuntimeError(f"Fold {fold} completed without an aggregated global trunk")
                final_shared = parameters_to_ndarrays(strategy.latest_parameters)
                torch.save(
                    {
                        "round": strategy.latest_round,
                        "keys": shared_keys(
                            _build_model(config, datasets[0], "cpu"),
                            fed.get("share_stem", True),
                        ),
                        "arrays": [torch.as_tensor(array).cpu() for array in final_shared],
                    },
                    fold_dir / "global_shared.pt",
                )

            logging.info(
                f"[fold {fold}] simulation complete; diagnostic aggregate best round="
                f"{strategy.best_round}; evaluating final-round "
                f"{'global trunk + personalized states' if not standalone else 'local models'}"
            )
        fold_rows, fold_pred_frames = [], []
        for client_id, dataset, task in roster:
            row, preds = _test_client(
                config,
                device,
                str(master_file),
                run_path,
                fold,
                client_id,
                dataset,
                task,
                setup,
                shared_arrays=final_shared,
            )
            fold_rows.append(row)
            if preds is not None:
                fold_pred_frames.append(preds)
        pd.DataFrame(fold_rows).to_csv(fold_results_path, index=False)
        if fold_pred_frames:
            pd.concat(fold_pred_frames, ignore_index=True, sort=False).to_csv(
                fold_predictions_path, index=False
            )
        test_rows.extend(fold_rows)
        pred_frames.extend(fold_pred_frames)

    _save_results(pd.DataFrame(test_rows), pred_frames, run_path, setup)
    logging.info(f"Total {setup} time: {time.perf_counter() - init_time:.2f}s")
    return run_path


def _save_results(df, pred_frames, run_path, setup):
    results_csv = Path(run_path) / f"{setup}_test_results.csv"
    df.to_csv(results_csv, index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True, sort=False).to_csv(
            Path(run_path) / f"{setup}_cls_predictions.csv", index=False
        )

    glance_col = {"seg": "dice", "cls": "acc"}
    for (dataset, task), group in df.groupby(["dataset", "task"]):
        metric = glance_col[task]
        scheme = group["evaluation_scheme"].iloc[0]
        unit = "clients in one holdout" if scheme == "holdout" else "client-fold observations"
        mean_value = group[metric].mean()
        std_value = group[metric].std()
        estimate = (
            f"{mean_value:.4f}"
            if pd.isna(std_value)
            else f"{mean_value:.4f} ± {std_value:.4f}"
        )
        logging.info(
            f"[{setup}] dataset={dataset} task={task}: {metric} "
            f"{estimate} (n={len(group)} {unit})"
        )
    logging.info(f"Saved {results_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
