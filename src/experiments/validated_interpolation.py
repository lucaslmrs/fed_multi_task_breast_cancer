"""Validated per-client interpolation between a paired local-only and federated run.

For every client the full model is interpolated, ``theta(l) = (1 - l) * theta_local + l *
theta_federated``, where ``theta_federated`` is the final global shared trunk plus the client's
federated personalized state and ``theta_local`` is its final local-only model. Both arms start
from the same initialization, so ``l = 0`` reproduces local-only and ``l = 1`` reproduces the
federated arm exactly.

``l*`` is chosen on the client's own validation slice by the same Delta_m the test comparison
uses (seg ``dice_positive``, cls ``balanced_acc``, relative to ``l = 0``), ties going to the
smaller ``l``. Only ``l`` in [0, 1] competes; extrapolated values are scored as a dilution
diagnostic. The whole test grid is written too, but it is an oracle and never feeds ``l*``.

Frozen runs are only read. Outputs go to ``--out``.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.dataset.federated_dataloader import build_client_loader, client_tasks, list_clients
from src.federated import unified_eval
from src.federated.config import dataset_config, shared_modules
from src.federated.model_split import set_personalized_state, set_shared_state
from src.training_federated import _build_model, _load_final_shared, _resolve_orchestrator_device
from src.utils.supervision import attach_saved_task_uncertainty
from src.utils.training_runtime import dataloader_kwargs, precision_policy, runtime_config

SELECTABLE = tuple(round(0.1 * i, 1) for i in range(11))
DIAGNOSTIC = (1.25, 1.5)
PRIMARY_METRIC = {"seg": "dice_positive", "cls": "balanced_acc"}
# Keys allowed to differ between the configs of the two arms of one pair.
ARM_KEYS = {("federated", "standalone"), ("experiment", "arm_id"),
            ("experiment", "method_id"), ("experiment", "setup")}


def _load_config(run_dir):
    with open(Path(run_dir) / "config.yaml") as handle:
        return yaml.safe_load(handle)


def check_pair(fed_config, local_config):
    """Refuse a pair whose configs differ in anything but the arm identity."""
    if fed_config["federated"].get("standalone") or not local_config["federated"].get("standalone"):
        raise ValueError("Expected a federated run and a standalone (local-only) run, in that order")
    diffs = []
    for section in sorted(set(fed_config) | set(local_config)):
        left, right = fed_config.get(section), local_config.get(section)
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if left.get(key) != right.get(key) and (section, key) not in ARM_KEYS:
                    diffs.append(f"{section}.{key}")
        elif left != right:
            diffs.append(section)
    if diffs:
        raise ValueError(f"The two runs are not a matched pair; they differ in {diffs}")


def interpolate_state(local_state, fed_state, lam):
    """Interpolate every float tensor; anything else must match and is taken as is."""
    if set(local_state) != set(fed_state):
        raise ValueError("Local and federated states have different keys")
    out = {}
    for key, local in local_state.items():
        fed = fed_state[key]
        if local.shape != fed.shape:
            raise ValueError(f"Shape mismatch for {key}: {tuple(local.shape)} vs {tuple(fed.shape)}")
        if torch.is_floating_point(local):
            out[key] = torch.lerp(local, fed.to(local.dtype), float(lam))
        else:
            if not torch.equal(local, fed):
                raise ValueError(f"Non-float tensor {key} differs between the arms")
            out[key] = local.clone()
    return out


def endpoint_states(config, fed_run, local_run, fold, client_id, dataset, shared_arrays, device):
    """Return ``(model, local_state, fed_state)`` for one client, both as full state dicts."""
    expected_round = int(config["federated"]["rounds"])
    states = {}
    for name, run in (("fed", fed_run), ("local", local_run)):
        state = torch.load(Path(run) / f"fold_{fold}" / f"client_{client_id}" / "state.pt",
                           map_location=device)
        if state.get("round") != expected_round:
            raise ValueError(f"{name} state of {client_id} is from round {state.get('round')}, "
                             f"expected {expected_round}")
        states[name] = state

    model = _build_model(config, dataset, device)
    attach_saved_task_uncertainty(model, states["local"]["model"])
    model.load_state_dict(states["local"]["model"])
    local_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    sharing = shared_modules(config)
    set_personalized_state(model, states["fed"]["personalized"], sharing=sharing)
    set_shared_state(model, shared_arrays, sharing=sharing)
    fed_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, local_state, fed_state


def _loaders(config, master, fold, client_id, dataset, tasks):
    data_cfg = dataset_config(config, dataset)
    runtime = runtime_config(config)
    loaders = {}
    for split in ("val", "test"):
        for task in tasks:
            loaders[split, task] = build_client_loader(
                partition_file=master, fold=fold, client_id=client_id, dataset=dataset,
                split=split, batch_size=runtime["inference_batch_size"],
                channels=data_cfg["channels"], classes=data_cfg["classes"], augmentations=None,
                tasks=[task], max_samples=config["federated"].get("max_samples_per_split"),
                loader_options=dataloader_kwargs(config, federated=True),
            )
    return loaders


def score_grid(config, fed_run, local_run, device, lambdas=SELECTABLE + DIAGNOSTIC):
    """Score every client at every lambda on val and test. One row per (client, task, split, l)."""
    master = config["federated"]["partition_file"]
    precision = precision_policy(config, device)
    fold = 0 if int(config["training"]["CV"]) == 1 else None
    if fold is None:
        raise NotImplementedError("Only CV=1 holdout runs are supported")
    shared_arrays = _load_final_shared(fed_run, fold, int(config["federated"]["rounds"]))
    rows = []
    # The roster has one row per (client, task); a multi-task client is scored once.
    roster = list_clients(master).drop_duplicates("client_id")
    for client in roster.itertuples():
        client_id, dataset = client.client_id, client.dataset
        tasks = client_tasks(master, fold, client_id, dataset=dataset)
        data_cfg = dataset_config(config, dataset)
        model, local_state, fed_state = endpoint_states(
            config, fed_run, local_run, fold, client_id, dataset, shared_arrays, device)
        loaders = _loaders(config, master, fold, client_id, dataset, tasks)
        for lam in lambdas:
            model.load_state_dict(interpolate_state(local_state, fed_state, lam))
            for (split, task), loader in loaders.items():
                metrics, _ = unified_eval.evaluate(
                    model, loader, task, len(data_cfg["classes"]), device,
                    class_names=data_cfg["classes"], precision=precision)
                rows.append({"client_id": client_id, "dataset": dataset, "task": task,
                             "split": split, "lam": lam,
                             "metric": PRIMARY_METRIC[task],
                             "value": metrics[PRIMARY_METRIC[task]],
                             "n": metrics.get("n_test")})
        logging.info("[interp] %s scored on %d lambdas", client_id, len(lambdas))
    return pd.DataFrame(rows)


def relative_delta(grid):
    """Add ``rel`` = % change of the primary metric against l = 0, per client/task/split."""
    base = grid[grid["lam"] == 0.0].set_index(["client_id", "task", "split"])["value"]
    grid = grid.copy()
    ref = grid.set_index(["client_id", "task", "split"]).index.map(base)
    grid["rel"] = (grid["value"] - ref) / ref * 100.0
    return grid


def select_lambda(grid, selectable=SELECTABLE):
    """``l*`` per client: max mean val Delta_m over its tasks, ties to the smaller ``l``."""
    val = grid[(grid["split"] == "val") & grid["lam"].isin(selectable)]
    score = val.groupby(["client_id", "lam"])["rel"].mean().reset_index()
    score = score.sort_values(["client_id", "rel", "lam"], ascending=[True, False, True])
    best = score.drop_duplicates("client_id")
    return best.rename(columns={"lam": "lam_star", "rel": "val_dm"})[
        ["client_id", "lam_star", "val_dm"]]


def run(fed_run, local_run, out_dir, device="auto"):
    fed_config, local_config = _load_config(fed_run), _load_config(local_run)
    check_pair(fed_config, local_config)
    device = _resolve_orchestrator_device(device)
    grid = relative_delta(score_grid(fed_config, fed_run, local_run, device))
    chosen = select_lambda(grid)
    test = grid[grid["split"] == "test"]
    picked = test.merge(chosen, on="client_id")
    picked = picked[picked["lam"] == picked["lam_star"]]
    endpoint = test[test["lam"] == 1.0][["client_id", "task", "rel"]].rename(
        columns={"rel": "federated_rel"})
    picked = picked.merge(endpoint, on=["client_id", "task"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    experiment = fed_config.get("experiment", {})
    for frame in (grid, picked):
        frame.insert(0, "seed", experiment.get("seed"))
        frame.insert(0, "study_id", experiment.get("study_id"))
    grid.to_csv(out / "interpolation_grid.csv", index=False)
    picked.to_csv(out / "interpolation_selected.csv", index=False)
    logging.info("[interp] test Delta_m: interpolated %.2f%%, federated %.2f%%",
                 picked["rel"].mean(), picked["federated_rel"].mean())
    return grid, picked


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--federated", required=True, help="Federated run directory")
    parser.add_argument("--local", required=True, help="Paired local-only run directory")
    parser.add_argument("--out", required=True, help="Output directory (never inside a run)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    for run_dir in (args.federated, args.local):
        if Path(args.out).resolve().is_relative_to(Path(run_dir).resolve()):
            parser.error("--out must not be inside a frozen run")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.federated, args.local, args.out, device=args.device)


if __name__ == "__main__":
    main()
