"""
Controls for the gradient-conflict telemetry (plano.md, stage E0.1).

The server measures conflict as the pairwise cosine between client trunk DELTAS
(``FedPerStrategy._gradient_conflict``). On the 2026-09-29 run that cosine is ~0 for every pair,
including clients of the same dataset and task on an almost-IID partition, which should be aligned.
This script decides whether that is real or a blind metric, by re-creating the final state of a
finished run and switching ONE factor at a time:

    real_adam10    warm Adam state, 10 local steps  -> what the server measured
    adam1          warm Adam state, 1 step           -> removes step accumulation (H2)
    fresh_adam10   fresh Adam, 10 steps              -> removes the warm optimizer state (H1)
    sgd10          plain SGD trajectory, 10 steps    -> removes Adam's per-coordinate scaling (H1)
    grad1          raw gradient at the trunk, 1 batch
    grad10         raw gradient at the trunk, averaged over the round's 10 batches
    grad10_refpers as grad10, but every client uses the personalized stem/heads of the FIRST
                   client of its (dataset, tasks) group -> isolates the personalized parts (H3)

Each condition is repeated on the NEXT round's batches (``*_b``). The cosine between a client's two
repetitions is the within-client ceiling: if even a client does not agree with itself, no cross-client
cosine can be read as (absence of) conflict. All conditions of a client use the same pre-fetched
batches, so they are paired.

Nothing in the source run is written: the client state is read from it, and every artifact goes to
``--out``. The production code paths are reused (``FederatedClient`` for loaders/criteria,
``train_local`` for the objective, ``FedPerStrategy._gradient_conflict`` for the cosine).

    python -m scripts.gradient_conflict_controls --run runs/<federated_run> [--fold 0] [--device cuda]
"""

import argparse
import copy
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from flwr.common import parameters_to_ndarrays

from src.dataset.federated_dataloader import list_clients
from src.federated import local_trainer
from src.federated.client import FederatedClient
from src.federated.config import partition_file
from src.federated.model_split import (get_personalized_state, get_shared_state,
                                       set_personalized_state, set_shared_state, shared_keys)
from src.federated.server import FedPerStrategy
from src.training_federated import _initial_shared_parameters
from src.utils.experiment_init import init_optimizer

CONDITIONS = ("real_adam10", "adam1", "fresh_adam10", "sgd10", "grad1", "grad10",
              "grad10_refpers")


def _roster(config, master_file):
    """Same (client, dataset, tasks) grouping as ``training_federated.run``."""
    frame = list_clients(str(master_file))
    frame = frame[frame["dataset"].isin(config["federated"]["datasets"])]
    grouped = (
        frame.groupby(["client_id", "dataset"], sort=False)["task"]
        .apply(lambda values: tuple(name for name in ("seg", "cls") if name in set(values)))
        .reset_index()
    )
    return [(row.client_id, row.dataset, row.task) for row in grouped.itertuples(index=False)]


def _round_batches(client, server_round):
    client.train_loader.batch_sampler.set_round(server_round)
    return list(client.train_loader)


def _train(client, batches, optimizer):
    return local_trainer.train_local(
        client.model, batches, optimizer, client.task, client.device, 1, client.num_classes,
        client.seg_criterion, client.cls_criterion, client.inversely_weighted,
        training_mode="steps", steps_per_round=len(batches), tasks=client.tasks,
        task_lambdas=client.task_lambdas, precision=client.precision,
    )


def _delta(client, theta):
    return [after - before for after, before in
            zip(get_shared_state(client.model, client.share_stem), theta)]


def _condition_delta(client, condition, batches, theta, model_state, optim_state, lr):
    """Trunk displacement of one condition, always starting from the identical saved state."""
    def reset():
        client.model.load_state_dict(model_state)
        set_shared_state(client.model, theta, client.share_stem)

    if condition in {"real_adam10", "adam1"}:
        reset()
        optimizer = init_optimizer(client.model, client.config["optimizer"]["opt"], lr)
        optimizer.load_state_dict(optim_state)
        _train(client, batches if condition == "real_adam10" else batches[:1], optimizer)
        return _delta(client, theta)
    if condition == "fresh_adam10":
        reset()
        _train(client, batches, init_optimizer(client.model, client.config["optimizer"]["opt"], lr))
        return _delta(client, theta)
    if condition == "sgd10":
        reset()
        _train(client, batches, torch.optim.SGD(client.model.parameters(), lr=lr))
        return _delta(client, theta)
    # Raw gradients: one SGD step with lr=1 from theta gives exactly -grad on the trunk. Averaging
    # single-batch gradients (never moving theta) is the low-noise gradient AT the sent trunk.
    # ``grad10_refpers`` receives the reference client's model state as ``model_state``.
    used = batches[:1] if condition == "grad1" else batches
    total = None
    for batch in used:
        reset()
        _train(client, [batch], torch.optim.SGD(client.model.parameters(), lr=1.0))
        delta = _delta(client, theta)
        total = delta if total is None else [a + b for a, b in zip(total, delta)]
    return [array / len(used) for array in total]


def _pair_class(a, b):
    (_, dataset_a, tasks_a), (_, dataset_b, tasks_b) = a, b
    if dataset_a == dataset_b:
        return "intra_same_tasks" if tasks_a == tasks_b else "intra_diff_tasks"
    return "inter_shared_task" if set(tasks_a) & set(tasks_b) else "inter_disjoint_tasks"


def _cosines(strategy, roster, theta, deltas):
    updates = [{
        "dataset": dataset, "task": "+".join(tasks), "client_id": client_id,
        "arrays": [base + delta for base, delta in zip(theta, client_delta)],
    } for (client_id, dataset, tasks), client_delta in zip(roster, deltas)]
    strategy._sent_arrays = theta
    return strategy._gradient_conflict(updates)["cosine"]


def _flat_cosine(first, second):
    a = np.concatenate([x.ravel() for x in first]).astype(np.float64)
    b = np.concatenate([x.ravel() for x in second]).astype(np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="finished federated run directory")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-clients", type=int, default=None, help="debug: first N clients")
    parser.add_argument("--at-init", action="store_true",
                        help="use the round-1 state (initial trunk, seeded stems/heads, fresh "
                             "optimizer) instead of the run's final state")
    args = parser.parse_args()

    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    if config["federated"].get("standalone", False):
        raise ValueError("controls need a federated run: standalone runs have no global trunk")
    out = Path(args.out or Path("runs/validation/gradient_conflict_controls")
               / f"{datetime.now():%Y%m%d_%H%M%S}_{run.name}")
    out.mkdir(parents=True, exist_ok=False)
    master_file = partition_file(config)
    roster = _roster(config, master_file)[: args.max_clients]

    if args.at_init:
        # Exactly the paired round-1 trunk the orchestrator sends (same seed policy).
        theta = parameters_to_ndarrays(_initial_shared_parameters(
            config, config["federated"]["datasets"], "cpu", args.fold))
        trunk_round, start_round = 0, 1
    else:
        global_state = torch.load(run / f"fold_{args.fold}" / "global_shared.pt",
                                  map_location="cpu")
        theta = [tensor.numpy() for tensor in global_state["arrays"]]
        trunk_round = int(global_state["round"])
        start_round = trunk_round + 1
    lr = float(config["optimizer"]["lr"])

    deltas = {condition: [] for condition in CONDITIONS}
    deltas_b = {condition: [] for condition in CONDITIONS}
    key_names = None
    fold_dir = run / f"fold_{args.fold}"
    reference, reference_personalized = {}, {}
    for client_id, dataset, tasks in roster:
        reference.setdefault((dataset, tasks), client_id)
    for index, (client_id, dataset, tasks) in enumerate(roster):
        print(f"[{index + 1}/{len(roster)}] {client_id} {tasks}", flush=True)
        client = FederatedClient(client_id, dataset, tasks[0], args.fold, config, args.device,
                                 str(master_file), str(out / "clients"))
        key_names = key_names or shared_keys(client.model, client.share_stem)
        if args.at_init:
            # The seeded construction IS the round-1 state: own stem/heads, fresh optimizer.
            personalized = get_personalized_state(client.model, client.share_stem)
            optim_state = copy.deepcopy(client.optimizer.state_dict())
        else:
            saved = torch.load(fold_dir / f"client_{client_id}" / "state.pt",
                               map_location=client.device)
            personalized, optim_state = saved["personalized"], saved["optimizer"]
        group_reference = reference[(dataset, tasks)]
        if group_reference == client_id:
            reference_personalized[group_reference] = copy.deepcopy(personalized)
        set_personalized_state(client.model, reference_personalized[group_reference],
                               client.share_stem)
        ref_state = copy.deepcopy(client.model.state_dict())
        set_personalized_state(client.model, personalized, share_stem=client.share_stem)
        model_state = copy.deepcopy(client.model.state_dict())
        batches = _round_batches(client, start_round)
        batches_b = _round_batches(client, start_round + 1)
        for condition in CONDITIONS:
            state = ref_state if condition == "grad10_refpers" else model_state
            deltas[condition].append(_condition_delta(
                client, condition, batches, theta, state, optim_state, lr))
            deltas_b[condition].append(_condition_delta(
                client, condition, batches_b, theta, state, optim_state, lr))
        del client, personalized, optim_state, model_state, ref_state, batches, batches_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    strategy = FedPerStrategy(task_weights={"seg": 1, "cls": 1}, shared_key_names=key_names)
    pair_rows, within_rows = [], []
    for condition in CONDITIONS:
        cosine = _cosines(strategy, roster, theta, deltas[condition])
        for block, matrix in cosine.items():
            for i, j in itertools.combinations(range(len(roster)), 2):
                pair_rows.append({
                    "condition": condition, "block": block,
                    "client_a": roster[i][0], "client_b": roster[j][0],
                    "pair_class": _pair_class(roster[i], roster[j]),
                    "cosine": np.nan if matrix[i][j] is None else matrix[i][j],
                })
        for (client_id, dataset, tasks), first, second in zip(
                roster, deltas[condition], deltas_b[condition]):
            within_rows.append({"condition": condition, "client_id": client_id,
                                "dataset": dataset, "tasks": "+".join(tasks),
                                "cosine": _flat_cosine(first, second)})

    pairs = pd.DataFrame(pair_rows)
    within = pd.DataFrame(within_rows)
    pairs.to_csv(out / "pair_cosines.csv", index=False)
    within.to_csv(out / "within_client_cosines.csv", index=False)

    total = pairs[pairs.block == "shared_total"]
    summary = (total.groupby(["condition", "pair_class"]).cosine
               .agg(["mean", "median", lambda s: s.quantile(0.05), lambda s: s.quantile(0.95),
                     lambda s: (s < 0).mean(), "count"]))
    summary.columns = ["mean", "median", "p5", "p95", "frac_negative", "pairs"]
    summary = summary.reset_index()
    ceiling = within.groupby("condition").cosine.agg(["mean", "min", "max"]).reset_index()
    ceiling.columns = ["condition", "within_mean", "within_min", "within_max"]
    summary = summary.merge(ceiling, on="condition")
    order = {name: rank for rank, name in enumerate(CONDITIONS)}
    summary = summary.sort_values(["condition", "pair_class"], key=lambda s: s.map(order)
                                  if s.name == "condition" else s)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(
        f"# Gradient-conflict controls\n\nSource run: `{run}` (fold {args.fold}, trunk from round "
        f"{trunk_round}{' (initialization)' if args.at_init else ''}, batches of rounds "
        f"{start_round}/{start_round + 1}).\n\n"
        + "```\n" + summary.to_string(index=False, float_format="%.4f") + "\n```\n",
        encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
