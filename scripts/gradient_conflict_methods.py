"""
Offline prototype of the candidate gradient-conflict methods (plano.md, stage E0.2.2-E0.2.3).

Every method is computed from the saved state of a finished federated run (or its round-1 state
with ``--at-init``), on the same paired batches, and checked against three controls:

    positive  clients that should agree (same dataset/tasks; within-client repetition = ceiling)
    negative  an ADVERSARY: a twin of a real client that sees the same images and personalization
              but inverted targets (mask -> 1 - mask, class -> (class + 1) mod K). Any method that
              does not report conflict between a client and its adversary is not measuring conflict.
    ceiling   each quantity is recomputed on the next round's batches; agreement between the two
              repetitions bounds what is detectable.

Methods (sources and exact definitions in docs/GRADIENT_CONFLICT_METHODS.md):

    M1  cosine and gradient magnitude similarity Phi = 2|gi||gj| / (|gi|^2 + |gj|^2) of the raw
        trunk gradient at the sent trunk, averaged over the round's batches (PCGrad).
    M3  pairwise sign consistency sum|gi + gj| / sum(|gi| + |gj|), the two-vector form of the
        GradDrop purity |2P - 1| weighted by magnitude, against a sign-randomised baseline.
    M2  lookahead affinity Z(i->j) = 1 - L_j(theta + Delta_i) / L_j(theta) (TAG), with Delta_i the
        REAL round update of client i (warm Adam, 10 steps) instead of a single small SGD step, and
        L_j the validation loss of client j under j's own pre-round personalized parameters.
    M2' federation gain G_j = Z(agg->j) - Z(j->j): whether the aggregated trunk helped client j more
        than its own step did (adaptation; precedent FedFomo). Uses the run's real aggregation
        weights. Recomputed with the adversary admitted into the aggregation.

Nothing in the source run is written; artifacts go to ``--out``.

    python -m scripts.gradient_conflict_methods --run runs/<federated_run> [--at-init]
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

from scripts.gradient_conflict_controls import (_condition_delta, _pair_class, _round_batches,
                                                _roster)
from src.federated import local_trainer
from src.federated.client import FederatedClient
from src.federated.config import aggregation_config, partition_file
from src.federated.model_split import (get_personalized_state, set_personalized_state,
                                       set_shared_state)
from src.federated.server import FedPerStrategy
from src.training_federated import _initial_shared_parameters

ADVERSARY_SUFFIX = "__adversary"


class _Batches(list):
    """A list of batches that ``evaluate_local`` can consume like a DataLoader."""

    @property
    def dataset(self):
        return range(sum(int(batch["image"].shape[0]) for batch in self))


def _invert_targets(batch, num_classes):
    inverted = dict(batch)
    if "mask" in batch:
        inverted["mask"] = 1 - batch["mask"]
    if "label" in batch:
        inverted["label"] = ((batch["label"] + 1) % num_classes).to(batch["label"].dtype)
    return inverted


def _flatten(arrays):
    return np.concatenate([array.ravel() for array in arrays]).astype(np.float64)


def _cos(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def _magnitude_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(2 * na * nb / (na ** 2 + nb ** 2)) if na + nb > 0 else float("nan")


def _sign_consistency(a, b):
    total = np.abs(a).sum() + np.abs(b).sum()
    return float(np.abs(a + b).sum() / total) if total > 0 else float("nan")


def _val_loss(participant, trunk):
    client = participant["client"]
    client.model.load_state_dict(participant["model_state"])
    set_shared_state(client.model, trunk, client.share_stem)
    result = local_trainer.evaluate_local(
        client.model, participant["val"], client.task, client.device, client.num_classes,
        client.seg_criterion, client.cls_criterion, client.inversely_weighted,
        tasks=client.tasks, task_lambdas=client.task_lambdas, precision=client.precision,
    )
    return float(result["loss"])


def _apply(theta, delta):
    return [base + step for base, step in zip(theta, delta)]


def _aggregate(theta, participants, strategy):
    records = [{"dataset": p["dataset"], "base_weight": p["base_weight"]} for p in participants]
    weights = strategy._hierarchical_weights(records) if strategy.aggregation_mode == "hierarchical" \
        else strategy._flat_weights(records)
    trunk = [np.zeros_like(array, dtype=np.float64) for array in theta]
    for weight, participant in zip(weights, participants):
        for index, step in enumerate(participant["delta"]):
            trunk[index] += weight * (theta[index] + step)
    return [array.astype(theta[index].dtype) for index, array in enumerate(trunk)], weights


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--at-init", action="store_true")
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--adversaries", nargs="*", default=None,
                        help="client ids to twin with an inverted-target adversary "
                             "(default: the first client of each (dataset, tasks) group)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    tag = f"{run.name}_{'init' if args.at_init else 'final'}"
    out = Path(args.out or Path("runs/validation/gradient_conflict_methods")
               / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}")
    out.mkdir(parents=True, exist_ok=False)
    master_file = partition_file(config)
    roster = _roster(config, master_file)
    fold_dir = run / f"fold_{args.fold}"
    lr = float(config["optimizer"]["lr"])
    aggregation = aggregation_config(config)
    strategy = FedPerStrategy(
        task_weights=aggregation["task_weights"], dataset_weights=aggregation["dataset_weights"],
        aggregation_mode=aggregation["mode"], client_weighting=aggregation["client_weighting"],
    )

    if args.at_init:
        theta = parameters_to_ndarrays(_initial_shared_parameters(
            config, config["federated"]["datasets"], "cpu", args.fold))
        start_round = 1
    else:
        state = torch.load(fold_dir / "global_shared.pt", map_location="cpu")
        theta = [tensor.numpy() for tensor in state["arrays"]]
        start_round = int(state["round"]) + 1

    twins = set(args.adversaries) if args.adversaries is not None else {
        next(cid for cid, ds, ts in roster if (ds, ts) == group)
        for group in dict.fromkeys((ds, ts) for _, ds, ts in roster)
    }

    participants = []
    for index, (client_id, dataset, tasks) in enumerate(roster):
        print(f"[{index + 1}/{len(roster)}] {client_id}", flush=True)
        client = FederatedClient(client_id, dataset, tasks[0], args.fold, config, args.device,
                                 str(master_file), str(out / "clients"))
        if args.at_init:
            personalized = get_personalized_state(client.model, client.share_stem)
            optim_state = copy.deepcopy(client.optimizer.state_dict())
        else:
            saved = torch.load(fold_dir / f"client_{client_id}" / "state.pt",
                               map_location=client.device)
            personalized, optim_state = saved["personalized"], saved["optimizer"]
        set_personalized_state(client.model, personalized, client.share_stem)
        model_state = copy.deepcopy(client.model.state_dict())
        batches = _round_batches(client, start_round)
        batches_b = _round_batches(client, start_round + 1)
        val = _Batches(itertools.islice(client.val_loader, args.val_batches))
        metrics = {f"task_mass_{name}": value for name, value in client.task_mass.items()}
        base_weight = strategy._base_weight(client.n_train, client.task, dataset, metrics)

        variants = [(client_id, False)]
        if client_id in twins:
            variants.append((client_id + ADVERSARY_SUFFIX, True))
        for participant_id, adversary in variants:
            def view(items):
                return _Batches(_invert_targets(b, client.num_classes) for b in items) \
                    if adversary else items
            run_batches, run_batches_b = view(batches), view(batches_b)
            participant = {
                "id": participant_id, "dataset": dataset, "tasks": tasks, "client": client,
                "model_state": model_state, "val": view(val), "base_weight": base_weight,
                "adversary": adversary, "twin": client_id,
            }
            for suffix, items in (("", run_batches), ("_b", run_batches_b)):
                participant["grad" + suffix] = _flatten(_condition_delta(
                    client, "grad10", list(items), theta, model_state, optim_state, lr))
                participant["delta" + suffix] = _condition_delta(
                    client, "real_adam10", list(items), theta, model_state, optim_state, lr)
            participants.append(participant)
        del personalized, optim_state

    real = [p for p in participants if not p["adversary"]]

    def pair_kind(a, b):
        if a["adversary"] or b["adversary"]:
            return "adversary_vs_twin" if a["twin"] == b["twin"] else "adversary_vs_other"
        return _pair_class((a["id"], a["dataset"], a["tasks"]), (b["id"], b["dataset"], b["tasks"]))

    # ---- M1 / M3 -------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    ceiling = {p["id"]: _cos(p["grad"], p["grad_b"]) for p in participants}
    geometric = []
    for a, b in itertools.combinations(participants, 2):
        random_signs = rng.choice([-1.0, 1.0], size=b["grad"].shape)
        ceil = ceiling[a["id"]] * ceiling[b["id"]]
        cos = _cos(a["grad"], b["grad"])
        geometric.append({
            "a": a["id"], "b": b["id"], "pair_class": pair_kind(a, b),
            "m1_cos": cos, "m1_cos_b": _cos(a["grad_b"], b["grad_b"]),
            "m1_cos_normalized": cos / np.sqrt(ceil) if ceil > 0.0025 else np.nan,
            "m1_magnitude_similarity": _magnitude_similarity(a["grad"], b["grad"]),
            "m3_sign_consistency": _sign_consistency(a["grad"], b["grad"]),
            "m3_baseline": _sign_consistency(a["grad"], b["grad"] * random_signs),
            "delta_cos": _cos(_flatten(a["delta"]), _flatten(b["delta"])),
        })
    geometric = pd.DataFrame(geometric)
    geometric["m3_excess"] = geometric.m3_sign_consistency - geometric.m3_baseline

    # ---- M2 / M2' ------------------------------------------------------------------------
    print("lookahead evaluations...", flush=True)
    agg, weights = _aggregate(theta, real, strategy)
    agg_b, _ = _aggregate(theta, [{**p, "delta": p["delta_b"]} for p in real], strategy)
    twin_weight = {p["id"]: w for p, w in zip(real, weights)}
    lookahead, gains = [], []
    for j in participants:
        base = _val_loss(j, theta)
        z = {}
        for i in participants:
            for suffix in ("", "_b"):
                z[(i["id"], suffix)] = 1 - _val_loss(j, _apply(theta, i["delta" + suffix])) / base
            lookahead.append({"source": i["id"], "target": j["id"], "pair_class": pair_kind(i, j)
                              if i is not j else "self", "z": z[(i["id"], "")],
                              "z_b": z[(i["id"], "_b")]})
        z_agg = 1 - _val_loss(j, agg) / base
        z_agg_b = 1 - _val_loss(j, agg_b) / base
        row = {"client": j["id"], "dataset": j["dataset"], "adversary": j["adversary"],
               "base_loss": base, "z_self": z[(j["id"], "")], "z_agg": z_agg,
               "gain": z_agg - z[(j["id"], "")], "gain_b": z_agg_b - z[(j["id"], "_b")]}
        if not j["adversary"] and j["id"] in twins:
            adversary = next(p for p in participants if p["twin"] == j["id"] and p["adversary"])
            agg_adv, _ = _aggregate(theta, real + [adversary], strategy)
            row["gain_with_adversary"] = 1 - _val_loss(j, agg_adv) / base - z[(j["id"], "")]
        gains.append(row)
    lookahead = pd.DataFrame(lookahead)
    gains = pd.DataFrame(gains)

    geometric.to_csv(out / "m1_m3_pairs.csv", index=False)
    lookahead.to_csv(out / "m2_lookahead.csv", index=False)
    gains.to_csv(out / "m2prime_federation_gain.csv", index=False)
    pd.DataFrame([{"client": k, "ceiling": v} for k, v in ceiling.items()]).to_csv(
        out / "within_client_ceiling.csv", index=False)

    def stability(frame, a, b):
        valid = frame[[a, b]].dropna()
        return float(np.corrcoef(valid[a], valid[b])[0, 1]) if len(valid) > 2 else float("nan")

    geo = geometric.groupby("pair_class")[[
        "m1_cos", "m1_cos_normalized", "m1_magnitude_similarity", "m3_excess", "delta_cos"
    ]].mean()
    look = lookahead.groupby("pair_class")[["z", "z_b"]].mean()
    lines = [
        f"# Gradient-conflict methods — {tag}", "",
        f"Trunk state: {'round 1 (initialization)' if args.at_init else f'round {start_round - 1}'}; "
        f"batches of rounds {start_round}/{start_round + 1}; {args.val_batches} validation batches "
        f"per client; adversaries twinned with: {sorted(twins)}.", "",
        "## M1 / M3 by pair class (means)", "```", geo.to_string(float_format="%.4f"), "```", "",
        "## M2 lookahead affinity Z(source -> target) by pair class (means)", "```",
        look.to_string(float_format="%.5f"), "```", "",
        "## M2' federation gain per client", "```",
        gains.to_string(index=False, float_format="%.5f"), "```", "",
        "## Stability across the two batch repetitions (Pearson over pairs/clients)", "```",
        f"M1 cosine          {stability(geometric, 'm1_cos', 'm1_cos_b'):.3f}",
        f"M2 Z               {stability(lookahead[lookahead.pair_class != 'self'], 'z', 'z_b'):.3f}",
        f"M2' gain           {stability(gains, 'gain', 'gain_b'):.3f}", "```", "",
        "## Within-client ceiling of M1 (cosine between the two repetitions)", "```",
        pd.Series(ceiling).to_string(float_format="%.3f"), "```",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
