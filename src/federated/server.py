"""Aggregation strategies for the FedPer shared trunk.

Only the shared model state reaches the server. ``flat`` mode is a FedAvg-compatible weighted
average. ``hierarchical`` mode first averages clients inside each dataset and then mixes the
dataset updates, so a configured 50/50 modality contribution does not depend on sample counts or
oversampling. Personalized stems, decoders, and heads never reach the server.
"""

import logging

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

TASKS = ("seg", "cls")


class FedPerStrategy(FedAvg):
    def __init__(
        self,
        task_weights,
        dataset_weights=None,
        aggregation_mode="flat",
        client_weighting="num_examples",
        shared_key_names=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights
        self.dataset_weights = dataset_weights or {}
        if aggregation_mode not in {"flat", "hierarchical"}:
            raise ValueError("aggregation_mode must be 'flat' or 'hierarchical'")
        if client_weighting not in {"uniform", "num_examples"}:
            raise ValueError("client_weighting must be 'uniform' or 'num_examples'")
        self.aggregation_mode = aggregation_mode
        self.client_weighting = client_weighting
        # JSON-serializable diagnostics persisted by the training orchestrator. Keeping this
        # separate from Flower's scalar History allows auditing every client's effective weight.
        self.aggregation_history = []
        # Kept as diagnostics for legacy callers. Final evaluation deliberately uses the common
        # last-round global trunk, not the per-client local ``best.pt`` snapshots.
        self.best_mean_val = float("inf")
        self.best_round = -1
        self.latest_parameters = None
        self.latest_round = 0
        # Names of the shared tensors, used only to label the per-block gradient-conflict blocks.
        self.shared_key_names = list(shared_key_names) if shared_key_names else None
        # The trunk the server SENT this round. Cosines must be taken between client *deltas*;
        # the returned trunks all start from this point, so cosines between trunks are ~1 and
        # meaningless. Nothing else persists a client's post-fit trunk, so this is collected live.
        self._sent_arrays = None

    def configure_fit(self, server_round, parameters, client_manager):
        # Round 1 configures from ``initial_parameters``; ``latest_parameters`` is still None then.
        self._sent_arrays = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if failures and not self.accept_failures:
            logging.error(
                "[round %s] refusing to aggregate because %s client fit(s) failed",
                server_round, len(failures),
            )
            return None, {}
        if not results:
            return None, {}

        updates = []
        for _, fit_res in results:
            task = fit_res.metrics.get("task", "seg")
            dataset = fit_res.metrics.get("dataset", "default")
            base_weight = self._base_weight(
                fit_res.num_examples, task, dataset, metrics=fit_res.metrics
            )
            updates.append({
                "dataset": dataset,
                "task": task,
                "client_id": str(fit_res.metrics.get("client_id", "?")),
                "num_examples": int(fit_res.num_examples),
                "base_weight": float(base_weight),
                "arrays": parameters_to_ndarrays(fit_res.parameters),
                "metrics": dict(fit_res.metrics),
            })

        _validate_update_shapes(updates)
        if self.aggregation_mode == "hierarchical":
            aggregated, final_weights = self._aggregate_hierarchical(updates)
        else:
            aggregated, final_weights = self._aggregate_flat(updates)

        telemetry = self._record_aggregation(server_round, "fit", updates, final_weights)
        conflict = self._gradient_conflict(updates)
        if conflict is not None:
            telemetry["gradient_conflict"] = conflict
        summary = telemetry["group_participation"]
        logging.info(
            f"[round {server_round}] {self.aggregation_mode}/{self.client_weighting} "
            f"aggregation from "
            f"{len(results)} clients | effective participation={summary}"
        )
        self.latest_parameters = ndarrays_to_parameters(aggregated)
        self.latest_round = server_round
        return self.latest_parameters, self._participation_metrics(telemetry)

    def _base_weight(self, num_examples, task, dataset, metrics=None):
        if int(num_examples) <= 0:
            raise ValueError(
                f"Client from dataset={dataset}, task={task} reported non-positive "
                f"num_examples={num_examples}"
            )
        client_factor = float(num_examples) if self.client_weighting == "num_examples" else 1.0
        base_weight = client_factor * self._task_factor(num_examples, task, metrics)
        if base_weight <= 0:
            raise ValueError(
                f"Non-positive aggregation weight for dataset={dataset}, task={task}: "
                f"{base_weight}"
            )
        return base_weight

    def _task_factor(self, num_examples, task, metrics):
        """Task weighting generalised from one scalar to a per-task supervision mass.

        ``sum_t task_weights[t] * mass_t / num_examples``. A single-task client reports its whole
        slice under its own task, so the sum collapses to ``task_weights[task]`` and reproduces the
        historical weights exactly. A client that predates this telemetry falls back to the scalar.
        """
        mass = {}
        for name in TASKS:
            value = (metrics or {}).get(f"task_mass_{name}")
            if value is not None:
                mass[name] = float(value)
        if not mass or sum(mass.values()) <= 0:
            return float(self.task_weights.get(task, 1.0))
        weighted = sum(float(self.task_weights.get(name, 1.0)) * value
                       for name, value in mass.items())
        return weighted / float(num_examples)

    def _aggregate_flat(self, updates):
        # Dataset weights belong exclusively to the hierarchical mixing stage. With task weights
        # set to one and num_examples weighting this is ordinary FedAvg.
        weights = self._flat_weights(updates)
        return _weighted_arrays([update["arrays"] for update in updates], weights), weights

    def _aggregate_hierarchical(self, updates):
        weights = self._hierarchical_weights(updates)
        return _weighted_arrays([update["arrays"] for update in updates], weights), weights

    def _normalized_weights(self, records):
        if self.aggregation_mode == "hierarchical":
            return self._hierarchical_weights(records)
        return self._flat_weights(records)

    @staticmethod
    def _flat_weights(records):
        raw = np.asarray([record["base_weight"] for record in records], dtype=np.float64)
        if np.any(raw <= 0) or raw.sum() <= 0:
            raise ValueError("All configured flat aggregation weights must be positive")
        return raw / raw.sum()

    def _hierarchical_weights(self, records):
        datasets = list(dict.fromkeys(record["dataset"] for record in records))
        missing = set(self.dataset_weights) - set(datasets)
        if missing:
            raise ValueError(
                f"Hierarchical aggregation received no successful client from: {sorted(missing)}"
            )
        configured = np.asarray(
            [self.dataset_weights.get(dataset, 1.0) for dataset in datasets], dtype=np.float64
        )
        if np.any(configured <= 0) or configured.sum() <= 0:
            raise ValueError("All participating hierarchical dataset weights must be positive")
        dataset_mix = configured / configured.sum()

        final_weights = np.zeros(len(records), dtype=np.float64)
        for dataset, dataset_weight in zip(datasets, dataset_mix):
            indices = [i for i, record in enumerate(records) if record["dataset"] == dataset]
            raw = np.asarray([records[i]["base_weight"] for i in indices], dtype=np.float64)
            within_dataset = raw / raw.sum()
            final_weights[indices] = dataset_weight * within_dataset
        return final_weights

    def aggregate_evaluate(self, server_round, results, failures):
        if failures and not self.accept_failures:
            logging.error(
                "[round %s] refusing validation aggregate because %s client(s) failed",
                server_round, len(failures),
            )
            return None, {}
        if not results:
            return None, {}

        evaluations = []
        for _, result in results:
            task = result.metrics.get("task", "seg")
            dataset = result.metrics.get("dataset", "default")
            evaluations.append({
                "dataset": dataset,
                "task": task,
                "client_id": str(result.metrics.get("client_id", "?")),
                "num_examples": int(result.num_examples),
                "base_weight": self._base_weight(
                    result.num_examples, task, dataset, metrics=result.metrics
                ),
                "loss": float(result.loss),
                "metrics": dict(result.metrics),
            })

        final_weights = self._normalized_weights(evaluations)
        mean_val = float(np.dot(final_weights, [item["loss"] for item in evaluations]))
        telemetry = self._record_aggregation(server_round, "evaluate", evaluations, final_weights)

        per_group = {}
        for index, item in enumerate(evaluations):
            key = f"{item['dataset']}/{item['task']}"
            per_group.setdefault(key, []).append(
                (
                    float(item["metrics"].get("val_metric", float("nan"))),
                    float(final_weights[index]),
                )
            )
        group_summary = {
            key: round(_weighted_nanmean(values), 4) for key, values in per_group.items()
        }

        improved = mean_val < self.best_mean_val
        if improved:
            self.best_mean_val = mean_val
            self.best_round = server_round

        logging.info(
            f"[round {server_round}] {self.aggregation_mode}/{self.client_weighting} "
            f"diagnostic mean val loss {mean_val:.4f} "
            f"| effective participation={telemetry['group_participation']} "
            f"| per-dataset/task metric {group_summary}"
        )
        return mean_val, {
            "mean_val_loss": mean_val,
            "diagnostic_best_round": self.best_round,
            **self._participation_metrics(telemetry),
            **group_summary,
        }

    def _record_aggregation(self, server_round, stage, records, weights):
        group_participation = {}
        dataset_participation = {}
        clients = []
        optional_metrics = (
            "raw_num_examples",
            "effective_num_examples",
            "optimizer_steps",
            "examples_processed",
            # Per-task supervision mass and exercised batches: the audit trail for how a
            # multi-task client earned its aggregation weight.
            "task_mass_seg",
            "task_mass_cls",
            "task_batches_seg",
            "task_batches_cls",
        )
        for record, weight in zip(records, weights):
            dataset = str(record["dataset"])
            task = str(record["task"])
            group = f"{dataset}/{task}"
            group_participation[group] = group_participation.get(group, 0.0) + float(weight)
            dataset_participation[dataset] = dataset_participation.get(dataset, 0.0) + float(weight)
            client = {
                "client_id": str(record.get("client_id", "?")),
                "dataset": dataset,
                "task": task,
                "reported_num_examples": int(record["num_examples"]),
                "base_weight": float(record["base_weight"]),
                "final_weight": float(weight),
            }
            metrics = record.get("metrics", {})
            for key in optional_metrics:
                if key in metrics:
                    client[key] = float(metrics[key])
            clients.append(client)

        telemetry = {
            "round": int(server_round),
            "stage": stage,
            "aggregation_mode": self.aggregation_mode,
            "client_weighting": self.client_weighting,
            "client_count": len(records),
            "dataset_participation": dict(sorted(dataset_participation.items())),
            "group_participation": dict(sorted(group_participation.items())),
            "clients": clients,
        }
        self.aggregation_history.append(telemetry)
        return telemetry

    def _gradient_conflict(self, updates):
        """Pairwise cosine between client trunk DELTAS, per block and overall.

        FedBone motivates this: aggregating updates from heterogeneous tasks skews the shared
        trunk's direction. Measuring it is only possible here -- no artifact persists a client's
        post-fit trunk, so the matrix cannot be recovered after the run.

        The FULL matrix is stored rather than a seg-vs-cls mean, because only the matrix can later
        separate conflict between tasks (BUSI-seg vs BUSI-cls) from conflict between domains
        (BUSI-seg vs ISIC-seg).
        """
        sent = self._sent_arrays
        if not sent or len(sent) != len(updates[0]["arrays"]):
            return None

        blocks = {}
        if self.shared_key_names and len(self.shared_key_names) == len(sent):
            for index, key in enumerate(self.shared_key_names):
                blocks.setdefault(str(key).split(".")[0], []).append(index)
        else:
            blocks["shared"] = list(range(len(sent)))

        n_clients = len(updates)
        grams = {}
        for block, indices in blocks.items():
            gram = np.zeros((n_clients, n_clients), dtype=np.float64)
            for index in indices:
                # One tensor at a time: never materialise every client's full delta at once.
                deltas = np.stack([
                    (update["arrays"][index] - sent[index]).ravel().astype(np.float32)
                    for update in updates
                ])
                gram += (deltas @ deltas.T).astype(np.float64)
            grams[block] = gram

        overall = np.zeros((n_clients, n_clients), dtype=np.float64)
        for gram in grams.values():
            overall += gram
        grams["shared_total"] = overall

        return {
            "clients": [
                f"{update['dataset']}/{update['task']}/{update['client_id']}"
                for update in updates
            ],
            "cosine": {block: _cosine_from_gram(gram) for block, gram in grams.items()},
        }

    @staticmethod
    def _participation_metrics(telemetry):
        metrics = {
            "aggregation_clients": int(telemetry["client_count"]),
        }
        metrics.update({
            f"participation/dataset/{key}": float(value)
            for key, value in telemetry["dataset_participation"].items()
        })
        metrics.update({
            f"participation/group/{key}": float(value)
            for key, value in telemetry["group_participation"].items()
        })
        return metrics


def _weighted_nanmean(values_and_weights):
    values = np.asarray([value for value, _ in values_and_weights], dtype=np.float64)
    weights = np.asarray([weight for _, weight in values_and_weights], dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    weights = weights[finite]
    return float(np.dot(values[finite], weights) / weights.sum())


def _validate_update_shapes(updates):
    reference = [tuple(array.shape) for array in updates[0]["arrays"]]
    for update in updates[1:]:
        shapes = [tuple(array.shape) for array in update["arrays"]]
        if shapes != reference:
            raise ValueError(
                "Clients returned incompatible shared states; check share_stem/channels. "
                f"Expected {reference[:2]}..., got {shapes[:2]}... from {update['dataset']}"
            )


def _weighted_arrays(updates, weights):
    """Layer-wise weighted average that preserves the original tensor dtype."""
    aggregated = []
    for layers in zip(*updates):
        stacked = np.stack(layers, axis=0)
        value = np.tensordot(weights, stacked, axes=(0, 0))
        aggregated.append(value.astype(layers[0].dtype, copy=False))
    return aggregated


def _cosine_from_gram(gram):
    """Cosine matrix from a Gram matrix, with NaN for clients whose delta is exactly zero."""
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    degenerate = norms <= 0
    safe = np.where(degenerate, 1.0, norms)
    cosine = gram / np.outer(safe, safe)
    cosine[degenerate, :] = np.nan
    cosine[:, degenerate] = np.nan
    cosine = np.clip(cosine, -1.0, 1.0)
    return [
        [None if np.isnan(value) else round(float(value), 6) for value in row]
        for row in cosine
    ]
