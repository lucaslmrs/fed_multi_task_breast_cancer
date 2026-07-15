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


class FedPerStrategy(FedAvg):
    def __init__(self, task_weights, dataset_weights=None, aggregation_mode="flat", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights
        self.dataset_weights = dataset_weights or {}
        if aggregation_mode not in {"flat", "hierarchical"}:
            raise ValueError("aggregation_mode must be 'flat' or 'hierarchical'")
        self.aggregation_mode = aggregation_mode
        # Kept as diagnostics for legacy callers. Final evaluation deliberately uses the common
        # last-round global trunk, not the per-client local ``best.pt`` snapshots.
        self.best_mean_val = float("inf")
        self.best_round = -1
        self.latest_parameters = None
        self.latest_round = 0

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
            base_weight = fit_res.num_examples * self.task_weights.get(task, 1.0)
            if base_weight <= 0:
                raise ValueError(
                    f"Non-positive aggregation weight for dataset={dataset}, task={task}: "
                    f"{base_weight}"
                )
            updates.append({
                "dataset": dataset,
                "task": task,
                "base_weight": float(base_weight),
                "arrays": parameters_to_ndarrays(fit_res.parameters),
            })

        _validate_update_shapes(updates)
        if self.aggregation_mode == "hierarchical":
            aggregated, final_weights = self._aggregate_hierarchical(updates)
        else:
            aggregated, final_weights = self._aggregate_flat(updates)

        participation = {}
        for update, weight in zip(updates, final_weights):
            key = f"{update['dataset']}/{update['task']}"
            participation[key] = participation.get(key, 0.0) + float(weight)
        summary = {key: round(value, 4) for key, value in sorted(participation.items())}
        logging.info(
            f"[round {server_round}] {self.aggregation_mode} aggregation from "
            f"{len(results)} clients | effective participation={summary}"
        )
        self.latest_parameters = ndarrays_to_parameters(aggregated)
        self.latest_round = server_round
        return self.latest_parameters, {}

    def _aggregate_flat(self, updates):
        # Flat is the global sample-weighted ablation. Dataset weights belong exclusively to the
        # hierarchical mixing stage; with task weights set to one this is ordinary FedAvg.
        raw = np.asarray([update["base_weight"] for update in updates], dtype=np.float64)
        if np.any(raw <= 0) or raw.sum() <= 0:
            raise ValueError("All configured flat aggregation weights must be positive")
        weights = raw / raw.sum()
        return _weighted_arrays([update["arrays"] for update in updates], weights), weights

    def _aggregate_hierarchical(self, updates):
        datasets = list(dict.fromkeys(update["dataset"] for update in updates))
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

        dataset_arrays = []
        final_weights = np.zeros(len(updates), dtype=np.float64)
        for dataset, dataset_weight in zip(datasets, dataset_mix):
            indices = [i for i, update in enumerate(updates) if update["dataset"] == dataset]
            raw = np.asarray([updates[i]["base_weight"] for i in indices], dtype=np.float64)
            within_dataset = raw / raw.sum()
            dataset_arrays.append(
                _weighted_arrays([updates[i]["arrays"] for i in indices], within_dataset)
            )
            final_weights[indices] = dataset_weight * within_dataset

        return _weighted_arrays(dataset_arrays, dataset_mix), final_weights

    def aggregate_evaluate(self, server_round, results, failures):
        if failures and not self.accept_failures:
            logging.error(
                "[round %s] refusing validation aggregate because %s client(s) failed",
                server_round, len(failures),
            )
            return None, {}
        if not results:
            return None, {}

        n_total = sum(result.num_examples for _, result in results)
        mean_val = sum(result.num_examples * result.loss for _, result in results) / n_total

        per_group = {}
        for _, result in results:
            key = f"{result.metrics.get('dataset', 'default')}/{result.metrics.get('task', '?')}"
            per_group.setdefault(key, []).append(
                result.metrics.get("val_metric", float("nan"))
            )
        group_summary = {
            key: round(float(np.nanmean(values)), 4) for key, values in per_group.items()
        }

        improved = mean_val < self.best_mean_val
        if improved:
            self.best_mean_val = mean_val
            self.best_round = server_round

        logging.info(
            f"[round {server_round}] diagnostic mean val loss {mean_val:.4f} "
            f"| per-dataset/task metric {group_summary}"
        )
        return mean_val, {
            "mean_val_loss": mean_val,
            "diagnostic_best_round": self.best_round,
            **group_summary,
        }


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
