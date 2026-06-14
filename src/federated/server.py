"""
Custom FedAvg strategy for the FedPer multi-task setup.

The exchanged parameters are ONLY the shared encoder. Aggregation is a weighted average where
each client's weight is ``num_examples * task_weight[task]`` -- the sample term keeps standard
FedAvg behaviour, the task term (configurable) lets us rebalance the seg/cls contribution to the
shared encoder if one task's loss scale dominates. Personalized heads never reach the server.
"""

import logging

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


class FedPerStrategy(FedAvg):
    def __init__(self, task_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights
        self.best_mean_val = float("inf")
        self.best_round = -1

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        layer_stacks, weights = None, []
        for _, fit_res in results:
            task = fit_res.metrics.get("task", "seg")
            weight = fit_res.num_examples * self.task_weights.get(task, 1.0)
            weights.append(weight)
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if layer_stacks is None:
                layer_stacks = [[] for _ in ndarrays]
            for i, layer in enumerate(ndarrays):
                layer_stacks[i].append(layer)

        total = float(sum(weights))
        w = np.asarray(weights, dtype=np.float64) / total
        aggregated = [np.tensordot(w, np.stack(layers, axis=0), axes=(0, 0))
                      for layers in layer_stacks]

        logging.info(f"[round {server_round}] aggregated encoder from {len(results)} clients "
                     f"(weights={[round(x, 3) for x in w.tolist()]})")
        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        n_total = sum(r.num_examples for _, r in results)
        mean_val = sum(r.num_examples * r.loss for _, r in results) / n_total

        # per-task mean validation metric for logging
        per_task = {}
        for _, r in results:
            per_task.setdefault(r.metrics.get("task", "?"), []).append(r.metrics.get("val_metric", float("nan")))
        task_summary = {t: round(float(np.nanmean(v)), 4) for t, v in per_task.items()}

        improved = mean_val < self.best_mean_val
        if improved:
            self.best_mean_val = mean_val
            self.best_round = server_round

        logging.info(f"[round {server_round}] mean val loss {mean_val:.4f} "
                     f"{'(best)' if improved else ''} | per-task metric {task_summary} "
                     f"| best round {self.best_round}")
        return mean_val, {"mean_val_loss": mean_val, "best_round": self.best_round, **task_summary}
