import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from flwr.common import ndarrays_to_parameters

from src.federated.negative_transfer import (
    MISSING_STATUS,
    SUMMARY_FILE,
    cancellation_from_gram,
    report_run,
    summarize_run,
)
from src.federated.server import FedPerStrategy


def _gram(deltas):
    deltas = np.asarray(deltas, dtype=np.float64)
    return deltas @ deltas.T


class CancellationTests(unittest.TestCase):
    def test_opposite_updates_cancel_completely(self):
        result = cancellation_from_gram(_gram([[1, 0], [-1, 0]]), [0.5, 0.5], ["A", "A"])
        self.assertAlmostEqual(result["overall"], 1.0)
        self.assertAlmostEqual(result["intra"], 1.0)

    def test_identical_updates_do_not_cancel(self):
        result = cancellation_from_gram(_gram([[1, 2], [1, 2]]), [0.3, 0.7], ["A", "B"])
        for key in ("overall", "intra", "inter", "intra_A", "intra_B"):
            self.assertAlmostEqual(result[key], 0.0)

    def test_conflict_inside_a_dataset_is_intra(self):
        # A's clients oppose each other on axis 1, but both datasets agree on axis 0.
        deltas = [[1, 1], [1, -1], [1, 0]]
        result = cancellation_from_gram(_gram(deltas), [0.25, 0.25, 0.5], ["A", "A", "B"])
        self.assertGreater(result["intra_A"], 0.2)
        self.assertAlmostEqual(result["intra_B"], 0.0)
        self.assertAlmostEqual(result["inter"], 0.0)

    def test_conflict_between_datasets_is_inter(self):
        deltas = [[1, 0], [1, 0], [-1, 0]]
        result = cancellation_from_gram(_gram(deltas), [0.25, 0.25, 0.5], ["A", "A", "B"])
        self.assertAlmostEqual(result["intra"], 0.0)
        self.assertAlmostEqual(result["inter"], 1.0)

    def test_overall_factorises_into_intra_and_inter(self):
        rng = np.random.default_rng(0)
        deltas = rng.normal(size=(6, 20))
        weights = rng.random(6)
        weights /= weights.sum()
        result = cancellation_from_gram(_gram(deltas), weights, ["A", "A", "B", "B", "B", "C"])
        self.assertAlmostEqual(
            1 - result["overall"], (1 - result["intra"]) * (1 - result["inter"]), places=9
        )

    def test_zero_updates_are_undefined(self):
        result = cancellation_from_gram(_gram([[0, 0], [0, 0]]), [0.5, 0.5], ["A", "B"])
        self.assertTrue(np.isnan(result["overall"]))


class ServerTelemetryTests(unittest.TestCase):
    def test_aggregate_fit_records_negative_transfer(self):
        strategy = FedPerStrategy(
            task_weights={"seg": 1, "cls": 1},
            dataset_weights={"A": 1, "B": 1},
            aggregation_mode="hierarchical",
            client_weighting="uniform",
        )
        strategy._sent_arrays = [np.zeros(2, np.float32)]
        results = [
            (None, SimpleNamespace(
                num_examples=10,
                metrics={"dataset": dataset, "task": "seg", "client_id": name},
                parameters=ndarrays_to_parameters([np.asarray(delta, np.float32)]),
            ))
            for dataset, name, delta in [("A", "a", [1, 0]), ("B", "b", [-1, 0])]
        ]
        strategy.aggregate_fit(1, results, [])
        record = strategy.aggregation_history[-1]["negative_transfer"]
        self.assertAlmostEqual(record["overall"], 1.0)
        self.assertAlmostEqual(record["inter"], 1.0)
        self.assertAlmostEqual(record["intra"], 0.0)


class SummaryTests(unittest.TestCase):
    @staticmethod
    def _write(run, fold, values):
        path = Path(run) / fold / "aggregation_history.json"
        path.parent.mkdir(parents=True)
        history = []
        for round_, value in enumerate(values, start=1):
            event = {"round": round_, "stage": "fit"}
            if value is not None:
                event["negative_transfer"] = {"overall": value, "intra": value, "inter": 0.0}
            history.append(event)
            history.append({"round": round_, "stage": "evaluate"})
        path.write_text(json.dumps(history), encoding="utf-8")

    def test_thirds_folds_and_incomplete_archives(self):
        with tempfile.TemporaryDirectory() as run:
            self._write(run, "fold_0", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            self._write(run, "fold_1", [0.3] * 6)
            self._write(run, "fold_1.incomplete_20260101_000000", [0.9] * 6)
            rounds, summary = summarize_run(run)
            self.assertEqual(len(rounds), 12)
            fold0 = summary[summary["fold"] == 0].iloc[0]
            self.assertAlmostEqual(fold0["overall_mean"], 0.35)
            self.assertAlmostEqual(fold0["overall_first_third"], 0.15)
            self.assertAlmostEqual(fold0["overall_last_third"], 0.55)
            total = summary.iloc[-1]
            self.assertEqual(total["fold"], "all_folds")
            self.assertEqual(total["status"], "ok")
            self.assertAlmostEqual(total["overall_mean"], 0.325)

            report_run(run)
            self.assertTrue((Path(run) / SUMMARY_FILE).exists())

    def test_old_history_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as run:
            self._write(run, "fold_0", [None, None])
            _, summary = summarize_run(run)
            self.assertEqual(summary.iloc[0]["status"], MISSING_STATUS)
            self.assertEqual(summary.iloc[-1]["status"], MISSING_STATUS)


if __name__ == "__main__":
    unittest.main()
