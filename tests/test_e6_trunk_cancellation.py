"""E6: cancellation of the historical trunk is kept beside the whole-shared-set one."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.federated import negative_transfer as nt
from src.federated.negative_transfer import cancellation_from_gram
from src.federated.server import FedPerStrategy


def _updates(decoder_b):
    return [
        {"dataset": "BUSI", "task": "seg", "client_id": "a",
         "arrays": [np.asarray([1.0, 0.0], np.float32), np.asarray([1.0, 0.0], np.float32)]},
        {"dataset": "ISIC", "task": "seg", "client_id": "b",
         "arrays": [np.asarray([1.0, 0.0], np.float32), np.asarray(decoder_b, np.float32)]},
    ]


class TrunkGramTests(unittest.TestCase):
    def _strategy(self, keys):
        strategy = FedPerStrategy(task_weights={"seg": 1, "cls": 1}, shared_key_names=keys)
        strategy._sent_arrays = [np.zeros(2, np.float32), np.zeros(2, np.float32)]
        return strategy

    def test_wide_sharing_keeps_a_trunk_only_gram(self):
        strategy = self._strategy(["encoder2.weight", "decoder1.weight"])
        strategy._gradient_conflict(_updates([-1.0, 0.0]))
        trunk = strategy._last_trunk_gram
        self.assertIsNotNone(trunk)
        # Trunk deltas agree; the decoder ones cancel and must not leak into the trunk gram.
        np.testing.assert_allclose(trunk, [[1.0, 1.0], [1.0, 1.0]])
        weights = [0.5, 0.5]
        datasets = ["BUSI", "ISIC"]
        self.assertAlmostEqual(cancellation_from_gram(trunk, weights, datasets)["overall"], 0.0)
        self.assertGreater(
            cancellation_from_gram(strategy._last_total_gram, weights, datasets)["overall"], 0.2)

    def test_trunk_only_sharing_records_nothing_extra(self):
        strategy = self._strategy(["encoder2.weight", "bottleneck.weight"])
        strategy._gradient_conflict(_updates([1.0, 0.0]))
        self.assertIsNone(strategy._last_trunk_gram)

    def test_unnamed_keys_record_nothing_extra(self):
        strategy = self._strategy(None)
        strategy._gradient_conflict(_updates([1.0, 0.0]))
        self.assertIsNone(strategy._last_trunk_gram)


class TrunkSummaryTests(unittest.TestCase):
    def test_summary_reads_the_trunk_field_and_writes_its_csvs(self):
        history = [
            {"stage": "fit", "round": r,
             "negative_transfer": {"overall": 0.3, "intra": 0.1, "inter": 0.2},
             "negative_transfer_trunk": {"overall": 0.05, "intra": 0.01, "inter": 0.04}}
            for r in (1, 2, 3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "fold_0").mkdir()
            (run / "fold_0" / "aggregation_history.json").write_text(json.dumps(history))
            _, whole = nt.summarize_run(run)
            _, trunk = nt.summarize_run(run, field=nt.TRUNK_FIELD)
            self.assertAlmostEqual(whole.iloc[-1]["overall_mean"], 0.3)
            self.assertAlmostEqual(trunk.iloc[-1]["overall_mean"], 0.05)
            nt._report_trunk(run)
            self.assertTrue((run / nt.TRUNK_ROUNDS_FILE).exists())
            self.assertTrue((run / nt.TRUNK_SUMMARY_FILE).exists())

    def test_history_without_trunk_field_writes_no_trunk_files(self):
        history = [{"stage": "fit", "round": 1,
                    "negative_transfer": {"overall": 0.3, "intra": 0.1, "inter": 0.2}}]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "fold_0").mkdir()
            (run / "fold_0" / "aggregation_history.json").write_text(json.dumps(history))
            self.assertIsNone(nt._report_trunk(run))
            self.assertFalse((run / nt.TRUNK_SUMMARY_FILE).exists())


if __name__ == "__main__":
    unittest.main()
