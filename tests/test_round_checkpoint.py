"""Round-level resume: checkpoint contents, client restore, and the strategy's round offset."""
import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

from src.federated import round_checkpoint
from src.federated.server import FedPerStrategy
from src.utils.training_runtime import validate_runtime_config

CLIENTS = ["A_mt_0", "B_seg_0"]


def _write_state(fold_dir, client_id, server_round, value):
    client_dir = Path(fold_dir) / f"client_{client_id}"
    client_dir.mkdir(parents=True, exist_ok=True)
    path = client_dir / "state.pt"
    temporary = client_dir / "state.pt.tmp"
    torch.save({"round": server_round, "model": {"w": torch.full((3,), float(value))}}, temporary)
    os.replace(temporary, path)  # the same write-then-replace the client uses
    return path


def _write_history(fold_dir, client_id, rounds):
    path = Path(fold_dir) / f"client_{client_id}" / "training_history.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["round", "metric_name", "value"])
        writer.writeheader()
        writer.writerows({"round": r, "metric_name": "loss", "value": r} for r in rounds)
    return path


class RoundCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fold = Path(self.tmp.name) / "fold_0"
        for client in CLIENTS:
            _write_state(self.fold, client, 4, 4.0)

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self, server_round=4):
        round_checkpoint.save(self.fold, server_round, [np.ones(2)], [{"round": 1}], 0.5, 3, CLIENTS)

    def test_checkpoint_survives_later_rounds_and_restores(self):
        self._save()
        for client in CLIENTS:  # rounds 5 and 6 run, then the run crashes
            _write_state(self.fold, client, 6, 6.0)
            _write_history(self.fold, client, range(1, 7))
        payload = round_checkpoint.load(self.fold, CLIENTS, total_rounds=10)
        self.assertEqual(payload["round"], 4)
        np.testing.assert_array_equal(payload["arrays"][0], np.ones(2))
        self.assertEqual((payload["best_mean_val"], payload["best_round"]), (0.5, 3))
        round_checkpoint.restore_clients(self.fold, payload)
        for client in CLIENTS:
            state = torch.load(self.fold / f"client_{client}" / "state.pt")
            self.assertEqual(state["round"], 4)
            self.assertTrue(torch.equal(state["model"]["w"], torch.full((3,), 4.0)))
            with (self.fold / f"client_{client}" / "training_history.csv").open() as stream:
                self.assertEqual([int(r["round"]) for r in csv.DictReader(stream)], [1, 2, 3, 4])
        # Restored states are links into the checkpoint, so the next round must not corrupt it.
        _write_state(self.fold, CLIENTS[0], 5, 5.0)
        again = round_checkpoint.load(self.fold, CLIENTS, total_rounds=10)
        self.assertEqual(again["round"], 4)

    def test_refuses_clients_from_another_round(self):
        _write_state(self.fold, CLIENTS[1], 3, 3.0)
        with self.assertRaisesRegex(RuntimeError, "not from round 4"):
            self._save()
        self.assertIsNone(round_checkpoint.load(self.fold, CLIENTS, total_rounds=10))

    def test_newer_checkpoint_replaces_older(self):
        self._save()
        for client in CLIENTS:
            _write_state(self.fold, client, 8, 8.0)
        self._save(8)
        self.assertEqual(round_checkpoint.load(self.fold, CLIENTS, total_rounds=10)["round"], 8)
        self.assertFalse((self.fold / "round_checkpoint.old").exists())

    def test_final_round_or_other_roster_is_not_a_resume_point(self):
        self._save()
        self.assertIsNone(round_checkpoint.load(self.fold, CLIENTS, total_rounds=4))
        with self.assertRaisesRegex(ValueError, "roster"):
            round_checkpoint.load(self.fold, CLIENTS[:1], total_rounds=10)

    def test_discard(self):
        self._save()
        round_checkpoint.discard(self.fold)
        self.assertIsNone(round_checkpoint.load(self.fold, CLIENTS, total_rounds=10))


class _Result:
    def __init__(self, client_id):
        self.num_examples = 4
        self.loss = 1.0
        self.metrics = {"task": "seg", "dataset": "A", "client_id": client_id, "val_metric": 0.5}
        self.parameters = ndarrays_to_parameters([np.ones(2)])


class StrategyOffsetTests(unittest.TestCase):
    def _strategy(self, offset, calls):
        return FedPerStrategy(
            task_weights={"seg": 1.0, "cls": 1.0}, aggregation_mode="flat",
            round_offset=offset, checkpoint_fn=calls.append,
            on_fit_config_fn=lambda r: {"server_round": r},
            on_evaluate_config_fn=lambda r: {"server_round": r},
            accept_failures=False,
        )

    def test_clients_and_history_see_the_global_round(self):
        calls = []
        strategy = self._strategy(30, calls)
        manager = mock.Mock()
        manager.num_available.return_value = 1
        manager.sample.return_value = [mock.Mock()]
        params = ndarrays_to_parameters([np.zeros(2)])
        (_, fit_ins), = strategy.configure_fit(1, params, manager)
        self.assertEqual(fit_ins.config["server_round"], 31)
        (_, eval_ins), = strategy.configure_evaluate(1, params, manager)
        self.assertEqual(eval_ins.config["server_round"], 31)

        strategy.aggregate_fit(1, [(None, _Result("c0"))], [])
        self.assertEqual(strategy.latest_round, 31)
        strategy.aggregate_evaluate(1, [(None, _Result("c0"))], [])
        self.assertEqual(calls, [31])
        self.assertEqual({row["round"] for row in strategy.aggregation_history}, {31})

    def test_no_checkpoint_after_a_failed_round(self):
        calls = []
        strategy = self._strategy(0, calls)
        strategy.aggregate_fit(1, [], [RuntimeError("actor died")])
        strategy.aggregate_evaluate(1, [(None, _Result("c0"))], [])
        self.assertEqual(calls, [])  # fit was not aggregated
        strategy.aggregate_fit(2, [(None, _Result("c0"))], [])
        strategy.aggregate_evaluate(2, [], [RuntimeError("actor died")])
        self.assertEqual(calls, [])

    def test_checkpoint_errors_never_fail_the_round(self):
        strategy = self._strategy(0, [])
        strategy.checkpoint_fn = mock.Mock(side_effect=OSError("disk full"))
        strategy.aggregate_fit(1, [(None, _Result("c0"))], [])
        loss, _ = strategy.aggregate_evaluate(1, [(None, _Result("c0"))], [])
        self.assertAlmostEqual(loss, 1.0)


class RuntimeConfigTests(unittest.TestCase):
    def test_checkpoint_interval_is_validated(self):
        for value in (10, 0):
            validate_runtime_config({"runtime": {"federated": {"checkpoint_every_rounds": value}}})
        for value in (-1, True, 2.5):
            with self.assertRaisesRegex(ValueError, "checkpoint_every_rounds"):
                validate_runtime_config(
                    {"runtime": {"federated": {"checkpoint_every_rounds": value}}}
                )


if __name__ == "__main__":
    unittest.main()
