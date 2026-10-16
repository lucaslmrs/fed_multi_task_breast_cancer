"""Online, read-only gradient guards (plano E1.2): they must never change a healthy run."""

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader

from src.federated.local_trainer import train_local
from src.federated.server import FedPerStrategy
from src.utils.training_runtime import runtime_config, validate_runtime_config
from tests.test_conflict_diagnostics import _client
from tests.test_local_training_budget import _DictionaryDataset, _TinySegmentationModel


def _train(client, grad_norm, record_step_history=True):
    optimizer = torch.optim.Adam(client.model.parameters(), lr=1e-3)
    result = train_local(client.model, client.train_loader, optimizer, client.task, "cpu", 1, 3,
                         client.seg_criterion, client.cls_criterion, True, training_mode="steps",
                         steps_per_round=len(client.train_loader), tasks=client.tasks,
                         task_lambdas=client.task_lambdas, precision=client.precision,
                         record_step_history=record_step_history, grad_norm=grad_norm)
    return result, optimizer


class GradientNormTests(unittest.TestCase):
    def test_measuring_the_norm_does_not_perturb_training(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, measured = _client(directory), _client(directory)
            torch.manual_seed(42)
            plain, plain_optimizer = _train(reference, grad_norm=False)
            torch.manual_seed(42)
            guarded, guarded_optimizer = _train(measured, grad_norm=True)
        for (name, a), (_, b) in zip(reference.model.state_dict().items(),
                                     measured.model.state_dict().items()):
            self.assertTrue(torch.equal(a, b), name)
        for a, b in zip(plain_optimizer.state.values(), guarded_optimizer.state.values()):
            self.assertTrue(torch.equal(a["exp_avg_sq"], b["exp_avg_sq"]))
        self.assertEqual(plain["loss"], guarded["loss"])
        self.assertEqual(plain["task_losses"], guarded["task_losses"])
        non_norm = [row for row in guarded["history"] if row["metric_name"] != "grad_norm"]
        self.assertEqual(plain["history"], non_norm)

    def test_norm_is_recorded_per_step_and_summarised(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = _train(_client(directory), grad_norm=True)
        norms = [row["value"] for row in result["history"] if row["metric_name"] == "grad_norm"]
        self.assertEqual(len(norms), 4)
        self.assertTrue(all(np.isfinite(norms)) and min(norms) > 0)
        self.assertAlmostEqual(result["grad_norm_mean"], float(np.mean(norms)))
        self.assertEqual(result["grad_norm_max"], max(norms))

    def test_disabled_norm_reports_nan_and_no_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = _train(_client(directory), grad_norm=False)
        self.assertTrue(np.isnan(result["grad_norm_mean"]))
        self.assertFalse(any(row["metric_name"] == "grad_norm" for row in result["history"]))

    def test_epochs_mode_records_mean_and_max_per_epoch(self):
        model = _TinySegmentationModel()
        loader = DataLoader(_DictionaryDataset(4), batch_size=2, shuffle=False)
        result = train_local(model, loader, torch.optim.SGD(model.parameters(), lr=1e-3), "seg",
                             "cpu", 2, 2, seg_criterion=torch.nn.MSELoss(), training_mode="epochs",
                             inversely_weighted=False)
        names = [(row["metric_name"], row["unit_index"]) for row in result["history"]
                 if row["metric_name"].startswith("grad_norm")]
        self.assertEqual(names, [("grad_norm_mean", 1), ("grad_norm_max", 1),
                                 ("grad_norm_mean", 2), ("grad_norm_max", 2)])

    def test_infinite_gradient_behind_a_finite_loss_stops_before_the_step(self):
        model = _TinySegmentationModel()
        model.scale.register_hook(lambda grad: grad * float("inf"))
        loader = DataLoader(_DictionaryDataset(4), batch_size=2, shuffle=False)
        with self.assertRaisesRegex(FloatingPointError, "non-finite gradient norm"):
            train_local(model, loader, torch.optim.SGD(model.parameters(), lr=1e-3), "seg",
                        "cpu", 1, 2, seg_criterion=torch.nn.MSELoss(), training_mode="epochs",
                        inversely_weighted=False)
        self.assertEqual(float(model.scale), 1.0)


class ServerFiniteGuardTests(unittest.TestCase):
    @staticmethod
    def _results(*arrays):
        return [(None, SimpleNamespace(
            num_examples=10, parameters=ndarrays_to_parameters([array]),
            metrics={"dataset": "BUSI", "task": "seg", "client_id": f"c{index}"},
        )) for index, array in enumerate(arrays)]

    def test_non_finite_client_update_is_refused(self):
        strategy = FedPerStrategy(task_weights={"seg": 1})
        results = self._results(np.zeros(3, np.float32), np.asarray([0, np.nan, 1], np.float32))
        with self.assertRaisesRegex(FloatingPointError, "client c1 .* 1 non-finite"):
            strategy.aggregate_fit(1, results, [])

    def test_finite_updates_aggregate_as_before(self):
        strategy = FedPerStrategy(task_weights={"seg": 1})
        results = self._results(np.zeros(3, np.float32), np.full(3, 2.0, np.float32))
        parameters, _ = strategy.aggregate_fit(1, results, [])
        np.testing.assert_allclose(parameters_to_ndarrays(parameters)[0], np.ones(3))


class RuntimeConfigTests(unittest.TestCase):
    def test_new_diagnostics_defaults(self):
        diagnostics = runtime_config({})["diagnostics"]
        self.assertTrue(diagnostics["grad_norm"]["enabled"])
        self.assertEqual(diagnostics["training"],
                         {"enabled": True, "baseline_root": "runs/baseline"})

    def test_invalid_diagnostics_values_are_rejected(self):
        for runtime in ({"diagnostics": {"grad_norm": {"enabled": "yes"}}},
                        {"diagnostics": {"training": {"enabled": 1}}},
                        {"diagnostics": {"training": {"baseline_root": ""}}}):
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                validate_runtime_config({"runtime": runtime})


if __name__ == "__main__":
    unittest.main()
