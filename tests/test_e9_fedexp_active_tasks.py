"""E9: FedExP server step and the single-task ablation (``federated.active_tasks``).

FedExP (Jhunjhunwala et al., ICLR 2023, Eq. 9) extrapolates the weighted mean update by
``max(1, sum w_i ||d_i||^2 / (2 (||sum w_i d_i||^2 + eps)))``. Absent, both keys must leave the
historical FedAvg path byte for byte, and both stay out of ``src/config.yaml`` (study hash).
"""
import copy
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.federated.config import (active_tasks, server_optimizer, validate_federated_config)
from src.federated.server import FedPerStrategy, average_iterates, fedexp_step

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "studies" / "e8_layer_mixing" / "resnet34_256_control.yaml"
E9 = ROOT / "studies" / "e9_esann"


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _arrays(*values):
    return [np.asarray(values, dtype=np.float32)]


class ServerOptimizerConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = _load(CONTROL)

    def _with(self, value):
        config = copy.deepcopy(self.config)
        config["federated"]["server_optimizer"] = value
        return config

    def test_absent_or_fedavg_is_none(self):
        self.assertIsNone(server_optimizer(self.config))
        self.assertIsNone(server_optimizer(self._with({"name": "fedavg"})))

    def test_fedexp_defaults_follow_the_paper(self):
        resolved = server_optimizer(self._with({"name": "fedexp"}))
        self.assertEqual(resolved, {"name": "fedexp", "epsilon": 1e-3,
                                    "final_iterate": "average_last_two"})

    def test_invalid_values_are_refused(self):
        for bad in ({"name": "fedadam"}, {"name": "fedexp", "epsilon": -1},
                    {"name": "fedexp", "epsilon": True},
                    {"name": "fedexp", "final_iterate": "best"}, "fedexp"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                server_optimizer(self._with(bad))

    def test_refused_on_standalone_and_with_layer_mixing(self):
        config = self._with({"name": "fedexp"})
        config["federated"]["standalone"] = True
        with self.assertRaises(ValueError):
            server_optimizer(config)
        mixing = _load(ROOT / "studies" / "e8_layer_mixing" / "resnet34_256_layer_mixing.yaml")
        mixing["federated"]["server_optimizer"] = {"name": "fedexp"}
        with self.assertRaises(ValueError):
            server_optimizer(mixing)

    def test_not_in_the_working_default_config(self):
        default = _load(ROOT / "src" / "config.yaml")
        self.assertNotIn("server_optimizer", default["federated"])
        self.assertNotIn("active_tasks", default["federated"])


class ActiveTasksConfigTests(unittest.TestCase):
    def test_default_is_both_tasks_in_canonical_order(self):
        config = {"federated": {}}
        self.assertEqual(active_tasks(config), ("seg", "cls"))
        config["federated"]["active_tasks"] = ["cls", "seg"]
        self.assertEqual(active_tasks(config), ("seg", "cls"))
        config["federated"]["active_tasks"] = ["cls"]
        self.assertEqual(active_tasks(config), ("cls",))

    def test_invalid_values_are_refused(self):
        for bad in ([], ["det"], ["seg", "seg"], "seg"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                active_tasks({"federated": {"active_tasks": bad}})


class FedExpStepTests(unittest.TestCase):
    def test_identical_updates_reduce_to_fedavg(self):
        sent = _arrays(0.0, 0.0)
        clients = [_arrays(1.0, 2.0), _arrays(1.0, 2.0)]
        weights = np.array([0.5, 0.5])
        new, step = fedexp_step(sent, clients, weights, _arrays(1.0, 2.0), epsilon=0.0)
        self.assertEqual(step["eta_g"], 1.0)
        self.assertAlmostEqual(step["eta_raw"], 0.5)
        np.testing.assert_allclose(new[0], [1.0, 2.0])
        self.assertAlmostEqual(step["effective_step_ratio"], 1.0)

    def test_orthogonal_equal_updates_extrapolate_by_m_over_two(self):
        m = 4
        sent = [np.zeros(m, np.float32)]
        clients = [[np.eye(m, dtype=np.float32)[i]] for i in range(m)]
        weights = np.full(m, 1.0 / m)
        mean = [np.full(m, 1.0 / m, np.float32)]
        new, step = fedexp_step(sent, clients, weights, mean, epsilon=0.0)
        self.assertAlmostEqual(step["eta_g"], m / 2)
        np.testing.assert_allclose(new[0], np.full(m, 0.5), rtol=1e-6)
        # FedAvg moves 1/sqrt(M) of a client step; FedExP moves M/2 times that.
        self.assertAlmostEqual(step["effective_step_ratio"], (m / 2) / np.sqrt(m), places=6)

    def test_uniform_weights_match_the_paper_formula(self):
        rng = np.random.default_rng(0)
        m, eps = 5, 1e-3
        sent = [rng.normal(size=(3, 4)).astype(np.float32), rng.normal(size=7).astype(np.float32)]
        clients = [[a + rng.normal(scale=0.1, size=a.shape).astype(np.float32) for a in sent]
                   for _ in range(m)]
        weights = np.full(m, 1.0 / m)
        mean = [np.mean([c[i] for c in clients], axis=0) for i in range(len(sent))]
        _, step = fedexp_step(sent, clients, weights, mean, epsilon=eps)
        deltas = [np.concatenate([(c[i] - sent[i]).ravel() for i in range(len(sent))])
                  .astype(np.float64) for c in clients]
        mean_delta = np.mean(deltas, axis=0)
        expected = max(1.0, sum(d @ d for d in deltas) / (2 * m * (mean_delta @ mean_delta + eps)))
        self.assertAlmostEqual(step["eta_g"], expected, places=4)

    def test_large_epsilon_falls_back_to_fedavg_and_zero_update_is_safe(self):
        sent = _arrays(0.0, 0.0)
        clients = [_arrays(1.0, 0.0), _arrays(0.0, 1.0)]
        mean = _arrays(0.5, 0.5)
        _, step = fedexp_step(sent, clients, np.array([0.5, 0.5]), mean, epsilon=1e6)
        self.assertEqual(step["eta_g"], 1.0)
        new, step = fedexp_step(sent, [sent, sent], np.array([0.5, 0.5]), sent, epsilon=0.0)
        self.assertEqual(step["eta_g"], 1.0)
        np.testing.assert_array_equal(new[0], sent[0])

    def test_dtype_preserved_and_average_iterates(self):
        sent = [np.zeros(2, np.float32)]
        new, _ = fedexp_step(sent, [_arrays(1.0, 0.0), _arrays(0.0, 1.0)], np.array([0.5, 0.5]),
                             _arrays(0.5, 0.5), epsilon=0.0)
        self.assertEqual(new[0].dtype, np.float32)
        mean = average_iterates([np.asarray([0.0, 2.0], np.float32)], new)
        np.testing.assert_allclose(mean[0], (np.asarray([0.0, 2.0]) + new[0]) / 2)
        self.assertEqual(mean[0].dtype, np.float32)


class StrategyTests(unittest.TestCase):
    def _run_round(self, optimizer):
        from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

        strategy = FedPerStrategy(
            task_weights={"seg": 1, "cls": 1}, aggregation_mode="flat",
            client_weighting="uniform", server_optimizer=optimizer,
        )
        strategy._sent_arrays = [np.zeros(2, np.float32)]
        results = []
        for index, values in enumerate(([1.0, 0.0], [0.0, 1.0])):
            res = FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters([np.asarray(values, np.float32)]),
                num_examples=10,
                metrics={"task": "seg", "dataset": "BUSI", "client_id": f"c{index}"},
            )
            results.append((None, res))
        strategy.aggregate_fit(1, results, [])
        return strategy

    def test_fedavg_path_is_unchanged(self):
        strategy = self._run_round(None)
        np.testing.assert_allclose(strategy.final_arrays()[0], [0.5, 0.5])
        self.assertNotIn("server_optimizer", strategy.aggregation_history[-1])

    def test_fedexp_round_records_eta_and_averages_the_final_iterate(self):
        optimizer = {"name": "fedexp", "epsilon": 0.0, "final_iterate": "average_last_two"}
        strategy = self._run_round(optimizer)
        record = strategy.aggregation_history[-1]["server_optimizer"]
        self.assertAlmostEqual(record["eta_g"], 1.0)  # M=2 orthogonal: M/2 = 1
        # Three clients would extrapolate; here the trajectory equals FedAvg, the final is the
        # mean of the sent zeros and the new iterate.
        np.testing.assert_allclose(strategy.final_arrays()[0], [0.25, 0.25])
        optimizer["final_iterate"] = "last"
        np.testing.assert_allclose(self._run_round(optimizer).final_arrays()[0], [0.5, 0.5])


class ManifestTests(unittest.TestCase):
    def test_arms_differ_from_the_control_only_where_declared(self):
        control = _load(CONTROL)
        allowed = {"standalone", "server_optimizer", "active_tasks"}
        for path in sorted(E9.glob("*.yaml")):
            with self.subTest(arm=path.stem):
                arm = _load(path)
                validate_federated_config(arm)
                self.assertEqual(arm["experiment"]["arm_id"], path.stem)
                for section in set(control) | set(arm):
                    if section in {"experiment", "federated"}:
                        continue
                    self.assertEqual(arm.get(section), control.get(section), section)
                fed_arm, fed_control = arm["federated"], control["federated"]
                for key in set(fed_arm) | set(fed_control):
                    if key not in allowed:
                        self.assertEqual(fed_arm.get(key), fed_control.get(key), key)
                self.assertEqual(arm["experiment"]["setup"],
                                 "standalone" if fed_arm["standalone"] else "federated")

    def test_the_five_planned_arms_exist(self):
        self.assertEqual(
            sorted(path.stem for path in E9.glob("*.yaml")),
            sorted(f"resnet34_256_{name}" for name in
                   ("fedexp", "fed_seg", "fed_cls", "local_seg", "local_cls")),
        )


if __name__ == "__main__":
    unittest.main()
