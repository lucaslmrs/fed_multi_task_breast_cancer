import unittest

import pandas as pd
import torch

from src.experiments.validated_interpolation import (
    check_pair,
    interpolate_state,
    relative_delta,
    select_lambda,
)


class InterpolateStateTests(unittest.TestCase):
    def setUp(self):
        self.local = {"w": torch.tensor([0.0, 2.0]), "steps": torch.tensor(3)}
        self.fed = {"w": torch.tensor([1.0, 4.0]), "steps": torch.tensor(3)}

    def test_endpoints_reproduce_the_arms(self):
        self.assertTrue(torch.equal(interpolate_state(self.local, self.fed, 0.0)["w"], self.local["w"]))
        self.assertTrue(torch.equal(interpolate_state(self.local, self.fed, 1.0)["w"], self.fed["w"]))

    def test_midpoint_and_extrapolation(self):
        self.assertTrue(torch.allclose(interpolate_state(self.local, self.fed, 0.5)["w"],
                                       torch.tensor([0.5, 3.0])))
        self.assertTrue(torch.allclose(interpolate_state(self.local, self.fed, 1.5)["w"],
                                       torch.tensor([1.5, 5.0])))

    def test_non_float_tensors_must_match(self):
        fed = {**self.fed, "steps": torch.tensor(4)}
        with self.assertRaises(ValueError):
            interpolate_state(self.local, fed, 0.5)

    def test_key_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            interpolate_state(self.local, {"w": self.fed["w"]}, 0.5)


def _grid(rows):
    return relative_delta(pd.DataFrame(rows, columns=["client_id", "task", "split", "lam", "value"]))


class SelectLambdaTests(unittest.TestCase):
    def test_multitask_client_gets_one_lambda_from_the_task_mean(self):
        grid = _grid([
            ("a", "seg", "val", 0.0, 0.50), ("a", "cls", "val", 0.0, 0.50),
            ("a", "seg", "val", 0.5, 0.60), ("a", "cls", "val", 0.5, 0.45),  # mean +5%
            ("a", "seg", "val", 1.0, 0.45), ("a", "cls", "val", 1.0, 0.60),  # mean +5%
        ])
        chosen = select_lambda(grid, selectable=(0.0, 0.5, 1.0))
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen.iloc[0]["lam_star"], 0.5)  # tie goes to the smaller lambda

    def test_never_worse_than_local_on_val(self):
        grid = _grid([("b", "cls", "val", 0.0, 0.70), ("b", "cls", "val", 1.0, 0.60)])
        chosen = select_lambda(grid, selectable=(0.0, 1.0))
        self.assertEqual(chosen.iloc[0]["lam_star"], 0.0)
        self.assertEqual(chosen.iloc[0]["val_dm"], 0.0)

    def test_diagnostic_lambdas_do_not_compete(self):
        grid = _grid([("c", "seg", "val", 0.0, 0.5), ("c", "seg", "val", 1.0, 0.6),
                      ("c", "seg", "val", 1.5, 0.9)])
        self.assertEqual(select_lambda(grid, selectable=(0.0, 1.0)).iloc[0]["lam_star"], 1.0)

    def test_test_split_never_feeds_selection(self):
        grid = _grid([("d", "seg", "val", 0.0, 0.5), ("d", "seg", "val", 1.0, 0.4),
                      ("d", "seg", "test", 0.0, 0.5), ("d", "seg", "test", 1.0, 0.9)])
        self.assertEqual(select_lambda(grid, selectable=(0.0, 1.0)).iloc[0]["lam_star"], 0.0)


class CheckPairTests(unittest.TestCase):
    def _configs(self):
        fed = {"federated": {"standalone": False, "rounds": 200},
               "experiment": {"arm_id": "primary", "setup": "federated", "seed": 1}}
        local = {"federated": {"standalone": True, "rounds": 200},
                 "experiment": {"arm_id": "local_only", "setup": "standalone", "seed": 1}}
        return fed, local

    def test_matched_pair_passes(self):
        check_pair(*self._configs())

    def test_any_other_difference_is_refused(self):
        fed, local = self._configs()
        local["federated"]["rounds"] = 100
        with self.assertRaises(ValueError):
            check_pair(fed, local)

    def test_arm_order_is_enforced(self):
        fed, local = self._configs()
        with self.assertRaises(ValueError):
            check_pair(local, fed)


if __name__ == "__main__":
    unittest.main()
