"""E3.4 (a)/(b): batch Dice and the ``loss.task_lambdas`` decoupling.

Both keys are optional and absent from ``src/config.yaml``: every study hashes its whole config, so
adding them there would invalidate completed arms. Absent, behaviour must be identical to before.
"""
import copy
import unittest
from pathlib import Path

import torch
import yaml
from monai.losses import DiceLoss

from src.federated.client import _task_lambdas, client_task_lambdas
from src.federated.config import aggregation_config
from src.utils.experiment_init import init_criterion_segmentation, init_criterion_segmentation_from_config
from src.utils.segmentation_loss import DiceBCELoss

ROOT = Path(__file__).resolve().parents[1]
BCE = torch.nn.functional.binary_cross_entropy_with_logits


def _batch(seed=0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(4, 1, 8, 8, generator=generator)
    target = (torch.rand(4, 1, 8, 8, generator=generator) > 0.7).float()
    target[0] = 0  # an empty mask, the case batch Dice changes
    return logits, target


def _manual_dice(logits, target, reduce_dims):
    p = torch.sigmoid(logits)
    intersection = (p * target).sum(reduce_dims)
    denominator = (p ** 2).sum(reduce_dims) + (target ** 2).sum(reduce_dims)
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


class BatchDiceTests(unittest.TestCase):
    def test_default_is_unchanged_per_sample_dice(self):
        logits, target = _batch()
        legacy = DiceLoss(include_background=True, sigmoid=True, smooth_dr=1, smooth_nr=1,
                          squared_pred=True)
        expected = .5 * legacy(logits, target) + .5 * BCE(logits, target)
        self.assertTrue(torch.equal(DiceBCELoss()(logits, target), expected))
        self.assertTrue(torch.equal(DiceBCELoss(batch=False)(logits, target), expected))

    def test_batch_pools_intersection_and_union_over_the_batch(self):
        logits, target = _batch(1)
        per_sample = _manual_dice(logits, target, reduce_dims=(1, 2, 3))
        pooled = _manual_dice(logits, target, reduce_dims=(0, 1, 2, 3))
        torch.testing.assert_close(DiceBCELoss(1, 0)(logits, target), per_sample)
        torch.testing.assert_close(DiceBCELoss(1, 0, batch=True)(logits, target), pooled)
        torch.testing.assert_close(DiceBCELoss(.8, .2, batch=True)(logits, target),
                                   .8 * pooled + .2 * BCE(logits, target))

    def test_empty_target_no_longer_scores_a_free_dice(self):
        # Predicting "nothing" everywhere: per-sample Dice rewards the empty image, batch Dice does not.
        logits = torch.full((2, 1, 8, 8), -20.0)
        target = torch.zeros(2, 1, 8, 8)
        target[1, :, 2:6, 2:6] = 1
        self.assertLess(DiceBCELoss(1, 0)(logits, target), 0.6)
        self.assertGreater(DiceBCELoss(1, 0, batch=True)(logits, target), 0.9)

    def test_gradient_flows_with_batch_dice(self):
        logits, target = _batch(2)
        logits.requires_grad_(True)
        DiceBCELoss(batch=True)(logits, target).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum(), 0)

    def test_batch_must_be_boolean(self):
        with self.assertRaises(ValueError):
            DiceBCELoss(batch=1)


class SegmentationCriterionFromConfigTests(unittest.TestCase):
    def test_absent_key_keeps_per_sample_dice(self):
        criterion = init_criterion_segmentation_from_config(
            {"function": "DiceBCE", "dice_weight": .5, "bce_weight": .5})
        self.assertFalse(criterion.dice.batch)

    def test_keys_reach_the_criterion(self):
        criterion = init_criterion_segmentation_from_config(
            {"function": "DiceBCE", "dice_weight": .8, "bce_weight": .2, "dice_batch": True})
        self.assertTrue(criterion.dice.batch)
        self.assertEqual((criterion.dice_weight, criterion.bce_weight), (.8, .2))

    def test_batch_dice_refused_outside_dicebce(self):
        with self.assertRaises(ValueError):
            init_criterion_segmentation("FocalDICE", dice_batch=True)
        with self.assertRaises(ValueError):
            init_criterion_segmentation_from_config({"function": "DICE", "dice_batch": True})


def _config(task_weights=None, task_lambdas=None):
    config = {"federated": {"aggregation": {"task_weights": task_weights or {"seg": 4.0, "cls": 1.0}}},
              "loss": {"function": "DiceBCE"}}
    if task_lambdas is not None:
        config["loss"]["task_lambdas"] = task_lambdas
    return config


class TaskLambdaTests(unittest.TestCase):
    def test_absent_key_follows_aggregation_task_weights(self):
        config = _config()
        expected = _task_lambdas({"seg": 4.0, "cls": 1.0}, ("seg", "cls"))
        self.assertEqual(client_task_lambdas(config, ("seg", "cls")), expected)
        self.assertEqual(expected, {"seg": .8, "cls": .2})

    def test_explicit_lambdas_leave_aggregation_untouched(self):
        config = _config(task_lambdas={"seg": 1, "cls": 1})
        before = copy.deepcopy(aggregation_config(config)["task_weights"])
        self.assertEqual(client_task_lambdas(config, ("seg", "cls")), {"seg": .5, "cls": .5})
        self.assertEqual(aggregation_config(config)["task_weights"], before)
        self.assertEqual(before, {"seg": 4.0, "cls": 1.0})

    def test_single_task_client_gets_unit_weight(self):
        config = _config(task_lambdas={"seg": 0.3, "cls": 0.7})
        self.assertEqual(client_task_lambdas(config, ("cls",)), {"cls": 1.0})

    def test_invalid_explicit_lambdas_are_refused(self):
        for bad in ({"seg": 1.0}, {"seg": -1.0, "cls": 1.0}, {"seg": float("nan"), "cls": 1.0},
                    {"seg": True, "cls": 1.0}, [0.5, 0.5]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                client_task_lambdas(_config(task_lambdas=bad), ("seg", "cls"))


class StudyHashGuardTests(unittest.TestCase):
    def test_base_config_does_not_define_the_new_keys(self):
        # Defining them in src/config.yaml would change config_sha256 for every manifest built on it.
        loss = yaml.safe_load((ROOT / "src" / "config.yaml").read_text(encoding="utf-8"))["loss"]
        self.assertNotIn("dice_batch", loss)
        self.assertNotIn("task_lambdas", loss)


if __name__ == "__main__":
    unittest.main()
