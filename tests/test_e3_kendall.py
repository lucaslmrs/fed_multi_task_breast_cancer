"""E3.4 (c): Kendall uncertainty weighting (``loss.task_weighting: uncertainty``).

Kendall et al. 2018, Eq. 10, classification form for both tasks: ``sum_t lambda_t e^{-s_t} L_t +
s_t/2`` with ``s_t = log sigma_t^2`` learned per task. The log-variances are personalized (never in
the shared trunk), absent tasks contribute nothing, and ``fixed`` keeps today's objective.
"""
import unittest

import torch

from src.federated.local_trainer import local_objective, train_local
from src.federated.model_split import get_personalized_state, set_personalized_state, shared_keys
from src.models.multitask.MTnnUNet import MTnnUNet
from src.utils.experiment_init import init_criterion_classification, init_criterion_segmentation
from src.utils.supervision import (LOG_VARS_ATTR, attach_saved_task_uncertainty, attach_task_uncertainty,
                                   task_weighting_mode, uncertainty_weighted_loss)
from src.utils.training_runtime import PrecisionPolicy

LAMBDAS = {"seg": 0.8, "cls": 0.2}


def _batch(seed=0, has_label=None, size=64):
    generator = torch.Generator().manual_seed(seed)
    return {"image": torch.rand(2, 1, size, size, generator=generator),
            "mask": (torch.rand(2, 1, size, size, generator=generator) > 0.7).float(),
            "label": torch.randint(0, 3, (2,), generator=generator),
            "has_mask": torch.ones(2, dtype=torch.bool),
            "has_label": torch.ones(2, dtype=torch.bool) if has_label is None else has_label}


def _model(seed=0, uncertainty=False, tasks=("seg", "cls")):
    torch.manual_seed(seed)
    model = MTnnUNet(sequences=1, regions=1, n_classes=3)
    if uncertainty:
        attach_task_uncertainty(model, tasks)
    return model


def _objective(model, data):
    return local_objective(
        model, data, data["image"], ("seg", "cls"), LAMBDAS, "cpu", 3,
        init_criterion_segmentation("DiceBCE"), init_criterion_classification(n_classes=3, device="cpu"),
        True, PrecisionPolicy("fp32", "cpu"))


class ModeTests(unittest.TestCase):
    def test_default_is_fixed_and_typos_are_refused(self):
        self.assertEqual(task_weighting_mode({}), "fixed")
        self.assertEqual(task_weighting_mode({"loss": {"task_weighting": "Uncertainty"}}), "uncertainty")
        with self.assertRaises(ValueError):
            task_weighting_mode({"loss": {"task_weighting": "kendal"}})


class AttachTests(unittest.TestCase):
    def test_single_task_client_is_untouched(self):
        model = _model(uncertainty=True, tasks=("seg",))
        self.assertFalse(hasattr(model, LOG_VARS_ATTR))

    def test_log_vars_start_at_zero_and_are_personalized(self):
        model = _model(uncertainty=True)
        log_vars = getattr(model, LOG_VARS_ATTR)
        self.assertEqual(set(log_vars), {"seg", "cls"})
        self.assertTrue(all(float(v) == 0.0 for v in log_vars.values()))
        keys = {f"{LOG_VARS_ATTR}.seg", f"{LOG_VARS_ATTR}.cls"}
        for share_stem in (False, True):
            self.assertFalse(keys & set(shared_keys(model, share_stem)))
            self.assertTrue(keys <= set(get_personalized_state(model, share_stem)))

    def test_attaching_does_not_change_the_shared_trunk(self):
        self.assertEqual(shared_keys(_model(), False), shared_keys(_model(uncertainty=True), False))

    def test_saved_state_loads_strictly_into_a_fresh_model(self):
        # Final evaluation rebuilds the model and loads state.pt strictly (regression: E3 smoke).
        trained = _model(uncertainty=True)
        with torch.no_grad():
            getattr(trained, LOG_VARS_ATTR)["seg"].fill_(0.25)
        personalized = get_personalized_state(trained, sharing=False)
        fresh = attach_saved_task_uncertainty(_model(seed=1), personalized)
        set_personalized_state(fresh, personalized, sharing=False)
        self.assertEqual(float(getattr(fresh, LOG_VARS_ATTR)["seg"]), 0.25)
        full = attach_saved_task_uncertainty(_model(seed=1), trained.state_dict())
        full.load_state_dict(trained.state_dict())

    def test_saved_state_without_log_vars_is_a_no_op(self):
        fresh = attach_saved_task_uncertainty(_model(), _model(seed=1).state_dict())
        self.assertFalse(hasattr(fresh, LOG_VARS_ATTR))


class LossTests(unittest.TestCase):
    def test_at_initialisation_equals_the_fixed_objective(self):
        raw = {"seg": torch.tensor(0.7), "cls": torch.tensor(1.3)}
        log_vars = {name: torch.zeros(()) for name in raw}
        torch.testing.assert_close(uncertainty_weighted_loss(raw, LAMBDAS, log_vars),
                                   torch.tensor(0.8 * 0.7 + 0.2 * 1.3))

    def test_formula_and_gradient_on_log_var(self):
        raw = {"seg": torch.tensor(0.7), "cls": torch.tensor(1.3)}
        log_vars = {"seg": torch.tensor(0.5, requires_grad=True),
                    "cls": torch.tensor(-0.3, requires_grad=True)}
        loss = uncertainty_weighted_loss(raw, LAMBDAS, log_vars)
        expected = sum(LAMBDAS[t] * torch.exp(-log_vars[t].detach()) * raw[t] + 0.5 * log_vars[t].detach()
                       for t in raw)
        torch.testing.assert_close(loss.detach(), expected)
        loss.backward()
        for task in raw:  # d/ds = -lambda e^{-s} L + 1/2
            torch.testing.assert_close(
                log_vars[task].grad,
                -LAMBDAS[task] * torch.exp(-log_vars[task].detach()) * raw[task] + 0.5)

    def test_absent_task_contributes_nothing(self):
        log_vars = {"seg": torch.tensor(0.0, requires_grad=True),
                    "cls": torch.tensor(2.0, requires_grad=True)}
        loss = uncertainty_weighted_loss({"seg": torch.tensor(0.7)}, LAMBDAS, log_vars)
        torch.testing.assert_close(loss.detach(), torch.tensor(0.8 * 0.7))
        loss.backward()
        self.assertIsNone(log_vars["cls"].grad)


class LocalObjectiveTests(unittest.TestCase):
    def test_fixed_model_keeps_the_combined_loss(self):
        data = _batch()
        loss, _, raw, _, _ = _objective(_model(), data)
        torch.testing.assert_close(loss, LAMBDAS["seg"] * raw["seg"] + LAMBDAS["cls"] * raw["cls"])

    def test_uncertainty_model_uses_the_weighted_loss(self):
        data = _batch()
        model = _model(uncertainty=True)
        with torch.no_grad():
            getattr(model, LOG_VARS_ATTR)["cls"].fill_(1.0)
        loss, _, raw, _, _ = _objective(model, data)
        expected = (LAMBDAS["seg"] * raw["seg"]
                    + LAMBDAS["cls"] * torch.exp(torch.tensor(-1.0)) * raw["cls"] + 0.5)
        torch.testing.assert_close(loss, expected)

    def test_batch_without_labels_skips_the_cls_term(self):
        data = _batch(has_label=torch.zeros(2, dtype=torch.bool))
        model = _model(uncertainty=True)
        loss, counted, raw, _, _ = _objective(model, data)
        self.assertEqual(counted, ["seg"])
        torch.testing.assert_close(loss, LAMBDAS["seg"] * raw["seg"])

    def test_training_updates_the_log_vars(self):
        model = _model(uncertainty=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        batches = [_batch(seed) for seed in range(3)]
        train_local(model, batches, optimizer, "seg+cls", "cpu", 1, 3,
                    init_criterion_segmentation("DiceBCE"),
                    init_criterion_classification(n_classes=3, device="cpu"), True,
                    training_mode="steps", steps_per_round=3, tasks=("seg", "cls"),
                    task_lambdas=LAMBDAS, precision=PrecisionPolicy("fp32", "cpu"))
        log_vars = getattr(model, LOG_VARS_ATTR)
        self.assertTrue(all(float(v) != 0.0 for v in log_vars.values()))
        self.assertTrue(all(torch.isfinite(v) for v in log_vars.values()))


if __name__ == "__main__":
    unittest.main()
