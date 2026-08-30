import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.federated.config import aggregation_config, local_training_config, validate_federated_config
from src.federated.client import stable_client_seed
from src.federated.model_split import get_shared_state, set_shared_state, shared_keys
from src.federated.server import FedPerStrategy
from src.models.multitask.MTnnUNet import MTnnUNet


class ModelSplitTests(unittest.TestCase):
    def test_paired_arms_start_from_identical_client_state(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder1 = torch.nn.Linear(1, 2)
                self.encoder2 = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 1)

        seed = stable_client_seed(42, 1, "ISIC_2018_cls_0")
        torch.manual_seed(seed)
        federated = TinyModel()
        torch.manual_seed(seed)
        standalone = TinyModel()

        global_arrays = [np.full_like(array, 0.25) for array in get_shared_state(federated, False)]
        set_shared_state(federated, global_arrays, False)
        set_shared_state(standalone, global_arrays, False)
        for fed_value, local_value in zip(
            federated.state_dict().values(), standalone.state_dict().values()
        ):
            self.assertTrue(torch.equal(fed_value, local_value))

        self.assertEqual(
            stable_client_seed(42, 1, "ISIC_2018_cls_0", "fit", 2),
            stable_client_seed(42, 1, "ISIC_2018_cls_0", "fit", 2),
        )
        self.assertNotEqual(
            stable_client_seed(42, 1, "ISIC_2018_cls_0", "fit", 1),
            stable_client_seed(42, 1, "ISIC_2018_cls_0", "fit", 2),
        )

    def test_multimodal_trunk_shapes_match(self):
        gray = MTnnUNet(sequences=1, regions=1, n_classes=3)
        rgb = MTnnUNet(sequences=3, regions=1, n_classes=7)

        self.assertEqual(shared_keys(gray, False), shared_keys(rgb, False))
        self.assertEqual(
            [array.shape for array in get_shared_state(gray, False)],
            [array.shape for array in get_shared_state(rgb, False)],
        )
        self.assertNotEqual(
            [array.shape for array in get_shared_state(gray, True)],
            [array.shape for array in get_shared_state(rgb, True)],
        )

    def test_incompatible_shared_stem_fails_before_flower(self):
        config = {
            "model": {"sequences": 1},
            "data": {"dataset": "gray", "classes": ["a"], "augmentation": {}},
            "datasets": {
                "gray": {"channels": 1, "classes": ["a"]},
                "rgb": {"channels": 3, "classes": ["b"]},
            },
            "federated": {
                "datasets": ["gray", "rgb"],
                "share_stem": True,
                "aggregation": {
                    "mode": "hierarchical",
                    "task_weights": {"seg": 1, "cls": 1},
                    "dataset_weights": {"gray": 1, "rgb": 1},
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "share_stem"):
            validate_federated_config(config)


class AggregationTests(unittest.TestCase):
    @staticmethod
    def _updates():
        return [
            {"dataset": "BUSI", "task": "seg", "base_weight": 1.0,
             "arrays": [np.asarray([0.0], dtype=np.float32)]},
            {"dataset": "BUSI", "task": "cls", "base_weight": 3.0,
             "arrays": [np.asarray([8.0], dtype=np.float32)]},
            {"dataset": "ISIC", "task": "cls", "base_weight": 100.0,
             "arrays": [np.asarray([100.0], dtype=np.float32)]},
        ]

    def test_hierarchical_is_fifty_fifty_between_datasets(self):
        strategy = FedPerStrategy(
            task_weights={"seg": 1, "cls": 1},
            dataset_weights={"BUSI": 1, "ISIC": 1},
            aggregation_mode="hierarchical",
        )
        arrays, client_weights = strategy._aggregate_hierarchical(self._updates())
        # BUSI average = (0*1 + 8*3)/4 = 6; ISIC = 100; dataset average = 53.
        self.assertAlmostEqual(float(arrays[0][0]), 53.0)
        self.assertAlmostEqual(float(client_weights[:2].sum()), 0.5)
        self.assertAlmostEqual(float(client_weights[2]), 0.5)

    def test_flat_remains_sample_weighted(self):
        strategy = FedPerStrategy(
            task_weights={"seg": 1, "cls": 1},
            # Hierarchical-only weights must not distort the flat FedAvg ablation.
            dataset_weights={"BUSI": 1000, "ISIC": 0.001},
            aggregation_mode="flat",
        )
        arrays, client_weights = strategy._aggregate_flat(self._updates())
        expected = (0 * 1 + 8 * 3 + 100 * 100) / 104
        self.assertAlmostEqual(float(arrays[0][0]), expected, places=5)
        self.assertGreater(float(client_weights[2]), 0.9)

    def test_uniform_weighting_ignores_cardinality_but_keeps_task_weights(self):
        strategy = FedPerStrategy(
            task_weights={"seg": 4, "cls": 1},
            dataset_weights={"BUSI": 1, "ISIC": 1},
            aggregation_mode="hierarchical",
            client_weighting="uniform",
        )
        records = [
            {"dataset": "BUSI", "task": "seg", "base_weight": strategy._base_weight(1, "seg", "BUSI")},
            {"dataset": "BUSI", "task": "cls", "base_weight": strategy._base_weight(1000, "cls", "BUSI")},
            {"dataset": "ISIC", "task": "seg", "base_weight": strategy._base_weight(10000, "seg", "ISIC")},
            {"dataset": "ISIC", "task": "cls", "base_weight": strategy._base_weight(2, "cls", "ISIC")},
        ]
        weights = strategy._hierarchical_weights(records)
        np.testing.assert_allclose(weights, [0.4, 0.1, 0.4, 0.1])
        self.assertAlmostEqual(float(weights[:2].sum()), 0.5)
        self.assertAlmostEqual(float(weights[2:].sum()), 0.5)

    def test_task_mass_reproduces_legacy_single_task_weights(self):
        """A single-task client reports its whole slice under its own task.

        The mass sum then collapses to ``task_weights[task]``, so every arm run before the
        telemetry existed keeps its exact aggregation weights.
        """
        strategy = FedPerStrategy(task_weights={"seg": 4, "cls": 1})
        for task, examples in (("seg", 120), ("cls", 371)):
            legacy = strategy._base_weight(examples, task, "BUSI")
            with_mass = strategy._base_weight(
                examples, task, "BUSI",
                metrics={"task_mass_seg": examples if task == "seg" else 0,
                         "task_mass_cls": examples if task == "cls" else 0},
            )
            self.assertAlmostEqual(legacy, with_mass)
            self.assertAlmostEqual(legacy, examples * (4 if task == "seg" else 1))

    def test_multitask_client_weight_sums_the_mass_of_every_owned_task(self):
        strategy = FedPerStrategy(task_weights={"seg": 4, "cls": 1})
        # 62 images, 53 of them mask-supervised (BUSI's `normal` class has no usable mask).
        weight = strategy._base_weight(
            62, "seg+cls", "BUSI", metrics={"task_mass_seg": 53, "task_mass_cls": 62}
        )
        self.assertAlmostEqual(weight, 4 * 53 + 1 * 62)
        # Strictly between a pure-seg and a pure-cls client of the same size.
        self.assertGreater(weight, strategy._base_weight(62, "cls", "BUSI"))
        self.assertLess(weight, strategy._base_weight(62, "seg", "BUSI") + 62)

    def test_missing_task_mass_falls_back_to_the_scalar_task_weight(self):
        strategy = FedPerStrategy(task_weights={"seg": 4, "cls": 1})
        self.assertAlmostEqual(strategy._base_weight(10, "seg", "BUSI", metrics={}), 40.0)

    def test_validation_uses_the_same_hierarchical_uniform_policy(self):
        strategy = FedPerStrategy(
            task_weights={"cls": 1}, dataset_weights={"BUSI": 1, "ISIC": 1},
            aggregation_mode="hierarchical", client_weighting="uniform",
        )
        results = [
            (None, SimpleNamespace(
                num_examples=1, loss=0.0,
                metrics={
                    "dataset": "BUSI", "task": "cls", "client_id": "b",
                    "val_metric": 1.0, "val_balanced_accuracy_cls": 0.75,
                    "val_macro_f1_cls": 0.6,
                },
            )),
            (None, SimpleNamespace(
                num_examples=10000, loss=2.0,
                metrics={"dataset": "ISIC", "task": "cls", "client_id": "i", "val_metric": 0.0},
            )),
        ]
        loss, metrics = strategy.aggregate_evaluate(1, results, [])
        self.assertAlmostEqual(loss, 1.0)
        self.assertAlmostEqual(metrics["participation/dataset/BUSI"], 0.5)
        self.assertAlmostEqual(metrics["participation/dataset/ISIC"], 0.5)
        busi = strategy.aggregation_history[-1]["clients"][0]
        self.assertEqual(busi["val_loss"], 0.0)
        self.assertEqual(busi["val_balanced_accuracy_cls"], 0.75)
        self.assertEqual(busi["val_macro_f1_cls"], 0.6)

    def test_new_config_defaults_preserve_legacy_behavior(self):
        config = {"federated": {"local_epochs": 2}}
        self.assertEqual(aggregation_config(config)["client_weighting"], "num_examples")
        self.assertEqual(local_training_config(config), {
            "mode": "epochs", "steps_per_round": 10, "local_epochs": 2,
        })


if __name__ == "__main__":
    unittest.main()


class GradientConflictTests(unittest.TestCase):
    """The cosine must be taken between client DELTAS, never between the returned trunks."""

    @staticmethod
    def _strategy(sent):
        strategy = FedPerStrategy(task_weights={"seg": 1, "cls": 1})
        strategy._sent_arrays = [np.asarray(sent, dtype=np.float32)]
        return strategy

    @staticmethod
    def _update(dataset, task, arrays):
        return {"dataset": dataset, "task": task, "client_id": f"{dataset}_{task}",
                "arrays": [np.asarray(arrays, dtype=np.float32)]}

    def test_opposite_updates_are_detected_as_conflict(self):
        # Both clients start from the same large trunk, so the cosine BETWEEN TRUNKS would be ~+1.
        # Their updates point in opposite directions, which is what must be reported.
        strategy = self._strategy([100.0, 100.0])
        conflict = strategy._gradient_conflict([
            self._update("BUSI", "seg", [101.0, 100.0]),
            self._update("BUSI", "cls", [99.0, 100.0]),
        ])
        matrix = conflict["cosine"]["shared_total"]
        self.assertAlmostEqual(matrix[0][1], -1.0, places=5)
        self.assertEqual(conflict["clients"], ["BUSI/seg/BUSI_seg", "BUSI/cls/BUSI_cls"])

    def test_matrix_is_symmetric_with_unit_diagonal(self):
        strategy = self._strategy([0.0, 0.0, 0.0])
        conflict = strategy._gradient_conflict([
            self._update("BUSI", "seg", [1.0, 0.0, 0.0]),
            self._update("BUSI", "cls", [0.0, 2.0, 0.0]),
            self._update("ISIC", "seg", [1.0, 1.0, 0.0]),
        ])
        matrix = conflict["cosine"]["shared_total"]
        for index in range(3):
            self.assertAlmostEqual(matrix[index][index], 1.0, places=6)
        self.assertAlmostEqual(matrix[0][1], 0.0, places=6)
        self.assertAlmostEqual(matrix[0][2], matrix[2][0], places=6)
        self.assertAlmostEqual(matrix[0][2], 2 ** -0.5, places=5)

    def test_zero_delta_reports_undefined_instead_of_a_fake_direction(self):
        strategy = self._strategy([5.0])
        conflict = strategy._gradient_conflict([
            self._update("BUSI", "seg", [5.0]),   # trained to a standstill
            self._update("BUSI", "cls", [6.0]),
        ])
        matrix = conflict["cosine"]["shared_total"]
        self.assertIsNone(matrix[0][0])
        self.assertIsNone(matrix[0][1])
        self.assertAlmostEqual(matrix[1][1], 1.0, places=6)

    def test_named_blocks_are_reported_separately(self):
        strategy = FedPerStrategy(
            task_weights={"seg": 1, "cls": 1},
            shared_key_names=["encoder2.weight", "bottleneck.weight"],
        )
        strategy._sent_arrays = [np.zeros(2, np.float32), np.zeros(2, np.float32)]
        updates = [
            {"dataset": "BUSI", "task": "seg", "client_id": "a",
             "arrays": [np.asarray([1.0, 0.0], np.float32), np.asarray([1.0, 0.0], np.float32)]},
            {"dataset": "BUSI", "task": "cls", "client_id": "b",
             "arrays": [np.asarray([1.0, 0.0], np.float32), np.asarray([-1.0, 0.0], np.float32)]},
        ]
        cosine = strategy._gradient_conflict(updates)["cosine"]
        self.assertEqual(sorted(cosine), ["bottleneck", "encoder2", "shared_total"])
        self.assertAlmostEqual(cosine["encoder2"][0][1], 1.0, places=5)
        self.assertAlmostEqual(cosine["bottleneck"][0][1], -1.0, places=5)
        # The blocks cancel exactly, which the aggregate view must show.
        self.assertAlmostEqual(cosine["shared_total"][0][1], 0.0, places=5)

