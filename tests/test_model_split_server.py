import unittest

import numpy as np
import torch

from src.federated.config import validate_federated_config
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


if __name__ == "__main__":
    unittest.main()
