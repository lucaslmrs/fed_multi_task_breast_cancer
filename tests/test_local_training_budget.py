import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.federated_dataloader import RotatingBatchSampler
from src.federated.local_trainer import train_local


class _DictionaryDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.full((1, 2, 2), float(index + 1))
        return {"image": image, "mask": torch.zeros_like(image)}


class _TinySegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        segmentation = inputs * self.scale
        logits = torch.zeros((inputs.shape[0], 1), device=inputs.device)
        return logits, segmentation


class RotatingBatchSamplerTests(unittest.TestCase):
    def test_round_has_exactly_full_batches_and_crosses_cycle_boundary(self):
        sampler = RotatingBatchSampler(
            dataset_size=5, batch_size=4, steps_per_round=2, seed=1993
        )
        batches = list(sampler)

        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch) == 4 for batch in batches))
        # The first five stream positions are one complete permutation even though the second
        # batch crosses into the next permutation.
        self.assertEqual(sorted(batches[0] + batches[1][:1]), list(range(5)))

    def test_consecutive_rounds_continue_one_permutation_stream(self):
        sampler = RotatingBatchSampler(
            dataset_size=5, batch_size=3, steps_per_round=2, seed=77
        )
        stream = []
        for server_round in range(1, 4):
            sampler.set_round(server_round)
            stream.extend(index for batch in sampler for index in batch)

        # Every aligned five-example cycle visits all indices before a new cycle begins.
        for offset in range(0, 15, 5):
            self.assertEqual(sorted(stream[offset:offset + 5]), list(range(5)))

    def test_round_is_reproducible_without_in_memory_cursor(self):
        first = RotatingBatchSampler(7, 3, 4, seed=11, server_round=3)
        recreated = RotatingBatchSampler(7, 3, 4, seed=11, server_round=3)
        self.assertEqual(list(first), list(recreated))

    def test_invalid_arguments_fail_early(self):
        for kwargs in (
            {"dataset_size": 0, "batch_size": 2, "steps_per_round": 1, "seed": 1},
            {"dataset_size": 2, "batch_size": 0, "steps_per_round": 1, "seed": 1},
            {"dataset_size": 2, "batch_size": 2, "steps_per_round": 0, "seed": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RotatingBatchSampler(**kwargs)


class LocalTrainingBudgetTests(unittest.TestCase):
    @staticmethod
    def _components(size, batch_sampler=None, batch_size=None):
        dataset = _DictionaryDataset(size)
        if batch_sampler is not None:
            loader = DataLoader(dataset, batch_sampler=batch_sampler)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model = _TinySegmentationModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        return loader, model, optimizer

    def test_steps_mode_reports_exact_optimizer_and_example_budget(self):
        sampler = RotatingBatchSampler(5, 4, 3, seed=42)
        loader, model, optimizer = self._components(5, batch_sampler=sampler)
        result = train_local(
            model, loader, optimizer, "seg", "cpu", local_epochs=9, num_classes=2,
            seg_criterion=torch.nn.MSELoss(), training_mode="steps", steps_per_round=3,
        )

        self.assertEqual(result["optimizer_steps"], 3)
        self.assertEqual(result["examples_processed"], 12)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))

    def test_epochs_mode_preserves_full_loader_passes(self):
        loader, model, optimizer = self._components(5, batch_size=2)
        result = train_local(
            model, loader, optimizer, "seg", "cpu", local_epochs=2, num_classes=2,
            seg_criterion=torch.nn.MSELoss(), training_mode="epochs",
        )

        self.assertEqual(result["optimizer_steps"], 6)
        self.assertEqual(result["examples_processed"], 10)

    def test_steps_mode_rejects_a_loader_that_does_not_meet_budget(self):
        loader, model, optimizer = self._components(3, batch_size=2)
        with self.assertRaisesRegex(RuntimeError, "expected 4 batches"):
            train_local(
                model, loader, optimizer, "seg", "cpu", local_epochs=1, num_classes=2,
                seg_criterion=torch.nn.MSELoss(), training_mode="steps", steps_per_round=4,
            )


if __name__ == "__main__":
    unittest.main()
