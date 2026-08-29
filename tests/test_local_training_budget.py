import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.federated_dataloader import RotatingBatchSampler
from src.federated.local_trainer import evaluate_local, train_local


class _DictionaryDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.full((1, 2, 2), float(index + 1))
        return {"image": image, "mask": torch.zeros_like(image)}


class _PartiallySupervisedDataset(Dataset):
    """Mimics a multi-task client slice: some rows have no mask, some have no label.

    Row 0 stands for BUSI's ``normal`` class -- it carries an all-zero mask on disk but is NOT
    segmentation-supervised, which is exactly the case a path-based check would get wrong.
    """

    def __init__(self, size, unmasked=(0,), unlabelled=()):
        self.size = size
        self.unmasked = set(unmasked)
        self.unlabelled = set(unlabelled)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.full((1, 2, 2), float(index + 1))
        return {
            "image": image,
            "mask": torch.zeros_like(image),
            "label": torch.tensor([float(index % 2)]),
            "has_mask": torch.tensor(index not in self.unmasked),
            "has_label": torch.tensor(index not in self.unlabelled),
        }


class _TinyMultiTaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        segmentation = inputs * self.scale
        logits = inputs.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1) * self.scale
        return logits, segmentation


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


class MultiTaskLocalTrainingTests(unittest.TestCase):
    @staticmethod
    def _components(dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model = _TinyMultiTaskModel()
        return loader, model, torch.optim.SGD(model.parameters(), lr=0.001)

    def _train(self, dataset, batch_size=2, **kwargs):
        loader, model, optimizer = self._components(dataset, batch_size)
        return train_local(
            model, loader, optimizer, None, "cpu", local_epochs=1, num_classes=2,
            seg_criterion=torch.nn.MSELoss(), cls_criterion=torch.nn.MSELoss(),
            tasks=("seg", "cls"), task_lambdas={"seg": 0.8, "cls": 0.2}, **kwargs,
        )

    def test_unsupervised_samples_are_excluded_from_their_task(self):
        # Row 0 has no mask; the first batch is [0, 1] so the seg term must see only row 1.
        result = self._train(_PartiallySupervisedDataset(4, unmasked=(0,)))
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["task_batches"], {"seg": 2, "cls": 2})

    def test_a_task_with_no_supervised_sample_in_a_batch_is_omitted(self):
        # Rows 0 and 1 carry no mask, so the first batch has no seg term at all -- but the round
        # still exercises seg through the second batch.
        result = self._train(_PartiallySupervisedDataset(4, unmasked=(0, 1)))
        self.assertEqual(result["task_batches"]["seg"], 1)
        self.assertEqual(result["task_batches"]["cls"], 2)

    def test_a_declared_task_that_never_trains_is_rejected(self):
        # Every row lacks a mask: the client would silently degrade into a cls-only client.
        with self.assertRaises(RuntimeError) as context:
            self._train(_PartiallySupervisedDataset(4, unmasked=(0, 1, 2, 3)))
        self.assertIn("seg", str(context.exception))

    def test_single_task_path_is_untouched_by_the_multitask_arguments(self):
        dataset = _PartiallySupervisedDataset(4, unmasked=())
        loader, model, optimizer = self._components(dataset, 2)
        baseline = train_local(
            model, loader, optimizer, "seg", "cpu", local_epochs=1, num_classes=2,
            seg_criterion=torch.nn.MSELoss(),
        )
        loader, model, optimizer = self._components(dataset, 2)
        explicit = train_local(
            model, loader, optimizer, None, "cpu", local_epochs=1, num_classes=2,
            seg_criterion=torch.nn.MSELoss(), tasks=("seg",),
        )
        self.assertAlmostEqual(baseline["loss"], explicit["loss"])
        self.assertEqual(baseline["optimizer_steps"], explicit["optimizer_steps"])

    def test_evaluation_reports_one_metric_per_owned_task(self):
        dataset = _PartiallySupervisedDataset(4, unmasked=(0,))
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        result = evaluate_local(
            _TinyMultiTaskModel(), loader, None, "cpu", 2,
            seg_criterion=torch.nn.MSELoss(), cls_criterion=torch.nn.MSELoss(),
            tasks=("seg", "cls"), task_lambdas={"seg": 0.8, "cls": 0.2},
        )
        self.assertEqual(result["metric_name"], "multitask")
        self.assertIn("metric_seg", result)
        self.assertIn("metric_cls", result)
        self.assertEqual(result["n"], 4)


