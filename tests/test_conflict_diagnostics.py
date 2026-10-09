import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from src.federated import conflict_diagnostics as cd
from src.federated.client import _preserve_rng_state
from src.federated.local_trainer import train_local
from src.federated.model_split import get_shared_state, shared_keys
from src.federated.server import FedPerStrategy
from src.models.multitask.MTnnUNet import MTnnUNet
from src.utils.experiment_init import init_criterion_classification, init_criterion_segmentation
from src.utils.training_runtime import PrecisionPolicy, validate_runtime_config


def _config(enabled=True, every=2, rounds=5, standalone=False, lookahead=True):
    return {
        "federated": {"rounds": rounds, "standalone": standalone},
        "runtime": {"diagnostics": {"gradient_conflict": {
            "enabled": enabled, "every_n_rounds": every, "probe_batches": 4, "sketch_dim": 4096,
            "lookahead": {"enabled": lookahead, "val_batches": 2, "train_batches": 2},
        }}},
    }


def _batches(seed, n=4, size=64):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n):
        image = torch.rand(2, 1, size, size, generator=generator)
        mask = (torch.rand(2, 1, size, size, generator=generator) > 0.7).float()
        batches.append({"image": image, "mask": mask,
                        "label": torch.randint(0, 3, (2,), generator=generator),
                        "has_mask": torch.ones(2, dtype=torch.bool),
                        "has_label": torch.ones(2, dtype=torch.bool)})
    return batches


def _client(directory, client_id="BUSI_mt_0", seed=0, config=None):
    torch.manual_seed(seed)
    state_dir = Path(directory) / "fold_0" / f"client_{client_id}"
    state_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        client_id=client_id, model=MTnnUNet(sequences=1, regions=1, n_classes=3),
        share_stem=False, tasks=("seg", "cls"), task="seg+cls",
        task_lambdas={"seg": 0.8, "cls": 0.2}, precision=PrecisionPolicy("fp32", "cpu"),
        device="cpu", num_classes=3, inversely_weighted=True,
        seg_criterion=init_criterion_segmentation("DiceBCE"),
        cls_criterion=init_criterion_classification(n_classes=3, device="cpu"),
        config=config or _config(), state_dir=state_dir,
        train_loader=_batches(seed), val_loader=_batches(seed + 100, n=2),
    )


def _trunk(client):
    # Copies: on CPU, ``get_shared_state`` returns views of the live weights.
    return [array.copy() for array in get_shared_state(client.model, False)]


def _train(client, batches):
    optimizer = torch.optim.Adam(client.model.parameters(), lr=1e-3)
    return train_local(client.model, batches, optimizer, client.task, "cpu", 1, 3,
                       client.seg_criterion, client.cls_criterion, True, training_mode="steps",
                       steps_per_round=len(batches), tasks=client.tasks,
                       task_lambdas=client.task_lambdas, precision=client.precision)


class SamplingTests(unittest.TestCase):
    def test_sampled_rounds_include_the_first_every_nth_and_the_last(self):
        config = _config(every=10, rounds=25)
        self.assertEqual([r for r in range(1, 26) if cd.is_sampled_round(config, r)], [1, 11, 21, 25])

    def test_disabled_or_standalone_never_samples(self):
        for config in (_config(enabled=False), _config(standalone=True)):
            self.assertFalse(any(cd.is_sampled_round(config, r) for r in range(1, 6)))

    def test_runtime_validation_rejects_a_single_probe_batch(self):
        config = _config()
        config["runtime"]["diagnostics"]["gradient_conflict"]["probe_batches"] = 1
        with self.assertRaisesRegex(ValueError, "probe_batches"):
            validate_runtime_config(config)


class SketchTests(unittest.TestCase):
    def test_sketched_cosine_tracks_the_exact_cosine(self):
        rng = np.random.default_rng(0)
        base, noise = rng.normal(size=200_000), rng.normal(size=200_000)
        sketch = cd.CountSketch(200_000, 65536, seed=7)
        for target in (-0.5, 0.0, 0.5, 0.99):
            other = target * base + np.sqrt(1 - target ** 2) * noise
            exact = base @ other / (np.linalg.norm(base) * np.linalg.norm(other))
            estimate = float(sketch(base) @ sketch(other)) / (
                np.linalg.norm(base) * np.linalg.norm(other))
            self.assertAlmostEqual(estimate, exact, delta=0.03)

    def test_same_seed_gives_identical_sketches(self):
        vector = torch.arange(1000, dtype=torch.float64)
        first = cd.CountSketch(1000, 64, seed=3)(vector)
        second = cd.CountSketch(1000, 64, seed=3)(vector)
        self.assertTrue(torch.equal(first, second))


class FormulaTests(unittest.TestCase):
    def test_spearman_brown_and_normalisation(self):
        self.assertAlmostEqual(cd.spearman_brown(0.5), 2 / 3)
        self.assertTrue(np.isnan(cd.spearman_brown(-0.1)))
        # Two clients with identical halves (ceiling 1) and a known dot product.
        exact = np.asarray([1.0, 1.0, 1.0, 1.0])
        row = cd._m1_row(0, 1, "shared_total", "a", "b", {}, 0.3, exact, exact)
        self.assertAlmostEqual(row["cos"], 0.3)
        self.assertAlmostEqual(row["cos_normalized"], 0.3)
        self.assertAlmostEqual(row["magnitude_similarity"], 1.0)

    def test_magnitude_similarity_penalises_dominance(self):
        small, large = np.asarray([1.0, 1.0, 1.0, 1.0]), np.asarray([1.0, 1.0, 1.0, 100.0])
        row = cd._m1_row(0, 1, "shared_total", "a", "b", {}, 0.0, small, large)
        self.assertAlmostEqual(row["magnitude_similarity"], 2 * 10 / 101)

    def test_pair_classes(self):
        meta = {"a": ("BUSI", ("seg", "cls")), "b": ("BUSI", ("seg", "cls")),
                "c": ("ISIC", ("seg",)), "d": ("ISIC", ("cls",))}
        self.assertEqual(cd.pair_class(meta, "a", "a"), "self")
        self.assertEqual(cd.pair_class(meta, "a", "b"), "intra_same_tasks")
        self.assertEqual(cd.pair_class(meta, "c", "d"), "intra_diff_tasks")
        self.assertEqual(cd.pair_class(meta, "a", "c"), "inter")
        self.assertEqual(cd.pair_class(meta, cd.AGGREGATE_SOURCE, "a"), "aggregate")


class ClientProbeTests(unittest.TestCase):
    def test_probes_do_not_perturb_training(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = _client(directory)
            probed = _client(directory)
            batches = reference.train_loader
            torch.manual_seed(42)
            _train(reference, batches)
            torch.manual_seed(42)
            _preserve_rng_state(lambda: cd.collect_fit_probes(probed, 1))
            _train(probed, batches)
            for (name, a), (_, b) in zip(reference.model.state_dict().items(),
                                         probed.model.state_dict().items()):
                self.assertTrue(torch.equal(a, b), name)
            directory = cd.round_dir(Path(directory) / "fold_0", 1)
            self.assertTrue((directory / "BUSI_mt_0_m1.npz").exists())
            self.assertTrue((directory / "BUSI_mt_0_pre.pt").exists())

    def test_identical_halves_give_a_unit_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            client = _client(directory)
            client.train_loader = [client.train_loader[0]] * 4
            cd.collect_fit_probes(client, 1)
            with np.load(cd.round_dir(Path(directory) / "fold_0", 1) / "BUSI_mt_0_m1.npz") as f:
                aa, bb, ab, _ = sum(f[f"{block}::exact"] for block in f["blocks"])
            self.assertAlmostEqual(ab / np.sqrt(aa * bb), 1.0, places=5)


class LookaheadTests(unittest.TestCase):
    def test_zero_update_and_unchanged_aggregate_have_zero_affinity(self):
        with tempfile.TemporaryDirectory() as directory:
            client = _client(directory)
            cd.collect_fit_probes(client, 1)
            theta = _trunk(client)
            rng = np.random.default_rng(0)
            updates = [
                {"client_id": "BUSI_mt_0", "arrays": [a.copy() for a in theta]},
                {"client_id": "ISIC_seg_0",
                 "arrays": [a + 0.05 * rng.normal(size=a.shape).astype(a.dtype) for a in theta]},
            ]
            fold_dir = Path(directory) / "fold_0"
            cd.write_round_deltas(cd.round_dir(fold_dir, 1), theta, updates)
            cd.collect_lookahead(client, theta, 1)
            rows = pd.read_csv(cd.round_dir(fold_dir, 1) / "BUSI_mt_0_m2.csv")
            self.assertEqual(set(rows.split), {"train", "val"})
            for source in ("BUSI_mt_0", cd.AGGREGATE_SOURCE):
                self.assertTrue(np.allclose(rows[rows.source == source].z, 0.0, atol=1e-6))
            self.assertTrue(np.isfinite(rows[rows.source == "ISIC_seg_0"].z).all())
            self.assertFalse(np.allclose(rows[rows.source == "ISIC_seg_0"].z, 0.0))
            cd.cleanup_round(cd.round_dir(fold_dir, 1))
            self.assertFalse((cd.round_dir(fold_dir, 1) / cd.DELTAS_FILE).exists())
            self.assertFalse(list(cd.round_dir(fold_dir, 1).glob("*_pre.pt")))

    def test_server_writes_deltas_only_on_sampled_lookahead_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            strategy = FedPerStrategy(task_weights={"seg": 1, "cls": 1},
                                      conflict_config=_config(every=2, rounds=5),
                                      conflict_fold_dir=Path(directory))
            self.assertEqual([r for r in range(1, 6) if strategy._lookahead_round(r)], [1, 3, 5])
            self.assertFalse(FedPerStrategy(task_weights={"seg": 1, "cls": 1},
                                            conflict_config=_config(lookahead=False),
                                            conflict_fold_dir=Path(directory))._lookahead_round(1))
            self.assertFalse(FedPerStrategy(task_weights={"seg": 1})._lookahead_round(1))


class ReportTests(unittest.TestCase):
    def test_run_without_diagnostics_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "fold_0").mkdir()
            self.assertIsNone(cd.report_run(directory))

    def test_report_consolidates_m1_and_m2(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = Path(directory) / "fold_0"
            clients = [_client(directory, "BUSI_mt_0", seed=0), _client(directory, "BUSI_mt_1", seed=1)]
            for client in clients:
                (client.state_dir / "metadata.yaml").write_text(yaml.safe_dump(
                    {"client_id": client.client_id, "dataset": "BUSI", "task": "seg+cls",
                     "tasks": ["seg", "cls"]}), encoding="utf-8")
                cd.collect_fit_probes(client, 1)
            theta = _trunk(clients[0])
            cd.write_round_deltas(cd.round_dir(fold_dir, 1), theta, [
                {"client_id": c.client_id, "arrays": _trunk(c)}
                for c in clients])
            for client in clients:
                cd.collect_lookahead(client, theta, 1)
            summary = cd.report_run(directory)
            self.assertTrue((Path(directory) / cd.M1_FILE).exists())
            m1 = pd.read_csv(Path(directory) / cd.M1_FILE)
            self.assertIn(cd.TOTAL_BLOCK, set(m1.block))
            self.assertEqual(set(m1.pair_class), {"intra_same_tasks"})
            self.assertIn("m2_z_train", set(summary.metric))
            self.assertIn("self", set(summary[summary.metric == "m2_z_val"].group))


if __name__ == "__main__":
    unittest.main()
