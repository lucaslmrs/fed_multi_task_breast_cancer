import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import yaml

from src.dataset.federated_dataloader import (
    build_client_loader,
    client_tasks,
    resolve_class_weights,
)
from src.dataset.federated_partition import build_multi_dataset_partition
from src.experiments.analyze import method_comparisons, pooled_auc, summary_by_seed, summary_table
from src.utils.experiment_init import init_criterion_classification


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data/federated_multi/federated_mapping.csv"


@unittest.skipUnless(MASTER.exists(), "generate the multi-dataset partition first")
class PartitionAndLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = pd.read_csv(MASTER, low_memory=False)

    def test_lesions_never_cross_client_or_split(self):
        rows = self.master[(self.master.task == "cls") & self.master.lesion_id.notna()]
        ownership = rows.groupby(["dataset", "fold", "lesion_id"]).agg(
            clients=("client_id", "nunique"), splits=("split", "nunique")
        )
        self.assertEqual(int((ownership.clients > 1).sum()), 0)
        self.assertEqual(int((ownership.splits > 1).sum()), 0)

    def test_busi_master_canonicalizes_to_frozen_partition(self):
        frozen = pd.read_csv(ROOT / "data/Curated_BUSI/federated/federated_mapping.csv")
        current = self.master[self.master.dataset == "Curated_BUSI"].copy()
        current["client_id"] = current.client_id.str.removeprefix("Curated_BUSI_")
        current = current[frozen.columns]
        sort = ["fold", "client_id", "split", "img_path"]
        pd.testing.assert_frame_equal(
            current.sort_values(sort).reset_index(drop=True),
            frozen.sort_values(sort).reset_index(drop=True),
            check_dtype=False,
        )

    def test_channel_shapes_and_missing_supervision_placeholders(self):
        busi = build_client_loader(
            MASTER, 0, "Curated_BUSI_cls_0", "train", 1,
            dataset="Curated_BUSI", channels=1,
            classes=["benign", "malignant", "normal"], max_samples=1,
        )
        isic_cls = build_client_loader(
            MASTER, 0, "ISIC_2018_cls_0", "train", 1,
            dataset="ISIC_2018", channels=3,
            classes=["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"], max_samples=1,
        )
        isic_seg = build_client_loader(
            MASTER, 0, "ISIC_2018_seg_0", "train", 1,
            dataset="ISIC_2018", channels=3,
            classes=["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"], max_samples=1,
        )
        busi_batch, cls_batch, seg_batch = next(iter(busi)), next(iter(isic_cls)), next(iter(isic_seg))
        self.assertEqual(tuple(busi_batch["image"].shape[1:]), (1, 128, 128))
        self.assertEqual(tuple(cls_batch["image"].shape[1:]), (3, 128, 128))
        self.assertEqual(tuple(seg_batch["image"].shape[1:]), (3, 128, 128))
        self.assertEqual(int(torch.count_nonzero(cls_batch["mask"])), 0)
        self.assertEqual(float(seg_batch["label"].item()), -1.0)

    def test_balanced_fold_weights_are_common_and_train_only(self):
        classes = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
        first = resolve_class_weights(MASTER, 0, "ISIC_2018_cls_0", "ISIC_2018", classes,
                                      "balanced_fold")
        second = resolve_class_weights(MASTER, 0, "ISIC_2018_cls_1", "ISIC_2018", classes,
                                       "balanced_fold")
        self.assertEqual(first, second)

        rows = self.master[
            (self.master.dataset == "ISIC_2018") & (self.master.fold == 0)
            & (self.master.task == "cls") & (self.master.split == "train")
        ]
        counts = rows["class"].value_counts()
        expected = [len(rows) / (len(classes) * counts[name]) for name in classes]
        np.testing.assert_allclose(first, expected)

    def test_weighted_focal_loss_runs_on_cpu(self):
        criterion = init_criterion_classification(
            n_classes=7,
            classification_criterion="Focal",
            class_weights=[1, 2, 3, 4, 5, 6, 7],
            device="cpu",
        )
        logits = torch.randn(2, 7, requires_grad=True)
        labels = torch.nn.functional.one_hot(torch.tensor([0, 6]), num_classes=7).float()
        loss = criterion(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_weighted_focal_loss_runs_on_cuda(self):
        criterion = init_criterion_classification(
            n_classes=7,
            classification_criterion="Focal",
            class_weights=[1, 2, 3, 4, 5, 6, 7],
            device="cuda",
        )
        logits = torch.randn(2, 7, device="cuda", requires_grad=True)
        labels = torch.nn.functional.one_hot(
            torch.tensor([0, 6], device="cuda"), num_classes=7
        ).float()
        loss = criterion(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def _predictions(dataset, n_classes):
        labels = np.tile(np.arange(n_classes), 2)
        probabilities = np.full((len(labels), n_classes), 0.02 / max(n_classes - 1, 1))
        probabilities[np.arange(len(labels)), labels] = 0.98
        frame = pd.DataFrame({
            "setup": "federated", "fold": 0, "dataset": dataset,
            "ground_truth": labels, "predicted": labels,
        })
        for index in range(n_classes):
            frame[f"prob_{index}"] = probabilities[:, index]
            frame[f"class_name_{index}"] = f"{dataset}_{index}"
        return frame

    def test_pooled_auc_keeps_three_and_seven_class_spaces_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            three = Path(directory) / "three.csv"
            seven = Path(directory) / "seven.csv"
            self._predictions("BUSI", 3).to_csv(three, index=False)
            self._predictions("ISIC", 7).to_csv(seven, index=False)
            result = pooled_auc([three, seven])
        self.assertEqual(set(result.dataset), {"BUSI", "ISIC"})
        self.assertEqual(set(result.num_classes), {3, 7})
        self.assertTrue(np.allclose(result.auc_pooled, 1.0))

    def test_summary_discovers_all_isic_class_metrics(self):
        frame = pd.DataFrame([
            {"dataset": "BUSI", "setup": "federated", "task": "cls",
             "precision_class_0": 0.5, "precision_class_6": np.nan, "acc": 0.5},
            {"dataset": "ISIC", "setup": "federated", "task": "cls",
             "precision_class_0": 0.6, "precision_class_6": 0.7, "acc": 0.6},
        ])
        result = summary_table(frame)
        class_six = result[result.metric == "precision_class_6"]
        self.assertEqual(class_six.dataset.tolist(), ["ISIC"])

    def test_manifest_comparison_pairs_method_seed_fold_and_client(self):
        rows = []
        for seed in (1993, 1994):
            for method, setup, value in (("primary", "federated", 0.8), ("local", "standalone", 0.6)):
                rows.append({
                    "study_id": "study", "seed": seed, "dataset": "BUSI", "fold": 0,
                    "client_id": "BUSI_cls_0", "task": "cls", "method_id": method,
                    "setup": setup, "acc": value,
                })
        comparisons = [{
            "comparison_id": "primary_vs_local",
            "left": {"method_id": "primary", "setup": "federated"},
            "right": {"method_id": "local", "setup": "standalone"},
        }]
        aggregate, observations = method_comparisons(pd.DataFrame(rows), comparisons)
        acc = aggregate[aggregate.metric == "acc"].iloc[0]
        self.assertEqual(int(acc.n_pairs), 2)
        self.assertEqual(int(acc.n_seeds), 2)
        self.assertAlmostEqual(float(acc.mean_delta), 0.2)
        self.assertEqual(len(observations), 2)

    def test_summary_by_seed_keeps_repetitions_separate(self):
        frame = pd.DataFrame([
            {"study_id": "s", "seed": 1, "dataset": "BUSI", "method_id": "m",
             "setup": "federated", "task": "seg", "dice": 0.2},
            {"study_id": "s", "seed": 2, "dataset": "BUSI", "method_id": "m",
             "setup": "federated", "task": "seg", "dice": 0.8},
        ])
        result = summary_by_seed(frame)
        dice = result[result.metric == "dice"]
        self.assertEqual(set(dice.seed), {"1", "2"})
        self.assertEqual(set(dice["mean"]), {0.2, 0.8})


if __name__ == "__main__":
    unittest.main()


BUSI_CLASSES = ["benign", "malignant", "normal"]


class MultiTaskClientTopologyTests(unittest.TestCase):
    """One client owns every task over the SAME images -- no image lives in two silos."""

    @classmethod
    def setUpClass(cls):
        config = yaml.safe_load((ROOT / "src/config.yaml").read_text(encoding="utf-8"))
        # BUSI alone keeps the fixture fast; ISIC cannot host this topology anyway.
        config["federated"]["datasets"] = ["Curated_BUSI"]
        config["datasets"]["Curated_BUSI"]["client_topology"] = "multi_task"
        config["federated"]["n_clients"] = {"Curated_BUSI": {"seg": 3, "cls": 3}}
        cls._directory = tempfile.TemporaryDirectory()
        cls.master = str(Path(cls._directory.name) / "multitask.csv")
        cls.frame = build_multi_dataset_partition(config, output_path=cls.master)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_no_image_belongs_to_two_clients(self):
        ownership = self.frame.groupby(["fold", "img_path"])["client_id"].nunique()
        self.assertEqual(int((ownership > 1).sum()), 0)

    def test_every_client_owns_both_tasks(self):
        owned = self.frame.groupby("client_id")["task"].apply(lambda values: set(values))
        self.assertEqual(len(owned), 3)
        for tasks in owned:
            self.assertEqual(tasks, {"seg", "cls"})
        self.assertEqual(
            client_tasks(self.master, 0, "Curated_BUSI_mt_0", dataset="Curated_BUSI"),
            ("seg", "cls"),
        )

    def test_seg_rows_exclude_the_class_without_a_usable_mask(self):
        seg = self.frame[self.frame.task == "seg"]
        cls_rows = self.frame[self.frame.task == "cls"]
        self.assertNotIn("normal", set(seg["class"]))
        self.assertIn("normal", set(cls_rows["class"]))

    def test_pivoted_loader_marks_normal_images_as_unsupervised_for_segmentation(self):
        """``normal`` images DO carry a mask_path (an all-zero mask) in the source mapping.

        Supervision therefore has to come from the partition; trusting the path would train the
        Dice term against empty targets.
        """
        loader = build_client_loader(
            self.master, 0, "Curated_BUSI_mt_0", "train", 256,
            dataset="Curated_BUSI", channels=1, classes=BUSI_CLASSES, tasks=("seg", "cls"),
        )
        batch = next(iter(loader))
        has_mask = batch["has_mask"].tolist()
        is_normal = [name == "normal" for name in batch["class"]]
        self.assertTrue(any(is_normal), "fixture needs at least one normal image")
        for normal, masked in zip(is_normal, has_mask):
            self.assertEqual(masked, not normal)
        self.assertTrue(all(batch["has_label"].tolist()))

    def test_task_mass_counts_supervision_not_rows(self):
        loader = build_client_loader(
            self.master, 0, "Curated_BUSI_mt_0", "train", 32,
            dataset="Curated_BUSI", channels=1, classes=BUSI_CLASSES, tasks=("seg", "cls"),
        )
        mass = loader.task_mass
        self.assertEqual(mass["cls"], loader.effective_num_samples)
        self.assertLess(mass["seg"], mass["cls"])

    def test_single_task_slice_reports_its_whole_mass_under_its_own_task(self):
        loader = build_client_loader(
            self.master, 0, "Curated_BUSI_mt_0", "train", 32,
            dataset="Curated_BUSI", channels=1, classes=BUSI_CLASSES, tasks=("seg",),
        )
        self.assertEqual(loader.task_mass["cls"], 0)
        self.assertEqual(loader.task_mass["seg"], loader.effective_num_samples)

    def test_oversampling_is_refused_on_a_multitask_slice(self):
        with self.assertRaises(ValueError) as context:
            build_client_loader(
                self.master, 0, "Curated_BUSI_mt_0", "train", 32,
                dataset="Curated_BUSI", channels=1, classes=BUSI_CLASSES,
                tasks=("seg", "cls"), oversampling=True,
            )
        self.assertIn("class_weighting", str(context.exception))

    def test_a_dataset_with_disjoint_task_pools_cannot_be_multitask(self):
        config = yaml.safe_load((ROOT / "src/config.yaml").read_text(encoding="utf-8"))
        config["federated"]["datasets"] = ["ISIC_2018"]
        config["datasets"]["ISIC_2018"]["client_topology"] = "multi_task"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as context:
                build_multi_dataset_partition(
                    config, output_path=str(Path(directory) / "isic.csv")
                )
        self.assertIn("disjoint image sets", str(context.exception))


