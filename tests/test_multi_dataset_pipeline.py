import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset.federated_dataloader import build_client_loader, resolve_class_weights
from src.experiments.analyze import pooled_auc, summary_table
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


if __name__ == "__main__":
    unittest.main()
