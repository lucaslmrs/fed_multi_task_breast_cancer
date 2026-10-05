import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.experiments.run_examples import (
    CANDIDATES_FILE,
    client_examples,
    select_candidates,
    write_fold_candidates,
)
from src.experiments.run_report import REPORT_FILE, build_run_report
from src.federated import unified_eval

CLASSES = ["benign", "malignant"]


def _seg(rows):
    """rows: (patient_id, dice_positive, target_empty, pred_pixels)"""
    return pd.DataFrame([
        {"patient_id": pid, "dice": dice if not empty else float("nan"),
         "dice_positive": dice if not empty else float("nan"), "target_empty": empty,
         "target_pixels": 0 if empty else 50, "pred_pixels": pred, "image_pixels": 100}
        for pid, dice, empty, pred in rows
    ])


def _cls(rows):
    """rows: (patient_id, ground_truth, predicted, confidence of the prediction)"""
    out = []
    for pid, gt, pred, conf in rows:
        probs = [1 - conf, conf] if pred == 1 else [conf, 1 - conf]
        out.append({"patient_id": pid, "ground_truth": gt, "predicted": pred,
                    "prob_0": probs[0], "prob_1": probs[1]})
    return pd.DataFrame(out)


class SelectionTests(unittest.TestCase):
    def test_quadrants_and_deterministic_ordering(self):
        seg = _seg([(1, .9, False, 40), (2, .8, False, 40), (3, .1, False, 5),
                    (4, 0, True, 0), (5, 0, True, 30), (6, .95, False, 50)])
        cls = _cls([(1, 0, 0, .9), (2, 0, 0, .95), (3, 1, 0, .8),
                    (4, 1, 1, .6), (5, 0, 1, .99), (6, 1, 0, .7)])
        chosen = select_candidates(seg, cls, CLASSES)
        by_quadrant = chosen.groupby("quadrant")["patient_id"].apply(list).to_dict()
        # Best segmentation first among good ones; the empty target with no prediction is good.
        self.assertEqual(by_quadrant["cls_ok__seg_ok"], [4, 1])
        self.assertEqual(by_quadrant["cls_err__seg_ok"], [6])
        # Bad: the empty target with a false positive (score 0.7) ranks after Dice 0.1.
        self.assertEqual(by_quadrant["cls_err__seg_bad"], [3, 5])
        self.assertLessEqual(chosen.groupby("quadrant").size().max(), 2)
        again = select_candidates(seg.sample(frac=1, random_state=3), cls, CLASSES)
        pd.testing.assert_frame_equal(chosen, again)

    def test_single_task_clients(self):
        seg_only = select_candidates(_seg([(1, .9, False, 40), (2, .2, False, 9)]), None, CLASSES)
        self.assertEqual(set(seg_only["quadrant"]), {"seg_ok", "seg_bad"})
        cls_only = select_candidates(None, _cls([(1, 0, 0, .9), (2, 0, 1, .8)]), CLASSES)
        self.assertEqual(set(cls_only["quadrant"]), {"cls_ok", "cls_err"})
        self.assertEqual(cls_only.loc[cls_only.quadrant == "cls_err", "predicted_class"].item(),
                         "malignant")


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        mask = torch.zeros(1, 8, 8)
        if index % 2:
            mask[:, 2:6, 2:6] = 1
        return {"patient_id": index + 10, "label": index % 2, "class": str(index % 2),
                "has_mask": torch.tensor(True), "has_label": torch.tensor(True),
                "image": mask.clone() * 0.8 + 0.1, "mask": mask}


class _TinyModel(torch.nn.Module):
    def forward(self, image):
        logits = torch.stack([image.mean((1, 2, 3)) * 0, image.mean((1, 2, 3)) * 20 - 2], dim=1)
        return [logits], [(image - 0.5) * 10]


class EndToEndTests(unittest.TestCase):
    def test_samples_keep_metrics_and_feed_examples_and_report(self):
        loader = DataLoader(_TinyDataset(), batch_size=3)
        for task in ("seg", "cls"):
            plain = unified_eval.evaluate(_TinyModel(), loader, task, 2, "cpu")[0]
            with_samples = unified_eval.evaluate(
                _TinyModel(), loader, task, 2, "cpu", return_samples=True
            )[0]
            for key, value in plain.items():
                np.testing.assert_equal(value, with_samples[key])

        outputs = {}
        for task in ("seg", "cls"):
            _, preds, samples = unified_eval.evaluate(
                _TinyModel(), loader, task, 2, "cpu", class_names=CLASSES, return_samples=True
            )
            outputs[task] = (preds, samples)
        self.assertEqual(len(outputs["seg"][1]["frame"]), 4)

        with tempfile.TemporaryDirectory() as run:
            run = Path(run)
            rows = client_examples(run, "federated", 0, "BUSI_mt_0", "BUSI", outputs, CLASSES)
            self.assertTrue(rows)
            self.assertTrue(all((run / row["image_path"]).exists() for row in rows))
            (run / "fold_0").mkdir()
            write_fold_candidates(run / "fold_0", "federated", rows)
            pd.DataFrame([
                {"fold": 0, "client_id": "BUSI_mt_0", "dataset": "BUSI", "task": "seg",
                 "n_test": 4, "dice_positive": .9, "dice": .9, "evaluation_scheme": "holdout"},
                {"fold": 0, "client_id": "BUSI_mt_0", "dataset": "BUSI", "task": "cls",
                 "n_test": 4, "acc": .5, "balanced_acc": .5, "evaluation_scheme": "holdout"},
            ]).to_csv(run / "federated_test_results.csv", index=False)
            report = build_run_report(run).read_text(encoding="utf-8")
            self.assertIn("BUSI_mt_0", report)
            self.assertIn("Exemplos de acertos e erros", report)
            self.assertIn(rows[0]["image_path"], report)

    def test_report_flags_folds_without_candidates(self):
        with tempfile.TemporaryDirectory() as run:
            run = Path(run)
            (run / "fold_0").mkdir()
            (run / "fold_0" / CANDIDATES_FILE.format(setup="standalone")).write_text("")
            pd.DataFrame([{"fold": 0, "client_id": "c", "dataset": "D", "task": "seg",
                           "n_test": 1, "dice": .5}]).to_csv(
                run / "standalone_test_results.csv", index=False)
            report = (build_run_report(run)).read_text(encoding="utf-8")
            self.assertIn("Exemplos indisponíveis", report)
            self.assertNotIn("negative_transfer.html", report)
            self.assertEqual((run / REPORT_FILE).name, "run_report.html")


if __name__ == "__main__":
    unittest.main()
