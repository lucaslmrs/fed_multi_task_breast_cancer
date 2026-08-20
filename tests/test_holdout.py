import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

from src.dataset.BUSI_dataloader import BUSI_dataloader_CV, BUSI_dataloader_CV_prod
from src.dataset.federated_partition import build_federated_partition
from src.dataset.splitting import (
    CROSS_VALIDATION,
    HOLDOUT,
    evaluation_settings,
    outer_split_indices,
    validate_master_splits,
)
from src.experiments.analyze import method_comparisons, pooled_auc, summary_table
from src.utils.miscellany import _add_evaluation_summary_columns


class OuterSplitTests(unittest.TestCase):
    def test_stratified_holdout_is_deterministic_disjoint_and_seventy_thirty(self):
        frame = pd.DataFrame({
            "id": np.arange(100),
            "class": np.repeat(["a", "b"], 50),
        })
        first = outer_split_indices(
            frame, n_splits=1, seed=1993, strategy="stratified", holdout_test_size=0.30
        )[0]
        second = outer_split_indices(
            frame, n_splits=1, seed=1993, strategy="stratified", holdout_test_size=0.30
        )[0]
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual((len(first[0]), len(first[1])), (70, 30))
        self.assertFalse(set(first[0]).intersection(first[1]))
        self.assertEqual(set(first[0]).union(first[1]), set(range(100)))
        self.assertEqual(frame.iloc[first[1]]["class"].value_counts().to_dict(), {"a": 15, "b": 15})

    def test_grouped_holdout_never_splits_a_lesion(self):
        rows = []
        for group in range(40):
            size = 1 + group % 3
            rows.extend(
                {"lesion_id": f"lesion_{group}", "class": "a" if group < 20 else "b"}
                for _ in range(size)
            )
        frame = pd.DataFrame(rows)
        train_ix, test_ix = outer_split_indices(
            frame,
            n_splits=1,
            seed=1993,
            strategy="stratified_group",
            holdout_test_size=0.30,
            group_col="lesion_id",
        )[0]
        train_groups = set(frame.iloc[train_ix].lesion_id)
        test_groups = set(frame.iloc[test_ix].lesion_id)
        self.assertFalse(train_groups.intersection(test_groups))
        self.assertEqual(len(test_groups), 12)
        self.assertLess(abs(len(test_ix) / len(frame) - 0.30), 0.04)

    def test_cross_validation_indices_remain_identical(self):
        frame = pd.DataFrame({"class": np.repeat(["a", "b"], 20)})
        expected = list(
            StratifiedKFold(n_splits=4, shuffle=True, random_state=1993).split(
                frame, frame["class"]
            )
        )
        actual = outer_split_indices(
            frame, n_splits=4, seed=1993, strategy="stratified"
        )
        for (expected_train, expected_test), (actual_train, actual_test) in zip(expected, actual):
            np.testing.assert_array_equal(expected_train, actual_train)
            np.testing.assert_array_equal(expected_test, actual_test)

        random_expected = list(
            KFold(n_splits=4, shuffle=True, random_state=1993).split(frame)
        )
        random_actual = outer_split_indices(
            frame, n_splits=4, seed=1993, strategy="random"
        )
        for expected_pair, actual_pair in zip(random_expected, random_actual):
            np.testing.assert_array_equal(expected_pair[0], actual_pair[0])
            np.testing.assert_array_equal(expected_pair[1], actual_pair[1])

        grouped = frame.assign(lesion_id=[f"g{index}" for index in range(len(frame))])
        group_expected = list(
            StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=1993).split(
                grouped, grouped["class"], groups=grouped["lesion_id"]
            )
        )
        group_actual = outer_split_indices(
            grouped,
            n_splits=4,
            seed=1993,
            strategy="stratified_group",
            group_col="lesion_id",
        )
        for expected_pair, actual_pair in zip(group_expected, group_actual):
            np.testing.assert_array_equal(expected_pair[0], actual_pair[0])
            np.testing.assert_array_equal(expected_pair[1], actual_pair[1])

    def test_evaluation_validation_and_master_compatibility(self):
        self.assertEqual(evaluation_settings({"CV": 1})[1:], (HOLDOUT, 0.30))
        self.assertEqual(evaluation_settings({"CV": 4, "holdout_test_size": 2})[1], CROSS_VALIDATION)
        with self.assertRaises(ValueError):
            evaluation_settings({"CV": 0})
        with self.assertRaises(ValueError):
            evaluation_settings({"CV": 1, "holdout_test_size": 1.0})
        master = pd.DataFrame({"fold": [0, 1], "split": ["train", "test"]})
        with self.assertRaisesRegex(ValueError, "Regenerate"):
            validate_master_splits(master, {"CV": 1, "holdout_test_size": 0.30})


class LoaderAndPartitionTests(unittest.TestCase):
    @staticmethod
    def _mapping(size_per_class=20):
        classes = np.repeat(["benign", "malignant", "normal"], size_per_class)
        return pd.DataFrame({
            "img_path": [f"image_{index}.png" for index in range(len(classes))],
            "mask_path": [f"mask_{index}.png" for index in range(len(classes))],
            "class": classes,
            "id": np.arange(len(classes)),
        })

    def test_classic_and_prod_loaders_use_the_same_holdout_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._mapping().to_csv(root / "mapping.csv", index=False)
            common = dict(
                seed=1993,
                batch_size=8,
                transforms=None,
                classes=["benign", "malignant", "normal"],
                n_folds=1,
                path_images=root,
                holdout_test_size=0.30,
                oversampling=False,
            )
            train, val, test = BUSI_dataloader_CV(**common)
            prod_train, prod_test = BUSI_dataloader_CV_prod(**common)
            self.assertEqual(len(test[0].dataset), 18)
            self.assertEqual(len(prod_test[0].dataset), 18)
            self.assertEqual(len(train[0].dataset) + len(val[0].dataset), 42)
            self.assertEqual(len(prod_train[0].dataset), 42)
            self.assertEqual(
                set(test[0].dataset.mapping_file.id), set(prod_test[0].dataset.mapping_file.id)
            )

    def test_federated_holdout_writes_only_fold_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.csv"
            output = root / "federated_mapping.csv"
            self._mapping().to_csv(mapping, index=False)
            master = build_federated_partition(
                mapping_path=mapping,
                output_path=output,
                n_folds=1,
                seed=1993,
                n_clients_seg=2,
                n_clients_cls=2,
                dirichlet_alpha=100.0,
                val_size=0.2,
                seg_exclude_classes=["normal"],
                holdout_test_size=0.30,
            )
            self.assertEqual(set(master.fold), {0})
            self.assertEqual(set(master.split), {"train", "val", "test"})
            for task in ("seg", "cls"):
                task_rows = master[master.task == task]
                ownership = task_rows.groupby("img_path").split.nunique()
                self.assertEqual(int((ownership > 1).sum()), 0)


class HoldoutAnalysisTests(unittest.TestCase):
    @staticmethod
    def _result_rows():
        rows = []
        for method, setup, value in (
            ("fed", "federated", 0.8), ("local", "standalone", 0.6)
        ):
            for client in ("c0", "c1"):
                rows.append({
                    "study_id": "s",
                    "seed": 1993,
                    "evaluation_scheme": HOLDOUT,
                    "n_splits": 1,
                    "holdout_test_size": 0.30,
                    "dataset": "BUSI",
                    "method_id": method,
                    "setup": setup,
                    "fold": 0,
                    "client_id": client,
                    "task": "cls",
                    "acc": value,
                })
        return pd.DataFrame(rows)

    def test_holdout_comparison_is_descriptive_without_wilcoxon(self):
        comparison = [{
            "comparison_id": "fed_vs_local",
            "left": {"method_id": "fed", "setup": "federated"},
            "right": {"method_id": "local", "setup": "standalone"},
        }]
        aggregate, observations = method_comparisons(self._result_rows(), comparison)
        row = aggregate[aggregate.metric == "acc"].iloc[0]
        self.assertEqual(row.inference_scope, "descriptive_only_single_holdout")
        self.assertTrue(pd.isna(row.wilcoxon_stat))
        self.assertTrue(pd.isna(row.wilcoxon_p))
        self.assertEqual(set(observations.evaluation_scheme), {HOLDOUT})

    def test_summary_keeps_evaluation_design_separate(self):
        frame = self._result_rows()
        cv = frame.copy()
        cv["evaluation_scheme"] = CROSS_VALIDATION
        cv["n_splits"] = 4
        result = summary_table(pd.concat([frame, cv], ignore_index=True))
        self.assertEqual(set(result.evaluation_scheme), {HOLDOUT, CROSS_VALIDATION})
        self.assertEqual(len(result[result.metric == "acc"]), 4)

    def test_single_split_latex_has_no_nan_dispersion(self):
        summary = pd.DataFrame({"fold 0": [0.8123]}, index=["dice"])
        summary["mean"] = summary.mean(axis=1)
        summary["std"] = summary[["fold 0"]].std(axis=1)
        result = _add_evaluation_summary_columns(summary, n_splits=1)
        self.assertTrue(pd.isna(result.loc["dice", "std"]))
        self.assertNotIn("nan", result.loc["dice", "latex"].lower())
        self.assertNotIn("pm", result.loc["dice", "latex"].lower())

    def test_pooled_auc_marks_holdout_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            labels = np.tile(np.arange(3), 2)
            frame = pd.DataFrame({
                "study_id": "s",
                "method_id": "m",
                "setup": "federated",
                "seed": 1993,
                "evaluation_scheme": HOLDOUT,
                "n_splits": 1,
                "fold": 0,
                "dataset": "BUSI",
                "ground_truth": labels,
            })
            for index in range(3):
                frame[f"prob_{index}"] = (labels == index).astype(float)
            frame.to_csv(path, index=False)
            result = pooled_auc([path])
        self.assertEqual(set(result.aggregation_scope), {"holdout_test"})


if __name__ == "__main__":
    unittest.main()
