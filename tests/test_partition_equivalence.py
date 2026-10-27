"""E4 C2: ``scripts/partition_equivalence`` on synthetic masters.

A candidate is equivalent when it is the reference master with only the variant prefix of the
paths and the mask-derived numbers of the rescaled datasets changed. Every other difference --
a class flip, a vanished lesion, a reordered row, a touched non-rescaled row -- is a problem.
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from scripts.partition_equivalence import compare_masters, main, read_master
from src.experiments.study_runner import (_partition_metadata_path, _partition_signature,
                                          _sha256_file, _stable_hash)

COLUMNS = ["img_path", "mask_path", "class", "id", "dim1", "dim2", "tumor_pixels", "y_max",
           "y_min", "x_max", "x_min", "y_size", "x_size", "official_split", "lesion_id",
           "dataset", "fold", "client_id", "task", "split"]


def _row(dataset, image_id, cls, pixels, client, task, split, variant="processed_128", size=128,
         lesion=None):
    stem = f"data/{dataset}/{variant}"
    scale = size // 128
    return {
        "img_path": f"{stem}/images/{image_id}.png", "mask_path": f"{stem}/masks/{image_id}_mask.png",
        "class": cls, "id": image_id, "dim1": size, "dim2": size,
        "tumor_pixels": float(pixels * scale * scale), "y_max": float(10 * scale) if pixels else 0.0,
        "y_min": 0.0, "x_max": float(10 * scale) if pixels else 0.0, "x_min": 0.0,
        "y_size": float(10 * scale) if pixels else 0.0, "x_size": float(10 * scale) if pixels else 0.0,
        "official_split": None, "lesion_id": lesion, "dataset": dataset, "fold": 0,
        "client_id": client, "task": task, "split": split,
    }


def _master(rescaled_variant="processed_128", size=128) -> pd.DataFrame:
    rows = [_row("Curated_BUSI", 1, "benign", 40, "Curated_BUSI_mt_0", task, "train")
            for task in ("seg", "cls")]
    rows += [_row("Curated_BUSI", 2, "normal", 0, "Curated_BUSI_mt_0", task, "test")
             for task in ("seg", "cls")]
    for image_id, cls, pixels, split in ((3, "tumor", 25, "train"), (4, "no_tumor", 0, "val"),
                                          (5, "tumor", 3, "test")):
        rows += [_row("TCGA_LGG", image_id, cls, pixels, "TCGA_LGG_mt_0", task, split,
                      rescaled_variant, size, lesion="TCGA_AA_0001") for task in ("seg", "cls")]
    return pd.DataFrame(rows, columns=COLUMNS)


def _as_csv_text(frame: pd.DataFrame, directory: Path, name: str) -> pd.DataFrame:
    """Round-trip through CSV exactly as the real masters are read."""
    path = directory / name
    frame.to_csv(path, index=False)
    return read_master(path)


class CompareMastersTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dir = Path(temporary.name)
        self.reference = _as_csv_text(_master(), self.dir, "reference.csv")
        self.candidate_frame = _master("processed_256", 256)

    def _compare(self, candidate_frame=None, **kwargs):
        frame = self.candidate_frame if candidate_frame is None else candidate_frame
        candidate = _as_csv_text(frame, self.dir, "candidate.csv")
        return compare_masters(self.reference, candidate, kwargs.pop("rescaled", ["TCGA_LGG"]),
                               check_files=False, **kwargs)

    def test_rescaled_copy_is_equivalent(self):
        report = self._compare()
        self.assertEqual(report["problems"], [])
        tcga = report["datasets"]["TCGA_LGG"]
        self.assertEqual(tcga["positive_images_reference"], 2)
        self.assertEqual(tcga["positive_images_candidate"], 2)
        self.assertEqual(tcga["positive_flips"], 0)
        self.assertEqual(tcga["images"], 3)
        self.assertEqual(report["datasets"]["Curated_BUSI"]["rows_differing"], 0)

    def test_class_flip_is_a_problem(self):
        frame = self.candidate_frame.copy()
        frame.loc[(frame["id"] == 5), "class"] = "no_tumor"
        report = self._compare(frame)
        self.assertTrue(any("change 'class'" in problem for problem in report["problems"]))

    def test_vanished_lesion_is_a_problem(self):
        frame = self.candidate_frame.copy()
        frame.loc[(frame["id"] == 5), "tumor_pixels"] = 0.0
        report = self._compare(frame)
        self.assertEqual(report["datasets"]["TCGA_LGG"]["positive_flips"], 2)
        self.assertTrue(any("vanished" in problem for problem in report["problems"]))

    def test_changed_row_of_a_dataset_not_rescaled_is_a_problem(self):
        frame = self.candidate_frame.copy()
        frame.loc[0, "tumor_pixels"] = 41.0
        report = self._compare(frame)
        self.assertEqual(report["datasets"]["Curated_BUSI"]["rows_differing"], 1)
        self.assertTrue(any(problem.startswith("Curated_BUSI") for problem in report["problems"]))

    def test_reordered_rows_are_a_problem(self):
        frame = self.candidate_frame.iloc[[1, 0] + list(range(2, len(self.candidate_frame)))]
        report = self._compare(frame.reset_index(drop=True))
        self.assertTrue(any("row order" in problem for problem in report["problems"]))

    def test_split_or_client_change_is_a_problem(self):
        for column, value in (("split", "val"), ("client_id", "TCGA_LGG_mt_1")):
            with self.subTest(column=column):
                frame = self.candidate_frame.copy()
                frame.loc[frame["id"] == 3, column] = value
                self.assertTrue(any("row order" in problem
                                    for problem in self._compare(frame)["problems"]))

    def test_path_under_another_variant_is_a_problem(self):
        frame = self.candidate_frame.copy()
        frame.loc[frame["id"] == 3, "img_path"] = "data/TCGA_LGG/processed_512/images/3.png"
        report = self._compare(frame)
        self.assertEqual(report["datasets"]["TCGA_LGG"]["img_path_differing"], 2)

    def test_wrong_size_is_a_problem(self):
        frame = self.candidate_frame.copy()
        frame.loc[frame["id"] == 4, "dim2"] = 128
        report = self._compare(frame)
        self.assertTrue(any("dim2 != 256" in problem for problem in report["problems"]))

    def test_unrescaled_dataset_left_at_the_old_variant_is_a_problem(self):
        report = self._compare(_master())  # the candidate never moved to processed_256
        self.assertTrue(report["problems"])

    def test_schema_and_row_count_must_match(self):
        self.assertTrue(self._compare(self.candidate_frame.iloc[:-1])["problems"])
        self.assertTrue(self._compare(self.candidate_frame.drop(columns=["lesion_id"]))["problems"])

    def test_unknown_rescaled_dataset_is_a_problem(self):
        report = self._compare(rescaled=["TCGA_LGG", "SIIM_ACR"])
        self.assertTrue(any("absent from the reference" in problem
                            for problem in report["problems"]))

    def test_missing_rescaled_files_are_a_problem(self):
        candidate = _as_csv_text(self.candidate_frame, self.dir, "candidate.csv")
        report = compare_masters(self.reference, candidate, ["TCGA_LGG"], check_files=True)
        self.assertEqual(report["datasets"]["TCGA_LGG"]["img_path_missing_files"], 3)


class MainTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dir = Path(temporary.name)
        self.reference = self.dir / "reference" / "federated_mapping.csv"
        self.candidate = self.dir / "candidate" / "federated_mapping.csv"
        self.reference.parent.mkdir()
        self.candidate.parent.mkdir()
        _master().to_csv(self.reference, index=False)
        _master("processed_256", 256).to_csv(self.candidate, index=False)
        _partition_metadata_path(self.reference).write_text(
            yaml.safe_dump({"csv_sha256": _sha256_file(self.reference)}), encoding="utf-8")
        self.config = {
            "training": {"seed": 1993, "CV": 1, "holdout_test_size": 0.3},
            "data": {"root": "data", "dataset": "Curated_BUSI", "variant": "processed_128"},
            "datasets": {"Curated_BUSI": {"variant": "processed_128"},
                         "TCGA_LGG": {"variant": "processed_256",
                                      "fold_strategy": "stratified_group"}},
            "federated": {"datasets": ["Curated_BUSI", "TCGA_LGG"],
                          "n_clients": {"Curated_BUSI": {"seg": 1, "cls": 1},
                                        "TCGA_LGG": {"seg": 1, "cls": 1}},
                          "dirichlet_alpha": 100.0, "val_size": 0.2,
                          "partition_file": str(self.candidate)},
        }
        self.config_path = self.dir / "run.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def _main(self, *extra):
        argv = ["--reference", str(self.reference), "--candidate", str(self.candidate),
                "--rescaled", "TCGA_LGG", "--no-file-check", *extra]
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            status = main(argv)
        return status, out.getvalue()

    def test_write_metadata_follows_the_study_runner_schema_and_is_written_once(self):
        status, out = self._main("--write-metadata", "e4_preprocessing", "1993",
                                 "--config", str(self.config_path))
        self.assertEqual(status, 0, out)
        self.assertIn("EQUIVALENCE_OK", out)
        metadata = yaml.safe_load(_partition_metadata_path(self.candidate).read_text())
        signature = _partition_signature(self.config)
        self.assertEqual(metadata["study_id"], "e4_preprocessing")
        self.assertEqual(metadata["seed"], 1993)
        self.assertEqual(metadata["partition_variant"], "base")
        self.assertEqual(metadata["partition_signature"], signature)
        self.assertEqual(metadata["partition_signature_sha256"], _stable_hash(signature))
        self.assertEqual(metadata["csv_sha256"], _sha256_file(self.candidate))
        self.assertEqual(metadata["rows"], 10)
        report = yaml.safe_load(Path(metadata["equivalence_report"]).read_text())
        self.assertEqual(report["status"], "ok")

        status, out = self._main("--write-metadata", "e4_preprocessing", "1993",
                                 "--config", str(self.config_path))
        self.assertEqual(status, 1)
        self.assertIn("written once", out)

    def test_changed_reference_fails_its_metadata_check(self):
        _partition_metadata_path(self.reference).write_text(
            yaml.safe_dump({"csv_sha256": "0" * 64}), encoding="utf-8")
        status, out = self._main()
        self.assertEqual(status, 1)
        self.assertIn("does not match its metadata", out)

    def test_metadata_needs_the_run_config_of_the_candidate(self):
        self.config["datasets"]["TCGA_LGG"]["variant"] = "processed_128"
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        status, out = self._main("--write-metadata", "e4_preprocessing", "1993",
                                 "--config", str(self.config_path))
        self.assertEqual(status, 1)
        self.assertIn("datasets.TCGA_LGG.variant", out)
        self.assertFalse(_partition_metadata_path(self.candidate).exists())

    def test_a_problem_never_writes_metadata(self):
        frame = _master("processed_256", 256)
        frame.loc[frame["id"] == 5, "class"] = "no_tumor"
        frame.to_csv(self.candidate, index=False)
        status, _ = self._main("--write-metadata", "e4_preprocessing", "1993",
                               "--config", str(self.config_path))
        self.assertEqual(status, 1)
        self.assertFalse(_partition_metadata_path(self.candidate).exists())


if __name__ == "__main__":
    unittest.main()
