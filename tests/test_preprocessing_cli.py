"""E4 C1: the preprocessing command line that writes a NEW variant without editing config.yaml.

``preprocessing_data_config`` without flags must return config.yaml's ``data`` block untouched (the
historical workflow). With ``--variant processed_<N> --image-size N`` it selects the script's own
dataset, takes the classes from the ``datasets`` registry and refuses any existing, non-empty
variant folder, so ``processed_128`` can never be overwritten from the command line.
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd
import yaml

from src.dataset import SIIM_ACR_preprocessing, TCGA_LGG_preprocessing, paths
from src.dataset.preprocessing_utils import assert_new_variant, preprocessing_data_config

FLAGS_256 = ["--variant", "processed_256", "--image-size", "256"]


class _TempConfig:
    """A config.yaml whose data.root is a temporary folder, as the real one is for `data/`."""

    def __init__(self, test: unittest.TestCase):
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.dir = Path(temporary.name)
        self.root = self.dir / "data"
        self.data = {
            "root": str(self.root), "dataset": "Curated_BUSI", "variant": "processed_128",
            "image_size": 128, "batch_size": 32, "classes": ["benign", "malignant", "normal"],
        }
        self.config = {
            "data": dict(self.data),
            "datasets": {
                "TCGA_LGG": {"variant": "processed_128", "classes": ["no_tumor", "tumor"]},
                "SIIM_ACR": {"variant": "processed_128",
                             "classes": ["no_pneumothorax", "pneumothorax"]},
            },
        }
        self.path = self.dir / "config.yaml"
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def variant_dir(self, dataset: str, variant: str) -> Path:
        return paths.processed_dir({"root": str(self.root), "dataset": dataset, "variant": variant})


def _resolve(config: _TempConfig, dataset: str, argv):
    with contextlib.redirect_stderr(io.StringIO()):
        return preprocessing_data_config(dataset, argv, str(config.path))


class PreprocessingDataConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = _TempConfig(self)

    def test_no_flags_returns_the_data_block_unchanged(self):
        self.assertEqual(_resolve(self.config, "TCGA_LGG", []), self.config.data)

    def test_flags_must_be_given_together(self):
        for argv in (["--variant", "processed_256"], ["--image-size", "256"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                _resolve(self.config, "TCGA_LGG", argv)

    def test_variant_name_must_spell_the_size(self):
        for argv in (["--variant", "processed_128", "--image-size", "256"],
                     ["--variant", "hires_256", "--image-size", "256"],
                     ["--variant", "processed_0", "--image-size", "0"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                _resolve(self.config, "TCGA_LGG", argv)

    def test_flags_select_the_dataset_variant_size_and_registry_classes(self):
        resolved = _resolve(self.config, "SIIM_ACR", FLAGS_256)
        self.assertEqual(resolved["dataset"], "SIIM_ACR")
        self.assertEqual(resolved["variant"], "processed_256")
        self.assertEqual(resolved["image_size"], 256)
        self.assertEqual(resolved["classes"], ["no_pneumothorax", "pneumothorax"])
        self.assertEqual(resolved["root"], self.config.data["root"])
        self.assertEqual(paths.processed_dir(resolved),
                         self.config.variant_dir("SIIM_ACR", "processed_256"))

    def test_dataset_without_registry_classes_aborts(self):
        with self.assertRaises(SystemExit) as raised:
            _resolve(self.config, "ISIC_2018", FLAGS_256)
        self.assertIn("datasets.ISIC_2018.classes", str(raised.exception.code))

    def test_existing_variant_with_a_mapping_aborts(self):
        variant = self.config.variant_dir("TCGA_LGG", "processed_256")
        variant.mkdir(parents=True)
        (variant / paths.MAPPING_FILENAME).write_text("img_path\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            _resolve(self.config, "TCGA_LGG", FLAGS_256)
        message = str(raised.exception.code)
        self.assertTrue(message.startswith("[ABORT]"))
        self.assertIn("already exists", message)
        self.assertIn(paths.MAPPING_FILENAME, message)

    def test_non_empty_variant_without_mapping_aborts(self):
        # A partial earlier run: images were written, the mapping never was.
        (self.config.variant_dir("TCGA_LGG", "processed_256") / "images").mkdir(parents=True)
        with self.assertRaises(SystemExit) as raised:
            _resolve(self.config, "TCGA_LGG", FLAGS_256)
        self.assertIn("not empty", str(raised.exception.code))

    def test_existing_empty_folder_is_accepted(self):
        self.config.variant_dir("TCGA_LGG", "processed_256").mkdir(parents=True)
        self.assertEqual(_resolve(self.config, "TCGA_LGG", FLAGS_256)["variant"], "processed_256")

    def test_assert_new_variant_rejects_a_file(self):
        target = self.config.dir / "processed_256"
        target.write_text("", encoding="utf-8")
        with self.assertRaises(SystemExit):
            assert_new_variant(target)


def _write_tcga_raw(root: Path) -> None:
    """Two 256x256 slices of one patient: one with a 20x20 tumour, one without."""
    patient = "TCGA_AA_0001_20000101"
    folder = root / "TCGA_LGG" / "raw" / "kaggle_3m" / patient
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for slice_number, tumour in ((1, False), (2, True)):
        image = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        if tumour:
            mask[100:120, 60:80] = 255
        cv2.imwrite(str(folder / f"{patient}_{slice_number}.tif"), image)
        cv2.imwrite(str(folder / f"{patient}_{slice_number}_mask.tif"), mask)


class ScriptEntryPointTests(unittest.TestCase):
    def setUp(self):
        self.config = _TempConfig(self)

    def _run(self, module, argv):
        with mock.patch.object(module, "CONFIG_FILE", str(self.config.path)), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            module.main(argv)
        return out.getvalue()

    def test_tcga_writes_a_new_variant_without_touching_config_yaml(self):
        before = self.config.path.read_bytes()
        _write_tcga_raw(self.config.root)
        log = self._run(TCGA_LGG_preprocessing, FLAGS_256)

        mapping_path = self.config.variant_dir("TCGA_LGG", "processed_256") / paths.MAPPING_FILENAME
        mapping = pd.read_csv(mapping_path)
        self.assertEqual(len(mapping), 2)
        self.assertEqual(set(mapping["dim1"]) | set(mapping["dim2"]), {256})
        self.assertEqual(mapping.sort_values("id")["class"].tolist(), ["no_tumor", "tumor"])
        self.assertEqual(int(mapping["tumor_pixels"].max()), 400)
        self.assertNotIn("became empty", log)
        self.assertNotIn("[WARN] config.yaml has data.classes", log)
        self.assertEqual(self.config.path.read_bytes(), before)

        # A second run with the same flags must refuse the variant it just wrote.
        with self.assertRaises(SystemExit) as raised:
            self._run(TCGA_LGG_preprocessing, FLAGS_256)
        self.assertIn("already exists", str(raised.exception.code))

    def test_scripts_refuse_an_existing_variant_before_reading_raw_data(self):
        for module, dataset in ((TCGA_LGG_preprocessing, "TCGA_LGG"),
                                (SIIM_ACR_preprocessing, "SIIM_ACR")):
            variant = self.config.variant_dir(dataset, "processed_128")
            variant.mkdir(parents=True)
            (variant / paths.MAPPING_FILENAME).write_text("img_path\n", encoding="utf-8")
            with self.subTest(dataset=dataset), self.assertRaises(SystemExit) as raised:
                self._run(module, ["--variant", "processed_128", "--image-size", "128"])
            self.assertIn("already exists", str(raised.exception.code))
            self.assertFalse(paths.raw_dir({"root": str(self.config.root), "dataset": dataset}).exists())

    def test_without_flags_the_dataset_guard_still_applies(self):
        # config.yaml selects Curated_BUSI, so the TCGA script must refuse to run.
        with self.assertRaises(SystemExit) as raised:
            self._run(TCGA_LGG_preprocessing, [])
        self.assertIn("This script preprocesses 'TCGA_LGG'", str(raised.exception.code))


if __name__ == "__main__":
    unittest.main()
