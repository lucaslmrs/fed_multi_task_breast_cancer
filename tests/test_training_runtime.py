import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.experiments.study_runner import (
    _config_signature,
    build_execution_plan,
    load_manifest,
)
from src.federated import unified_eval
from src.federated.client import FederatedClient
from src.training_federated import _resume_config_signature
from src.utils.models import inference_binary_classification, inference_binary_segmentation
from src.utils.training_runtime import (
    GpuTelemetry,
    PrecisionPolicy,
    dataloader_kwargs,
    precision_name,
    runtime_config,
    validate_runtime_config,
)


class RuntimeConfigTests(unittest.TestCase):
    def test_missing_precision_keeps_legacy_fp32(self):
        self.assertEqual(precision_name({"training": {}}), "fp32")
        PrecisionPolicy("fp32", "cpu")

    def test_bf16_rejects_cpu_and_unsupported_cuda(self):
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
            PrecisionPolicy("bf16", "cpu")
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.is_bf16_supported", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "native BF16 support"):
                PrecisionPolicy("bf16", "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_bf16_autocast_is_active_on_supported_cuda(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("GPU has no BF16 support")
        policy = PrecisionPolicy("bf16", "cuda")
        with policy.autocast():
            result = torch.nn.Linear(4, 4).cuda()(torch.ones((2, 4), device="cuda"))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_loader_kwargs_omit_worker_only_options_at_zero(self):
        config = {
            "training": {},
            "federated": {},
            "runtime": {"dataloader": {"federated_num_workers": 0}},
        }
        self.assertEqual(
            dataloader_kwargs(config, federated=True),
            {"num_workers": 0, "pin_memory": True},
        )
        classic = dataloader_kwargs(config)
        self.assertEqual(classic["num_workers"], 4)
        self.assertEqual(classic["prefetch_factor"], 2)
        self.assertTrue(classic["persistent_workers"])

    def test_runtime_is_excluded_but_precision_remains_scientific(self):
        base = {"training": {"CV": 1, "precision": "fp32"}, "federated": {}}
        operational = copy.deepcopy(base)
        operational["runtime"] = {
            "federated": {"client_resources": {"num_cpus": 2, "num_gpus": 0.5}}
        }
        self.assertEqual(_config_signature(base), _config_signature(operational))
        self.assertEqual(_resume_config_signature(base), _resume_config_signature(operational))

        bf16 = copy.deepcopy(base)
        bf16["training"]["precision"] = "bf16"
        self.assertNotEqual(_config_signature(base), _config_signature(bf16))
        self.assertNotEqual(_resume_config_signature(base), _resume_config_signature(bf16))

    def test_invalid_runtime_values_fail_validation(self):
        config = {
            "training": {"precision": "fp32"},
            "federated": {},
            "runtime": {"inference_batch_size": 0},
        }
        with self.assertRaisesRegex(ValueError, "inference_batch_size"):
            validate_runtime_config(config)


class _EvalDataset(Dataset):
    def __init__(self, task):
        self.task = task

    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.full((1, 4, 4), float(index % 2))
        mask = torch.full((1, 4, 4), float(index % 2))
        return {
            "patient_id": torch.tensor(index),
            "class": "benign" if index % 2 == 0 else "malignant",
            "image": image,
            "mask": mask,
            "label": torch.tensor([index % 2], dtype=torch.float32),
        }


class _EvalModel(torch.nn.Module):
    def forward(self, inputs):
        score = inputs.mean(dim=(1, 2, 3))
        logits = torch.stack((1.0 - score, score), dim=1) * 4.0
        segmentation = (inputs - 0.5) * 8.0
        return logits, [segmentation]


class _ClassicSegModel(torch.nn.Module):
    def forward(self, inputs):
        return [(inputs - 0.5) * 8.0]


class _ClassicClsModel(torch.nn.Module):
    def forward(self, inputs):
        return (inputs.mean(dim=(1, 2, 3), keepdim=True) - 0.5) * 8.0


class BatchedEvaluationTests(unittest.TestCase):
    def _evaluate(self, task, batch_size):
        return unified_eval.evaluate(
            _EvalModel(),
            DataLoader(_EvalDataset(task), batch_size=batch_size, shuffle=False),
            task,
            2,
            "cpu",
        )

    def test_classification_batching_preserves_metrics_and_rows(self):
        one, one_preds = self._evaluate("cls", 1)
        many, many_preds = self._evaluate("cls", 4)
        self.assertEqual(one.keys(), many.keys())
        for key in one:
            if isinstance(one[key], float) and np.isnan(one[key]):
                self.assertTrue(np.isnan(many[key]))
            else:
                self.assertEqual(one[key], many[key])
        self.assertEqual(len(one_preds), 4)
        self.assertEqual(one_preds.to_dict("records"), many_preds.to_dict("records"))

    def test_segmentation_batching_preserves_per_image_metrics(self):
        one, _ = self._evaluate("seg", 1)
        many, _ = self._evaluate("seg", 4)
        for key in one:
            if isinstance(one[key], float):
                self.assertAlmostEqual(one[key], many[key], places=7)
            else:
                self.assertEqual(one[key], many[key])

    def test_classic_inference_batching_preserves_rows_and_segmentation_files(self):
        with tempfile.TemporaryDirectory() as root:
            outputs = []
            files = []
            for batch_size in (1, 4):
                path = Path(root) / str(batch_size)
                (path / "features_map").mkdir(parents=True)
                (path / "segs").mkdir()
                loader = DataLoader(_EvalDataset("seg"), batch_size=batch_size, shuffle=False)
                outputs.append(
                    inference_binary_segmentation(
                        _ClassicSegModel(), loader, f"{path}/", fill_holes=False
                    )
                )
                files.append(sorted(item.name for item in (path / "segs").iterdir()))
            pd.testing.assert_frame_equal(outputs[0], outputs[1])
            self.assertEqual(files[0], files[1])
            self.assertEqual(len(files[0]), len(_EvalDataset("seg")))

    def test_classic_classification_batching_preserves_rows(self):
        with tempfile.TemporaryDirectory() as root:
            frames = []
            for batch_size in (1, 4):
                path = Path(root) / str(batch_size)
                path.mkdir()
                frames.append(
                    inference_binary_classification(
                        _ClassicClsModel(),
                        DataLoader(_EvalDataset("cls"), batch_size=batch_size, shuffle=False),
                        f"{path}/",
                    )
                )
            pd.testing.assert_frame_equal(frames[0], frames[1])


class FederatedValidationContractTests(unittest.TestCase):
    def test_fit_has_no_post_local_validation_or_best_checkpoint(self):
        source = inspect.getsource(FederatedClient.fit)
        self.assertNotIn("evaluate_local", source)
        self.assertNotIn("best.pt", source)
        self.assertIn("_save_state", source)

        evaluate_source = inspect.getsource(FederatedClient.evaluate)
        self.assertIn("evaluate_local", evaluate_source)
        self.assertIn("post_aggregation_round", evaluate_source)


class StudyPrecisionTests(unittest.TestCase):
    def test_example_study_pins_the_bf16_protocol(self):
        manifest = load_manifest("studies/example_multi_dataset.yaml")
        row = build_execution_plan(manifest, [1993], [manifest["arms"][0]])[0][0]
        self.assertEqual(row["config"]["training"]["precision"], "bf16")
        self.assertTrue(row["config"]["training"]["cuda_benchmark"])
        self.assertEqual(row["config"]["training"]["CV"], 1)
        # Precision is a scientific hash input: flipping it must change the arm signature.
        fp32 = copy.deepcopy(row["config"])
        fp32["training"]["precision"] = "fp32"
        self.assertNotEqual(_config_signature(row["config"]), _config_signature(fp32))

    def test_cpu_smoke_overrides_bf16(self):
        manifest = load_manifest("studies/example_multi_dataset.yaml")
        row = build_execution_plan(
            manifest, [1993], [manifest["arms"][0]], smoke=True
        )[0][0]
        self.assertEqual(row["config"]["training"]["precision"], "fp32")
        self.assertEqual(
            row["config"]["runtime"]["federated"]["client_resources"]["num_gpus"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
