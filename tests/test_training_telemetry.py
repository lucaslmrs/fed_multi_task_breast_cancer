import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.experiments.study_runner import _config_signature
from src.experiments.training_curves import build_run_artifacts, collect_history
from src.federated.client import FederatedClient, _preserve_rng_state
from src.federated.config import training_telemetry_config, validate_federated_config
from src.federated.local_trainer import train_local
from src.training_federated import _resume_config_signature


class _RandomSegDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.rand((1, 2, 2))
        return {"image": image, "mask": torch.zeros_like(image)}


class _TinySegModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, inputs):
        return torch.zeros((inputs.shape[0], 1)), inputs * self.scale


def _train(callback=None, record_steps=False, mode="epochs"):
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    model = _TinySegModel()
    loader = DataLoader(_RandomSegDataset(), batch_size=2, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_local(
        model, loader, optimizer, "seg", "cpu", local_epochs=2, num_classes=2,
        seg_criterion=torch.nn.MSELoss(), training_mode=mode,
        steps_per_round=2, epoch_end_callback=callback,
        record_step_history=record_steps,
    )
    return model, result


class TelemetryCollectionTests(unittest.TestCase):
    def test_epoch_and_step_histories_have_the_selected_unit(self):
        _, epochs = _train()
        self.assertEqual(
            sorted({row["unit_index"] for row in epochs["history"]}), [1, 2]
        )
        self.assertEqual({row["unit_type"] for row in epochs["history"]}, {"epoch"})

        _, steps = _train(record_steps=True, mode="steps")
        self.assertEqual(
            sorted({row["unit_index"] for row in steps["history"]}), [1, 2]
        )
        self.assertEqual({row["unit_type"] for row in steps["history"]}, {"step"})

    def test_observational_callback_does_not_change_final_parameters(self):
        baseline, _ = _train()

        def diagnostic(_epoch):
            return _preserve_rng_state(lambda: torch.rand(50))

        observed, result = _train(callback=diagnostic)
        self.assertEqual(len(result["validation_history"]), 2)
        self.assertTrue(torch.equal(baseline.scale, observed.scale))

    def test_telemetry_is_excluded_from_resume_and_study_signatures(self):
        base = {"training": {"CV": 2}, "federated": {"rounds": 2}}
        changed = yaml.safe_load(yaml.safe_dump(base))
        changed["federated"]["training_telemetry"] = {
            "enabled": True, "granularity": "epoch_and_round", "formats": ["csv"]
        }
        self.assertEqual(_resume_config_signature(base), _resume_config_signature(changed))
        self.assertEqual(_config_signature(base), _config_signature(changed))

    def test_default_telemetry_contract(self):
        resolved = training_telemetry_config({"federated": {}})
        self.assertEqual(resolved, {
            "enabled": True,
            "granularity": "epoch_and_round",
            "formats": ["csv", "html", "png"],
        })

    def test_client_histories_are_isolated_and_resume_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            clients = []
            for client_id in ("first", "second"):
                client = object.__new__(FederatedClient)
                client.telemetry = {"enabled": True}
                client.history_path = Path(directory) / client_id / "training_history.csv"
                client.history_path.parent.mkdir()
                clients.append(client)

            def row(client_id, round_number, value):
                return {
                    "setup": "federated", "fold": 0, "round": round_number,
                    "dataset": "BUSI", "client_id": client_id, "task": "seg",
                    "phase": "post_local_round", "split": "train",
                    "unit_type": "round", "unit_index": round_number,
                    "metric_name": "loss", "value": value, "n_samples": 2,
                    "optimizer_steps": 1, "examples_processed": 2,
                }

            clients[0]._append_history([row("first", 2, 0.8), row("first", 1, 1.0)])
            clients[0]._append_history([row("first", 2, 0.7)])
            clients[1]._append_history([row("second", 1, 2.0)])

            first = pd.read_csv(clients[0].history_path)
            second = pd.read_csv(clients[1].history_path)
            self.assertEqual(first["round"].tolist(), [1, 2])
            self.assertEqual(first["value"].tolist(), [1.0, 0.7])
            self.assertEqual(second["client_id"].tolist(), ["second"])


def _history_rows(dataset, client_id, task, metric):
    rows = []
    for round_number in (1, 2):
        for metric_name, value in (("loss", 1.0 / round_number), (metric, 0.4 + round_number / 10)):
            rows.append({
                "setup": "federated", "fold": 0, "round": round_number,
                "dataset": dataset, "client_id": client_id, "task": task,
                "phase": "post_aggregation_round", "split": "val", "unit_type": "round",
                "unit_index": round_number, "metric_name": metric_name, "value": value,
                "n_samples": 4, "optimizer_steps": 0, "examples_processed": 0,
            })
        rows.append({
            "setup": "federated", "fold": 0, "round": round_number,
            "dataset": dataset, "client_id": client_id, "task": task,
            "phase": "local_step", "split": "train", "unit_type": "step",
            "unit_index": 1, "metric_name": "loss", "value": 1.2 / round_number,
            "n_samples": 2, "optimizer_steps": 1, "examples_processed": 2,
        })
    return rows


class TrainingCurveArtifactTests(unittest.TestCase):
    def _run_fixture(self, root):
        run = Path(root) / "run"
        clients = (
            ("Curated_BUSI", "busi_seg_0", "seg", "dice"),
            ("ISIC_2018", "isic_cls_0", "cls", "balanced_accuracy"),
        )
        for dataset, client_id, task, metric in clients:
            directory = run / "fold_0" / f"client_{client_id}"
            directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(_history_rows(dataset, client_id, task, metric)).to_csv(
                directory / "training_history.csv", index=False
            )
        events = []
        for round_number in (1, 2):
            events.append({
                "round": round_number, "stage": "evaluate",
                "clients": [
                    {"dataset": dataset, "client_id": client_id, "final_weight": 0.5}
                    for dataset, client_id, _, _ in clients
                ],
            })
        (run / "fold_0" / "aggregation_history.json").write_text(
            json.dumps(events), encoding="utf-8"
        )
        (run / "config.yaml").write_text(yaml.safe_dump({
            "federated": {
                "local_training": {"mode": "steps", "steps_per_round": 1, "local_epochs": 2},
                "training_telemetry": {
                    "enabled": True, "granularity": "epoch_and_round",
                    "formats": ["csv", "html", "png"],
                },
            }
        }), encoding="utf-8")
        return run

    def test_builds_csv_html_and_all_three_plot_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_fixture(directory)
            output = build_run_artifacts(run)
            self.assertTrue((output / "history.csv").exists())
            self.assertTrue((output / "dashboard.html").exists())
            self.assertTrue((output / "plots" / "overview" / "overview_metrics.png").exists())
            self.assertEqual(len(list((output / "plots" / "clients").rglob("*.png"))), 2)
            self.assertEqual(len(list((output / "plots" / "datasets").glob("*.png"))), 2)
            history = collect_history(run)
            post_aggregation = history[history.phase == "post_aggregation_round"]
            self.assertTrue(post_aggregation["aggregation_weight"].notna().all())

    def test_legacy_run_is_labeled_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "legacy"
            run.mkdir()
            output = build_run_artifacts(run, force=True)
            html = (output / "dashboard.html").read_text(encoding="utf-8")
            self.assertIn("Telemetria indisponível", html)


if __name__ == "__main__":
    unittest.main()
