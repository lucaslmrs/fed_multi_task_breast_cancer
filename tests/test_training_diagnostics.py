import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.experiments import training_diagnostics as td

ROUNDS = 30
HISTORY_FIELDS = ["setup", "fold", "round", "dataset", "client_id", "task", "phase", "split",
                  "unit_type", "unit_index", "metric_name", "value", "n_samples"]


def _decreasing(start=1.0, end=0.3, n=ROUNDS):
    return list(np.linspace(start, end, n))


def _config(master, rounds=ROUNDS, criterion="CE", epochs=None):
    config = {
        "model": {"sequences": 1},
        "data": {"root": "data", "dataset": "DS", "variant": "v", "classes": ["a", "b"],
                 "batch_size": 4},
        "datasets": {"DS": {"classes": ["a", "b"], "batch_size": 4,
                            "classification_criterion": criterion}},
        "federated": {"rounds": rounds, "datasets": ["DS"], "partition_file": str(master)},
        "training": {"epochs": epochs or rounds},
    }
    return config


class RunBuilder:
    """A synthetic federated run whose default curves are healthy for every check."""

    def __init__(self, root, kind="federated", criterion="CE"):
        self.root = Path(root) / f"{kind}_run"
        self.kind = kind
        self.master = Path(root) / "master.csv"
        self.master.write_text("fold\n", encoding="utf-8")
        self.config = _config(self.master, criterion=criterion)
        self.clients = {}
        self.results = []
        self.predictions = []

    def client(self, client_id, tasks=("seg", "cls"), train=None, val=None, steps=10,
               extra_rows=(), effective=100):
        rows = []
        train = train or {}
        val = val or {}
        for task in tasks:
            curve = train.get((task, "loss"), _decreasing())
            for round_index, value in enumerate(curve, start=1):
                for step in range(1, steps + 1):
                    rows.append(self._row(client_id, task, "local_step", "train", "step", step,
                                          "loss", value, round_index))
            # Validation improves, then plateaus: no overfitting and no round ceiling.
            half = ROUNDS // 2
            defaults = {"loss": _decreasing(0.9, 0.4, half) + [0.4] * (ROUNDS - half)}
            defaults[td.PRIMARY_METRIC[task]] = _decreasing(0.3, 0.8, half) + [0.8] * (
                ROUNDS - half)
            if task == "seg":
                defaults["empty_fp_image_rate"] = [0.1] * ROUNDS
                defaults["n_empty"] = [10] * ROUNDS
            for metric, curve in {**defaults, **{m: v for (t, m), v in val.items()
                                                 if t == task}}.items():
                for round_index, value in enumerate(curve, start=1):
                    rows.append(self._row(client_id, task, "post_aggregation_round", "val",
                                          "round", round_index, metric, value, round_index))
        for (task, metric), curve in train.items():
            if metric != "loss":
                for round_index, value in enumerate(curve, start=1):
                    rows.append(self._row(client_id, task, "local_step", "train", "step", 1,
                                          metric, value, round_index))
        rows.extend(extra_rows)
        self.clients[client_id] = (tasks, rows, effective)
        return self

    def _row(self, client_id, task, phase, split, unit, index, metric, value, round_index):
        return {"setup": self.kind, "fold": 0, "round": round_index, "dataset": "DS",
                "client_id": client_id, "task": task, "phase": phase, "split": split,
                "unit_type": unit, "unit_index": index, "metric_name": metric, "value": value,
                "n_samples": 4}

    def result(self, client_id, task, **metrics):
        self.results.append({"setup": self.kind, "fold": 0, "client_id": client_id,
                             "dataset": "DS", "task": task, **metrics})
        return self

    def predictions_for(self, truth, predicted):
        for index, (t, p) in enumerate(zip(truth, predicted)):
            self.predictions.append({"setup": self.kind, "fold": 0, "client_id": "c0",
                                     "dataset": "DS", "patient_id": index, "ground_truth": t,
                                     "predicted": p, "class_name_0": "a", "class_name_1": "b"})
        return self

    def write(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")
        for client_id, (tasks, rows, effective) in self.clients.items():
            directory = self.root / "fold_0" / f"client_{client_id}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "metadata.yaml").write_text(yaml.safe_dump({
                "client_id": client_id, "dataset": "DS", "tasks": list(tasks),
                "task": "+".join(tasks), "class_names": ["a", "b"],
                "classification_criterion": self.config["datasets"]["DS"][
                    "classification_criterion"],
                "local_training_mode": "steps", "steps_per_round": 10,
                "effective_train_examples": effective,
            }), encoding="utf-8")
            pd.DataFrame(rows, columns=HISTORY_FIELDS).to_csv(
                directory / "training_history.csv", index=False)
        if not self.results:
            self.result("c0", "seg", dice_positive=0.7, empty_fp_image_rate=0.1, n_empty=10)
            self.result("c0", "cls", balanced_acc=0.8, macro_f1=0.8)
        pd.DataFrame(self.results).to_csv(self.root / f"{self.kind}_test_results.csv",
                                          index=False)
        if self.predictions:
            pd.DataFrame(self.predictions).to_csv(
                self.root / f"{self.kind}_cls_predictions.csv", index=False)
        return self.root


def _diagnose(root, **kwargs):
    kwargs.setdefault("baseline_root", Path(root).parent / "no_baselines")
    frame, summary = td.diagnose(root, **kwargs)
    return frame, summary


def _ids(frame, check=None):
    return set(frame.check_id) if check is None else frame[frame.check_id == check]


class HealthyRunTests(unittest.TestCase):
    def test_healthy_run_has_no_curve_findings(self):
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client("c0").client("c1").write()
            frame, summary = _diagnose(run)
            self.assertFalse(_ids(frame) & {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
                                            "D13", "D14", "D15"}, frame)
            self.assertEqual(summary["checks"]["D3"]["state"], "ok")
            self.assertEqual(summary["checks"]["D11"]["state"], "n/a")
            self.assertEqual(summary["status"], "ok")

    def test_post_local_round_is_not_read_as_a_second_training_series(self):
        with tempfile.TemporaryDirectory() as root:
            builder = RunBuilder(root)
            flat = [builder._row("c0", "seg", "post_local_round", "train", "round", r,
                                 "loss", 1.0, r) for r in range(1, ROUNDS + 1)]
            run = builder.client("c0", tasks=("seg",), extra_rows=flat).write()
            frame, _ = _diagnose(run)
            self.assertFalse(_ids(frame) & {"D3", "D4"}, frame)


class CurveCheckTests(unittest.TestCase):
    def _run(self, root, criterion="CE", **client):
        return RunBuilder(root, criterion=criterion).client("c0", **client).write()

    def test_d1_infinite_value(self):
        with tempfile.TemporaryDirectory() as root:
            curve = _decreasing()
            curve[5] = math.inf
            frame, _ = _diagnose(self._run(root, train={("seg", "loss"): curve}))
            self.assertEqual(_ids(frame, "D1").severity.tolist(), ["critical"])

    def test_d2_chance_level_only_for_ce(self):
        chance = [math.log(2)] * ROUNDS
        with tempfile.TemporaryDirectory() as root:
            frame, _ = _diagnose(self._run(root, train={("cls", "loss"): chance}))
            self.assertIn("D2", _ids(frame))
        with tempfile.TemporaryDirectory() as root:
            frame, summary = _diagnose(self._run(root, criterion="Focal",
                                                 train={("cls", "loss"): chance}))
            self.assertNotIn("D2", _ids(frame))
            self.assertEqual(summary["checks"]["D2"]["state"], "n/a")

    def test_d3_stagnation(self):
        with tempfile.TemporaryDirectory() as root:
            frame, _ = _diagnose(self._run(root, train={("seg", "loss"): [0.5] * ROUNDS}))
            self.assertEqual(_ids(frame, "D3").task.tolist(), ["seg"])

    def test_d4_divergence_and_spikes(self):
        with tempfile.TemporaryDirectory() as root:
            frame, _ = _diagnose(self._run(root, train={("seg", "loss"): _decreasing(0.3, 1.0)}))
            self.assertIn("D4", _ids(frame))
        with tempfile.TemporaryDirectory() as root:
            curve = _decreasing()
            curve[12] = curve[15] = 50.0
            frame, _ = _diagnose(self._run(root, train={("seg", "loss"): curve}))
            self.assertEqual(_ids(frame, "D4").metric.tolist(), ["train loss spikes"])

    def test_d5_overfitting(self):
        val = list(np.linspace(0.9, 0.4, 10)) + list(np.linspace(0.4, 0.8, ROUNDS - 10))
        with tempfile.TemporaryDirectory() as root:
            frame, _ = _diagnose(self._run(root, val={("cls", "loss"): val}))
            found = _ids(frame, "D5")
            self.assertEqual((found.task.tolist(), found.severity.tolist()), (["cls"], ["high"]))

    def test_d6_d7_collapse_and_floor(self):
        with tempfile.TemporaryDirectory() as root:
            frame, _ = _diagnose(self._run(root, val={
                ("seg", "dice_positive"): [0.0] * ROUNDS,
                ("cls", "balanced_accuracy"): [0.52] * ROUNDS,
            }))
            self.assertTrue({"D6", "D7"} <= _ids(frame))

    def test_d11_gradient_norm(self):
        with tempfile.TemporaryDirectory() as root:
            frame, summary = _diagnose(self._run(root, train={
                ("combined", "grad_norm"): [1.0] * 10 + [5.0] * 10 + [50.0] * 10}))
            self.assertEqual(_ids(frame, "D11").severity.tolist(), ["high"])
        with tempfile.TemporaryDirectory() as root:
            frame, _ = _diagnose(self._run(root, train={
                ("combined", "grad_norm"): [1.0] * (ROUNDS - 1) + [math.inf]}))
            self.assertEqual(_ids(frame, "D11").severity.tolist(), ["critical"])

    def test_d14_validation_and_test_empty_false_positives(self):
        with tempfile.TemporaryDirectory() as root:
            builder = RunBuilder(root).client(
                "c0", val={("seg", "empty_fp_image_rate"): [1.0] * ROUNDS})
            builder.result("c0", "seg", dice_positive=0.7, empty_fp_image_rate=1.0, n_empty=19)
            frame, _ = _diagnose(builder.write())
            self.assertEqual(sorted(_ids(frame, "D14").metric),
                             ["test empty_fp_image_rate", "val empty_fp_image_rate"])

    def test_d15_still_improving_at_the_round_budget(self):
        with tempfile.TemporaryDirectory() as root:
            frame, _ = _diagnose(self._run(root, tasks=("seg",), val={
                ("seg", "loss"): _decreasing(0.9, 0.3), ("seg", "dice_positive"): _decreasing(
                    0.2, 0.8)}))
            self.assertEqual(_ids(frame, "D15").severity.tolist(), ["info"])


class ResultCheckTests(unittest.TestCase):
    def test_d8_heterogeneity_requires_a_failed_client(self):
        with tempfile.TemporaryDirectory() as root:
            builder = RunBuilder(root).client("c0", tasks=("seg",)).client(
                "c1", tasks=("seg",), val={("seg", "dice_positive"): [0.0] * ROUNDS})
            builder.result("c0", "seg", dice_positive=0.6).result("c1", "seg", dice_positive=0.1)
            frame, _ = _diagnose(builder.write())
            self.assertEqual(_ids(frame, "D8").client_id.tolist(), ["c1"])
        with tempfile.TemporaryDirectory() as root:
            builder = RunBuilder(root).client("c0", tasks=("seg",)).client("c1", tasks=("seg",))
            builder.result("c0", "seg", dice_positive=0.6).result("c1", "seg", dice_positive=0.1)
            frame, _ = _diagnose(builder.write())
            self.assertNotIn("D8", _ids(frame))

    def test_d13_prediction_collapse(self):
        truth = [0] * 80 + [1] * 20
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client("c0").predictions_for(truth, [1] * 60 + [0] * 40).write()
            frame, _ = _diagnose(run)
            self.assertEqual(_ids(frame, "D13").severity.tolist(), ["high"])
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client("c0").predictions_for(truth, [1] * 100).write()
            self.assertEqual(_ids(_diagnose(run)[0], "D13").severity.tolist(), ["critical"])
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client("c0").predictions_for(truth, truth).write()
            self.assertNotIn("D13", _ids(_diagnose(run)[0]))

    def test_d10_exposure_imbalance(self):
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client("c0", effective=40).client("c1", effective=4000).write()
            frame, _ = _diagnose(run)
            self.assertEqual(_ids(frame, "D10").severity.tolist(), ["info"])


class BaselineTests(unittest.TestCase):
    def _baseline(self, root, master, dataset="DS", mode="classification", **metrics):
        directory = Path(root) / "baselines" / mode / f"{dataset}_20261015_120000"
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(yaml.safe_dump(_config(master)), encoding="utf-8")
        pd.DataFrame([{"setup": "baseline", "fold": 0, "client_id": "c0", "dataset": dataset,
                       "task": "cls", "best_epoch": 5, **metrics}]).to_csv(
            directory / "baseline_test_results.csv", index=False)
        return Path(root) / "baselines"

    def test_d12_below_baseline_beyond_noise(self):
        with tempfile.TemporaryDirectory() as root:
            builder = RunBuilder(root).client("c0", tasks=("cls",))
            builder.result("c0", "cls", balanced_acc=0.60, macro_f1=0.60)
            run = builder.write()
            baselines = self._baseline(root, builder.master, balanced_acc=0.80, macro_f1=0.605)
            frame, _ = _diagnose(run, baseline_root=baselines)
            self.assertEqual(_ids(frame, "D12").metric.tolist(), ["test balanced_acc"])

    def test_d12_needs_the_same_master(self):
        with tempfile.TemporaryDirectory() as root:
            builder = RunBuilder(root).client("c0", tasks=("cls",))
            builder.result("c0", "cls", balanced_acc=0.60, macro_f1=0.60)
            run = builder.write()
            baselines = self._baseline(root, Path(root) / "other.csv", balanced_acc=0.80)
            frame, summary = _diagnose(run, baseline_root=baselines)
            self.assertNotIn("D12", _ids(frame))
            self.assertIn("different test slices", summary["checks"]["D12"]["reason"])

    def test_baseline_run_history_and_epoch_ceiling(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "seg_run"
            run.mkdir()
            config = _config(Path(root) / "m.csv", epochs=20)
            config["experiment"] = {"baseline_mode": "segmentation", "baseline_dataset": "DS"}
            (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            epochs = range(1, 21)
            pd.DataFrame({"fold": 0, "epoch": list(epochs),
                          "train_loss": _decreasing(1.0, 0.2, 20),
                          "val_loss": _decreasing(1.0, 0.3, 20), "best": True,
                          "val_dice_positive_seg": [0.0] * 20}).to_csv(run / "history.csv",
                                                                     index=False)
            pd.DataFrame([{"setup": "baseline", "fold": 0, "client_id": "c0", "dataset": "DS",
                           "task": "seg", "best_epoch": 19, "dice_positive": 0.0}]).to_csv(
                run / "baseline_test_results.csv", index=False)
            frame, summary = _diagnose(run)
            self.assertEqual(summary["kind"], "baseline")
            self.assertEqual(_ids(frame, "D15").severity.tolist(), ["high"])
            self.assertIn("D6", _ids(frame))
            self.assertEqual(summary["checks"]["D12"]["state"], "n/a")


class ConflictTests(unittest.TestCase):
    def _summary(self, run, rows):
        pd.DataFrame(rows, columns=["metric", "scope", "group", "mean", "std", "rounds"]).to_csv(
            run / "conflict_summary.csv", index=False)

    def test_d9_reads_no_interaction_interference_and_overfitting(self):
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client("c0").write()
            self._summary(run, [
                ("m1_cos", "pair_class", "intra_same_tasks", 0.003, 0.007, 21),
                ("m2_z_train", "dataset_pair", "A -> B", -1e-3, 1e-4, 21),
                ("m2_z_val", "dataset_pair", "A -> B", -5e-4, 1e-4, 21),
                ("m2_z_train", "dataset_pair", "B -> A", -1e-3, 1e-2, 21),
                ("m2_z_val", "dataset_pair", "B -> A", -5e-4, 1e-4, 21),
                ("m2_z_train", "pair_class", "self", 1e-3, 1e-3, 21),
                ("m2_z_val", "pair_class", "self", -3e-3, 1e-3, 21),
                ("m2_z_train", "pair_class", "aggregate", 8e-4, 1e-4, 21),
            ])
            found = _ids(_diagnose(run)[0], "D9")
            self.assertEqual(
                sorted(zip(found.metric, found.severity)),
                [("M1 cos intra_same_tasks", "info"), ("M2 Z A -> B", "info"),
                 ("M2 Z self", "high")])
            interference = found[found.metric == "M2 Z A -> B"].evidence.iloc[0]
            self.assertIn("overstates", interference)
            self.assertIn("no interaction", found[found.severity == "info"].evidence.iloc[0])

    def test_d9_falls_back_to_cancellation(self):
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client("c0").write()
            pd.DataFrame([{"fold": "all_folds", "overall_mean": 0.80,
                           "orthogonal_reference_mean": 0.66}]).to_csv(
                run / "negative_transfer_summary.csv", index=False)
            found = _ids(_diagnose(run)[0], "D9")
            self.assertEqual((found.metric.tolist(), found.severity.tolist()),
                             (["cancellation excess"], ["high"]))

    def test_d9_falls_back_to_the_history_cosine(self):
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client("c0").write()
            entry = {"stage": "fit", "gradient_conflict": {
                "clients": ["DS/seg/c0", "DS/seg/c1", "X/cls/c2"],
                "cosine": {"shared_total": [[1, 0.001, 0.5], [0.001, 1, 0.5], [0.5, 0.5, 1]]}}}
            (run / "fold_0" / "aggregation_history.json").write_text(json.dumps([entry]))
            found = _ids(_diagnose(run)[0], "D9")
            self.assertAlmostEqual(found.value.iloc[0], 0.001)


class ToleranceAndCliTests(unittest.TestCase):
    def test_old_run_without_histories_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "old"
            (run / "fold_0" / "client_c0").mkdir(parents=True)
            (run / "fold_0" / "client_c0" / "metadata.yaml").write_text("client_id: c0\n")
            pd.DataFrame([{"setup": "federated", "fold": 0, "client_id": "c0", "dataset": "DS",
                           "task": "seg", "dice_positive": 0.5}]).to_csv(
                run / "federated_test_results.csv", index=False)
            frame, summary = _diagnose(run)
            for check in ("D1", "D3", "D9", "D11"):
                self.assertEqual(summary["checks"][check]["state"], "n/a", check)
            self.assertTrue(frame.empty)

    def test_execution_log_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "old"
            run.mkdir()
            lines = ["************* FOLD 0 (federated) *************"]
            for r in range(1, 21):
                dice = 0.01
                lines.append(f"[x] - [round {r}] h/u diagnostic mean val loss {1.0 - r / 40:.3f} "
                             f"| effective participation={{}} | per-dataset/task metric "
                             f"{{'DS/seg': {dice}, 'DS/cls': {r / 20:.3f}}}")
            (run / "execution.log").write_text("\n".join(lines))
            pd.DataFrame([{"setup": "federated", "fold": 0, "client_id": "c0", "dataset": "DS",
                           "task": "seg", "dice_positive": 0.5}]).to_csv(
                run / "federated_test_results.csv", index=False)
            frame, _ = _diagnose(run)
            self.assertEqual(set(frame.source), {"execution_log"})
            self.assertTrue({"D6", "D15"} <= _ids(frame))

    def test_study_discovery_skips_incomplete_and_finds_unindexed_arms(self):
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "study"
            arms = study / "runs" / "seed_1"
            for arm in ("a", "b"):
                (arms / arm).mkdir(parents=True)
                pd.DataFrame([{"task": "seg"}]).to_csv(arms / arm / "federated_test_results.csv")
            (arms / "a" / "fold_0.incomplete_1").mkdir()
            pd.DataFrame([{"arm_id": "a", "seed": 1, "run_path": str(arms / "a")}]).to_csv(
                study / "run_index.csv", index=False)
            found = {labels["arm_id"]: path for path, labels in td.discover_runs(study)}
            self.assertEqual(set(found), {"a", "b"})
            self.assertEqual(td._fold_dirs(found["a"]), [])

    def test_cli_writes_outputs_and_accepts_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            run = RunBuilder(root).client(
                "c0", val={("seg", "dice_positive"): [0.01] * ROUNDS}).write()
            with contextlib.redirect_stdout(io.StringIO()):
                td.main([str(run), "--baseline-root", str(Path(root) / "none")])
            findings = pd.read_csv(run / "diagnostics" / "findings.csv")
            self.assertIn("D6", set(findings.check_id))
            summary = json.loads((run / "diagnostics" / "summary.json").read_text())
            self.assertEqual(summary["status"], "critical")
            self.assertTrue((run / "diagnostics" / "summary.md").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                td.main([str(run), "--threshold", "D6_dice=0.001",
                         "--baseline-root", str(Path(root) / "none")])
            findings = pd.read_csv(run / "diagnostics" / "findings.csv")
            self.assertNotIn("D6", set(findings.check_id))
            with self.assertRaises(SystemExit):
                td.main([str(run), "--threshold", "nope=1"])


if __name__ == "__main__":
    unittest.main()
