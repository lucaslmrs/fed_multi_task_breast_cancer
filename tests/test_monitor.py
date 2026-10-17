import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

from src.monitor import catalog, compare, readers
from src.monitor.server import make_server
from tests.test_training_diagnostics import RunBuilder


def _age(path, seconds_ago=3 * 3600):
    """Back-date every file under ``path`` so the run reads as idle."""
    stamp = time.time() - seconds_ago
    for item in [Path(path), *Path(path).rglob("*")]:
        os.utime(item, (stamp, stamp))


def _federated(root, name, mode="steps", master="master.csv", topology="multi_task",
               results=True):
    Path(root).mkdir(parents=True, exist_ok=True)
    builder = RunBuilder(root)
    builder.root = Path(root) / name
    builder.config["federated"]["local_training"] = {"mode": mode}
    builder.config["federated"]["partition_file"] = str(Path(root) / master)
    builder.config["datasets"]["DS"]["client_topology"] = topology
    builder.client("c0").client("c1")
    run = builder.write()
    if not results:
        (run / "federated_test_results.csv").unlink()
    return run


def _baseline(root):
    run = Path(root) / "baseline" / "multitask" / "DS_20261015_224047"
    run.mkdir(parents=True)
    config = {"experiment": {"baseline_mode": "multitask", "baseline_dataset": "DS"},
              "datasets": {"DS": {"classes": ["a", "b"]}}, "training": {"epochs": 3}}
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    pd.DataFrame({"fold": [0] * 3, "epoch": [1, 2, 3], "train_loss": [1.0, 0.8, 0.6],
                  "val_loss": [1.1, 0.9, 0.8], "val_dice_positive_seg": [0.2, 0.4, 0.5]}
                 ).to_csv(run / "history.csv", index=False)
    pd.DataFrame([{"fold": 0, "client_id": "pool", "dataset": "DS", "task": "seg",
                   "dice_positive": 0.5}]).to_csv(run / "baseline_test_results.csv",
                                                  index=False)
    return run


def _study(root):
    study = Path(root) / "studies" / "toy"
    indexed = _federated(study / "runs" / "seed_1", "primary")
    _federated(study / "runs" / "seed_1", "outside")
    columns = ["study_id", "arm_id", "seed", "run_path", "status"]
    pd.DataFrame([["toy", "primary", "1", str(indexed), "complete"]], columns=columns).to_csv(
        study / "run_index.csv", index=False)
    pd.DataFrame([["toy", "primary", "1", str(indexed), "planned"],
                  ["toy", "planned_arm", "1", str(study / "runs/seed_1/planned_arm"), "planned"]],
                 columns=columns).to_csv(study / "execution_plan.csv", index=False)
    return study


class CatalogTests(unittest.TestCase):
    def test_statuses_cover_every_vintage(self):
        with tempfile.TemporaryDirectory() as root:
            _federated(root, "complete_run")
            running = _federated(root, "running_run", results=False)
            stale = _federated(root, "stale_run", results=False)
            _age(stale)
            _baseline(root)
            _study(root)
            data = catalog.scan(root)
            by_name = {(e["study_id"], e["name"]): e for e in data["runs"]}
            self.assertEqual(by_name[("", "complete_run")]["status"], "complete")
            self.assertEqual(by_name[("", running.name)]["status"], "running")
            self.assertEqual(by_name[("", "stale_run")]["status"], "stale")
            baseline = by_name[("", "DS_20261015_224047")]
            self.assertEqual((baseline["status"], baseline["kind"], baseline["group"]),
                             ("complete", "baseline", "baseline"))
            self.assertEqual(by_name[("toy", "primary")]["status"], "complete")
            self.assertEqual(by_name[("toy", "outside")]["arm_id"], "outside")
            self.assertEqual(by_name[("toy", "planned_arm")]["status"], "pending")
            # Study arms are listed once, under the study, not again as loose runs.
            self.assertEqual(sum(e["name"] == "primary" for e in data["runs"]), 1)
            self.assertEqual(by_name[("", "complete_run")]["budget_mode"], "steps")
            self.assertEqual(by_name[("", "complete_run")]["topology"], {"DS": "multi_task"})

    def test_resolve_refuses_paths_outside_the_root(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(PermissionError):
                catalog.resolve(root, "../etc")
            self.assertEqual(catalog.resolve(root, "a/b"), (Path(root) / "a/b").resolve())


class TaskModeTests(unittest.TestCase):
    def _run(self, root, name, clients, topology):
        builder = RunBuilder(root)
        builder.root = Path(root) / name
        builder.config["datasets"]["DS"]["client_topology"] = topology
        for client_id, tasks in clients:
            builder.client(client_id, tasks=tasks)
        return builder.write()

    def test_multitask_clients_read_from_the_fold(self):
        with tempfile.TemporaryDirectory() as root:
            run = self._run(root, "mt", [("DS_mt_0", ("seg", "cls")), ("DS_mt_1", ("seg", "cls"))],
                            "multi_task")
            entry = catalog.asdict(catalog.describe_run(run, root, "runs"))
            self.assertEqual(entry["task_mode"], "MTL")
            self.assertEqual(entry["composition"], [{"dataset": "DS", "tasks": ["seg", "cls"],
                                                     "topology": "multi_task",
                                                     "clients": {"mt": 2}}])
            self.assertTrue(entry["display_name"].startswith("Federado · DS"))

    def test_separate_task_clients_are_single_task(self):
        with tempfile.TemporaryDirectory() as root:
            run = self._run(root, "st", [("DS_seg_0", ("seg",)), ("DS_cls_0", ("cls",))],
                            "single_task")
            entry = catalog.asdict(catalog.describe_run(run, root, "runs"))
            self.assertEqual(entry["task_mode"], "ST")
            self.assertEqual(entry["composition"][0]["clients"], {"seg": 1, "cls": 1})

    def test_config_fallback_before_any_fold_and_seg_only(self):
        with tempfile.TemporaryDirectory() as root:
            run = self._run(root, "early", [("c0", ("seg",))], "multi_task")
            shutil.rmtree(run / "fold_0")
            config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
            config["federated"]["n_clients"] = {"DS": {"seg": 3, "cls": 3}}
            (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            entry = catalog.asdict(catalog.describe_run(run, root, "runs"))
            self.assertEqual(entry["composition"][0]["clients"], {"mt": 3})
            self.assertEqual(entry["task_mode"], "MTL")
            config["datasets"]["DS"]["client_topology"] = "single_task"
            config["federated"]["n_clients"] = {"DS": {"seg": 2, "cls": 0}}
            (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertEqual(catalog.describe_run(run, root, "runs").task_mode, "SEG")

    def test_baseline_mode_and_name(self):
        with tempfile.TemporaryDirectory() as root:
            entry = catalog.describe_run(_baseline(root), root, "baseline")
            self.assertEqual((entry.task_mode, entry.display_name), ("MTL", "DS · 15/10 22:40"))
            self.assertEqual(entry.composition[0]["topology"], "pooled")


class CurveTests(unittest.TestCase):
    def test_validation_reads_post_aggregation_round_only(self):
        with tempfile.TemporaryDirectory() as root:
            run = _federated(root, "run")
            history = run / "fold_0" / "client_c0" / "training_history.csv"
            frame = pd.read_csv(history)
            local = frame[frame.phase == "post_aggregation_round"].copy()
            local["phase"] = "post_local_round"
            local["value"] = 99.0
            pd.concat([frame, local]).to_csv(history, index=False)
            payload = readers.curves(run)
            self.assertTrue(payload["available"])
            client = next(s for s in payload["series"] if s["client_id"] == "c0")
            self.assertNotIn(99.0, client["val"]["seg/loss"]["y"])
            self.assertEqual(len(client["val"]["seg/loss"]["x"]), 30)
            self.assertIn("seg/loss", client["train"])

    def test_old_history_falls_back_to_the_execution_log(self):
        with tempfile.TemporaryDirectory() as root:
            run = _federated(root, "run")
            for path in run.glob("fold_0/client_*/training_history.csv"):
                path.unlink()
            (run / "execution.log").write_text(
                "************* FOLD 0 (federated) *************\n"
                "[round 1] flat diagnostic mean val loss 1.5 | x | per-dataset/task metric "
                "{'DS/seg': 0.4}\n"
                "[round 2] flat diagnostic mean val loss 1.2 | x | per-dataset/task metric "
                "{'DS/seg': 0.5}\n", encoding="utf-8")
            payload = readers.curves(run)
            self.assertEqual(payload["source"], "execution.log")
            group = next(s for s in payload["series"] if s["dataset"] == "DS")
            self.assertEqual(group["val"]["seg/primary_metric"]["y"], [0.4, 0.5])

    def test_baseline_curves_are_per_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            payload = readers.curves(_baseline(root))
            self.assertEqual(payload["unit"], "epoch")
            self.assertEqual(payload["series"][0]["val"]["seg/dice_positive"]["y"],
                             [0.2, 0.4, 0.5])


class ToleranceTests(unittest.TestCase):
    def test_missing_artifacts_report_a_reason(self):
        with tempfile.TemporaryDirectory() as root:
            run = _federated(root, "run")
            fits = [{"round": r, "stage": "fit", "clients": [
                {"client_id": "c0", "final_weight": 0.5}, {"client_id": "c1",
                                                           "final_weight": 0.5}]}
                    for r in (1, 2)]
            (run / "fold_0" / "aggregation_history.json").write_text(json.dumps(fits),
                                                                     encoding="utf-8")
            agg = readers.aggregation(run)
            self.assertTrue(agg["available"])
            self.assertEqual(agg["folds"]["0"]["weights"]["c0"], [0.5, 0.5])
            self.assertIn("predates", agg["negative_transfer_reason"])
            for reader in (readers.conflict, readers.resources, readers.diagnostics):
                payload = reader(run)
                self.assertFalse(payload["available"])
                self.assertTrue(payload["reason"])

    def test_running_run_has_no_final_metrics_but_curves(self):
        with tempfile.TemporaryDirectory() as root:
            run = _federated(root, "run", results=False)
            self.assertFalse(readers.final_metrics(run)["available"])
            self.assertTrue(readers.curves(run)["available"])
            self.assertIn("end of the fold", readers.aggregation(run)["reason"])

    def test_clean_makes_nan_json_safe(self):
        payload = readers.clean({"a": float("nan"), "b": [float("inf"), 1.0]})
        self.assertEqual(json.dumps(payload, allow_nan=False), '{"a": null, "b": [null, 1.0]}')


class CacheTests(unittest.TestCase):
    def test_unchanged_files_are_not_reread(self):
        with tempfile.TemporaryDirectory() as root:
            run = _federated(root, "run")
            readers.curves(run)
            loads = readers.calls["loads"]
            readers.curves(run)
            self.assertEqual(readers.calls["loads"], loads)
            history = run / "fold_0" / "client_c0" / "training_history.csv"
            stamp = history.stat().st_mtime + 5
            os.utime(history, (stamp, stamp))
            readers.curves(run)
            self.assertEqual(readers.calls["loads"], loads + 1)


class CompareTests(unittest.TestCase):
    def _entries(self, root, *runs):
        return [catalog.asdict(catalog.describe_run(r, root, "runs")) for r in runs]

    def test_budget_and_master_mismatches_are_flagged(self):
        with tempfile.TemporaryDirectory() as root:
            a = _federated(root, "a", mode="steps", master="m1.csv")
            b = _federated(root, "b", mode="epochs", master="m2.csv", topology="single_task")
            result = compare.compare([a, b], self._entries(root, a, b))
            codes = {w["code"] for w in result["warnings"]}
            self.assertEqual(codes, {"budget_mode", "master", "topology"})
            self.assertIn("não devem ser lidas como significativas", result["note"])
            reference, other = result["runs"]
            self.assertTrue(all(r["delta_vs_reference"] is None for r in reference["final"]))
            self.assertTrue(all(r["delta_vs_reference"] == 0.0 for r in other["final"]))
            self.assertIn("val/seg/loss", other["curves"]["DS"])

    def test_same_protocol_has_no_warning(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = _federated(root, "a"), _federated(root, "b")
            self.assertEqual(compare.compare([a, b], self._entries(root, a, b))["warnings"], [])


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        self.root.mkdir()
        self.run = _federated(self.root, "run")
        (self.root.parent / "secret.txt").write_text("x", encoding="utf-8")
        (self.run / "run_report.html").write_text("<html>report</html>", encoding="utf-8")
        self.server = make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _get(self, path, method="GET"):
        request = urllib.request.Request(self.base + path, method=method)
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()

    def _status(self, path, method="GET"):
        try:
            return self._get(path, method)[0]
        except urllib.error.HTTPError as error:
            return error.code

    def test_api_and_page(self):
        status, body = self._get("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual([e["name"] for e in json.loads(body)["runs"]], ["run"])
        status, body = self._get("/api/curves?path=run")
        self.assertTrue(json.loads(body)["available"])
        self.assertIn(b"<html", self._get("/")[1].lower())
        self.assertEqual(self._get("/files/run/run_report.html")[1], b"<html>report</html>")

    def test_study_arm_keeps_its_name_when_opened(self):
        _study(self.root)
        status, body = self._get("/api/run?path=studies/toy/runs/seed_1/primary")
        entry = json.loads(body)["entry"]
        self.assertEqual((entry["display_name"], entry["group"], entry["study_id"]),
                         ("primary", "study", "toy"))
        self.assertIn("MTL", json.loads(self._get("/api/config")[1])["task_modes"])

    def test_paths_outside_the_root_or_unserved_are_refused(self):
        self.assertEqual(self._status("/api/curves?path=../"), 403)
        self.assertEqual(self._status("/files/%2e%2e/secret.txt"), 403)
        self.assertEqual(self._status("/files/run/config.yaml"), 403)
        self.assertEqual(self._status("/api/curves?path=missing"), 404)
        self.assertEqual(self._status("/static/../server.py"), 404)

    def test_on_demand_diagnosis_writes_nothing(self):
        before = {p: p.stat().st_mtime_ns for p in self.run.rglob("*")}
        status, body = self._get("/api/diagnose?path=run", method="POST")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["computed"])
        self.assertIn("D1", payload["checks"])
        after = {p: p.stat().st_mtime_ns for p in self.run.rglob("*")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
