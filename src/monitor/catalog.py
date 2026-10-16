"""Discover runs under a runs root and describe each one: kind, status, protocol, reports.

``training_diagnostics.discover_runs`` only sees finished runs (it keys on
``{kind}_test_results.csv``). The catalog also lists runs still training, runs that stopped
without results, and study arms that are planned but not started, without changing that function.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.experiments.training_diagnostics import (
    OUTPUT_DIRNAME,
    _master_path,
    _read_yaml,
    run_kind,
)
from src.federated.config import dataset_config

STATUSES = ("running", "stale", "complete", "pending")
STALE_MINUTES = 15
# Directories that belong to a run (or to the app's own outputs) and never contain another run.
_SKIP_DIRS = {OUTPUT_DIRNAME, "report", "training_curves", "conflict", "plots", "resolved",
              "analysis", "analysis_partial", "paired_analysis", "__pycache__"}
_ACTIVITY_GLOBS = ("execution.log", "history.csv", "gpu_telemetry.csv",
                   "fold_*/client_*/training_history.csv", "fold_*/aggregation_history.json")
_REPORTS = ("run_report.html", "negative_transfer.html", "training_curves/dashboard.html",
            "report.html", "analysis/report.html")


@dataclass
class RunEntry:
    path: str                 # relative to the runs root, with "/" separators
    name: str
    group: str                # "study", "baseline", "validation" or "runs"
    kind: str = ""            # federated / standalone / baseline
    status: str = "complete"
    study_id: str = ""
    seed: str = ""
    arm_id: str = ""
    datasets: list = field(default_factory=list)
    budget_mode: str = ""     # federated.local_training.mode; empty for a baseline
    topology: dict = field(default_factory=dict)
    master: str = ""
    rounds: int = 0
    folds: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    reports: list = field(default_factory=list)
    last_activity: float = 0.0


def is_run_dir(path):
    """A run directory: a ``config.yaml`` next to results, folds, a history or a log."""
    path = Path(path)
    if not (path / "config.yaml").is_file():
        return False
    if run_kind(path) or (path / "history.csv").exists() or (path / "execution.log").exists():
        return True
    return any(p.is_dir() for p in path.glob("fold_*"))


def infer_kind(run_dir, config):
    kind = run_kind(run_dir)
    if kind:
        return kind
    if (config.get("experiment") or {}).get("baseline_mode"):
        return "baseline"
    return "standalone" if (config.get("federated") or {}).get("standalone") else "federated"


def last_activity(run_dir):
    """Latest modification time among the files a training run rewrites as it goes."""
    run_dir = Path(run_dir)
    times = [run_dir.stat().st_mtime]
    for pattern in _ACTIVITY_GLOBS:
        for path in run_dir.glob(pattern):
            try:
                times.append(path.stat().st_mtime)
            except OSError:
                continue
    return max(times)


def _datasets(config, kind):
    if kind == "baseline":
        dataset = (config.get("experiment") or {}).get("baseline_dataset")
        return [str(dataset)] if dataset else []
    configured = (config.get("federated") or {}).get("datasets")
    if configured:
        return [str(d) for d in configured]
    dataset = (config.get("data") or {}).get("dataset")
    return [str(dataset)] if dataset else []


def _topology(config, datasets):
    topology = {}
    for dataset in datasets:
        try:
            topology[dataset] = str(dataset_config(config, dataset).get("client_topology",
                                                                        "single_task"))
        except (KeyError, TypeError):
            continue
    return topology


def _diagnostics_badge(run_dir):
    summary = Path(run_dir) / OUTPUT_DIRNAME / "summary.json"
    if not summary.exists():
        return {}
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if "runs" in data:   # consolidated study/baseline summary: the worst run sets the badge
        runs = data.get("runs") or []
        order = ("critical", "warning", "ok")
        statuses = [r.get("status") for r in runs if r.get("status") in order]
        counts = {}
        for run in runs:
            for severity, count in (run.get("counts") or {}).items():
                counts[severity] = counts.get(severity, 0) + int(count)
        status = min(statuses, key=order.index) if statuses else ""
        return {"status": status, "counts": counts}
    return {"status": data.get("status", ""), "counts": data.get("counts", {})}


def describe_run(run_dir, root, group, labels=None, stale_minutes=STALE_MINUTES, now=None):
    run_dir = Path(run_dir)
    root = Path(root)
    labels = labels or {}
    config = _read_yaml(run_dir / "config.yaml")
    kind = infer_kind(run_dir, config)
    datasets = _datasets(config, kind)
    federated = config.get("federated") or {}
    activity = last_activity(run_dir)
    if run_kind(run_dir):
        status = "complete"
    else:
        age = (now if now is not None else time.time()) - activity
        status = "running" if age < stale_minutes * 60 else "stale"
    master = _master_path(config)
    folds = sorted(p.name.split("_", 1)[1] for p in run_dir.glob("fold_*")
                   if p.is_dir() and ".incomplete" not in p.name)
    rounds = federated.get("rounds") if kind != "baseline" else (config.get("training") or {}
                                                                 ).get("epochs")
    return RunEntry(
        path=relative(run_dir, root), name=run_dir.name, group=group, kind=kind, status=status,
        study_id=labels.get("study_id", ""), seed=str(labels.get("seed", "")),
        arm_id=labels.get("arm_id", ""), datasets=datasets,
        budget_mode="" if kind == "baseline" else str(
            (federated.get("local_training") or {}).get("mode", "")),
        topology={} if kind == "baseline" else _topology(config, datasets),
        master=str(master) if master else "", rounds=int(rounds or 0), folds=folds,
        diagnostics=_diagnostics_badge(run_dir),
        reports=[relative(run_dir / r, root) for r in _REPORTS if (run_dir / r).is_file()],
        last_activity=activity,
    )


def relative(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _study_arms(study_dir):
    """``(run_dir_or_None, labels)`` for every arm of a study, including planned ones."""
    arms, seen = [], set()
    for name in ("run_index.csv", "execution_plan.csv"):
        index = study_dir / name
        if not index.exists():
            continue
        try:
            rows = pd.read_csv(index, dtype=str).fillna("").to_dict("records")
        except (OSError, ValueError):
            continue
        for row in rows:
            key = (row.get("seed", ""), row.get("arm_id", ""))
            if key in seen:
                continue
            seen.add(key)
            # Absolute paths in the index come from the machine that ran the study.
            candidates = [study_dir / "runs" / f"seed_{key[0]}" / key[1]]
            if row.get("run_path"):
                candidates.insert(0, Path(row["run_path"]))
            run_dir = next((c for c in candidates if c.is_dir() and is_run_dir(c)), None)
            arms.append((run_dir, {"seed": key[0], "arm_id": key[1]}))
    for seed_dir in sorted((study_dir / "runs").glob("seed_*")):
        for arm_dir in sorted(seed_dir.iterdir()):
            seed = seed_dir.name.removeprefix("seed_")
            if (seed, arm_dir.name) in seen or ".incomplete" in arm_dir.name:
                continue
            if arm_dir.is_dir() and is_run_dir(arm_dir):
                seen.add((seed, arm_dir.name))
                arms.append((arm_dir, {"seed": seed, "arm_id": arm_dir.name}))
    return arms


def _is_study_dir(path):
    return any((path / name).exists() for name in
               ("run_index.csv", "execution_plan.csv", "study_manifest.yaml"))


def _walk_runs(directory, max_depth):
    directory = Path(directory)
    if max_depth < 0 or not directory.is_dir():
        return
    for child in sorted(directory.iterdir()):
        if (not child.is_dir() or child.name in _SKIP_DIRS or child.name.startswith(".")
                or ".incomplete" in child.name or child.name.startswith("fold_")):
            continue
        if is_run_dir(child):
            yield child
        elif not (child.name == "studies" and child.parent == directory):
            yield from _walk_runs(child, max_depth - 1)


def scan(root, stale_minutes=STALE_MINUTES, now=None):
    """Return ``{"runs": [RunEntry...], "studies": [...]}`` for the runs root ``root``."""
    root = Path(root)
    entries, studies = [], []
    study_roots = []
    studies_dir = root / "studies"
    if studies_dir.is_dir():
        for study_dir in sorted(studies_dir.iterdir()):
            if study_dir.is_dir() and _is_study_dir(study_dir):
                study_roots.append((study_dir, study_dir.name))
                smoke = study_dir / "smoke"
                if smoke.is_dir() and _is_study_dir(smoke):
                    study_roots.append((smoke, f"{study_dir.name}/smoke"))
    for study_dir, study_id in study_roots:
        arms = _study_arms(study_dir)
        studies.append({
            "study_id": study_id, "path": relative(study_dir, root),
            "reports": [relative(study_dir / r, root) for r in _REPORTS
                        if (study_dir / r).is_file()],
            "diagnostics": _diagnostics_badge(study_dir), "arms": len(arms),
        })
        for run_dir, labels in arms:
            labels = {**labels, "study_id": study_id}
            if run_dir is None:
                entries.append(RunEntry(path="", name=labels["arm_id"], group="study",
                                        status="pending", study_id=study_id,
                                        seed=labels["seed"], arm_id=labels["arm_id"]))
            else:
                entries.append(describe_run(run_dir, root, "study", labels, stale_minutes, now))
    for run_dir in _walk_runs(root, max_depth=4):
        first = relative(run_dir, root).split("/", 1)[0]
        group = first if first in ("baseline", "validation") and first != run_dir.name else "runs"
        entries.append(describe_run(run_dir, root, group, None, stale_minutes, now))
    return {"runs": [asdict(e) for e in entries], "studies": studies}


def resolve(root, rel_path):
    """Resolve ``rel_path`` under ``root``; raise ``PermissionError`` if it escapes the root."""
    root = Path(root).resolve()
    target = (root / str(rel_path or "")).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(f"{rel_path} is outside the runs root")
    return target
