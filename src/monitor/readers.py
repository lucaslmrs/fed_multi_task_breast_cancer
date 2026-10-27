"""Turn one run's artifacts into JSON-ready payloads for the monitor.

Every reader is read-only and tolerant: a missing or older-vintage artifact yields
``{"available": False, "reason": ...}`` instead of an exception. Curves come from the E1 readers
(``read_client_series``, ``read_baseline_series``, ``read_log_series``), so the app and the
diagnostics never disagree on how a history is read.
"""

import json
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import training_diagnostics as td
from src.monitor.catalog import infer_kind

SEG_METRICS = ("dice_positive", "iou_positive", "dice", "iou", "empty_fp_image_rate")
CLS_METRICS = ("balanced_acc", "macro_f1", "acc", "auc")
TEST_METRICS = {"seg": SEG_METRICS, "cls": CLS_METRICS}
MAX_TELEMETRY_POINTS = 600

_cache = {}
_cache_lock = threading.Lock()
calls = {"loads": 0}   # counts real (uncached) loads; read by the tests


def _signature(paths):
    signature = []
    for path in paths:
        try:
            stat = Path(path).stat()
        except OSError:
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def cached(name, paths, loader):
    """``loader()`` memoized on the (mtime, size) of ``paths``: reread only when a file changes."""
    key = (name, tuple(str(p) for p in paths))
    signature = _signature(paths)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == signature:
            return hit[1]
    calls["loads"] += 1
    value = loader()
    with _cache_lock:
        _cache[key] = (signature, value)
    return value


def clean(value):
    """Recursively convert to JSON-safe Python: numpy scalars, NaN/inf -> ``None``."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def unavailable(reason):
    return {"available": False, "reason": reason}


def _xy(series):
    series = pd.to_numeric(series, errors="coerce")
    return {"x": [clean(x) for x in series.index.tolist()],
            "y": [clean(y) for y in series.tolist()]}


def _key(task, metric):
    return f"{task}/{metric}"


# ---- curves -------------------------------------------------------------------------------

def _client_payload(series):
    return {
        "fold": series.fold, "client_id": series.client_id, "dataset": series.dataset,
        "tasks": list(series.tasks),
        "train": {_key(*k): _xy(v) for k, v in series.train.items()},
        "val": {_key(*k): _xy(v) for k, v in series.val.items()},
    }


def curves(run_dir, fold=None):
    """Per-client curves. Train = mean of ``local_step``/``local_epoch`` per round; validation =
    ``post_aggregation_round`` only (``post_local_round`` repeats the train aggregate, plano A8)."""
    run_dir = Path(run_dir)
    config = td._read_yaml(run_dir / "config.yaml")
    kind = infer_kind(run_dir, config)
    if kind == "baseline":
        path = run_dir / "history.csv"
        series = cached("baseline_curves", [path, run_dir / "config.yaml"],
                        lambda: [_client_payload(s) for s in td.read_baseline_series(run_dir,
                                                                                     config)])
        unit, source = "epoch", "history.csv"
    else:
        series = []
        for fold_dir in td._fold_dirs(run_dir):
            fold_id = fold_dir.name.split("_", 1)[1]
            if fold is not None and str(fold) != fold_id:
                continue
            for client_dir in sorted(fold_dir.glob("client_*")):
                files = [client_dir / "training_history.csv", client_dir / "metadata.yaml"]
                payload = cached("client_curves", files, lambda c=client_dir, f=fold_id: (
                    lambda s: _client_payload(s) if s is not None else None)(
                    td.read_client_series(c, config, f)))
                if payload is not None:
                    series.append(payload)
        unit, source = "round", "training_history.csv"
        if not series:
            series = _log_payload(run_dir, fold)
            source = "execution.log" if series else ""
    if fold is not None:
        series = [s for s in series if str(s["fold"]) == str(fold)]
    if not series:
        return unavailable("no training history yet (no per-client history, history.csv or "
                           "per-round log line)")
    return {"available": True, "kind": kind, "unit": unit, "source": source, "series": series}


def _log_payload(run_dir, fold=None):
    """Older runs without per-client curves: the server's per-round log line, per group."""
    log = run_dir / "execution.log"
    folds = cached("log_series", [log], lambda: td.read_log_series(run_dir))
    payload = []
    for fold_id, record in folds.items():
        if fold is not None and str(fold) != str(fold_id):
            continue
        for key, values in record.items():
            if key == "mean_val_loss":
                payload.append({"fold": fold_id, "client_id": "server (mean)", "dataset": "all",
                                "tasks": ["server"], "train": {},
                                "val": {"server/loss": _xy(values)}})
                continue
            dataset, _, task = key.partition("/")
            # A multitask group ("seg+cls") logs one primary metric for both tasks together.
            task = "combined" if "+" in task else task
            payload.append({"fold": fold_id, "client_id": f"{key} (server)", "dataset": dataset,
                            "tasks": [task], "train": {},
                            "val": {f"{task}/primary_metric": _xy(values)}})
    return payload


# ---- final metrics ------------------------------------------------------------------------

def _results_frame(run_dir):
    kind = td.run_kind(run_dir)
    if not kind:
        return None, kind
    path = Path(run_dir) / f"{kind}_test_results.csv"
    return cached("results", [path], lambda: td._read_csv(path)), kind


def summarize_results(frame):
    """Mean, std and n over client x fold rows, per dataset/task and metric."""
    rows = []
    for (dataset, task), group in frame.groupby(["dataset", "task"], sort=True):
        for metric in TEST_METRICS.get(task, ()):
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({"dataset": dataset, "task": task, "metric": metric,
                         "mean": float(values.mean()),
                         "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                         "n": int(len(values))})
    return rows


def final_metrics(run_dir):
    frame, kind = _results_frame(run_dir)
    if frame is None or frame.empty:
        return unavailable("no test results yet (run in progress or interrupted)")
    keep = ["fold", "client_id", "dataset", "task"]
    metrics = [m for task in TEST_METRICS.values() for m in task if m in frame]
    clients = frame[[c for c in keep if c in frame] + metrics]
    return {"available": True, "kind": kind, "summary": clean(summarize_results(frame)),
            "clients": clean(clients.to_dict("records")), "metrics": metrics}


# ---- confusion matrices -------------------------------------------------------------------

def _class_names(group):
    """Class names in index order from the ``class_name_<k>`` columns of a prediction slice."""
    names = {}
    for column in group.columns:
        if column.startswith("class_name_"):
            values = group[column].dropna()
            if not values.empty:
                names[int(column.rsplit("_", 1)[1])] = str(values.iloc[0])
    labels = pd.to_numeric(pd.concat([group["ground_truth"], group["predicted"]]),
                           errors="coerce").dropna().astype(int)
    size = max([*names.keys(), *labels.tolist(), -1]) + 1
    return [names.get(i, str(i)) for i in range(size)]


def _counts(group, size):
    matrix = np.zeros((size, size), dtype=int)
    truth = pd.to_numeric(group["ground_truth"], errors="coerce")
    predicted = pd.to_numeric(group["predicted"], errors="coerce")
    for t, p in zip(truth, predicted):
        if pd.notna(t) and pd.notna(p) and 0 <= int(t) < size and 0 <= int(p) < size:
            matrix[int(t), int(p)] += 1
    return matrix.tolist()


def confusion(run_dir):
    """Test confusion matrices per dataset (rows = true class, columns = predicted), pooled over
    clients x folds and per client, from ``<kind>_cls_predictions.csv``."""
    run_dir = Path(run_dir)
    kind = td.run_kind(run_dir)
    path = run_dir / f"{kind}_cls_predictions.csv" if kind else None
    if path is None or not path.exists():
        return unavailable("no classification predictions yet (run in progress or no cls task)")

    def load():
        frame = td._read_csv(path)
        if frame is None or frame.empty or not {"dataset", "ground_truth", "predicted"} <= set(frame):
            return unavailable("classification predictions without ground_truth/predicted")
        datasets = []
        for dataset, group in frame.groupby("dataset", sort=True):
            classes = _class_names(group)
            clients = []
            if "client_id" in group:
                keys = ["client_id"] + (["fold"] if "fold" in group else [])
                for key, part in group.groupby(keys, sort=True):
                    key = key if isinstance(key, tuple) else (key,)
                    clients.append({"client_id": str(key[0]),
                                    "fold": str(key[1]) if len(key) > 1 else None,
                                    "counts": _counts(part, len(classes))})
            datasets.append({"dataset": str(dataset), "classes": classes,
                             "counts": _counts(group, len(classes)), "n": int(len(group)),
                             "clients": clients})
        return {"available": True, "kind": kind, "datasets": datasets}

    return cached("confusion", [path], load)


# ---- aggregation --------------------------------------------------------------------------

def _aggregation_fold(history):
    fits = [e for e in history if e.get("stage") == "fit"]
    rounds = [int(e.get("round", i + 1)) for i, e in enumerate(fits)]
    weights = {}
    for index, entry in enumerate(fits):
        for client in entry.get("clients") or []:
            column = weights.setdefault(str(client.get("client_id")), [None] * len(fits))
            column[index] = client.get("final_weight")
    transfer = {}
    with_transfer = [e.get("negative_transfer") for e in fits]
    if any(isinstance(t, dict) for t in with_transfer):
        keys = sorted({k for t in with_transfer if isinstance(t, dict) for k in t})
        transfer = {k: [(t or {}).get(k) for t in with_transfer] for k in keys}
    participation = {}
    for index, entry in enumerate(fits):
        for dataset, share in (entry.get("dataset_participation") or {}).items():
            participation.setdefault(dataset, [None] * len(fits))[index] = share
    return {"rounds": rounds, "weights": weights, "dataset_participation": participation,
            "negative_transfer": transfer or None}


def aggregation(run_dir):
    """Client weights, dataset participation and cancellation per round, per finished fold."""
    run_dir = Path(run_dir)
    folds, reasons = {}, []
    for fold_dir in td._fold_dirs(run_dir):
        path = fold_dir / "aggregation_history.json"
        if not path.exists():
            reasons.append(f"{fold_dir.name}: aggregation_history.json is written at the end of "
                           "the fold")
            continue

        def load(p=path):
            try:
                return _aggregation_fold(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                return None
        payload = cached("aggregation", [path], load)
        if payload is not None:
            folds[fold_dir.name.split("_", 1)[1]] = payload
    if not folds:
        return unavailable("; ".join(reasons) or "no aggregation history (baseline or "
                                                   "standalone-only run)")
    if not any(f["negative_transfer"] for f in folds.values()):
        csv = run_dir / "negative_transfer_rounds.csv"
        frame = td._read_csv(csv)
        if frame is not None and not frame.empty:
            for fold, group in frame.groupby("fold"):
                if str(fold) in folds:
                    group = group.sort_values("round")
                    folds[str(fold)]["negative_transfer"] = {
                        c: group[c].tolist() for c in group.columns if c not in ("fold", "round")}
    transfer_reason = "" if any(f["negative_transfer"] for f in folds.values()) else (
        "history predates the negative_transfer field")
    return clean({"available": True, "folds": folds, "pending": reasons,
                  "negative_transfer_reason": transfer_reason})


# ---- conflict -----------------------------------------------------------------------------

def conflict(run_dir):
    run_dir = Path(run_dir)
    summary_path = run_dir / "conflict_summary.csv"
    m1_path = run_dir / "conflict_m1_pairs.csv"
    m2_path = run_dir / "conflict_m2_lookahead.csv"
    if not any(p.exists() for p in (summary_path, m1_path, m2_path)):
        return unavailable("no conflict_*.csv (run predates E0.3, conflict disabled, or still "
                           "training)")

    def load():
        payload = {"available": True, "caveat": td.D9_FULL_DELTA_CAVEAT}
        summary = td._read_csv(summary_path)
        payload["summary"] = summary.to_dict("records") if summary is not None else []
        m1 = td._read_csv(m1_path)
        if m1 is not None and not m1.empty:
            mean = m1.groupby(["pair_class", "round"])["cos"].mean()
            payload["m1"] = {cls: _xy(group.droplevel(0)) for cls, group in
                             mean.groupby(level=0)}
            blocks = m1.groupby(["block", "pair_class"])["cos"].mean().unstack()
            payload["m1_blocks"] = {"blocks": blocks.index.tolist(),
                                    "classes": blocks.columns.tolist(),
                                    "values": blocks.values.tolist()}
        m2 = td._read_csv(m2_path)
        if m2 is not None and not m2.empty:
            payload["m2"] = {}
            for split, group in m2.groupby("split"):
                matrix = group.groupby(["source_dataset", "target_dataset"])["z"].mean().unstack()
                payload["m2"][split] = {"sources": matrix.index.tolist(),
                                        "targets": matrix.columns.tolist(),
                                        "values": matrix.values.tolist()}
        return clean(payload)
    return cached("conflict", [summary_path, m1_path, m2_path], load)


# ---- resources ----------------------------------------------------------------------------

def resources(run_dir):
    run_dir = Path(run_dir)
    gpu_path, events_path = run_dir / "gpu_telemetry.csv", run_dir / "runtime_events.csv"
    if not gpu_path.exists() and not events_path.exists():
        return unavailable("no gpu_telemetry.csv or runtime_events.csv (CPU run or telemetry "
                           "disabled); per-actor RSS memory is not recorded (plano E0.4 (e))")

    def load():
        payload = {"available": True}
        gpu = td._read_csv(gpu_path)
        if gpu is not None and not gpu.empty:
            payload["gpu"] = {}
            for index, group in gpu.groupby("gpu_index"):
                # Bin means rather than a stride: 1 Hz utilization is too spiky to read.
                bins = np.arange(len(group)) // max(1, len(group) // MAX_TELEMETRY_POINTS + 1)
                group = group.select_dtypes(include=[np.number]).groupby(bins).mean()
                payload["gpu"][str(index)] = {
                    c: group[c].tolist() for c in (
                        "elapsed_seconds", "utilization_gpu_percent", "memory_used_mb",
                        "memory_total_mb", "power_watts", "temperature_c") if c in group}
        events = td._read_csv(events_path)
        if events is not None and not events.empty:
            durations = pd.to_numeric(events["duration_seconds"], errors="coerce")
            table = events.assign(duration_seconds=durations).groupby("phase").agg(
                count=("duration_seconds", "size"), total_seconds=("duration_seconds", "sum"))
            payload["phases"] = table.reset_index().to_dict("records")
        return clean(payload)
    return cached("resources", [gpu_path, events_path], load)


# ---- diagnostics --------------------------------------------------------------------------

def _diagnostics_payload(frame, summary, computed):
    return clean({"available": True, "computed": computed, "status": summary.get("status"),
                  "counts": summary.get("counts", {}), "checks": summary.get("checks", {}),
                  "findings": frame.to_dict("records") if frame is not None else []})


def diagnostics(run_dir):
    """The diagnostics saved by E1 in ``<run>/diagnostics``."""
    out = Path(run_dir) / td.OUTPUT_DIRNAME
    summary_path, findings_path = out / "summary.json", out / "findings.csv"
    if not summary_path.exists():
        return unavailable("not diagnosed yet (run in progress, or predates E1): use "
                           "'recompute' to diagnose in memory")

    def load():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frame = td._read_csv(findings_path)
        return _diagnostics_payload(frame, summary, computed=False)
    return cached("diagnostics", [summary_path, findings_path], load)


def diagnose_now(run_dir, baseline_root):
    """Run the E1 diagnostics in memory. Nothing is written into the run directory."""
    try:
        frame, summary = td.diagnose(run_dir, baseline_root=baseline_root)
    except Exception as error:  # a partial run can miss what a check needs
        return unavailable(f"diagnostics failed: {type(error).__name__}: {error}")
    return _diagnostics_payload(frame, summary, computed=True)
