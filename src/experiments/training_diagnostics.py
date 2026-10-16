"""Offline training diagnostics: read a finished run's artifacts and flag training problems.

Every check reads durable artifacts only (per-client ``training_history.csv``, the baseline
``history.csv``, ``aggregation_history.json``, ``conflict_summary.csv``,
``negative_transfer_summary.csv``, test results and predictions), so it can be re-run on any run
of any vintage. A check whose inputs are missing reports ``n/a`` with the reason instead of
failing. Nothing here changes a run's numbers, and nothing is read from or written to the
scientific configuration: thresholds are the module constants below, overridable from the CLI.

Output, per run: ``diagnostics/{findings.csv, summary.json, summary.md}``. For a study or a
directory of runs, a consolidated ``diagnostics/`` is also written at the top level.

    python -m src.experiments.training_diagnostics runs/<run> [runs/studies/<id> runs/baseline]

Severities are descriptive flags for a single run on a single seed. They are not inferential
claims, and every threshold below is a screening heuristic, not a significance test.
"""

import argparse
import ast
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.federated.config import dataset_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
OUTPUT_DIRNAME = "diagnostics"
SEVERITIES = ("critical", "high", "info")

CHECKS = {
    "D1": "non-finite values",
    "D2": "classification loss at chance level",
    "D3": "stagnating training loss",
    "D4": "diverging training loss",
    "D5": "overfitting (validation loss rises after its minimum)",
    "D6": "collapsed segmentation",
    "D7": "classification at the balanced-accuracy floor",
    "D8": "intra-dataset heterogeneity with a failed client",
    "D9": "gradient conflict / client interaction",
    "D10": "training exposure imbalance across clients",
    "D11": "gradient norm",
    "D12": "below the non-federated baseline",
    "D13": "prediction collapse onto one class",
    "D14": "false positives on empty images",
    "D15": "epoch/round ceiling (still improving at the end)",
}

THRESHOLDS = {
    "min_points": 6,                 # shortest series the third-based rules accept
    "D2_chance_fraction": 0.95,      # last-third CE >= fraction * ln K
    "D3_min_drop": -0.05,            # (last / first third - 1) above this -> stagnation
    "D4_window": 11,                 # rolling-median window (rounds/epochs) for spikes
    "D4_mad_factor": 10.0,           # spike: round-mean loss > rolling median + factor * MAD
    "D4_spike_ratio": 3.0,           # ... and > ratio x the rolling median (healthy runs: < 1.9)
    "D4_spike_fraction": 0.01,       # ... in more than this fraction of rounds
    "D5_best_fraction": 0.60,        # best validation loss before this fraction of the run
    "D5_rise_factor": 1.15,          # final validation loss > factor * minimum
    "D6_dice": 0.05,                 # last-third validation positive Dice below this
    "D7_margin": 0.05,               # balanced accuracy <= 1/K + margin
    "D8_spread": 0.15,               # max - min test metric among clients of a dataset/task
    "D9_no_interaction": 0.02,       # |intra-dataset cosine| or |cancellation excess| below this
    "D9_first_order_conflict": -0.05,  # mean M1 cosine below this
    "D9_cancellation_excess": 0.05,  # cancellation above the orthogonal reference
    "D9_consistency_k": 2.0,         # mean + k * std / sqrt(rounds) < 0 -> consistent sign
    "D10_ratio": 5.0,                # max / min effective epochs across clients
    "D11_growth": 10.0,              # last-third / first-third median gradient norm
    "D13_excess": 0.20,              # predicted fraction - prevalence of the most predicted class
    "D13_constant": 0.95,            # predicted fraction of one class -> constant predictor
    "D13_min_samples": 20,
    "D14_rate": 0.5,                 # empty-image false-positive rate
    "D14_min_empty": 5,
    "D15_tail_fraction": 0.10,       # best in the last 10% of the federated rounds
    "D15_baseline_fraction": 0.80,   # baseline best epoch >= fraction * configured epochs
}

# Run-to-run GPU noise of a single run (E0.3.8): differences below it are not effects.
NOISE = {"Curated_BUSI": 0.05, "default": 0.01}

D9_FULL_DELTA_CAVEAT = (
    "Z applies the whole delta of the source client and overstates the effect; at the scale the "
    "aggregation applies (x w_i) the inter-dataset interference vanishes (plano E0.3.9)"
)

FINDING_FIELDS = (
    "run", "kind", "arm_id", "seed", "check_id", "severity", "fold", "dataset", "client_id",
    "task", "metric", "value", "threshold", "evidence", "source",
)
TRAIN_PHASES = ("local_step", "local_epoch")
VAL_PHASE = "post_aggregation_round"
PRIMARY_METRIC = {"seg": "dice_positive", "cls": "balanced_accuracy"}
TEST_METRIC = {"seg": "dice_positive", "cls": "balanced_acc"}
BASELINE_MODES = {"segmentation": ("seg",), "classification": ("cls",),
                  "multitask": ("seg", "cls")}
_TIMESTAMPED = re.compile(r"^(?P<dataset>.+)_(?P<stamp>\d{8}_\d{6})$")
_FOLD_MARK = re.compile(r"\*+ FOLD (\d+) \(")
_ROUND_LINE = re.compile(
    r"\[round (\d+)\].*?diagnostic mean val loss ([-\w.]+).*?per-dataset/task metric (\{.*\})"
)


# ---- data model ---------------------------------------------------------------------------

@dataclass
class Finding:
    check_id: str
    severity: str
    metric: str = ""
    value: float = float("nan")
    threshold: str = ""
    evidence: str = ""
    fold: str = ""
    dataset: str = ""
    client_id: str = ""
    task: str = ""
    source: str = ""


@dataclass
class Series:
    """One client's (or baseline pool's) curves: ``{(task, metric): pd.Series}`` by unit index."""
    fold: str
    client_id: str
    dataset: str
    tasks: tuple
    num_classes: int
    criterion: str
    train: dict = field(default_factory=dict)
    val: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)
    raw_values: object = None   # every numeric value read, for D1
    unit: str = "round"


class Diagnosis:
    """Accumulates findings and the state of every check for one run."""

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.findings = []
        self.state = {check: {"state": "n/a", "reason": "no applicable series or task"}
                      for check in CHECKS}

    def ran(self, check):
        if self.state[check]["state"] != "triggered":
            self.state[check] = {"state": "ok", "reason": ""}

    def na(self, check, reason):
        if self.state[check]["state"] == "n/a":
            self.state[check] = {"state": "n/a", "reason": reason}

    def add(self, finding):
        self.findings.append(finding)
        self.state[finding.check_id] = {"state": "triggered", "reason": ""}


# ---- small numeric helpers ----------------------------------------------------------------

def _thirds(series, min_points):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < min_points:
        return None
    third = max(1, len(values) // 3)
    return values[:third], values[-third:], values


def _fmt(value):
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _read_yaml(path):
    path = Path(path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _noise(dataset, noise):
    return float(noise.get(dataset, noise.get("default", 0.01)))


# ---- discovery ----------------------------------------------------------------------------

def run_kind(run_dir):
    for kind in ("federated", "standalone", "baseline"):
        if (Path(run_dir) / f"{kind}_test_results.csv").exists():
            return kind
    return None


def discover_runs(path, max_depth=3):
    """Return ``[(run_dir, labels)]`` for a run, a study (``run_index.csv``) or a folder of runs."""
    path = Path(path)
    if run_kind(path):
        return [(path, {})]
    index = path / "run_index.csv"
    if index.exists():
        found = []
        for row in pd.read_csv(index).to_dict("records"):
            candidates = [Path(str(row.get("run_path", "")))]
            candidates.append(path / "runs" / f"seed_{row.get('seed')}" / str(row.get("arm_id")))
            run_dir = next((c for c in candidates if run_kind(c)), None)
            if run_dir is not None:
                found.append((run_dir, {"arm_id": str(row.get("arm_id", "")),
                                        "seed": str(row.get("seed", ""))}))
        # An index can omit arms that were reused or launched outside the runner.
        listed = {run_dir.resolve() for run_dir, _ in found}
        for run_dir, labels in discover_runs(path / "runs", max_depth):
            if run_dir.resolve() not in listed:
                seed = run_dir.parent.name.removeprefix("seed_")
                found.append((run_dir, {"arm_id": run_dir.name, "seed": seed, **labels}))
        return found
    found = []
    if max_depth > 0 and path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_dir() and ".incomplete" not in child.name and child.name != OUTPUT_DIRNAME:
                found.extend(discover_runs(child, max_depth - 1))
    return found


def _fold_dirs(run_dir):
    return sorted(
        (p for p in Path(run_dir).glob("fold_*") if p.is_dir() and ".incomplete" not in p.name),
        key=lambda p: p.name,
    )


# ---- readers ------------------------------------------------------------------------------

def _client_classes(config, dataset, metadata):
    names = metadata.get("class_names")
    if names:
        return len(names)
    try:
        return len(dataset_config(config, dataset)["classes"])
    except (KeyError, TypeError):
        return 0


def _client_criterion(config, dataset, metadata):
    criterion = metadata.get("classification_criterion")
    if criterion:
        return str(criterion)
    try:
        return str(dataset_config(config, dataset).get("classification_criterion", "CE"))
    except (KeyError, TypeError):
        return "CE"


def read_client_series(client_dir, config, fold):
    """Curves of one federated/standalone client, or ``None`` without a training history."""
    client_dir = Path(client_dir)
    metadata = _read_yaml(client_dir / "metadata.yaml")
    history = _read_csv(client_dir / "training_history.csv")
    if history is None or history.empty:
        return None
    dataset = str(metadata.get("dataset") or history["dataset"].iloc[0])
    tasks = tuple(metadata.get("tasks") or [metadata.get("task", history["task"].iloc[0])])
    series = Series(
        fold=str(fold), client_id=str(metadata.get("client_id") or history["client_id"].iloc[0]),
        dataset=dataset, tasks=tasks, num_classes=_client_classes(config, dataset, metadata),
        criterion=_client_criterion(config, dataset, metadata),
    )
    history["value"] = pd.to_numeric(history["value"], errors="coerce")
    series.raw_values = history[["round", "phase", "task", "metric_name", "value"]]
    # ``post_local_round/train`` repeats the step aggregate exactly (plano A8): read each series once.
    train = history[history.phase.isin(TRAIN_PHASES)]
    for (task, metric), group in train.groupby(["task", "metric_name"]):
        series.train[(task, metric)] = group.groupby("round")["value"].mean().sort_index()
        series.steps[(task, metric)] = group.sort_values(["round", "unit_index"])["value"]
    val = history[history.phase == VAL_PHASE]
    for (task, metric), group in val.groupby(["task", "metric_name"]):
        series.val[(task, metric)] = group.groupby("round")["value"].mean().sort_index()
    return series


def read_baseline_series(run_dir, config):
    """Curves of a baseline run, converted from the wide per-epoch ``history.csv``."""
    history = _read_csv(Path(run_dir) / "history.csv")
    if history is None or history.empty:
        return []
    experiment = config.get("experiment", {})
    mode = str(experiment.get("baseline_mode", ""))
    dataset = str(experiment.get("baseline_dataset", ""))
    tasks = BASELINE_MODES.get(mode, ())
    loss_task = tasks[0] if len(tasks) == 1 else "combined"
    try:
        data_cfg = dataset_config(config, dataset)
        num_classes = len(data_cfg["classes"])
        criterion = str(data_cfg.get("classification_criterion", "CE"))
    except (KeyError, TypeError):
        num_classes, criterion = 0, "CE"
    result = []
    for fold, group in history.groupby("fold"):
        group = group.set_index("epoch").sort_index()
        series = Series(fold=str(fold), client_id="pool", dataset=dataset, tasks=tasks,
                        num_classes=num_classes, criterion=criterion, unit="epoch")
        numeric = group.select_dtypes(include=[np.number])
        series.raw_values = numeric
        series.train[(loss_task, "loss")] = group["train_loss"]
        series.steps[(loss_task, "loss")] = group["train_loss"]
        series.val[(loss_task, "loss")] = group["val_loss"]
        for column in group.columns:
            if column.startswith("train_grad_norm"):
                series.steps[("combined", column[len("train_"):])] = group[column]
            match = re.match(r"^val_(.+)_(seg|cls)$", column)
            if match:
                series.val[(match.group(2), match.group(1))] = group[column]
        result.append(series)
    return result


def read_log_series(run_dir):
    """Fallback for histories without per-client curves: the server's per-round log line.

    Returns ``{fold: {"mean_val_loss": Series, "<dataset>/<task>": Series}}``. A fold that was
    restarted keeps its last attempt only.
    """
    log = Path(run_dir) / "execution.log"
    if not log.exists():
        return {}
    folds, current = {}, "0"
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        mark = _FOLD_MARK.search(line)
        if mark:
            current = mark.group(1)
            folds[current] = {}
            continue
        match = _ROUND_LINE.search(line)
        if not match:
            continue
        server_round = int(match.group(1))
        record = folds.setdefault(current, {})
        record.setdefault("mean_val_loss", {})[server_round] = _float(match.group(2))
        try:
            groups = ast.literal_eval(match.group(3))
        except (ValueError, SyntaxError):
            continue
        for key, value in groups.items():
            record.setdefault(key, {})[server_round] = _float(value)
    return {fold: {key: pd.Series(values).sort_index() for key, values in record.items()}
            for fold, record in folds.items() if record}


def _master_path(config):
    """The frozen master a config trains on, resolved without requiring it to exist."""
    configured = (config.get("federated") or {}).get("partition_file")
    if configured:
        path = Path(configured)
    else:
        from src.dataset import paths
        try:
            path = paths.partition_file(config["data"])
        except (KeyError, TypeError):
            return None
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


# ---- checks over curves -------------------------------------------------------------------

def _scope(series, task):
    return dict(fold=series.fold, dataset=series.dataset, client_id=series.client_id, task=task)


def check_non_finite(diag, series, source):
    raw = series.raw_values
    if raw is None:
        return
    diag.ran("D1")
    if "value" in getattr(raw, "columns", []):
        # NaN is written for "not applicable" metrics (e.g. Dice on a batch without positives);
        # only +/-Inf, or NaN in a loss or gradient norm, is a numerical failure.
        values = raw["value"].to_numpy(dtype=float)
        numeric_metric = raw["metric_name"].astype(str).str.contains("loss|grad_norm").to_numpy()
        bad = raw[np.isinf(values) | (np.isnan(values) & numeric_metric)]
        if not bad.empty:
            first = bad.iloc[0]
            diag.add(Finding("D1", "critical", metric=str(first["metric_name"]), value=len(bad),
                             threshold="finite", source=source,
                             evidence=f"{len(bad)} non-finite values; first at round "
                                      f"{first['round']} {first['phase']}/{first['task']}",
                             **_scope(series, str(first["task"]))))
        return
    columns = [c for c in raw.columns if c.endswith("loss") or "grad_norm" in c]
    values = raw[columns].to_numpy(dtype=float) if columns else np.empty((0, 0))
    count = int((~np.isfinite(values)).sum())
    if count:
        diag.add(Finding("D1", "critical", metric=",".join(columns), value=count,
                         threshold="finite", source=source,
                         evidence=f"{count} non-finite loss/gradient values in history.csv",
                         **_scope(series, "combined")))


def check_training_curves(diag, series, source, baseline=False):
    t = diag.thresholds
    loss_keys = [key for key in series.train if key[1] == "loss"]
    task_keys = [key for key in loss_keys if key[0] != "combined"] or loss_keys
    for task, metric in task_keys:
        thirds = _thirds(series.train[(task, metric)], t["min_points"])
        if thirds is None:
            diag.na("D3", "series too short")
            continue
        first, last, _ = thirds
        first_mean, last_mean = float(first.mean()), float(last.mean())
        diag.ran("D3")
        diag.ran("D4")
        change = last_mean / first_mean - 1.0 if first_mean else float("nan")
        if last_mean > first_mean:
            diag.add(Finding("D4", "critical", metric="train loss", value=change,
                             threshold="last third <= first third", source=source,
                             evidence=f"train loss {first_mean:.4g} -> {last_mean:.4g}",
                             **_scope(series, task)))
        elif np.isfinite(change) and change > t["D3_min_drop"]:
            diag.add(Finding("D3", "high", metric="train loss", value=change,
                             threshold=f"> {t['D3_min_drop']}", source=source,
                             evidence=f"train loss {first_mean:.4g} -> {last_mean:.4g} "
                                      f"({change:+.1%})", **_scope(series, task)))
        # Spikes are read on the per-round mean (single-step segmentation losses are bimodal:
        # batches with and without lesion pixels) and against a rolling median, so that the
        # high early values of a loss that is simply decreasing do not count as spikes.
        curve = pd.Series(thirds[2])
        if len(curve) >= t["min_points"]:
            local = curve.rolling(int(t["D4_window"]), center=True, min_periods=1).median()
            residual = (curve - local).to_numpy()
            mad = float(np.median(np.abs(residual - np.median(residual))))
            spikes = float(np.mean((residual > t["D4_mad_factor"] * mad)
                                   & (curve.to_numpy() > t["D4_spike_ratio"] * local)))
            if spikes > t["D4_spike_fraction"]:
                diag.add(Finding("D4", "critical", metric="train loss spikes", value=spikes,
                                 threshold=f"> {t['D4_spike_fraction']}", source=source,
                                 evidence=f"{spikes:.1%} of {len(curve)} {series.unit}s jump "
                                          f"above {t['D4_spike_ratio']:g}x the rolling median "
                                          f"(and {t['D4_mad_factor']:g} MAD)",
                                 **_scope(series, task)))
        if task == "cls" and series.criterion.upper() == "CE" and series.num_classes > 1:
            chance = math.log(series.num_classes)
            diag.ran("D2")
            if last_mean >= t["D2_chance_fraction"] * chance:
                diag.add(Finding("D2", "critical", metric="train cls loss", value=last_mean,
                                 threshold=f">= {t['D2_chance_fraction']} ln {series.num_classes}"
                                           f" = {t['D2_chance_fraction'] * chance:.4g}",
                                 source=source,
                                 evidence=f"last-third CE {last_mean:.4g} vs chance {chance:.4g}",
                                 **_scope(series, task)))
        elif task == "cls":
            diag.na("D2", f"criterion {series.criterion} has no chance-level reference")
    if not any(key[0] == "cls" for key in task_keys):
        diag.na("D2", "no classification training loss")

    val_losses = [key for key in series.val if key[1] == "loss" and key[0] != "combined"]
    for task, _ in val_losses or [key for key in series.val if key[1] == "loss"]:
        _check_overfitting(diag, series, task, series.val[(task, "loss")], source, baseline)
    for (task, metric), values in series.val.items():
        thirds = _thirds(values, t["min_points"])
        if thirds is None:
            continue
        last_mean = float(thirds[1].mean())
        if task == "seg" and metric == "dice_positive":
            diag.ran("D6")
            if last_mean < t["D6_dice"]:
                diag.add(Finding("D6", "critical", metric="val dice_positive", value=last_mean,
                                 threshold=f"< {t['D6_dice']}", source=source,
                                 evidence=f"last-third val positive Dice {last_mean:.4g}; "
                                          f"max over the run {float(np.nanmax(thirds[2])):.4g}",
                                 **_scope(series, task)))
        if task == "cls" and metric == "balanced_accuracy" and series.num_classes > 1:
            floor = 1.0 / series.num_classes + t["D7_margin"]
            diag.ran("D7")
            if last_mean <= floor:
                diag.add(Finding("D7", "critical", metric="val balanced_accuracy",
                                 value=last_mean, threshold=f"<= {floor:.4g}", source=source,
                                 evidence=f"last-third val balanced accuracy {last_mean:.4g} "
                                          f"(K={series.num_classes})", **_scope(series, task)))
        if task == "seg" and metric == "empty_fp_image_rate":
            empties = series.val.get(("seg", "n_empty"))
            n_empty = float(empties.iloc[-1]) if empties is not None and len(empties) else np.nan
            if last_mean >= t["D14_rate"] and not (n_empty < t["D14_min_empty"]):
                diag.add(Finding("D14", "high", metric="val empty_fp_image_rate", value=last_mean,
                                 threshold=f">= {t['D14_rate']}", source=source,
                                 evidence=f"last-third validation rate {last_mean:.3g} over "
                                          f"{_fmt(n_empty)} empty images", **_scope(series, task)))
    _check_round_ceiling(diag, series, source, baseline)
    _check_grad_norm(diag, series, source)


def _check_overfitting(diag, series, task, values, source, baseline):
    t = diag.thresholds
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < t["min_points"]:
        return
    diag.ran("D5")
    position = int(np.argmin(values.to_numpy()))
    best, final = float(values.iloc[position]), float(values.iloc[-1])
    train = series.train.get((task, "loss"))
    if train is None:
        train = next((v for (k, m), v in series.train.items() if m == "loss"), None)
    train_thirds = _thirds(train, t["min_points"]) if train is not None else None
    falling = train_thirds is not None and train_thirds[1].mean() < train_thirds[0].mean()
    if (position < t["D5_best_fraction"] * len(values) and best > 0
            and final > t["D5_rise_factor"] * best and falling):
        diag.add(Finding(
            "D5", "info" if baseline else "high", metric="val loss", value=final / best,
            threshold=f"best before {t['D5_best_fraction']:.0%}, final > {t['D5_rise_factor']} x min",
            source=source,
            evidence=f"best {series.unit} {values.index[position]} of {len(values)} "
                     f"(val {best:.4g}) -> final {final:.4g} while train loss keeps falling"
                     + ("; early stopping evaluates the best checkpoint" if baseline else
                        "; the protocol evaluates the last round"),
            **_scope(series, task)))


def _check_round_ceiling(diag, series, source, baseline):
    if baseline:
        return  # the baseline ceiling is read from best_epoch vs the configured budget
    t = diag.thresholds
    for task in series.tasks:
        loss = pd.to_numeric(series.val.get((task, "loss")), errors="coerce")
        metric = pd.to_numeric(series.val.get((task, PRIMARY_METRIC.get(task, ""))),
                               errors="coerce")
        if loss is None or metric is None:
            continue
        loss, metric = loss.dropna(), metric.dropna()
        if len(loss) < t["min_points"] or len(metric) < t["min_points"]:
            continue
        diag.ran("D15")
        tail = max(1, int(math.ceil(t["D15_tail_fraction"] * len(loss))))
        best_loss = int(np.argmin(loss.to_numpy()))
        best_metric = int(np.argmax(metric.to_numpy()))
        if best_loss >= len(loss) - tail and best_metric >= len(metric) - tail:
            diag.add(Finding(
                "D15", "info", metric=f"val loss & {PRIMARY_METRIC[task]}",
                value=float(loss.index[best_loss]),
                threshold=f"best in the last {t['D15_tail_fraction']:.0%} of rounds", source=source,
                evidence=f"val loss minimum at round {loss.index[best_loss]} and best "
                         f"{PRIMARY_METRIC[task]} at round {metric.index[best_metric]} of "
                         f"{len(loss)}: still improving at the round budget",
                **_scope(series, task)))


def _check_grad_norm(diag, series, source):
    t = diag.thresholds
    keys = [key for key in series.steps if key[1] in ("grad_norm", "grad_norm_mean")]
    if not keys:
        diag.na("D11", "no grad_norm in the history (run predates E1.2)")
        return
    for key in keys:
        values = pd.to_numeric(series.steps[key], errors="coerce").to_numpy(dtype=float)
        diag.ran("D11")
        bad = int((~np.isfinite(values)).sum())
        if bad:
            diag.add(Finding("D11", "critical", metric=key[1], value=bad, threshold="finite",
                             source=source, evidence=f"{bad} non-finite gradient norms",
                             **_scope(series, key[0])))
            continue
        thirds = _thirds(pd.Series(values), t["min_points"])
        if thirds is None:
            continue
        first, last = float(np.median(thirds[0])), float(np.median(thirds[1]))
        if first > 0 and last > t["D11_growth"] * first:
            diag.add(Finding("D11", "high", metric=key[1], value=last / first,
                             threshold=f"> {t['D11_growth']} x first third", source=source,
                             evidence=f"median gradient norm {first:.4g} -> {last:.4g}",
                             **_scope(series, key[0])))


def check_log_series(diag, log_series):
    """Dataset/task-level fallback for runs that kept no per-client curves."""
    t = diag.thresholds
    source = "execution_log"
    for fold, record in log_series.items():
        loss = record.get("mean_val_loss")
        if loss is not None:
            fake = Series(fold=fold, client_id="server", dataset="*", tasks=(), num_classes=0,
                          criterion="", train={}, val={})
            values = loss.dropna()
            if len(values) >= t["min_points"]:
                diag.ran("D5")
                position = int(np.argmin(values.to_numpy()))
                best, final = float(values.iloc[position]), float(values.iloc[-1])
                if (position < t["D5_best_fraction"] * len(values) and best > 0
                        and final > t["D5_rise_factor"] * best):
                    diag.add(Finding("D5", "high", metric="server mean val loss",
                                     value=final / best, source=source,
                                     threshold=f"best before {t['D5_best_fraction']:.0%}",
                                     evidence=f"best round {values.index[position]} "
                                              f"({best:.4g}) -> final {final:.4g}",
                                     **_scope(fake, "combined")))
        for key, values in record.items():
            if key == "mean_val_loss" or "/" not in key:
                continue
            dataset, task = key.split("/", 1)
            values = values.dropna()
            if len(values) < t["min_points"]:
                continue
            scope = dict(fold=fold, dataset=dataset, client_id="*", task=task)
            diag.ran("D15")
            tail = max(1, int(math.ceil(t["D15_tail_fraction"] * len(values))))
            best = int(np.argmax(values.to_numpy()))
            if best >= len(values) - tail:
                diag.add(Finding("D15", "info", metric="logged val metric", source=source,
                                 value=float(values.index[best]),
                                 threshold=f"best in the last {t['D15_tail_fraction']:.0%}",
                                 evidence=f"weighted val metric best at round "
                                          f"{values.index[best]} of {len(values)}", **scope))
            if task == "seg":
                last = float(values.to_numpy()[-max(1, len(values) // 3):].mean())
                diag.ran("D6")
                if last < t["D6_dice"]:
                    diag.add(Finding("D6", "critical", metric="logged val dice", value=last,
                                     threshold=f"< {t['D6_dice']}", source=source,
                                     evidence="dataset-level weighted validation Dice", **scope))


# ---- checks over results ------------------------------------------------------------------

def check_heterogeneity(diag, results, failed):
    t = diag.thresholds
    if results is None or results.empty:
        diag.na("D8", "no test results")
        return
    for (dataset, task), group in results.groupby(["dataset", "task"]):
        column = TEST_METRIC.get(task)
        if column not in group or group["client_id"].nunique() < 2:
            continue
        diag.ran("D8")
        per_client = group.groupby("client_id")[column].mean().dropna()
        if per_client.empty:
            continue
        spread = float(per_client.max() - per_client.min())
        failing = sorted(c for c in per_client.index if (dataset, task, c) in failed)
        if spread > t["D8_spread"] and failing:
            diag.add(Finding("D8", "high", metric=f"test {column}", value=spread,
                             threshold=f"> {t['D8_spread']} with a D6/D7 client",
                             dataset=dataset, task=task, client_id=",".join(failing),
                             source="test_results",
                             evidence="per-client " + ", ".join(
                                 f"{c}={v:.3f}" for c, v in per_client.items())))


def check_empty_false_positives(diag, results):
    t = diag.thresholds
    if results is None or "empty_fp_image_rate" not in results:
        diag.na("D14", "no empty_fp_image_rate in test results")
        return
    seg = results[(results.task == "seg") & results["n_empty"].notna()]
    for dataset, group in seg.groupby("dataset"):
        n_empty = float(group["n_empty"].sum())
        if n_empty < t["D14_min_empty"]:
            continue
        diag.ran("D14")
        rate = float(np.nansum(group["empty_fp_image_rate"] * group["n_empty"]) / n_empty)
        if rate >= t["D14_rate"]:
            diag.add(Finding("D14", "high", metric="test empty_fp_image_rate", value=rate,
                             threshold=f">= {t['D14_rate']}", dataset=dataset, task="seg",
                             client_id="*", source="test_results",
                             evidence=f"{rate:.0%} of {int(n_empty)} empty test images get a "
                                      "predicted lesion pixel"))


def check_prediction_collapse(diag, predictions):
    t = diag.thresholds
    if predictions is None or predictions.empty:
        diag.na("D13", "no classification predictions")
        return
    for dataset, group in predictions.groupby("dataset"):
        truth = pd.to_numeric(group["ground_truth"], errors="coerce")
        predicted = pd.to_numeric(group["predicted"], errors="coerce")
        valid = truth.notna() & predicted.notna()
        truth, predicted = truth[valid].astype(int), predicted[valid].astype(int)
        if len(truth) < t["D13_min_samples"]:
            continue
        diag.ran("D13")
        names = {int(k): str(group[f"class_name_{k}"].dropna().iloc[0])
                 for k in range(16) if f"class_name_{k}" in group
                 and group[f"class_name_{k}"].notna().any()}
        classes = sorted(set(truth) | set(predicted) | set(names))
        predicted_fraction = {c: float((predicted == c).mean()) for c in classes}
        prevalence = {c: float((truth == c).mean()) for c in classes}
        top = max(classes, key=lambda c: predicted_fraction[c])
        excess = predicted_fraction[top] - prevalence[top]
        recall = {c: float((predicted[truth == c] == c).mean()) if (truth == c).any() else np.nan
                  for c in classes}
        label = names.get(top, str(top))
        evidence = (f"'{label}' predicted for {predicted_fraction[top]:.0%} of {len(truth)} "
                    f"images vs prevalence {prevalence[top]:.0%}; recall " + ", ".join(
                        f"{names.get(c, c)}={recall[c]:.2f}" for c in classes))
        if predicted_fraction[top] >= t["D13_constant"] and prevalence[top] < t["D13_constant"]:
            severity = "critical"
        elif excess >= t["D13_excess"]:
            severity = "high"
        else:
            continue
        diag.add(Finding("D13", severity, metric=f"predicted fraction of {label}", value=excess,
                         threshold=f"excess >= {t['D13_excess']} (constant >= "
                                   f"{t['D13_constant']})",
                         dataset=dataset, task="cls", client_id="*", source="cls_predictions",
                         evidence=evidence))


def check_exposure(diag, config, clients_meta):
    t = diag.thresholds
    rounds = _float((config.get("federated") or {}).get("rounds"))
    epochs = {}
    for meta in clients_meta:
        effective = _float(meta.get("effective_train_examples"))
        if not effective or not np.isfinite(rounds):
            continue
        if str(meta.get("local_training_mode")) == "steps":
            try:
                batch = float(dataset_config(config, meta.get("dataset"))["batch_size"])
            except (KeyError, TypeError):
                continue
            epochs[meta["client_id"]] = rounds * _float(meta.get("steps_per_round")) * batch / effective
        else:
            epochs[meta["client_id"]] = rounds * _float(meta.get("local_epochs"))
    epochs = {k: v for k, v in epochs.items() if np.isfinite(v) and v > 0}
    if len(epochs) < 2:
        diag.na("D10", "no client metadata with a training budget")
        return
    diag.ran("D10")
    low, high = min(epochs, key=epochs.get), max(epochs, key=epochs.get)
    ratio = epochs[high] / epochs[low]
    if ratio > t["D10_ratio"]:
        diag.add(Finding("D10", "info", metric="effective epochs max/min", value=ratio,
                         threshold=f"> {t['D10_ratio']}", client_id=f"{high} / {low}",
                         source="metadata.yaml",
                         evidence=f"{high} sees {epochs[high]:.1f} effective epochs, {low} "
                                  f"{epochs[low]:.1f}"))


def check_conflict(diag, run_dir, histories):
    t = diag.thresholds
    summary = _read_csv(Path(run_dir) / "conflict_summary.csv")
    if summary is not None and not summary.empty:
        _conflict_from_summary(diag, summary)
        return
    negative = _read_csv(Path(run_dir) / "negative_transfer_summary.csv")
    if negative is not None and "overall_mean" in negative:
        row = negative[negative["fold"].astype(str) == "all_folds"]
        row = (row if not row.empty else negative).iloc[0]
        excess = _float(row.get("overall_mean")) - _float(row.get("orthogonal_reference_mean"))
        if np.isfinite(excess):
            diag.ran("D9")
            _cancellation_finding(diag, excess, "negative_transfer_summary.csv")
            return
    cosines = []
    for history in histories:
        for entry in history:
            cosines.extend(_intra_same_task_cosines(entry.get("gradient_conflict")))
    cosines = [c for c in cosines if np.isfinite(c)]
    if cosines:
        diag.ran("D9")
        mean = float(np.mean(cosines))
        if abs(mean) < t["D9_no_interaction"]:
            diag.add(Finding("D9", "info", metric="intra-dataset delta cosine", value=mean,
                             threshold=f"|cos| < {t['D9_no_interaction']}", client_id="*",
                             source="aggregation_history.gradient_conflict",
                             evidence="clients do not interact through the trunk (orthogonal "
                                      "updates); this is no interaction, not 'no conflict'"))
        return
    diag.na("D9", "no conflict_summary, negative_transfer or gradient_conflict")


def _intra_same_task_cosines(conflict):
    """Delta cosines between clients of the same dataset and tasks, from one round's record.

    ``clients`` are ``dataset/task/client_id`` labels indexing the per-block cosine matrices.
    """
    if not isinstance(conflict, dict):
        return []
    labels = [str(label).split("/") for label in conflict.get("clients", [])]
    matrices = conflict.get("cosine") or {}
    matrix = matrices.get("shared_total")
    if matrix is None:
        return []
    values = []
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            if a[:2] == labels[j][:2]:
                values.append(_float(matrix[i][j]))
    return values


def _cancellation_finding(diag, excess, source):
    t = diag.thresholds
    if excess > t["D9_cancellation_excess"]:
        diag.add(Finding("D9", "high", metric="cancellation excess", value=excess,
                         threshold=f"> {t['D9_cancellation_excess']}", source=source,
                         client_id="*", evidence="weighted client updates cancel more than "
                                                 "mutually orthogonal updates would"))
    elif abs(excess) <= t["D9_no_interaction"]:
        diag.add(Finding("D9", "info", metric="cancellation excess", value=excess,
                         threshold=f"|excess| <= {t['D9_no_interaction']}", source=source,
                         client_id="*",
                         evidence="cancellation equals the orthogonal reference: clients do not "
                                  "interact through the trunk (no interaction, not 'no conflict')"))


def _conflict_from_summary(diag, summary):
    t = diag.thresholds
    source = "conflict_summary.csv"
    diag.ran("D9")

    def rows(metric, scope):
        return summary[(summary.metric == metric) & (summary.scope == scope)]

    def stat(metric, scope, group):
        row = summary[(summary.metric == metric) & (summary.scope == scope)
                      & (summary.group == group)]
        if row.empty:
            return None
        row = row.iloc[0]
        return _float(row["mean"]), _float(row["std"]), max(_float(row["rounds"], 1.0), 1.0)

    same = stat("m1_cos", "pair_class", "intra_same_tasks")
    if same is not None and abs(same[0]) < t["D9_no_interaction"]:
        diag.add(Finding("D9", "info", metric="M1 cos intra_same_tasks", value=same[0],
                         threshold=f"|cos| < {t['D9_no_interaction']}", client_id="*",
                         source=source,
                         evidence="clients of the same dataset and tasks do not interact through "
                                  "the trunk: no interaction, not 'no conflict'"))
    for scope in ("pair_class", "dataset_pair"):
        for row in rows("m1_cos", scope).itertuples():
            if _float(row.mean) < t["D9_first_order_conflict"]:
                diag.add(Finding("D9", "high", metric=f"M1 cos {row.group}", value=row.mean,
                                 threshold=f"< {t['D9_first_order_conflict']}", client_id="*",
                                 source=source, dataset=str(row.group),
                                 evidence=f"first-order gradient conflict over {row.rounds:g} "
                                          "sampled rounds"))

    def consistently_negative(value):
        mean, std, n = value
        return mean + t["D9_consistency_k"] * std / math.sqrt(n) < 0

    for row in rows("m2_z_train", "dataset_pair").itertuples():
        source_ds, _, target_ds = str(row.group).partition(" -> ")
        if source_ds == target_ds:
            continue
        train = stat("m2_z_train", "dataset_pair", row.group)
        val = stat("m2_z_val", "dataset_pair", row.group)
        if train and val and consistently_negative(train) and consistently_negative(val):
            diag.add(Finding("D9", "info", metric=f"M2 Z {row.group}", value=train[0],
                             threshold="Z_train < 0 and Z_val < 0, consistent sign",
                             client_id="*", dataset=str(row.group), source=source,
                             evidence=f"interference: Z_train {train[0]:.2e}, Z_val {val[0]:.2e}"
                                      f". {D9_FULL_DELTA_CAVEAT}"))
    own_train = stat("m2_z_train", "pair_class", "self")
    own_val = stat("m2_z_val", "pair_class", "self")
    if own_train and own_val and own_train[0] >= 0 > own_val[0]:
        diag.add(Finding("D9", "high", metric="M2 Z self", value=own_val[0],
                         threshold="Z_train >= 0 and Z_val < 0", client_id="*", source=source,
                         evidence=f"overfitting, not interference: a client's own step helps its "
                                  f"training loss (Z {own_train[0]:.2e}) and hurts its validation "
                                  f"loss (Z {own_val[0]:.2e})"))
    aggregate = stat("m2_z_train", "pair_class", "aggregate")
    if aggregate and aggregate[0] < 0:
        diag.add(Finding("D9", "high", metric="M2 Z aggregate", value=aggregate[0],
                         threshold="Z_train >= 0", client_id="*", source=source,
                         evidence="the update the server applies raises the clients' training "
                                  "loss on average"))


def latest_baselines(baseline_root):
    """``{(mode, dataset): run_dir}`` for the most recent baseline of every mode and dataset."""
    root = Path(baseline_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    latest = {}
    for mode in BASELINE_MODES:
        for run_dir in sorted((root / mode).glob("*")):
            match = _TIMESTAMPED.match(run_dir.name)
            if match and (run_dir / "baseline_test_results.csv").exists():
                latest[(mode, match.group("dataset"))] = run_dir   # sorted: the last one wins
    return latest


def check_baseline_gap(diag, kind, config, results, baseline_root, noise):
    if kind == "baseline":
        diag.na("D12", "the run is itself a baseline")
        return
    if results is None or results.empty:
        diag.na("D12", "no test results")
        return
    baselines = latest_baselines(baseline_root)
    if not baselines:
        diag.na("D12", f"no baseline runs under {baseline_root}")
        return
    master = _master_path(config)
    compared = False
    for (dataset, task), group in results.groupby(["dataset", "task"]):
        metrics = ("balanced_acc", "macro_f1") if task == "cls" else ("dice_positive",)
        references = {}
        for (mode, base_dataset), run_dir in baselines.items():
            if base_dataset != dataset or task not in BASELINE_MODES[mode]:
                continue
            base_config = _read_yaml(run_dir / "config.yaml")
            if master is None or _master_path(base_config) != master:
                continue
            base = _read_csv(run_dir / "baseline_test_results.csv")
            if base is not None:
                references[mode] = base[base.task == task]
        if not references:
            continue
        compared = True
        diag.ran("D12")
        margin = _noise(dataset, noise)
        for metric in metrics:
            if metric not in group:
                continue
            value = float(group[metric].mean())
            reference = {mode: float(frame[metric].mean()) for mode, frame in references.items()
                         if metric in frame and frame[metric].notna().any()}
            if not reference or not np.isfinite(value):
                continue
            best_mode = max(reference, key=reference.get)
            gap = value - reference[best_mode]
            if gap < -margin:
                diag.add(Finding(
                    "D12", "high" if kind == "federated" else "info", metric=f"test {metric}",
                    value=gap, threshold=f"< -{margin} (run-to-run noise)", dataset=dataset,
                    task=task, client_id="*", source="baseline_test_results",
                    evidence=f"{kind} {value:.3f} vs baseline " + ", ".join(
                        f"{mode} {score:.3f}" for mode, score in reference.items())))
    if not compared:
        diag.na("D12", "no baseline trained on the same frozen master (different test slices)")


def check_baseline_ceiling(diag, config, results, histories):
    t = diag.thresholds
    epochs = _float((config.get("training") or {}).get("epochs"))
    if results is None or "best_epoch" not in results or not np.isfinite(epochs):
        diag.na("D15", "no best_epoch / configured epochs")
        return
    last_epoch = max((float(s.val[next(iter(s.val))].index.max()) for s in histories if s.val),
                     default=float("nan"))
    best = float(results["best_epoch"].max())
    diag.ran("D15")
    if last_epoch >= epochs and best >= t["D15_baseline_fraction"] * epochs:
        experiment = config.get("experiment", {})
        diag.add(Finding("D15", "high", metric="best_epoch", value=best,
                         threshold=f">= {t['D15_baseline_fraction']} x {epochs:g} epochs, "
                                   "no early stop",
                         dataset=str(experiment.get("baseline_dataset", "")),
                         task="+".join(BASELINE_MODES.get(str(experiment.get("baseline_mode")),
                                                          ())),
                         client_id="pool", source="baseline_test_results",
                         evidence=f"best epoch {best:g} of {epochs:g} and the run used the whole "
                                  "budget: the baseline may be underestimated"))


# ---- orchestration ------------------------------------------------------------------------

def _load_aggregation_histories(run_dir):
    histories = []
    for fold_dir in _fold_dirs(run_dir):
        path = fold_dir / "aggregation_history.json"
        if path.exists():
            try:
                histories.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return histories


def diagnose(run_dir, labels=None, thresholds=None, noise=None, baseline_root="runs/baseline"):
    """Run every check on one run directory; return ``(findings_frame, summary_dict)``."""
    run_dir = Path(run_dir)
    labels = dict(labels or {})
    thresholds = {**THRESHOLDS, **(thresholds or {})}
    noise = {**NOISE, **(noise or {})}
    diag = Diagnosis(thresholds)
    kind = run_kind(run_dir)
    config = _read_yaml(run_dir / "config.yaml")
    results = _read_csv(run_dir / f"{kind}_test_results.csv")
    predictions = _read_csv(run_dir / f"{kind}_cls_predictions.csv")

    if kind == "baseline":
        series = read_baseline_series(run_dir, config)
        clients_meta = []
        source = "history.csv"
    else:
        series, clients_meta = [], []
        for fold_dir in _fold_dirs(run_dir):
            fold = fold_dir.name.split("_", 1)[1]
            for client_dir in sorted(fold_dir.glob("client_*")):
                meta = _read_yaml(client_dir / "metadata.yaml")
                if meta:
                    clients_meta.append(meta)
                client = read_client_series(client_dir, config, fold)
                if client is not None:
                    series.append(client)
        source = "training_history.csv"

    for client in series:
        check_non_finite(diag, client, source)
        check_training_curves(diag, client, source, baseline=kind == "baseline")
    if not series:
        reason = "no per-client training history (run predates training telemetry)"
        for check in ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D11", "D15"):
            diag.na(check, reason)
        if kind != "baseline":
            check_log_series(diag, read_log_series(run_dir))

    failed = {(f.dataset, f.task, f.client_id) for f in diag.findings
              if f.check_id in ("D6", "D7")}
    check_heterogeneity(diag, results, failed)
    check_empty_false_positives(diag, results)
    check_prediction_collapse(diag, predictions)
    check_baseline_gap(diag, kind, config, results, baseline_root, noise)
    if kind == "baseline":
        check_baseline_ceiling(diag, config, results, series)
        for check, reason in (("D9", "no federation"), ("D10", "single pooled training")):
            diag.na(check, reason)
    elif kind == "standalone":
        diag.na("D9", "standalone clients share no trunk")
        check_exposure(diag, config, clients_meta)
    else:
        check_conflict(diag, run_dir, _load_aggregation_histories(run_dir))
        check_exposure(diag, config, clients_meta)

    frame = pd.DataFrame([{**asdict(f), "run": str(run_dir), "kind": kind,
                           "arm_id": labels.get("arm_id", ""), "seed": labels.get("seed", "")}
                          for f in diag.findings], columns=list(FINDING_FIELDS))
    order = {severity: index for index, severity in enumerate(SEVERITIES)}
    frame = frame.sort_values(["severity", "check_id"], key=lambda s: s.map(
        lambda v: order.get(v, 9) if s.name == "severity" else int(str(v)[1:])), kind="stable")
    counts = {severity: int((frame.severity == severity).sum()) for severity in SEVERITIES}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": str(run_dir),
        "kind": kind,
        **labels,
        "status": "critical" if counts["critical"] else "warning" if counts["high"] else "ok",
        "counts": counts,
        "checks": {check: {"name": CHECKS[check], **state} for check, state in diag.state.items()},
        "thresholds": thresholds,
        "noise": noise,
    }
    return frame.reset_index(drop=True), summary


def _markdown(frame, summary):
    lines = [f"# Diagnóstico de treino — `{Path(summary['run']).name}`", "",
             f"- tipo: `{summary['kind']}`; status: **{summary['status']}**; "
             + ", ".join(f"{k}={v}" for k, v in summary["counts"].items()),
             "- Sinalizações descritivas de um run (uma seed); não são testes de significância.",
             "", "## Checagens", "", "| id | checagem | estado | motivo |", "|---|---|---|---|"]
    for check, state in summary["checks"].items():
        lines.append(f"| {check} | {state['name']} | {state['state']} | {state['reason']} |")
    lines += ["", "## Achados", ""]
    if frame.empty:
        lines.append("Nenhum.")
    else:
        lines += ["| sev. | id | dataset | cliente | tarefa | métrica | valor | evidência |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in frame.itertuples():
            lines.append(f"| {row.severity} | {row.check_id} | {row.dataset} | {row.client_id} | "
                         f"{row.task} | {row.metric} | {_fmt(row.value)} | "
                         f"{str(row.evidence).replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir, frame, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "findings.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    (out_dir / "summary.md").write_text(_markdown(frame, summary), encoding="utf-8")
    return out_dir


def consolidated_table(summaries, frames):
    rows = []
    for summary, frame in zip(summaries, frames):
        triggered = sorted(set(frame.check_id), key=lambda c: int(c[1:]))
        rows.append({"run": summary["run"], "arm_id": summary.get("arm_id", ""),
                     "seed": summary.get("seed", ""), "kind": summary["kind"],
                     "status": summary["status"], **summary["counts"],
                     "checks": " ".join(triggered)})
    return pd.DataFrame(rows)


def diagnose_run(run_path, **kwargs):
    """Diagnose one run and write ``<run>/diagnostics``; the hook called at the end of training."""
    frame, summary = diagnose(run_path, **kwargs)
    out = write_outputs(Path(run_path) / OUTPUT_DIRNAME, frame, summary)
    logging.info("[diagnostics] status=%s critical=%s high=%s info=%s -> %s", summary["status"],
                 summary["counts"]["critical"], summary["counts"]["high"],
                 summary["counts"]["info"], out)
    return summary


def run_training_diagnostics(config, run_path):
    """End-of-run hook: diagnose ``run_path`` unless disabled; never fails the run."""
    from src.utils.training_runtime import runtime_config

    try:
        settings = runtime_config(config)["diagnostics"]["training"]
        if not settings["enabled"]:
            logging.info("[diagnostics] disabled (runtime.diagnostics.training.enabled)")
            return None
        return diagnose_run(run_path, baseline_root=settings["baseline_root"])
    except Exception:
        logging.warning("Could not build the training diagnostics", exc_info=True)
        return None


def diagnose_paths(paths, thresholds=None, noise=None, baseline_root="runs/baseline"):
    """Diagnose every run under ``paths``; write per-run and consolidated outputs."""
    table = []
    for path in paths:
        path = Path(path)
        runs = discover_runs(path)
        summaries, frames = [], []
        for run_dir, labels in runs:
            frame, summary = diagnose(run_dir, labels, thresholds, noise, baseline_root)
            write_outputs(run_dir / OUTPUT_DIRNAME, frame, summary)
            summaries.append(summary)
            frames.append(frame)
        if len(runs) > 1 or (runs and runs[0][0] != path):
            out = path / OUTPUT_DIRNAME
            out.mkdir(parents=True, exist_ok=True)
            all_findings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=list(FINDING_FIELDS))
            all_findings.to_csv(out / "findings.csv", index=False)
            consolidated = consolidated_table(summaries, frames)
            consolidated.to_csv(out / "runs.csv", index=False)
            (out / "summary.json").write_text(json.dumps(
                {"schema_version": SCHEMA_VERSION, "root": str(path), "runs": summaries},
                indent=2, default=str), encoding="utf-8")
            (out / "summary.md").write_text(
                f"# Diagnóstico consolidado — `{path.name}`\n\n"
                + _table_markdown(consolidated), encoding="utf-8")
        table.append(consolidated_table(summaries, frames))
    return pd.concat(table, ignore_index=True) if table else pd.DataFrame()


def _table_markdown(table):
    if table.empty:
        return "Nenhum run encontrado.\n"
    columns = ["run", "arm_id", "kind", "status", "critical", "high", "info", "checks"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in table.to_dict("records"):
        row["run"] = Path(row["run"]).name
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def _parse_overrides(items, defaults, flag):
    parsed = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        if defaults is not None and key not in defaults:
            raise SystemExit(f"unknown {flag} key {key!r}; known: {', '.join(defaults)}")
        parsed[key] = float(value)
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="run, study or folder of runs")
    parser.add_argument("--threshold", action="append", metavar="KEY=VALUE",
                        help=f"override a threshold; keys: {', '.join(THRESHOLDS)}")
    parser.add_argument("--noise", action="append", metavar="DATASET=VALUE",
                        help="run-to-run noise margin for D12 (default BUSI 0.05, others 0.01)")
    parser.add_argument("--baseline-root", default="runs/baseline")
    args = parser.parse_args(argv)
    table = diagnose_paths(
        args.paths,
        thresholds=_parse_overrides(args.threshold, THRESHOLDS, "--threshold"),
        noise=_parse_overrides(args.noise, None, "--noise"),
        baseline_root=args.baseline_root,
    )
    print(_table_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
