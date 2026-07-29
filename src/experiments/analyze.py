"""Cross-setup analysis for the comparison experiment.

The input result files are produced by the shared evaluator for federated, standalone and
centralized runs.  Every aggregation is isolated by ``dataset`` and ``task`` so heterogeneous
label spaces (for example, three and seven classes) are never mixed.  Historical single-dataset
CSVs without a ``dataset`` column remain readable and are assigned the explicit ``legacy`` key.

Outputs:
  - summary_per_task_setup.csv: mean/std per dataset x task x setup;
  - federated_vs_local_wilcoxon.csv: exploratory paired Wilcoxon and deltas within each
    dataset/task (client x CV-fold pairs are not independent inferential replicates);
  - per_client_deltas.csv: primary-metric delta for every matched dataset/fold/client/task;
  - pooled_auc.csv: pooled OvR macro AUC within each dataset/setup/fold and across folds;
  - report.html: charts split by dataset/task plus the summary tables.

Run with ``python -m src.experiments.analyze`` or override paths with ``--results``, ``--preds``
and ``--out``.
"""

import argparse
import base64
import io
import json
import logging
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend selection)

# --- Edit these to point at each setup's run directory before running (CLI flags override them) ---
FED = "runs/20260622_192540_FEDERATED_MTnnUNet_2seg_2cls"
STD = "runs/20260622_195528_STANDALONE_MTnnUNet_2seg_2cls"
CEN = "runs/20260622_190616_CENTRALIZED_MTnnUNet"

RESULTS = [
    f"{FED}/federated_test_results.csv",
    f"{STD}/standalone_test_results.csv",
    f"{CEN}/centralized_test_results.csv",
]
PREDS = [
    f"{FED}/federated_cls_predictions.csv",
    f"{STD}/standalone_cls_predictions.csv",
    f"{CEN}/centralized_cls_predictions.csv",
]
OUT = "runs/comparison"
# --------------------------------------------------------------------------------------------------

SEG_METRICS = ["dice", "iou", "sensitivity", "specificity", "precision"]
CLS_BASE_METRICS = ["acc", "macro_f1", "balanced_acc", "auc"]
# Preserve the public legacy constant; runtime discovery extends it to any number of classes.
CLS_METRICS = CLS_BASE_METRICS + [
    metric
    for class_index in range(3)
    for metric in (f"precision_class_{class_index}", f"recall_class_{class_index}")
]
PRIMARY = {"seg": "dice", "cls": "acc"}
LEGACY_DATASET = "legacy"
LEGACY_STUDY = "legacy"
LEGACY_METHOD = "legacy"
LEGACY_SEED = "legacy"

_CLASS_METRIC_RE = re.compile(r"^(precision|recall)_class_(\d+)(?:_(.+))?$")
_PROB_RE = re.compile(r"^prob_(\d+)$")


def _with_dataset(df):
    """Return a copy with a non-null dataset key, including for historical CSVs."""
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = LEGACY_DATASET
    else:
        out["dataset"] = out["dataset"].fillna(LEGACY_DATASET).astype(str)
    return out


def _with_study_metadata(df):
    """Canonicalize study identifiers while keeping historical result CSVs readable."""
    out = _with_dataset(df)
    if "study_id" not in out:
        out["study_id"] = LEGACY_STUDY
    else:
        out["study_id"] = out["study_id"].fillna(LEGACY_STUDY).astype(str)
    if "method_id" not in out:
        out["method_id"] = (
            out["setup"].astype(str) if "setup" in out else LEGACY_METHOD
        )
    else:
        out["method_id"] = out["method_id"].fillna(LEGACY_METHOD).astype(str)
    if "arm_id" not in out:
        out["arm_id"] = out["method_id"]
    else:
        out["arm_id"] = out["arm_id"].fillna(out["method_id"]).astype(str)
    if "seed" not in out:
        out["seed"] = LEGACY_SEED
    else:
        out["seed"] = out["seed"].fillna(LEGACY_SEED).astype(str)
    return out


def _class_metric_sort_key(metric):
    match = _CLASS_METRIC_RE.match(str(metric))
    if match is None:
        base_index = (
            CLS_BASE_METRICS.index(metric)
            if metric in CLS_BASE_METRICS
            else len(CLS_BASE_METRICS)
        )
        return (0, base_index, 0)
    return (1, int(match.group(2)), 0 if match.group(1) == "precision" else 1)


def _metrics_for(task, columns=None):
    """Return task metrics, discovering all per-class columns when a schema is supplied."""
    if task == "seg":
        return SEG_METRICS
    if columns is None:
        return CLS_METRICS
    metrics = [metric for metric in CLS_BASE_METRICS if metric in columns]
    metrics.extend(str(column) for column in columns if _CLASS_METRIC_RE.match(str(column)))
    return sorted(set(metrics), key=_class_metric_sort_key)


def _add_class_metadata(row, metric, group):
    """Associate a long-format per-class metric with its optional human-readable name."""
    match = _CLASS_METRIC_RE.match(str(metric))
    if match is None:
        return
    class_index = int(match.group(2))
    row["class_index"] = class_index
    name_col = f"class_name_{class_index}"
    if name_col in group:
        names = group[name_col].dropna().astype(str).unique()
        if len(names) == 1:
            row["class_name"] = names[0]
            return
    # Tolerate third-party self-describing names such as precision_class_2_malignant.
    if match.group(3):
        row["class_name"] = match.group(3)


def summary_table(df):
    df = _with_study_metadata(df)
    rows = []
    group_keys = ["study_id", "dataset", "method_id", "setup", "task"]
    for keys, group in df.groupby(group_keys):
        study_id, dataset, method_id, setup, task = keys
        for metric in _metrics_for(task, group.columns):
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            # Concatenating 3- and 7-class result frames creates all-NaN class 3..6 columns for
            # the 3-class dataset.  Those columns belong to the other label space, not this group.
            if not values.notna().any():
                continue
            row = {
                "study_id": study_id,
                "dataset": dataset,
                "method_id": method_id,
                "setup": setup,
                "task": task,
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
                "n": int(values.notna().sum()),
                "n_seeds": int(group.loc[values.notna(), "seed"].nunique()),
            }
            _add_class_metadata(row, metric, group)
            rows.append(row)
    return pd.DataFrame(rows)


def summary_by_seed(df):
    """Mean/std across client-fold observations without hiding between-seed variation."""
    df = _with_study_metadata(df)
    rows = []
    keys = ["study_id", "seed", "dataset", "method_id", "setup", "task"]
    for values_key, group in df.groupby(keys):
        study_id, seed, dataset, method_id, setup, task = values_key
        for metric in _metrics_for(task, group.columns):
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            if not values.notna().any():
                continue
            rows.append({
                "study_id": study_id,
                "seed": seed,
                "dataset": dataset,
                "method_id": method_id,
                "setup": setup,
                "task": task,
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
                "n": int(values.notna().sum()),
            })
    return pd.DataFrame(rows)


def paired_deltas(df, a="federated", b="standalone"):
    """Exploratory paired deltas across matching dataset/fold/client/task observations.

    The synthetic clients and overlapping CV training folds are not independent replicates.
    Wilcoxon values are retained as diagnostics for compatibility, but must not be interpreted as
    confirmatory significance tests. Multiple independent seeds or sample-level OOF inference are
    required for that claim.
    """
    df = _with_dataset(df)
    key = ["dataset", "fold", "client_id", "task"]
    left_setup, right_setup = df[df.setup == a], df[df.setup == b]
    rows = []
    for (dataset, task), group in df.groupby(["dataset", "task"]):
        left = left_setup[(left_setup.dataset == dataset) & (left_setup.task == task)]
        right = right_setup[(right_setup.dataset == dataset) & (right_setup.task == task)]
        for metric in _metrics_for(task, group.columns):
            if metric not in left or metric not in right:
                continue
            paired = left[key + [metric]].merge(
                right[key + [metric]], on=key, suffixes=("_a", "_b")
            )
            va = pd.to_numeric(paired[f"{metric}_a"], errors="coerce").to_numpy()
            vb = pd.to_numeric(paired[f"{metric}_b"], errors="coerce").to_numpy()
            keep = ~(pd.isna(va) | pd.isna(vb))
            va, vb = va[keep].astype(float), vb[keep].astype(float)
            if len(va) == 0:
                continue
            diff = va - vb
            if np.all(diff == 0):
                stat, p_value = np.nan, np.nan
            else:
                try:
                    stat, p_value = wilcoxon(va, vb)
                except ValueError:
                    stat, p_value = np.nan, np.nan
            row = {
                "dataset": dataset,
                "task": task,
                "metric": metric,
                "n_pairs": len(va),
                f"{a}_mean": va.mean(),
                f"{b}_mean": vb.mean(),
                "mean_delta": diff.mean(),
                "median_delta": float(np.median(diff)),
                "wilcoxon_stat": stat,
                "wilcoxon_p": p_value,
                "inference_scope": "exploratory_only_non_independent_client_fold_pairs",
            }
            _add_class_metadata(row, metric, group)
            rows.append(row)
    return pd.DataFrame(rows)


def per_client_deltas(df, a="federated", b="standalone"):
    """Return setup ``a`` minus setup ``b`` for each task's primary metric."""
    df = _with_dataset(df)
    key = ["dataset", "fold", "client_id", "task"]
    left, right = df[df.setup == a], df[df.setup == b]
    rows = []
    for task, metric in PRIMARY.items():
        task_left, task_right = left[left.task == task], right[right.task == task]
        if metric not in task_left or metric not in task_right:
            continue
        paired = task_left[key + [metric]].merge(
            task_right[key + [metric]], on=key, suffixes=("_a", "_b")
        )
        for record in paired.itertuples(index=False):
            va = float(getattr(record, f"{metric}_a"))
            vb = float(getattr(record, f"{metric}_b"))
            rows.append({
                "dataset": record.dataset,
                "fold": record.fold,
                "client_id": record.client_id,
                "task": task,
                "metric": metric,
                a: va,
                b: vb,
                "delta": va - vb,
            })
    return pd.DataFrame(rows)


def _load_comparisons(manifest):
    if manifest is None:
        return []
    if isinstance(manifest, (str, Path)):
        with Path(manifest).open(encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream)
    return list((manifest or {}).get("comparisons", []))


def method_comparisons(df, comparisons):
    """Evaluate manifest-declared paired contrasts without pooling datasets or seeds."""
    df = _with_study_metadata(df)
    pair_rows, observation_rows = [], []
    key = ["study_id", "seed", "dataset", "fold", "client_id", "task"]
    for comparison in comparisons:
        left_spec, right_spec = comparison["left"], comparison["right"]
        left = df[
            (df.method_id == str(left_spec["method_id"]))
            & (df.setup == left_spec["setup"])
        ]
        right = df[
            (df.method_id == str(right_spec["method_id"]))
            & (df.setup == right_spec["setup"])
        ]
        for (dataset, task), group in df.groupby(["dataset", "task"]):
            metrics = _metrics_for(task, group.columns)
            task_left = left[(left.dataset == dataset) & (left.task == task)]
            task_right = right[(right.dataset == dataset) & (right.task == task)]
            for metric in metrics:
                if metric not in task_left or metric not in task_right:
                    continue
                paired = task_left[key + [metric]].merge(
                    task_right[key + [metric]], on=key, suffixes=("_left", "_right")
                )
                left_values = pd.to_numeric(paired[f"{metric}_left"], errors="coerce")
                right_values = pd.to_numeric(paired[f"{metric}_right"], errors="coerce")
                valid = left_values.notna() & right_values.notna()
                paired = paired.loc[valid].copy()
                if paired.empty:
                    continue
                left_array = left_values.loc[valid].to_numpy(float)
                right_array = right_values.loc[valid].to_numpy(float)
                differences = left_array - right_array
                if np.all(differences == 0):
                    statistic, p_value = np.nan, np.nan
                else:
                    try:
                        statistic, p_value = wilcoxon(left_array, right_array)
                    except ValueError:
                        statistic, p_value = np.nan, np.nan
                pair_rows.append({
                    "comparison_id": comparison["comparison_id"],
                    "interpretation": comparison.get("interpretation", ""),
                    "dataset": dataset,
                    "task": task,
                    "metric": metric,
                    "left_method": left_spec["method_id"],
                    "left_setup": left_spec["setup"],
                    "right_method": right_spec["method_id"],
                    "right_setup": right_spec["setup"],
                    "n_pairs": len(differences),
                    "n_seeds": int(paired["seed"].nunique()),
                    "left_mean": left_array.mean(),
                    "right_mean": right_array.mean(),
                    "mean_delta": differences.mean(),
                    "median_delta": float(np.median(differences)),
                    "wilcoxon_stat": statistic,
                    "wilcoxon_p": p_value,
                    "inference_scope": "exploratory_only_non_independent_client_fold_pairs",
                })
                if metric == PRIMARY.get(task):
                    for record, left_value, right_value, delta in zip(
                        paired.itertuples(index=False), left_array, right_array, differences
                    ):
                        observation_rows.append({
                            "comparison_id": comparison["comparison_id"],
                            "study_id": record.study_id,
                            "seed": record.seed,
                            "dataset": record.dataset,
                            "fold": record.fold,
                            "client_id": record.client_id,
                            "task": record.task,
                            "metric": metric,
                            "left_method": left_spec["method_id"],
                            "right_method": right_spec["method_id"],
                            "left": left_value,
                            "right": right_value,
                            "delta": delta,
                        })
    return pd.DataFrame(pair_rows), pd.DataFrame(observation_rows)


def _probability_columns(dataset_predictions):
    """Find the complete contiguous prob_0..prob_K-1 space for one dataset."""
    numbered = {
        int(match.group(1)): column
        for column in dataset_predictions.columns
        if (match := _PROB_RE.match(str(column)))
    }
    columns = []
    for index in range(len(numbered)):
        column = numbered.get(index)
        if column is None:
            break
        # Columns from a larger label space exist as NaN after concatenating heterogeneous CSVs.
        if not pd.to_numeric(dataset_predictions[column], errors="coerce").notna().all():
            break
        columns.append(column)
    return columns


def _class_names_for(dataset_predictions, num_classes):
    names = []
    for index in range(num_classes):
        column = f"class_name_{index}"
        if column not in dataset_predictions:
            return None
        unique = dataset_predictions[column].dropna().astype(str).unique()
        if len(unique) != 1:
            return None
        names.append(unique[0])
    return "|".join(names)


def _pooled_group_auc(group, probability_columns):
    labels = list(range(len(probability_columns)))
    try:
        return roc_auc_score(
            group["ground_truth"],
            group[probability_columns].to_numpy(float),
            multi_class="ovr",
            average="macro",
            labels=labels,
        )
    except (TypeError, ValueError):
        return np.nan


def pooled_auc(pred_paths):
    if not pred_paths:
        return pd.DataFrame()
    predictions = pd.concat(
        [_with_study_metadata(pd.read_csv(path)) for path in pred_paths], ignore_index=True
    )
    rows = []
    for dataset, dataset_predictions in predictions.groupby("dataset"):
        probability_columns = _probability_columns(dataset_predictions)
        if len(probability_columns) < 2:
            logging.warning(f"Skipping pooled AUC for {dataset}: no complete probability space")
            continue
        num_classes = len(probability_columns)
        class_names = _class_names_for(dataset_predictions, num_classes)
        group_keys = ["study_id", "method_id", "setup", "seed", "fold"]
        for values_key, group in dataset_predictions.groupby(group_keys):
            study_id, method_id, setup, seed, fold = values_key
            row = {
                "study_id": study_id,
                "dataset": dataset,
                "method_id": method_id,
                "setup": setup,
                "seed": seed,
                "fold": fold,
                "num_classes": num_classes,
                "n": len(group),
                "auc_pooled": _pooled_group_auc(group, probability_columns),
            }
            if class_names is not None:
                row["class_names"] = class_names
            rows.append(row)
        overall_keys = ["study_id", "method_id", "setup", "seed"]
        for values_key, group in dataset_predictions.groupby(overall_keys):
            study_id, method_id, setup, seed = values_key
            row = {
                "study_id": study_id,
                "dataset": dataset,
                "method_id": method_id,
                "setup": setup,
                "seed": seed,
                "fold": "all",
                "num_classes": num_classes,
                "n": len(group),
                "auc_pooled": _pooled_group_auc(group, probability_columns),
            }
            if class_names is not None:
                row["class_names"] = class_names
            rows.append(row)
    return pd.DataFrame(rows)


def _fig_to_b64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _metric_boxplot(df, task, metrics, dataset=None):
    """Plot metric distributions across client-fold, optionally isolated to one dataset."""
    subset = df[df.task == task]
    if dataset is not None:
        subset = subset[subset.dataset == dataset]
    metrics = [metric for metric in metrics if metric in subset.columns]
    subset = subset.copy()
    subset["series"] = subset["method_id"].astype(str) + " (" + subset["setup"] + ")"
    series = sorted(subset.series.unique())
    width = 0.8 / max(len(series), 1)
    colors = plt.cm.Set2.colors
    fig, axis = plt.subplots(figsize=(8, 4))
    for setup_index, label in enumerate(series):
        setup_rows = subset[subset.series == label]
        for metric_index, metric in enumerate(metrics):
            values = pd.to_numeric(setup_rows[metric], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                continue
            position = metric_index + setup_index * width - 0.4 + width / 2
            boxes = axis.boxplot(
                values,
                positions=[position],
                widths=width * 0.9,
                patch_artist=True,
                manage_ticks=False,
            )
            boxes["boxes"][0].set(
                facecolor=colors[setup_index % len(colors)], alpha=0.85
            )
        axis.plot(
            [], [], color=colors[setup_index % len(colors)], lw=6, label=label
        )
    axis.set_xticks(range(len(metrics)))
    axis.set_xticklabels(metrics, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    prefix = f"{dataset} / " if dataset is not None else ""
    axis.set_title(f"{prefix}{task} - distribution across client-fold")
    if series:
        axis.legend(fontsize=8)
    return _fig_to_b64(fig)


def _delta_bars(per_client, task, dataset=None):
    """Plot federated minus local for every matching client-fold."""
    subset = per_client[per_client.task == task].copy()
    if dataset is not None:
        subset = subset[subset.dataset == dataset]
    subset["label"] = subset["client_id"].astype(str) + " f" + subset["fold"].astype(str)
    colors = ["#2a9d8f" if delta >= 0 else "#e76f51" for delta in subset["delta"]]
    fig, axis = plt.subplots(figsize=(max(6, len(subset) * 0.5), 4))
    axis.bar(subset["label"], subset["delta"], color=colors)
    axis.axhline(0, color="black", lw=0.8)
    axis.tick_params(axis="x", rotation=60)
    prefix = f"{dataset} / " if dataset is not None else ""
    comparison = subset["comparison_id"].iloc[0] if not subset.empty else "comparison"
    axis.set_title(f"{prefix}{task} - {comparison} ({PRIMARY[task]}) per client-fold")
    return _fig_to_b64(fig)


def _auc_bars(pooled, dataset=None):
    if dataset is not None:
        pooled = pooled[pooled.dataset == dataset]
    overall = pooled[pooled.fold == "all"]
    fig, axis = plt.subplots(figsize=(5, 4))
    labels = overall["method_id"].astype(str) + " (" + overall["setup"] + ")"
    axis.bar(labels, overall["auc_pooled"], color="#264653")
    axis.tick_params(axis="x", rotation=55)
    axis.set_ylim(0, 1)
    prefix = f"{dataset} - " if dataset is not None else ""
    axis.set_title(f"{prefix}pooled OvR-macro AUC (all folds)")
    return _fig_to_b64(fig)


def build_html(df, summary, deltas, per_client, pooled, out):
    df = _with_study_metadata(df)
    charts = []
    dataset_tasks = (
        df[["dataset", "task"]].drop_duplicates().sort_values(["dataset", "task"])
    )
    for dataset, task in dataset_tasks.itertuples(index=False, name=None):
        overview_metrics = SEG_METRICS if task == "seg" else CLS_BASE_METRICS
        charts.append(
            (
                f"{dataset} / {task} metrics by setup",
                _metric_boxplot(df, task, overview_metrics, dataset=dataset),
            )
        )
    if not per_client.empty:
        client_dataset_tasks = (
            per_client[["dataset", "task"]]
            .drop_duplicates()
            .sort_values(["dataset", "task"])
        )
        for comparison_id in sorted(per_client.comparison_id.unique()):
            comparison_rows = per_client[per_client.comparison_id == comparison_id]
            for dataset, task in client_dataset_tasks.itertuples(index=False, name=None):
                selected = comparison_rows[
                    (comparison_rows.dataset == dataset) & (comparison_rows.task == task)
                ]
                if not selected.empty:
                    charts.append((
                        f"{comparison_id}: {dataset} / {task}",
                        _delta_bars(selected, task, dataset=dataset),
                    ))
    if not pooled.empty:
        charts.extend(
            (f"{dataset} pooled AUC", _auc_bars(pooled, dataset=dataset))
            for dataset in sorted(pooled.dataset.unique())
        )

    figures = "".join(
        f'<h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{image}"/>'
        for title, image in charts
    )
    tables = ""
    if not deltas.empty:
        tables += (
            "<h2>Manifest-declared exploratory paired deltas</h2>"
            "<p><strong>Inference warning:</strong> client × CV-fold pairs are not independent "
            "replicates. Wilcoxon p-values are descriptive diagnostics, not evidence of "
            "confirmatory statistical significance. Use independent seeds or paired OOF "
            "sample-level inference for such claims.</p>"
            + deltas.round(4).to_html(index=False)
        )
    tables += (
        "<h2>Summary (mean +/- std per dataset x task x method)</h2>"
        + summary.round(4).to_html(index=False)
    )

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Comparison report</title><style>
body{{font-family:system-ui,Arial,sans-serif;margin:2rem auto;max-width:1000px;color:#222}}
h1{{border-bottom:2px solid #264653}} h2{{margin-top:2rem;color:#264653}}
img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
table{{border-collapse:collapse;font-size:.82rem}} th,td{{border:1px solid #ccc;padding:4px 8px}}
th{{background:#f4f4f4}}</style></head><body>
<h1>Federated experiment comparison</h1>
{figures}{tables}
</body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")


def aggregation_audit(result_paths):
    """Flatten per-round aggregation histories stored beside result files."""
    rows = []
    for result_path in result_paths:
        result_path = Path(result_path)
        result_frame = _with_study_metadata(pd.read_csv(result_path, nrows=1))
        metadata = result_frame.iloc[0]
        for history_path in sorted(result_path.parent.glob("fold_*/aggregation_history.json")):
            # Resume-by-fold archives interrupted work as ``fold_N.incomplete_<timestamp>``.
            # Those histories are forensic artifacts, not completed experiment folds, and must
            # not be folded into the audit or parsed as a numeric fold identifier.
            fold_name = history_path.parent.name.removeprefix("fold_")
            if not fold_name.isdecimal():
                continue
            fold = int(fold_name)
            with history_path.open(encoding="utf-8") as stream:
                history = json.load(stream)
            for event in history:
                for client in event.get("clients", []):
                    rows.append({
                        "study_id": metadata.study_id,
                        "method_id": metadata.method_id,
                        "setup": metadata.setup,
                        "seed": metadata.seed,
                        "fold": fold,
                        "round": event["round"],
                        "stage": event["stage"],
                        "aggregation_mode": event["aggregation_mode"],
                        "client_weighting": event["client_weighting"],
                        **client,
                    })
    return pd.DataFrame(rows)


def partition_audits(result_paths):
    """Describe dataset scale and fold-local class weights from paired study partitions."""
    sources = {}
    for result_path in result_paths:
        run_dir = Path(result_path).parent
        metadata_path = run_dir / "study_run.yaml"
        if not metadata_path.exists():
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        partition_path = metadata.get("partition_path")
        if partition_path:
            sources[(str(metadata.get("study_id", LEGACY_STUDY)), str(metadata.get("seed")))] = Path(partition_path)

    composition_rows, class_rows = [], []
    for (study_id, seed), partition_path in sorted(sources.items()):
        mapping = pd.read_csv(partition_path, low_memory=False)
        if "dataset" not in mapping:
            mapping["dataset"] = LEGACY_DATASET
        unique_column = "img_path" if "img_path" in mapping else None
        counts = (
            mapping.groupby("dataset")[unique_column].nunique()
            if unique_column else mapping.groupby("dataset").size()
        )
        total = int(counts.sum())
        for dataset, count in counts.items():
            composition_rows.append({
                "study_id": study_id,
                "seed": seed,
                "dataset": dataset,
                "unique_images": int(count),
                "fraction_unique_images": float(count / total) if total else np.nan,
            })

        required = {"fold", "split", "class"}
        if required.issubset(mapping.columns):
            train = mapping[(mapping.split == "train")]
            if "task" in mapping:
                train = train[train.task == "cls"]
            for (fold, dataset), group in train.groupby(["fold", "dataset"]):
                class_counts = group["class"].dropna().value_counts().sort_index()
                n_samples, n_classes = int(class_counts.sum()), int(len(class_counts))
                for class_name, count in class_counts.items():
                    class_rows.append({
                        "study_id": study_id,
                        "seed": seed,
                        "fold": int(fold),
                        "dataset": dataset,
                        "class_name": class_name,
                        "train_examples": int(count),
                        "balanced_fold_weight": n_samples / (n_classes * int(count)),
                    })
    return pd.DataFrame(composition_rows), pd.DataFrame(class_rows)


def run(results, preds, out, manifest=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat(
        [_with_study_metadata(pd.read_csv(path)) for path in results], ignore_index=True
    )

    summary = summary_table(df)
    summary.to_csv(out / "summary_per_task_setup.csv", index=False)
    by_seed = summary_by_seed(df)
    by_seed.to_csv(out / "summary_by_seed.csv", index=False)

    deltas, per_client = pd.DataFrame(), pd.DataFrame()
    comparisons = _load_comparisons(manifest)
    if comparisons:
        deltas, per_client = method_comparisons(df, comparisons)
        deltas.to_csv(out / "method_comparisons.csv", index=False)
        per_client.to_csv(out / "per_client_method_deltas.csv", index=False)
    elif {"federated", "standalone"}.issubset(set(df.setup.unique())):
        logging.warning(
            "Wilcoxon client x CV-fold pairs are non-independent; treating p-values as "
            "exploratory diagnostics only"
        )
        deltas = paired_deltas(df)
        deltas.to_csv(out / "federated_vs_local_wilcoxon.csv", index=False)
        per_client = per_client_deltas(df)
        per_client["comparison_id"] = "federated_vs_local"
        per_client.to_csv(out / "per_client_deltas.csv", index=False)

    pooled = pooled_auc(preds)
    if not pooled.empty:
        pooled.to_csv(out / "pooled_auc.csv", index=False)

    audit = aggregation_audit(results)
    if not audit.empty:
        audit.to_csv(out / "aggregation_audit.csv", index=False)
    composition, class_balance = partition_audits(results)
    if not composition.empty:
        composition.to_csv(out / "data_composition.csv", index=False)
    if not class_balance.empty:
        class_balance.to_csv(out / "class_balance_audit.csv", index=False)

    build_html(df, summary, deltas, per_client, pooled, out)

    logging.info(
        "\nSummary (mean per dataset x task x method):\n%s",
        summary.pivot_table(
            index=["dataset", "task", "metric"], columns="method_id", values="mean"
        ).round(4),
    )
    logging.info(f"Wrote analysis tables + report.html to {out}")


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", nargs="+", default=RESULTS, help="per-setup *_test_results.csv files"
    )
    parser.add_argument(
        "--preds",
        nargs="*",
        default=PREDS,
        help="per-setup *_cls_predictions.csv files (for pooled AUC)",
    )
    parser.add_argument("--out", default=OUT, help="output directory for analysis tables")
    parser.add_argument("--manifest", default=None, help="study manifest with paired comparisons")
    args = parser.parse_args()
    run(args.results, args.preds, args.out, manifest=args.manifest)


if __name__ == "__main__":
    main()
