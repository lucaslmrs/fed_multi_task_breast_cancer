"""Negative transfer as update cancellation in the aggregated shared trunk.

For client trunk deltas ``Δᵢ`` and the final aggregation weights ``wᵢ`` (summing to one), the
cancellation ratio is ``1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖``: 0 when every client moves the trunk in the same
direction, 1 when the weighted updates annul each other completely.

It decomposes along the two stages of hierarchical aggregation. With ``π_d`` the total weight of
dataset ``d`` and ``D_d`` its weighted mean delta:

    intra = 1 − Σ π_d‖D_d‖ / Σ wᵢ‖Δᵢ‖        (clients of the same dataset)
    inter = 1 − ‖Σ π_d D_d‖ / Σ π_d‖D_d‖     (datasets against each other)

and ``(1 − overall) = (1 − intra)(1 − inter)`` exactly. Everything is computed from the Gram
matrix ``Gᵢⱼ = Δᵢ·Δⱼ`` of the whole shared trunk, which the server builds live every round.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("overall", "intra", "inter")
ROUNDS_FILE = "negative_transfer_rounds.csv"
SUMMARY_FILE = "negative_transfer_summary.csv"
MISSING_STATUS = "n/a: history without negative_transfer"


def _ratio(numerator, denominator):
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(np.clip(1.0 - numerator / denominator, 0.0, 1.0))


def cancellation_from_gram(gram, weights, datasets):
    """Cancellation ratios (overall, intra, inter, intra_<dataset>) for one aggregation."""
    gram = np.asarray(gram, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    aggregate_norm = math.sqrt(max(float(weights @ gram @ weights), 0.0))
    client_mass = float(np.dot(weights, norms))

    result = {}
    dataset_mass = 0.0
    for dataset in dict.fromkeys(datasets):
        indices = [i for i, name in enumerate(datasets) if name == dataset]
        share = float(weights[indices].sum())
        if share <= 0:
            result[f"intra_{dataset}"] = float("nan")
            continue
        within = weights[indices] / share
        mean_norm = math.sqrt(max(float(within @ gram[np.ix_(indices, indices)] @ within), 0.0))
        result[f"intra_{dataset}"] = _ratio(mean_norm, float(np.dot(within, norms[indices])))
        dataset_mass += share * mean_norm

    result["overall"] = _ratio(aggregate_norm, client_mass)
    result["intra"] = _ratio(dataset_mass, client_mass)
    result["inter"] = _ratio(aggregate_norm, dataset_mass)
    return result


def _history_paths(run_path):
    for path in sorted(Path(run_path).glob("fold_*/aggregation_history.json")):
        # ``fold_N.incomplete_<timestamp>`` are forensic archives of interrupted folds.
        fold = path.parent.name.removeprefix("fold_")
        if fold.isdecimal():
            yield int(fold), path


def summarize_run(run_path):
    """Per-round values and a per-fold / all-folds summary for one run directory."""
    round_rows, summary_rows = [], []
    for fold, path in _history_paths(run_path):
        history = json.loads(path.read_text(encoding="utf-8"))
        fold_rows = [
            {"fold": fold, "round": int(event["round"]), **event["negative_transfer"]}
            for event in history
            if event.get("stage") == "fit" and event.get("negative_transfer")
        ]
        if not fold_rows:
            summary_rows.append({"fold": fold, "n_rounds": 0, "status": MISSING_STATUS})
            continue
        frame = pd.DataFrame(fold_rows).sort_values("round").astype({"round": int})
        round_rows.extend(frame.to_dict("records"))
        columns = [c for c in frame.columns if c not in {"fold", "round"}]
        third = max(1, len(frame) // 3)
        row = {"fold": fold, "n_rounds": len(frame), "status": "ok"}
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_first_third"] = values.iloc[:third].mean()
            row[f"{column}_last_third"] = values.iloc[-third:].mean()
        summary_rows.append(row)

    rounds = pd.DataFrame(round_rows)
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return rounds, summary
    valid = summary[summary["status"] == "ok"]
    value_columns = [c for c in summary.columns if c not in {"fold", "n_rounds", "status"}]
    overall = {"fold": "all_folds", "n_rounds": int(summary["n_rounds"].sum()),
               "status": "ok" if len(valid) == len(summary) else
               ("partial" if len(valid) else MISSING_STATUS)}
    for column in value_columns:
        overall[column] = valid[column].mean() if len(valid) else float("nan")
        overall[f"{column}_std"] = valid[column].std() if len(valid) > 1 else float("nan")
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return rounds, summary


def _fmt(value):
    return "n/a" if value is None or pd.isna(value) else f"{value:.4f}"


def report_run(run_path):
    """Write both CSVs beside the run and log the negative transfer level."""
    run_path = Path(run_path)
    rounds, summary = summarize_run(run_path)
    if summary.empty:
        logging.info("[negative transfer] n/a: no aggregation history found")
        return summary
    rounds.to_csv(run_path / ROUNDS_FILE, index=False)
    summary.to_csv(run_path / SUMMARY_FILE, index=False)

    total = summary.iloc[-1]
    metrics = list(METRICS) + sorted(
        c.removesuffix("_mean") for c in summary.columns
        if c.startswith("intra_") and c.endswith("_mean") and c != "intra_mean"
    )
    lines = [f"[negative transfer] cancellation of shared-trunk updates "
             f"(0 = aligned, 1 = fully cancelled) | status={total['status']}"]
    for metric in metrics:
        mean = total.get(f"{metric}_mean")
        std = total.get(f"{metric}_mean_std")
        spread = "" if std is None or pd.isna(std) else f" ± {std:.4f}"
        lines.append(
            f"  {metric:<24} mean={_fmt(mean)}{spread}  "
            f"first_third={_fmt(total.get(f'{metric}_first_third'))}  "
            f"last_third={_fmt(total.get(f'{metric}_last_third'))}"
        )
    logging.info("\n".join(lines))
    return summary
