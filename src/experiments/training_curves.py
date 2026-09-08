"""Build durable training histories and task-separated federated learning dashboards."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from src.federated.config import training_telemetry_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.plot_style import (  # noqa: E402
    ACCENT_RED,
    EARTH_5,
    EMBED_DPI,
    EXPORT_DPI,
    fold_style,
    metric_color,
    split_style,
    style_axis,
    with_plot_style,
)


REQUIRED_COLUMNS = {
    "setup", "fold", "round", "dataset", "client_id", "task", "phase", "split",
    "unit_type", "unit_index", "metric_name", "value", "n_samples", "optimizer_steps",
    "examples_processed",
}
PRIMARY_METRIC = {"seg": "dice", "cls": "balanced_accuracy"}
SECONDARY_METRICS = {
    "seg": ("dice", "iou"),
    "cls": ("balanced_accuracy", "macro_f1", "accuracy"),
}


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _config(run_path):
    path = Path(run_path) / "config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _valid_fold(path):
    suffix = path.name.removeprefix("fold_")
    return suffix.isdecimal()


def _aggregation_weights(run_path):
    rows = []
    for history_path in sorted(Path(run_path).glob("fold_*/aggregation_history.json")):
        if not _valid_fold(history_path.parent):
            continue
        fold = int(history_path.parent.name.removeprefix("fold_"))
        try:
            events = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for event in events:
            phase = (
                "post_local_round" if event.get("stage") == "fit"
                else "post_aggregation_round"
            )
            for client in event.get("clients", []):
                rows.append({
                    "fold": fold,
                    "round": int(event["round"]),
                    "dataset": str(client.get("dataset", "unknown")),
                    "client_id": str(client.get("client_id", "unknown")),
                    "phase": phase,
                    "aggregation_weight": float(client.get("final_weight", np.nan)),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(
        ["fold", "round", "dataset", "client_id", "phase"], keep="last"
    )


def collect_history(run_path):
    """Collect completed-fold client histories and attach effective server weights."""
    frames = []
    for path in sorted(Path(run_path).glob("fold_*/client_*/training_history.csv")):
        if not _valid_fold(path.parents[1]):
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logging.warning("Skipping malformed training history %s: %s", path, exc)
            continue
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            logging.warning("Skipping %s; missing columns %s", path, sorted(missing))
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS | {"aggregation_weight", "progress"}))

    history = pd.concat(frames, ignore_index=True, sort=False)
    numeric = (
        "fold", "round", "unit_index", "value", "n_samples", "optimizer_steps",
        "examples_processed",
    )
    for column in numeric:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    weights = _aggregation_weights(run_path)
    if not weights.empty:
        history = history.merge(
            weights,
            on=["fold", "round", "dataset", "client_id", "phase"],
            how="left",
        )
    else:
        history["aggregation_weight"] = np.nan

    config = _config(run_path)
    local = config.get("federated", {}).get("local_training", {})
    local_epochs = max(int(local.get("local_epochs", config.get("federated", {}).get("local_epochs", 1))), 1)
    steps = max(int(local.get("steps_per_round", 1)), 1)

    def progress(row):
        if row.unit_type == "epoch":
            return float(row["round"] - 1 + row.unit_index / local_epochs)
        if row.unit_type == "step":
            return float(row["round"] - 1 + row.unit_index / steps)
        return float(row["round"])

    history["progress"] = history.apply(progress, axis=1)
    return history.sort_values(
        ["fold", "round", "dataset", "client_id", "phase", "split", "unit_index", "task", "metric_name"]
    ).reset_index(drop=True)


def _weighted_summary(frame):
    frame = frame[np.isfinite(frame["value"])].copy()
    if frame.empty:
        return np.nan, np.nan, np.nan
    weights = pd.to_numeric(frame.get("aggregation_weight"), errors="coerce")
    valid = np.isfinite(weights) & (weights > 0)
    if valid.any():
        values = frame.loc[valid, "value"].to_numpy(dtype=float)
        selected = weights.loc[valid].to_numpy(dtype=float)
        mean = float(np.dot(values, selected / selected.sum()))
    else:
        mean = float(frame["value"].mean())
    return mean, float(frame["value"].min()), float(frame["value"].max())


def _plot_fold_mean(ax, fold_points, metric_name, color=ACCENT_RED):
    """Overlay the descriptive mean trend when more than one CV fold is present."""
    if len(fold_points) < 2:
        return
    combined = pd.concat(fold_points, ignore_index=True)
    mean = combined.groupby("round", sort=True)["value"].mean().reset_index()
    ax.plot(
        mean["round"], mean["value"], color=color, linestyle="-.", linewidth=2.6,
        marker="X", markersize=5, label=f"média dos folds / {metric_name}", zorder=5,
    )


def _line_label(row):
    phase = str(row["phase"]).replace("_round", "").replace("local_", "")
    return f"{row['split']} / {phase} / {row['metric_name']}"


@with_plot_style
def _plot_client(frame, title):
    tasks = [task for task in ("seg", "cls") if task in set(frame["task"])]
    if not tasks:
        tasks = [str(frame["task"].iloc[0])]
    fig, axes = plt.subplots(
        len(tasks), 2, figsize=(13, 4.2 * len(tasks)), squeeze=False,
        constrained_layout=True,
    )
    for row_index, task in enumerate(tasks):
        task_frame = frame[frame.task == task]
        for column, metric_names in enumerate((("loss",), SECONDARY_METRICS.get(task, ()))):
            ax = axes[row_index, column]
            selected = task_frame[task_frame.metric_name.isin(metric_names)]
            if column == 0 and "combined" in set(frame["task"]):
                selected = pd.concat([
                    selected,
                    frame[(frame.task == "combined") & (frame.metric_name == "loss")],
                ], ignore_index=True)
            for index, (_, group) in enumerate(selected.groupby(
                ["task", "phase", "split", "metric_name"], sort=False
            )):
                group = group.sort_values("progress")
                label = _line_label(group.iloc[0])
                if len(set(selected["task"])) > 1:
                    label = f"{group.iloc[0]['task']} / {label}"
                first = group.iloc[0]
                line_style = split_style(first["split"], first["phase"], index)
                ax.plot(
                    group["progress"], group["value"], markersize=3.5,
                    markevery=max(len(group) // 12, 1), linewidth=1.7,
                    color=metric_color(first["metric_name"], index),
                    label=label, **line_style,
                )
            if column == 1:
                ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{task}: {'loss' if column == 0 else 'métricas'}")
            ax.set_xlabel("progresso em rounds")
            style_axis(ax)
            for boundary in range(1, int(frame["round"].max()) + 1):
                ax.axvline(boundary, color=EARTH_5[2], linewidth=0.6, alpha=0.30)
            if not selected.empty:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=14)
    return fig


@with_plot_style
def _plot_group(frame, dataset, task, title):
    metrics = ("loss",) + SECONDARY_METRICS[task]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    round_frame = frame[
        (frame.dataset == dataset)
        & (frame.task == task)
        & (frame.phase == "post_aggregation_round")
        & (frame.split == "val")
    ]
    for axis_index, names in enumerate((("loss",), metrics[1:])):
        ax = axes[axis_index]
        selected = round_frame[round_frame.metric_name.isin(names)]
        for (_, client_id, metric_name), client in selected.groupby(
            ["fold", "client_id", "metric_name"], sort=False
        ):
            client = client.sort_values("round")
            ax.plot(
                client["round"], client["value"], color=EARTH_5[2],
                alpha=0.30, linewidth=0.8,
            )
        for metric_index, metric_name in enumerate(names):
            metric = selected[selected.metric_name == metric_name]
            fold_points = []
            for fold, fold_frame in metric.groupby("fold", sort=True):
                points = []
                for round_number, group in fold_frame.groupby("round", sort=True):
                    mean, lower, upper = _weighted_summary(group)
                    points.append((round_number, mean, lower, upper))
                if not points:
                    continue
                values = np.asarray(points, dtype=float)
                label = f"fold {int(fold)} / {metric_name}"
                color = metric_color(metric_name, metric_index)
                line_style = fold_style(fold)
                line_style["color"] = color
                ax.plot(
                    values[:, 0], values[:, 1], linewidth=2.0,
                    markersize=4.5, color=color, label=label,
                    linestyle=line_style["linestyle"], marker=line_style["marker"],
                )
                ax.fill_between(values[:, 0], values[:, 2], values[:, 3], color=color, alpha=0.12)
                fold_points.append(pd.DataFrame({"round": values[:, 0], "value": values[:, 1]}))
            _plot_fold_mean(ax, fold_points, metric_name, color=color)
        if axis_index == 1:
            ax.set_ylim(-0.03, 1.03)
        ax.set_title("loss de validação" if axis_index == 0 else "métricas de validação")
        ax.set_xlabel("round")
        style_axis(ax)
        if not selected.empty:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    return fig


def _aggregate_primary(frame, extra_group=()):
    rows = []
    selected = frame[
        (frame.phase == "post_aggregation_round")
        & (frame.split == "val")
        & frame.apply(lambda row: row.metric_name == PRIMARY_METRIC.get(row.task), axis=1)
    ]
    keys = list(extra_group) + ["dataset", "task", "fold", "round", "metric_name"]
    for values, group in selected.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        mean, lower, upper = _weighted_summary(group)
        rows.append(dict(zip(keys, values), value=mean, lower=lower, upper=upper))
    return pd.DataFrame(rows)


@with_plot_style
def _plot_overview(frame, title="Visão geral do treinamento"):
    summary = _aggregate_primary(frame)
    groups = [(d, t) for d in sorted(frame.dataset.unique()) for t in ("seg", "cls")
              if ((frame.dataset == d) & (frame.task == t)).any()]
    columns = 2
    rows = max((len(groups) + columns - 1) // columns, 1)
    fig, axes = plt.subplots(
        rows, columns, figsize=(13, 4.2 * rows), squeeze=False,
        constrained_layout=True,
    )
    for ax, group in zip(axes.flat, groups):
        dataset, task = group
        selected = summary[(summary.dataset == dataset) & (summary.task == task)]
        fold_points = []
        for fold, fold_frame in selected.groupby("fold", sort=True):
            fold_frame = fold_frame.sort_values("round")
            line_style = fold_style(fold)
            ax.plot(
                fold_frame["round"], fold_frame["value"], linewidth=2,
                markersize=4.5, label=f"fold {int(fold)}", **line_style,
            )
            ax.fill_between(
                fold_frame["round"], fold_frame["lower"], fold_frame["upper"],
                color=line_style["color"], alpha=0.12,
            )
            fold_points.append(fold_frame[["round", "value"]])
        _plot_fold_mean(ax, fold_points, PRIMARY_METRIC[task])
        ax.set_title(f"{dataset} / {task} / {PRIMARY_METRIC[task]}")
        ax.set_xlabel("round")
        ax.set_ylim(-0.03, 1.03)
        style_axis(ax)
        if not selected.empty:
            ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    return fig


def _save_figure(fig, path, keep_file):
    path.parent.mkdir(parents=True, exist_ok=True)
    if keep_file:
        fig.savefig(path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
        source = path.as_posix()
    else:
        buffer = io.BytesIO()
        fig.savefig(
            buffer, format="png", dpi=EMBED_DPI, bbox_inches="tight", facecolor="white"
        )
        source = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    plt.close(fig)
    return source


def build_run_artifacts(run_path, *, force=False):
    """Consolidate one run and generate client, dataset/task, and overview artifacts."""
    run_path = Path(run_path)
    config = _config(run_path)
    telemetry = training_telemetry_config(config) if config else {
        "enabled": True, "granularity": "epoch_and_round", "formats": ["csv", "html", "png"]
    }
    if not telemetry["enabled"] and not force:
        return None
    formats = set(telemetry["formats"])
    output = run_path / "training_curves"
    output.mkdir(parents=True, exist_ok=True)
    history = collect_history(run_path)
    if "csv" in formats or force:
        history.to_csv(output / "history.csv", index=False)

    images = []
    keep_png = "png" in formats
    if not history.empty and formats.intersection({"html", "png"}):
        overview_path = output / "plots" / "overview" / "overview_metrics.png"
        source = _save_figure(_plot_overview(history), overview_path, keep_png)
        images.append(("Visão geral", source if not keep_png else overview_path.relative_to(output).as_posix()))

        for (fold, dataset, client_id), group in history.groupby(
            ["fold", "dataset", "client_id"], sort=True
        ):
            path = output / "plots" / "clients" / _safe_name(dataset) / (
                f"{_safe_name(client_id)}_fold_{int(fold)}.png"
            )
            source = _save_figure(
                _plot_client(group, f"{dataset} / {client_id} / fold {int(fold)}"), path, keep_png
            )
            images.append((
                f"Cliente: {dataset} / {client_id} / fold {int(fold)}",
                source if not keep_png else path.relative_to(output).as_posix(),
            ))

        for dataset in sorted(history.dataset.unique()):
            for task in ("seg", "cls"):
                if not ((history.dataset == dataset) & (history.task == task)).any():
                    continue
                path = output / "plots" / "datasets" / f"{_safe_name(dataset)}_{task}.png"
                source = _save_figure(
                    _plot_group(history, dataset, task, f"{dataset} / {task}"), path, keep_png
                )
                images.append((
                    f"Dataset: {dataset} / {task}",
                    source if not keep_png else path.relative_to(output).as_posix(),
                ))

    if "html" in formats or force:
        if history.empty:
            body = (
                "<div class='notice'>Telemetria indisponível: esta run não contém históricos "
                "por cliente. Runs antigas não são reconstruídas artificialmente.</div>"
            )
        else:
            cards = "".join(
                f"<section><h2>{title}</h2><img alt='{title}' src='{source}'></section>"
                for title, source in images
            )
            inventory = (
                history.groupby(["dataset", "task", "client_id", "fold"], dropna=False)
                .size().rename("measurements").reset_index().to_html(index=False)
            )
            body = cards + "<h2>Inventário da telemetria</h2>" + inventory
        html = f"""<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>
<title>Curvas de treinamento</title><style>
body{{font-family:system-ui,Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#584A47;background:#fff}}
h1{{border-bottom:3px solid #08519C;color:#584A47}} h2{{color:#08519C}}
section{{margin:2rem 0}} img{{width:100%;height:auto;border:1px solid #CABD91;border-radius:7px;background:#fff}}
.notice{{padding:1rem;border-left:5px solid #9F2B2C;background:#FCFAE1}} table{{border-collapse:collapse;font-size:.82rem}}
th,td{{border:1px solid #CABD91;padding:4px 8px}} th{{background:#EFF3FF;color:#584A47}}
a{{color:#08519C}}</style></head><body>
<h1>Desempenho do treinamento</h1><p>Loss e métricas são separadas por dataset e tarefa. As faixas mostram a dispersão entre clientes; a linha agregada usa os pesos efetivos da estratégia quando disponíveis.</p>
{body}</body></html>"""
        (output / "dashboard.html").write_text(html, encoding="utf-8")
    return output


def build_study_artifacts(result_paths, out):
    """Build/rebuild run dashboards and a study-level index without inventing legacy curves."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames, entries = [], []
    for result_path in result_paths:
        result_path = Path(result_path)
        run_path = result_path.parent
        build_run_artifacts(run_path, force=True)
        history_path = run_path / "training_curves" / "history.csv"
        result = pd.read_csv(result_path, nrows=1)
        metadata = result.iloc[0].to_dict() if not result.empty else {}
        label = str(metadata.get("method_id", metadata.get("setup", run_path.name)))
        available = history_path.exists() and history_path.stat().st_size > 0
        frame = pd.read_csv(history_path) if available else pd.DataFrame()
        if not frame.empty:
            for key in ("study_id", "arm_id", "method_id", "seed"):
                frame[key] = metadata.get(key, "legacy")
            frames.append(frame)
        dashboard = run_path / "training_curves" / "dashboard.html"
        entries.append({
            "label": label,
            "available": not frame.empty,
            "dashboard": os.path.relpath(dashboard, out),
        })
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    combined.to_csv(out / "training_history.csv", index=False)
    items = "".join(
        f"<li><a href='{entry['dashboard']}'>{entry['label']}</a> — "
        f"{'disponível' if entry['available'] else 'telemetria indisponível'}</li>"
        for entry in entries
    )
    (out / "training_dashboard.html").write_text(
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Treinamento do estudo</title>"
        "<style>body{font-family:system-ui,Arial;margin:2rem auto;max-width:900px;color:#584A47;"
        "background:#fff}h1{border-bottom:3px solid #08519C}li{margin:.6rem 0}"
        "a{color:#08519C}</style>"
        f"</head><body><h1>Dashboards de treinamento por braço</h1><ul>{items}</ul></body></html>",
        encoding="utf-8",
    )
    return out / "training_dashboard.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run directory containing fold_*/client_*/training_history.csv")
    parser.add_argument("--force", action="store_true", help="rebuild even when telemetry is disabled")
    args = parser.parse_args()
    output = build_run_artifacts(args.run, force=args.force)
    print(f"TRAINING_CURVES_OK output={output}")


if __name__ == "__main__":
    main()
