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

import html
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("overall", "intra", "inter")
REFERENCE = "orthogonal_reference"
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
    # What ``overall`` would be if the same updates were mutually orthogonal (no conflict, no
    # agreement). High-dimensional updates are near-orthogonal by default, so ``overall`` only
    # signals conflict when it exceeds this reference.
    result["orthogonal_reference"] = _ratio(
        math.sqrt(float(np.dot(weights ** 2, norms ** 2))), client_mass
    )
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


def _report_metrics(summary):
    """overall / intra / inter, the orthogonal reference when recorded, then each dataset."""
    reference = [REFERENCE] if f"{REFERENCE}_mean" in summary else []
    return list(METRICS) + reference + sorted(
        c.removesuffix("_mean") for c in summary.columns
        if c.startswith("intra_") and c.endswith("_mean") and c != "intra_mean"
    )


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
    metrics = _report_metrics(summary)
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
    build_html(run_path, rounds, summary)
    return summary


# ------------------------------------------------------------------------------------------------
# HTML report
# ------------------------------------------------------------------------------------------------

HTML_FILE = "negative_transfer.html"
PLOTS_DIR = Path("report") / "negative_transfer"
METRIC_LABELS = {
    "overall": "Total",
    "intra": "Intra-base (clientes da mesma base)",
    "inter": "Inter-base (entre bases)",
    REFERENCE: "Referência: updates ortogonais",
}


def mean_cosine_matrix(run_path):
    """Whole-trunk cosine between client deltas, averaged over every fit round of every fold.

    Clients are matched by label, so a round with a different client order still aligns.
    Returns ``(labels, matrix)`` or ``(None, None)`` when no history carries the matrix.
    """
    sums, counts = {}, {}
    for _, path in _history_paths(run_path):
        for event in json.loads(path.read_text(encoding="utf-8")):
            conflict = event.get("gradient_conflict") if event.get("stage") == "fit" else None
            if not conflict:
                continue
            labels = conflict["clients"]
            matrix = conflict["cosine"].get("shared_total")
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    if value is None:
                        continue
                    key = (labels[i], labels[j])
                    sums[key] = sums.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1
    if not sums:
        return None, None
    labels = sorted({a for a, _ in sums}, key=lambda label: (label.split("/")[0], label))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan)
    for (a, b), total in sums.items():
        matrix[index[a], index[b]] = total / counts[(a, b)]
    return labels, matrix


def pair_cosines(labels, matrix):
    """Mean off-diagonal cosine of same-dataset pairs vs cross-dataset pairs."""
    datasets = [label.split("/")[0] for label in labels]
    same, cross = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if np.isfinite(matrix[i, j]):
                (same if datasets[i] == datasets[j] else cross).append(matrix[i, j])
    return {
        "same_dataset": (float(np.mean(same)) if same else float("nan"), len(same)),
        "cross_dataset": (float(np.mean(cross)) if cross else float("nan"), len(cross)),
    }


def _plot_rounds(rounds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from src.utils.plot_style import PLOT_RC, categorical_color, fold_style, style_axis

    dataset_columns = sorted(c for c in rounds.columns if c.startswith("intra_"))
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)
        for fold, frame in rounds.groupby("fold"):
            style = fold_style(fold)
            for metric, colour in zip(METRICS, ("#08519C", "#6F6357", "#9F2B2C")):
                axes[0].plot(frame["round"], frame[metric], color=colour,
                             linestyle=style["linestyle"], linewidth=1.4,
                             label=f"{METRIC_LABELS[metric]} · fold {fold}")
            for index, column in enumerate(dataset_columns):
                axes[1].plot(frame["round"], frame[column], color=categorical_color(index),
                             linestyle=style["linestyle"], linewidth=1.4,
                             label=f"{column.removeprefix('intra_')} · fold {fold}")
            if REFERENCE in frame:
                axes[0].plot(frame["round"], frame[REFERENCE], color="#CABD91",
                             linestyle=":", linewidth=1.4,
                             label=f"{METRIC_LABELS[REFERENCE]} · fold {fold}")
        axes[0].set_title("Cancelamento por rodada")
        axes[1].set_title("Cancelamento intra-base, por base")
        for axis in axes:
            axis.set_xlabel("rodada")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_ylim(-0.02, 1.02)
            style_axis(axis)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7)
        axes[0].set_ylabel("razão de cancelamento (0 = alinhado, 1 = anulado)")
    return fig


def _plot_heatmap(labels, matrix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from src.utils.plot_style import PLOT_RC

    cmap = LinearSegmentedColormap.from_list(
        "diverging_blue_red", ["#9F2B2C", "#CABD91", "#FCFAE1", "#6BAED6", "#08519C"]
    )
    size = max(5.0, 0.55 * len(labels) + 2.5)
    with matplotlib.rc_context(PLOT_RC):
        fig, axis = plt.subplots(figsize=(size + 1.2, size), constrained_layout=True)
        image = axis.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-1, vmax=1))
        short = [label.split("/", 1)[0] + " / " + label.rsplit("/", 1)[-1] for label in labels]
        axis.set_xticks(range(len(labels)), short, rotation=60, ha="right", fontsize=8)
        axis.set_yticks(range(len(labels)), short, fontsize=8)
        if len(labels) <= 16:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if np.isfinite(matrix[i, j]):
                        axis.text(j, i, f"{round(matrix[i, j], 2) + 0.0:.2f}", ha="center", va="center",
                                  fontsize=7, color="white" if abs(matrix[i, j]) > 0.6 else "#584A47")
        datasets = [label.split("/")[0] for label in labels]
        for index in range(1, len(labels)):
            if datasets[index] != datasets[index - 1]:
                axis.axhline(index - 0.5, color="#584A47", linewidth=1.2)
                axis.axvline(index - 0.5, color="#584A47", linewidth=1.2)
        axis.set_title("Cosseno médio entre deltas do trunk (vermelho = conflito)")
        fig.colorbar(image, ax=axis, shrink=0.8, label="cosseno")
    return fig


def _save(fig, path):
    import matplotlib.pyplot as plt
    from src.utils.plot_style import EMBED_DPI
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=EMBED_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_html(run_path, rounds=None, summary=None):
    """``negative_transfer.html`` with the summary table, per-round curves and the cosine heatmap."""
    run_path = Path(run_path)
    if rounds is None or summary is None:
        rounds, summary = summarize_run(run_path)
    esc = html.escape
    parts = [
        "<h1>Transferência negativa</h1>",
        "<p class='meta'><a href='run_report.html'>Relatório da run</a></p>",
        "<p>Razão de cancelamento dos updates do trunk compartilhado na agregação: "
        "<code>1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖</code>, com os pesos reais da agregação. 0 = todos os "
        "clientes movem o trunk na mesma direção; 1 = os updates se anulam. Decompõe-se em "
        "<b>intra-base</b> (clientes da mesma base) e <b>inter-base</b> (bases entre si), com "
        "<code>(1 − total) = (1 − intra)(1 − inter)</code>.</p>"
        "<p>Leia o total contra a <b>referência ortogonal</b>: o valor que ele teria se os mesmos "
        "updates fossem mutuamente ortogonais (nem conflito nem concordância). Updates em alta "
        "dimensão já são quase ortogonais, e n updates ortogonais de mesma norma cancelam "
        "<code>1 − 1/√n</code> só pela geometria. Total acima da referência indica conflito; abaixo, "
        "concordância.</p>",
    ]
    if summary.empty or rounds.empty:
        parts.append("<div class='notice'>Indisponível: o histórico desta run não contém "
                     "<code>negative_transfer</code>.</div>")
    else:
        total = summary.iloc[-1]
        metrics = _report_metrics(summary)
        rows = []
        for metric in metrics:
            label = METRIC_LABELS.get(metric, f"Intra-base: {metric.removeprefix('intra_')}")
            std = total.get(f"{metric}_mean_std")
            spread = "" if std is None or pd.isna(std) else f" ± {std:.4f}"
            rows.append(
                f"<tr><td>{esc(label)}</td><td class='num'>{_fmt(total.get(f'{metric}_mean'))}"
                f"{spread}</td><td class='num'>{_fmt(total.get(f'{metric}_first_third'))}</td>"
                f"<td class='num'>{_fmt(total.get(f'{metric}_last_third'))}</td></tr>"
            )
        parts.append(
            "<h2>Resumo</h2><table><thead><tr><th>Componente</th><th>Média das rodadas"
            "</th><th>1º terço</th><th>Último terço</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
            f"<p class='meta'>Status: {esc(str(total['status']))} · {int(total['n_rounds'])} "
            "rodadas no total · ± = desvio entre folds. Uma base com um único cliente tem "
            "intra-base 0 por construção.</p>"
        )
        curves = PLOTS_DIR / "rounds.png"
        _save(_plot_rounds(rounds), run_path / curves)
        parts.append(f"<h2>Por rodada</h2><img alt='curvas por rodada' src='{curves.as_posix()}'>")

    labels, matrix = mean_cosine_matrix(run_path)
    if labels is not None:
        heatmap = PLOTS_DIR / "cosine_heatmap.png"
        _save(_plot_heatmap(labels, matrix), run_path / heatmap)
        pairs = pair_cosines(labels, matrix)
        parts.append(
            "<h2>Similaridade entre clientes</h2>"
            "<table><thead><tr><th>Pares</th><th>Cosseno médio</th><th>Pares</th></tr></thead>"
            "<tbody>"
            f"<tr><td>Mesma base</td><td class='num'>{_fmt(pairs['same_dataset'][0])}</td>"
            f"<td class='num'>{pairs['same_dataset'][1]}</td></tr>"
            f"<tr><td>Bases diferentes</td><td class='num'>{_fmt(pairs['cross_dataset'][0])}</td>"
            f"<td class='num'>{pairs['cross_dataset'][1]}</td></tr></tbody></table>"
            "<p class='meta'>Cosseno entre os deltas do trunk inteiro, média de todas as rodadas e "
            "folds. Linhas escuras separam as bases.</p>"
            f"<img alt='heatmap de cosseno' src='{heatmap.as_posix()}'>"
        )

    style = (
        "body{font-family:system-ui,Arial,sans-serif;margin:2rem auto;max-width:1100px;"
        "color:#584A47;background:#fff;padding:0 16px}h1{border-bottom:3px solid #08519C}"
        "h2{color:#08519C;margin-top:2rem}table{border-collapse:collapse;font-size:.85rem}"
        "th,td{border:1px solid #CABD91;padding:4px 8px}th{background:#EFF3FF}"
        "td.num{text-align:right;font-variant-numeric:tabular-nums}a{color:#08519C}"
        "img{max-width:100%;height:auto;border:1px solid #CABD91;border-radius:7px}"
        ".notice{padding:.8rem 1rem;border-left:5px solid #9F2B2C;background:#FCFAE1}"
        ".meta{color:#6F6357;font-size:.9rem}"
    )
    document = (
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Transferência negativa</title><style>{style}</style></head><body>"
        + "".join(parts) + "</body></html>"
    )
    output = run_path / HTML_FILE
    output.write_text(document, encoding="utf-8")
    return output
