"""Build a self-contained executive HTML report for the multi-dataset study."""

from __future__ import annotations

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


STUDY_ROOT = Path("runs/studies/example_multi_dataset_dice_bce")
DEFAULT_OUTPUT = Path("RELATORIO_EXECUTIVO_FEDERACAO_MULTI_DATASET.html")
# Display names for the arms of studies/example_multi_dataset.yaml; unknown ids are shown raw.
METHOD_LABELS = {
    "primary": "Federado principal (clientes multitarefa)",
    "local_only": "Local (clientes multitarefa)",
    "ablation_flat": "Federado — flat",
    "single_task_primary": "Federado — clientes monotarefa",
    "single_task_local": "Local — clientes monotarefa",
}
METRIC_LABELS = {
    "dice": "Dice global",
    "dice_positive": "Dice (alvos não vazios)",
    "iou_positive": "IoU (alvos não vazios)",
    "empty_fp_image_rate": "Falsos positivos em imagens vazias",
    "empty_predicted_area_fraction": "Área prevista em imagens vazias",
    "iou": "IoU global",
    "sensitivity": "Sensibilidade",
    "specificity": "Especificidade",
    "precision": "Precisão",
    "acc": "Acurácia",
    "balanced_acc": "Acurácia balanceada",
    "macro_f1": "Macro-F1",
    "auc": "AUC OvR macro",
}
COMPARISON_LABELS = {
    "primary_vs_local": "Federado principal vs local",
    "effect_flat_aggregation": "Hierárquica uniforme vs flat",
    "single_task_vs_local": "Federado monotarefa vs local monotarefa",
    "effect_client_topology": "Clientes multitarefa vs monotarefa",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(value, digits=4):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _esc(value):
    return html.escape(str(value))


def _table(headers, rows, numeric=None):
    numeric = set(numeric or [])
    head = "".join(f"<th>{_esc(column)}</th>" for column in headers)
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            cls = ' class="num"' if index in numeric else ""
            cells.append(f"<td{cls}>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _comparison_chart(summary: pd.DataFrame):
    if summary.empty:
        return '<p class="muted">Resultados ainda indisponíveis.</p>'
    selected = []
    for dataset in ("Curated_BUSI", "ISIC_2018"):
        for task, metric in (("seg", "dice_positive" if "dice_positive" in set(summary.metric) else "dice"), ("cls", "balanced_acc"), ("cls", "auc")):
            rows = summary[
                (summary.dataset == dataset)
                & (summary.task == task)
                & (summary.metric == metric)
                & (summary.method_id.isin(["primary", "local_only"]))
            ]
            values = {row.method_id: float(row.mean) for row in rows.itertuples() if pd.notna(row.mean)}
            if values:
                selected.append((dataset, task, metric, values))
    if not selected:
        return '<p class="muted">Resultados principais ainda indisponíveis.</p>'

    width, height = 920, 90 + 80 * len(selected)
    left, plot_width = 210, 630
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Comparação das métricas principais entre federado e local">',
        '<title>Comparação federado principal versus local</title>',
    ]
    for tick in range(6):
        x = left + plot_width * tick / 5
        parts.append(f'<line class="grid" x1="{x}" x2="{x}" y1="35" y2="{height-30}"/>')
        parts.append(f'<text class="tick" x="{x}" y="25" text-anchor="middle">{tick/5:.1f}</text>')
    for index, (dataset, task, metric, values) in enumerate(selected):
        y = 68 + 80 * index
        metric_label = 'Dice positivo' if metric == 'dice_positive' else METRIC_LABELS[metric]
        label = f"{dataset} · {'seg.' if task == 'seg' else 'cls.'} · {metric_label}"
        parts.append(f'<text class="label" x="0" y="{y+17}">{_esc(label)}</text>')
        for offset, method in ((0, "primary"), (26, "local_only")):
            value = values.get(method)
            if value is None:
                continue
            bar_width = max(0, min(plot_width, value * plot_width))
            css = "bar-fed" if method == "primary" else "bar-local"
            parts.append(
                f'<rect class="{css}" x="{left}" y="{y+offset}" width="{bar_width}" height="18" rx="3"/>'
            )
            parts.append(
                f'<text class="value" x="{left+bar_width+7}" y="{y+offset+14}">{value:.4f}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _composition_chart(composition: pd.DataFrame):
    if composition.empty:
        return ""
    rows = composition.drop_duplicates(["seed", "dataset"]).groupby("dataset", as_index=False)[
        "unique_images"
    ].mean()
    total = rows.unique_images.sum()
    parts = [
        '<svg class="composition" viewBox="0 0 920 150" role="img" '
        'aria-label="Composição das imagens únicas por dataset">',
        "<title>Composição das imagens únicas</title>",
    ]
    x = 20.0
    usable = 880.0
    for index, row in enumerate(rows.itertuples()):
        fraction = float(row.unique_images / total)
        width = usable * fraction
        css = "bar-fed" if index == 0 else "bar-local"
        parts.append(f'<rect class="{css}" x="{x}" y="45" width="{width}" height="42" rx="4"/>')
        x += width
    x = 20.0
    for index, row in enumerate(rows.itertuples()):
        fraction = float(row.unique_images / total)
        css = "legend-fed" if index == 0 else "legend-local"
        parts.append(f'<circle class="{css}" cx="{x+6}" cy="120" r="6"/>')
        parts.append(
            f'<text class="label" x="{x+18}" y="125">{_esc(row.dataset)}: '
            f'{int(row.unique_images):,} ({fraction*100:.2f}%)</text>'
        )
        x += 340
    parts.append("</svg>")
    return "".join(parts)


def _status_section(run_index: pd.DataFrame):
    if run_index.empty:
        return "<p>Índice de execução indisponível.</p>", False
    complete = run_index.status.isin(["complete", "reused"]).sum()
    all_complete = complete == len(run_index)
    rows = []
    for row in run_index.itertuples():
        status = str(row.status)
        completed_at = getattr(row, "completed_at_utc", "")
        if pd.isna(completed_at) or not completed_at:
            completed_at = "—"
        badge = f'<span class="status status-{_esc(status)}">{_esc(status)}</span>'
        rows.append(
            [
                _esc(METHOD_LABELS.get(row.method_id, row.method_id)),
                _esc(row.setup),
                badge,
                _esc(completed_at),
            ]
        )
    summary = (
        f'<p><strong>{complete}/{len(run_index)} braços concluídos.</strong> '
        + (
            "A matriz operacional está completa."
            if all_complete
            else "A análise exibida é provisória e usa somente braços completos."
        )
        + "</p>"
    )
    return summary + _table(["Abordagem", "Setup", "Status", "Conclusão UTC"], rows), all_complete


def _running_fold_progress(run_index: pd.DataFrame, n_splits: int, scheme: str):
    """Show durable per-fold progress without treating partial folds as results."""
    if run_index.empty:
        return ""

    rows = []
    for row in run_index[run_index.status == "running"].itertuples():
        run_path = Path(row.run_path)
        completed = 0
        for fold_path in sorted(run_path.glob("fold_[0-9]*")):
            if not fold_path.is_dir() or ".incomplete_" in fold_path.name:
                continue
            if any(fold_path.glob("*test_results.csv")):
                completed += 1
        rows.append(
            [
                _esc(METHOD_LABELS.get(row.method_id, row.method_id)),
                f"{completed}/{n_splits} "
                + ("holdout com resultado de teste persistido" if scheme == "holdout" else "folds com resultados de teste persistidos"),
                "O split em curso só entra nas tabelas após concluir e gravar seus artefatos.",
            ]
        )
    if not rows:
        return ""
    return (
        '<div class="notice"><strong>Progresso em execução.</strong> '
        "O treinamento segue em processo independente; resultados parciais não são usados para "
        "substituir splits incompletos.</div>"
        + _table(["Abordagem", "Progresso durável", "Regra de inclusão"], rows)
    )


def _results_table(summary: pd.DataFrame):
    if summary.empty:
        return '<p class="muted">Resultados ainda indisponíveis.</p>'
    wanted = {
        "seg": ["dice_positive", "iou_positive", "empty_fp_image_rate",
                "empty_predicted_area_fraction", "dice", "iou", "sensitivity", "specificity"],
        "cls": ["acc", "balanced_acc", "macro_f1", "auc"],
    }
    rows = []
    for dataset in ("Curated_BUSI", "ISIC_2018"):
        for task in ("seg", "cls"):
            for metric in wanted[task]:
                subset = summary[
                    (summary.dataset == dataset)
                    & (summary.task == task)
                    & (summary.metric == metric)
                ]
                for row in subset.itertuples():
                    rows.append(
                        [
                            _esc(dataset),
                            "Segmentação" if task == "seg" else "Classificação",
                            _esc(METRIC_LABELS[metric]),
                            _esc(METHOD_LABELS.get(row.method_id, row.method_id)),
                            _fmt(row.mean),
                            _fmt(row.std),
                            str(int(row.n)),
                        ]
                    )
    return _table(
        ["Dataset", "Tarefa", "Métrica", "Abordagem", "Média", "DP", "n"],
        rows,
        numeric={4, 5, 6},
    )


def _primary_deltas(comparisons: pd.DataFrame):
    if comparisons.empty:
        return '<p class="muted">Resultados ainda indisponíveis.</p>'
    subset = comparisons[comparisons.comparison_id == "primary_vs_local"]
    rows = []
    wanted = {"seg": (["dice_positive", "iou_positive"] if "dice_positive" in set(comparisons.metric) else ["dice", "iou"]), "cls": ["acc", "balanced_acc", "macro_f1", "auc"]}
    for dataset in ("Curated_BUSI", "ISIC_2018"):
        for task in ("seg", "cls"):
            for metric in wanted[task]:
                match = subset[
                    (subset.dataset == dataset)
                    & (subset.task == task)
                    & (subset.metric == metric)
                ]
                if match.empty:
                    continue
                row = match.iloc[0]
                delta = float(row.mean_delta)
                css = "delta-pos" if delta > 0 else ("delta-neg" if delta < 0 else "")
                rows.append(
                    [
                        _esc(dataset),
                        "Segmentação" if task == "seg" else "Classificação",
                        _esc(METRIC_LABELS[metric]),
                        _fmt(row.left_mean),
                        _fmt(row.right_mean),
                        f'<span class="{css}">{delta:+.4f}</span>',
                        str(int(row.n_pairs)),
                    ]
                )
    return _table(
        ["Dataset", "Tarefa", "Métrica", "Federado", "Local", "Δ Fed−Local", "Pares"],
        rows,
        numeric={3, 4, 5, 6},
    )


def _interpretation_notice(all_complete: bool, scheme: str):
    evidence = ('os contrastes no holdout são somente descritivos, sem Wilcoxon'
                if scheme == 'holdout' else 'os contrastes por cliente e fold são exploratórios')
    status = 'Matriz completa' if all_complete else 'Resultados parciais'
    return (f'<div class="notice"><strong>{status}:</strong> {evidence}. '
            'Compare apenas os braços concluídos e os pares declarados no manifesto. '
            'Alvos vazios e não vazios têm métricas separadas; nenhuma conclusão é inferida '
            'para execuções pendentes.</div>')


def _selected_effects(comparisons: pd.DataFrame):
    if comparisons.empty:
        return comparisons
    selected = comparisons[
        ((comparisons.task == "seg") & (comparisons.metric == ("dice_positive" if "dice_positive" in set(comparisons.metric) else "dice")))
        | ((comparisons.task == "cls") & comparisons.metric.isin(["balanced_acc", "auc"]))
    ].copy()
    order = {key: index for index, key in enumerate(COMPARISON_LABELS)}
    selected["_order"] = selected.comparison_id.map(order).fillna(len(order))
    return selected.sort_values(["_order", "dataset", "task", "metric"])


def _effects_table(comparisons: pd.DataFrame):
    selected = _selected_effects(comparisons)
    if selected.empty:
        return '<p class="muted">Contrastes completos ainda indisponíveis.</p>'
    rows = []
    for row in selected.itertuples():
        delta = float(row.mean_delta)
        css = "delta-pos" if delta > 0 else ("delta-neg" if delta < 0 else "")
        rows.append(
            [
                _esc(COMPARISON_LABELS.get(row.comparison_id, row.comparison_id)),
                _esc(row.dataset),
                "Segmentação" if row.task == "seg" else "Classificação",
                _esc(METRIC_LABELS.get(row.metric, row.metric)),
                _esc(METHOD_LABELS.get(row.left_method, row.left_method)),
                _esc(METHOD_LABELS.get(row.right_method, row.right_method)),
                _fmt(row.left_mean),
                _fmt(row.right_mean),
                f'<span class="{css}">{delta:+.4f}</span>',
                _fmt(row.wilcoxon_p),
            ]
        )
    return _table(
        [
            "Contraste",
            "Dataset",
            "Tarefa",
            "Métrica",
            "Primeira",
            "Segunda",
            "Média 1",
            "Média 2",
            "Δ 1−2",
            "p (quando aplicável)",
        ],
        rows,
        numeric={6, 7, 8, 9},
    )


def _effects_chart(comparisons: pd.DataFrame):
    if comparisons.empty:
        return ""
    mechanisms = [
        "effect_flat_aggregation",
        "effect_client_topology",
    ]
    selected = _selected_effects(comparisons)
    selected = selected[selected.comparison_id.isin(mechanisms)]
    if selected.empty:
        return ""

    short_comparison = {
        "effect_flat_aggregation": "Agregação",
        "effect_client_topology": "Topologia",
    }
    short_metric = {"dice": "Dice global", "dice_positive": "Dice positivo",
                    "balanced_acc": "BAcc", "auc": "AUC"}
    width = 1000
    row_height = 34
    top = 45
    height = top + row_height * len(selected) + 45
    center = 690
    half_width = 270
    limit = max(0.30, float(selected.mean_delta.abs().max()) * 1.1)
    parts = [
        f'<svg class="effect-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Efeitos das ablações disponíveis">',
        "<title>Efeitos pareados das ablações</title>",
        "<desc>Valores positivos favorecem a primeira condição do contraste e valores negativos favorecem a segunda.</desc>",
    ]
    for tick in (-limit, -limit/2, 0.0, limit/2, limit):
        x = center + (tick / limit) * half_width
        parts.append(f'<line class="grid" x1="{x}" x2="{x}" y1="25" y2="{height-28}"/>')
        parts.append(f'<text class="tick" x="{x}" y="18" text-anchor="middle">{tick:+.2f}</text>')
    for index, row in enumerate(selected.itertuples()):
        y = top + index * row_height
        task_label = "seg" if row.task == "seg" else "cls"
        label = (
            f"{short_comparison.get(row.comparison_id, row.comparison_id)} · "
            f"{'BUSI' if row.dataset == 'Curated_BUSI' else 'ISIC'} {task_label} "
            f"{short_metric.get(row.metric, row.metric)}"
        )
        delta = float(row.mean_delta)
        bar_width = min(half_width, abs(delta) / limit * half_width)
        x = center if delta >= 0 else center - bar_width
        css = "effect-positive" if delta >= 0 else "effect-negative"
        value_anchor = "start" if delta >= 0 else "end"
        value_x = center + bar_width + 6 if delta >= 0 else center - bar_width - 6
        parts.append(f'<text class="label" x="0" y="{y+13}">{_esc(label)}</text>')
        parts.append(f'<rect class="{css}" x="{x}" y="{y}" width="{bar_width}" height="17" rx="3"/>')
        parts.append(
            f'<text class="value" x="{value_x}" y="{y+13}" text-anchor="{value_anchor}">{delta:+.3f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _descriptive_rankings(summary: pd.DataFrame):
    if summary.empty:
        return '<p class="muted">Resultados ainda indisponíveis.</p>'
    outcomes = [
        ("Curated_BUSI", "seg", "dice"),
        ("Curated_BUSI", "cls", "balanced_acc"),
        ("Curated_BUSI", "cls", "auc"),
        ("ISIC_2018", "seg", "dice"),
        ("ISIC_2018", "cls", "balanced_acc"),
        ("ISIC_2018", "cls", "auc"),
    ]
    rows = []
    for dataset, task, metric in outcomes:
        if task == 'seg' and 'dice_positive' in set(summary.metric):
            metric = 'dice_positive'
        subset = summary[
            (summary.dataset == dataset) & (summary.task == task) & (summary.metric == metric)
        ].dropna(subset=["mean"]).sort_values("mean", ascending=False)
        if subset.empty:
            continue
        # BUSI outcomes are intentionally unchanged by the ISIC-only focal local arm. Collapse
        # exact duplicate means so the ranking does not present the same result twice.
        subset = subset.drop_duplicates(["mean", "std"])
        best = subset.iloc[0]
        second = subset.iloc[1] if len(subset) > 1 else None
        rows.append(
            [
                _esc(dataset),
                "Segmentação" if task == "seg" else "Classificação",
                _esc(METRIC_LABELS[metric]),
                _esc(METHOD_LABELS.get(best.method_id, best.method_id)),
                _fmt(best["mean"]),
                _esc(METHOD_LABELS.get(second.method_id, second.method_id)) if second is not None else "—",
                _fmt(second["mean"]) if second is not None else "—",
            ]
        )
    return _table(
        ["Dataset", "Tarefa", "Métrica", "Maior média", "Valor", "Segunda média", "Valor"],
        rows,
        numeric={4, 6},
    )


def _pooled_auc_table(pooled: pd.DataFrame):
    if pooled.empty:
        return ""
    subset = pooled[pooled.fold.astype(str) == "all"].sort_values(
        ["dataset", "auc_pooled"], ascending=[True, False]
    )
    rows = [
        [
            _esc(row.dataset),
            _esc(METHOD_LABELS.get(row.method_id, row.method_id)),
            _esc(row.setup),
            str(int(row.num_classes)),
            str(int(row.n)),
            _fmt(row.auc_pooled),
            _esc(getattr(row, "aggregation_scope", "out_of_fold")),
        ]
        for row in subset.itertuples()
    ]
    return _table(
        ["Dataset", "Abordagem", "Setup", "Classes", "n", "AUC agregada", "Escopo"],
        rows,
        {3, 4, 5},
    )


def _comparison_delta(comparisons: pd.DataFrame, comparison_id, dataset, task, metric):
    match = comparisons[
        (comparisons.comparison_id == comparison_id)
        & (comparisons.dataset == dataset)
        & (comparisons.task == task)
        & (comparisons.metric == metric)
    ]
    return None if match.empty else float(match.iloc[0].mean_delta)


def _key_findings(comparisons: pd.DataFrame):
    selected = _selected_effects(comparisons)
    if selected.empty:
        return '<p>Contrastes concluídos ainda indisponíveis.</p>'
    rows = []
    for row in selected.itertuples():
        rows.append(f'<li>{_esc(COMPARISON_LABELS.get(row.comparison_id, row.comparison_id))} '
                    f'— {_esc(row.dataset)}, {_esc(METRIC_LABELS.get(row.metric, row.metric))}: '
                    f'Δ = {float(row.mean_delta):+.4f} '
                    f'({int(row.n_pairs)} observações pareadas).</li>')
    return '<ul class="key-findings">' + ''.join(rows) + '</ul>'


def _class_weight_table(balance: pd.DataFrame):
    if balance.empty:
        return ""
    subset = balance[balance.dataset == "ISIC_2018"]
    rows = []
    for row in subset.sort_values(["fold", "class_name"]).itertuples():
        rows.append(
            [str(int(row.fold)), _esc(row.class_name), str(int(row.train_examples)), _fmt(row.balanced_fold_weight)]
        )
    return _table(["Fold", "Classe ISIC", "Exemplos de treino", "Peso balanced_fold"], rows, {0, 2, 3})


def _aggregation_audit(audit: pd.DataFrame):
    if audit.empty:
        return "<p>Auditoria ainda indisponível.</p>"
    selected = audit[
        audit.method_id.isin(["primary", "ablation_weighting", "ablation_flat"])
        & (audit.stage == "fit")
    ]
    grouped = selected.groupby(
        ["method_id", "fold", "round", "dataset"], as_index=False
    ).final_weight.sum()
    checks = (
        grouped.groupby(["method_id", "dataset"])
        .final_weight.agg(["min", "max", "mean"])
        .reset_index()
    )
    rows = [
        [
            _esc(METHOD_LABELS.get(row.method_id, row.method_id)),
            _esc(row.dataset),
            _fmt(row["min"]),
            _fmt(row["max"]),
            _fmt(row["mean"]),
        ]
        for _, row in checks.iterrows()
    ]
    return _table(
        ["Agregação", "Dataset", "Peso mínimo", "Peso máximo", "Peso médio"],
        rows,
        {2, 3, 4},
    )


def _report_design(run_index: pd.DataFrame, summary: pd.DataFrame, study_root=None):
    scheme = "cross_validation"
    n_splits = 0
    rounds = "não informado"
    if not summary.empty and "evaluation_scheme" in summary:
        schemes = summary["evaluation_scheme"].dropna().astype(str).unique()
        if len(schemes) == 1:
            scheme = schemes[0]
        if "n_splits" in summary and summary["n_splits"].notna().any():
            n_splits = int(summary["n_splits"].max())
    if not run_index.empty and "resolved_config" in run_index:
        for config_path in run_index["resolved_config"].dropna().astype(str):
            path = Path(config_path)
            if not path.exists():
                continue
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            n_splits = int(config.get("training", {}).get("CV", n_splits))
            rounds = int(config.get("federated", {}).get("rounds", rounds))
            if n_splits == 1:
                scheme = "holdout"
            break
    if study_root is not None and (Path(study_root) / 'config.yaml').is_file():
        config = yaml.safe_load((Path(study_root) / 'config.yaml').read_text())
        n_splits = int(config['training']['CV'])
        rounds = config.get('federated', {}).get('rounds', rounds)
        scheme = 'holdout' if n_splits == 1 else 'cross_validation'
    seeds = sorted(run_index["seed"].dropna().astype(str).unique()) if "seed" in run_index else []
    seed_label = ", ".join(seeds) if seeds else "não informada"
    return scheme, n_splits, rounds, seed_label


def build_report(study_root: Path, analysis_dir: Path, output: Path):
    summary = _read_csv(analysis_dir / "summary_per_task_setup.csv")
    comparisons = _read_csv(analysis_dir / "method_comparisons.csv")
    composition = _read_csv(analysis_dir / "data_composition.csv")
    balance = _read_csv(analysis_dir / "class_balance_audit.csv")
    audit = _read_csv(analysis_dir / "aggregation_audit.csv")
    pooled = _read_csv(analysis_dir / "pooled_auc.csv")
    run_index = _read_csv(study_root / "run_index.csv")
    if run_index.empty:
        run_index = _read_csv(study_root / "execution_plan.csv")
    scheme, n_splits, rounds, seed_label = _report_design(run_index, summary, study_root)
    split_label = "holdout 70/30" if scheme == "holdout" else f"{n_splits} folds"
    observation_label = "clientes no holdout" if scheme == "holdout" else "clientes/folds"
    inference_unit = "seed × cliente no holdout" if scheme == "holdout" else "seed × fold × cliente"
    inference_text = (
        "No holdout único, os contrastes são descritivos e o Wilcoxon não é calculado."
        if scheme == "holdout"
        else "Com uma única seed, testes de Wilcoxon por cliente-fold são exploratórios: clientes e folds não devem ser tratados como repetições totalmente independentes."
    )
    auc_title = "AUC agregada no teste holdout" if scheme == "holdout" else "AUC OOF agrupada"
    auc_text = (
        "A AUC abaixo agrega as predições do teste holdout, mantendo separados os espaços BUSI de 3 classes e ISIC de 7 classes."
        if scheme == "holdout"
        else "A AUC abaixo agrega as predições out-of-fold de todas as imagens, mantendo separados os espaços BUSI de 3 classes e ISIC de 7 classes."
    )
    status_html, all_complete = _status_section(run_index)
    running_html = _running_fold_progress(run_index, n_splits, scheme)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    label = "FINAL" if all_complete else "PROVISÓRIO"

    document = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relatório executivo — Federação multi-dataset</title>
<style>
:root{{--ink:#17202a;--muted:#5f6b76;--paper:#f5f7f9;--card:#fff;--line:#d9e0e6;--fed:#2468d6;--local:#e3982f;--ok:#147d4b;--bad:#b43a3a;--soft:#eaf1fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}} li,code{{overflow-wrap:anywhere;word-break:break-word}}
main{{max-width:1120px;margin:auto;padding:42px 24px 80px}} h1{{font-size:34px;line-height:1.15;margin:.2rem 0 .7rem}} h2{{font-size:23px;margin:2.3rem 0 .8rem;border-bottom:1px solid var(--line);padding-bottom:.35rem}} h3{{font-size:17px;margin:1.5rem 0 .45rem}} p{{max-width:88ch}} .eyebrow{{color:var(--fed);font-weight:700;letter-spacing:.08em;text-transform:uppercase}} .lead{{font-size:18px;color:#374451;max-width:82ch}} .meta{{color:var(--muted);font-size:13px}} .notice{{border-left:5px solid var(--local);background:#fff7e8;padding:14px 18px;margin:22px 0}} .grid2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:18px}} .kpi{{font-size:26px;font-weight:700}} .muted{{color:var(--muted)}} .table-wrap{{overflow-x:auto;margin:12px 0 22px}} table{{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{background:#edf1f5;font-weight:700}} td.num{{text-align:right;font-variant-numeric:tabular-nums}} .status{{display:inline-block;padding:2px 8px;border-radius:999px;background:#e9edf1}} .status-complete,.status-reused{{background:#dff4e9;color:#0d653b}} .status-running{{background:#e5efff;color:#1f58a8}} .status-failed{{background:#fde5e5;color:#972929}} .status-planned,.status-missing{{background:#f1f2f4;color:#5e6872}} .delta-pos{{color:var(--ok);font-weight:700}} .delta-neg{{color:var(--bad);font-weight:700}} .flow{{display:grid;grid-template-columns:repeat(5,1fr);align-items:center;gap:10px;margin:20px 0}} .flow .step{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:13px;text-align:center;min-height:92px;display:grid;place-items:center}} .arrow{{text-align:center;color:var(--muted);font-size:22px}} .chart,.composition,.effect-chart{{width:100%;height:auto;overflow:visible}} .chart .grid,.effect-chart .grid{{stroke:#d9e0e6;stroke-width:1}} .chart .tick,.chart .label,.chart .value,.composition .label,.effect-chart .tick,.effect-chart .label,.effect-chart .value{{fill:#26323d;font:13px Segoe UI,Arial,sans-serif}} .chart .tick,.effect-chart .tick{{fill:#66727d}} .effect-positive{{fill:var(--ok)}} .effect-negative{{fill:var(--bad)}} .bar-fed,.legend-fed{{fill:var(--fed)}} .bar-local,.legend-local{{fill:var(--local)}} .key-findings li{{margin:.65rem 0;max-width:94ch}} code{{background:#edf1f5;padding:2px 5px;border-radius:4px}} .formula{{font:16px Cambria,serif;background:var(--card);padding:12px 16px;border-left:4px solid var(--fed)}} footer{{margin-top:45px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:16px}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}.flow{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}h1{{font-size:28px}}main{{padding:26px 14px 60px}}}}
@media print{{body{{background:#fff}}main{{max-width:none}}.card{{break-inside:avoid}}}}
</style></head><body><main>
<div class="eyebrow">Relatório {label} · estudo controlado</div>
<h1>Federação multi-dataset BUSI + ISIC</h1>
<p class="lead">Avaliação de transferência entre modalidades com stems e cabeças personalizados e trunk compartilhado (<code>encoder2..bottleneck</code>). O desfecho principal compara aprendizado federado e local-only dentro de cada dataset, sem alegar equivalência ao benchmark oficial do ISIC.</p>
<p class="meta">Gerado em {generated} · seed(s) {seed_label} · {split_label} · {rounds} rodadas</p>
    {'' if all_complete else '<div class="notice"><strong>Documento provisório, atualizado durante a execução.</strong> Os resultados abaixo incluem somente braços completos; ablações e folds pendentes não são inferidos nem substituídos por smoke tests.</div>'}

<h2>1. Situação executiva</h2>{status_html}{running_html}

<h2>2. Pergunta de pesquisa e desenho</h2>
<div class="grid2"><div class="card"><h3>Pergunta principal</h3><p>Com o mesmo orçamento local, partições, transforms e pesos de classe, o compartilhamento de um trunk entre clientes BUSI e ISIC melhora o desempenho de teste de cada dataset em relação ao treinamento local?</p></div><div class="card"><h3>Unidade de inferência</h3><p>As comparações são pareadas por {inference_unit}. {inference_text}</p></div></div>
<div class="flow"><div class="step">Partição externa: {split_label}<br><small>lesion_id indivisível no ISIC</small></div><div class="arrow">→</div><div class="step">Clientes por dataset × tarefa<br><small>train/val/test sem vazamento</small></div><div class="arrow">→</div><div class="step">Treino pareado<br><small>federado ou local-only</small></div><div class="arrow">→</div><div class="step">Teste por cliente<br><small>3 e 7 classes separadas</small></div><div class="arrow">→</div><div class="step">Deltas pareados e auditorias</div></div>

<h2>3. Dados e proteção contra vazamento</h2>
<p>BUSI usa ultrassom em um canal; ISIC usa dermatoscopia RGB em três canais. O loader converte BGR→RGB, entrega tensores C×H×W e aplica transforms geométricas de modo conjunto a imagem e máscara. Placeholders de supervisão ausente não são consumidos pela tarefa oposta.</p>
{_composition_chart(composition)}
<p>A diferença de volume é deliberadamente dissociada da contribuição global no braço principal. No ISIC classificação, todas as imagens da mesma lesão permanecem no mesmo fold, cliente e split. Os splits oficiais do challenge são preservados apenas como metadado.</p>

<h2>4. Modelo e federação</h2>
<div class="grid2"><div class="card"><h3>Personalizado</h3><p><strong>Stem</strong> específico para 1 ou 3 canais; decoders de segmentação e cabeças classificadoras específicas para 3 ou 7 classes. Esses parâmetros nunca são agregados entre modalidades.</p></div><div class="card"><h3>Compartilhado</h3><p>O trunk <code>encoder2..bottleneck</code> tem chaves e shapes idênticos entre os modelos. Configurações com <code>share_stem=true</code> e canais heterogêneos falham antes do Flower.</p></div></div>
<h3>Agregação principal: hierárquica e sem cardinalidade do cliente</h3>
<p class="formula">w<sub>i|d</sub> = task_weight<sub>i</sub> / Σ task_weight &nbsp;&nbsp; e &nbsp;&nbsp; w<sub>i</sub> = dataset_weight<sub>d</sub> / Σ dataset_weight × w<sub>i|d</sub></p>
<p>Com pesos iguais de dataset e pesos de tarefa segmentação:classificação = 4:1, a participação esperada é BUSI 50% e ISIC 50%; nos clientes monotarefa, a contribuição se divide conforme os pesos de tarefa. Nos clientes multitarefa, a massa supervisionada de cada tarefa entra no cálculo. Consulte a auditoria para os pesos efetivos.</p>
{_aggregation_audit(audit)}

<h2>5. Desbalanceamento e orçamento</h2>
<p><code>balanced_fold</code> calcula N/(K×n<sub>c</sub>) somente no pool de treino do fold e aplica o mesmo vetor aos clientes ISIC federados e local-only. O protocolo Dice+BCE inclui máscaras vazias válidas. O manifesto atual compara federação, agregação flat e topologia de cliente, mantendo 10 passos por cliente e rodada.</p>
{_class_weight_table(balance)}

<h2>6. Resultados disponíveis</h2>
{_comparison_chart(summary)}
<p class="meta"><span style="color:var(--fed)">■</span> federado principal &nbsp; <span style="color:var(--local)">■</span> local-only. Valores são médias de {observation_label}; consulte a tabela para dispersão e n.</p>
<h3>Maiores médias descritivas por desfecho</h3>
<p class="meta">Ranking descritivo entre protocolos diferentes; não deve ser interpretado como contraste causal.</p>
{_descriptive_rankings(summary)}

<h3>{auc_title}</h3>
<p>{auc_text}</p>
{_pooled_auc_table(pooled)}

<h3>Todas as métricas agregadas</h3><p>Dice/IoU positivos usam somente alvos não vazios; as taxas de falsos positivos usam somente alvos vazios. Dice/IoU globais preservam a convenção histórica. Valores indisponíveis são exibidos como —.</p>
{_results_table(summary)}

<h3>Deltas pareados da comparação principal</h3>
{_primary_deltas(comparisons)}
{_interpretation_notice(all_complete, scheme)}

<h2>7. Efeitos das ablações concluídas</h2>
<p>Nos gráficos e tabelas desta seção, Δ = primeira condição − segunda condição. {inference_text}</p>
{_effects_chart(comparisons)}
<p class="meta">Barras à direita favorecem a primeira condição do contraste; barras à esquerda favorecem a segunda.</p>
{_effects_table(comparisons)}

<h2>8. Conclusões da matriz operacional</h2>
{_key_findings(comparisons)}

<h2>9. Validação e rastreabilidade</h2>
<ul><li>A suíte automatizada cobre shapes compartilhados, canais, placeholders, lesion_id, class weights sem teste, CPU/CUDA, agregação flat/hierárquica, AUC dinâmica, orçamento por passos e retomada por split.</li><li>As métricas retornam valor indefinido somente quando o denominador inexiste e 0 quando há casos sem acerto.</li><li>O run é reprodutível por configs resolvidas, checksums de partição, seed, estado inicial pareado e histórico de agregação por rodada.</li></ul>

<h2>10. Limitações e decisão</h2>
<ul><li>Uma única seed não sustenta uma conclusão inferencial definitiva, independentemente do desenho de avaliação.</li><li>{inference_text}</li><li>O protocolo ISIC é uma avaliação interna; não é resultado oficial do challenge.</li><li>O baseline centralizado multi-dataset permanece fora do escopo.</li><li>A comparação de maiores médias entre braços com orçamentos diferentes é descritiva.</li></ul>
<p><strong>Leitura do protocolo:</strong> avalie Dice e IoU dos alvos não vazios junto da
frequência e área de falsos positivos nos alvos vazios. A classificação não bloqueia a máscara.
Os resultados descrevem os braços concluídos; alegações sobre qualidade precisam de orçamento
adequado e replicação. Smokes comprovam funcionamento, não desempenho científico.</p>

<h2>11. Artefatos</h2>
<ul><li><code>{_esc(analysis_dir / 'summary_per_task_setup.csv')}</code></li><li><code>{_esc(analysis_dir / 'method_comparisons.csv')}</code></li><li><code>{_esc(analysis_dir / 'pooled_auc.csv')}</code></li><li><code>{_esc(analysis_dir / 'aggregation_audit.csv')}</code></li><li><code>{_esc(study_root / 'run_index.csv')}</code></li></ul>
<footer>Documento autogerado a partir dos CSVs persistidos; nenhuma métrica foi transcrita manualmente. Os gráficos usam os mesmos valores das tabelas.</footer>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--analysis", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analysis = args.analysis
    if analysis is None:
        final = args.study_root / "analysis"
        analysis = final if (final / "summary_per_task_setup.csv").exists() else args.study_root / "analysis_partial"
    output = build_report(args.study_root, analysis, args.output)
    print(f"REPORT_OK output={output}")


if __name__ == "__main__":
    main()
