"""Per-run HTML report: overall, per-dataset and per-client metrics plus hit/miss examples.

Reads only durable artifacts of one run directory (``{setup}_test_results.csv``,
``{setup}_cls_predictions.csv``, ``fold_*/{setup}_example_candidates.csv``, ``config.yaml``), so
it can be rebuilt at any time with ``python -m src.experiments.run_report <run_dir>``.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.experiments.run_examples import (
    CANDIDATES_FILE,
    PER_QUADRANT,
    QUADRANT_LABELS,
    QUADRANT_ORDER,
    rank_candidates,
)

REPORT_FILE = "run_report.html"
SEG_METRICS = {
    "dice_positive": "Dice (alvos com lesão)",
    "iou_positive": "IoU (alvos com lesão)",
    "empty_fp_image_rate": "Falso positivo em imagem vazia ↓",
    "dice": "Dice global",
}
CLS_METRICS = {
    "acc": "Acurácia",
    "balanced_acc": "Acurácia balanceada",
    "macro_f1": "Macro-F1",
    "auc": "AUC OvR macro",
}
TASK_METRICS = {"seg": SEG_METRICS, "cls": CLS_METRICS}
TASK_LABELS = {"seg": "Segmentação", "cls": "Classificação"}

STYLE = """
body{font-family:system-ui,Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#584A47;
background:#fff;padding:0 16px}
h1{border-bottom:3px solid #08519C} h2{color:#08519C;margin-top:2.2rem} h3{color:#584A47}
table{border-collapse:collapse;font-size:.82rem;margin:.6rem 0} th,td{border:1px solid #CABD91;
padding:4px 8px;text-align:left} th{background:#EFF3FF} td.num{text-align:right;
font-variant-numeric:tabular-nums} a{color:#08519C}
.notice{padding:.8rem 1rem;border-left:5px solid #9F2B2C;background:#FCFAE1;margin:1rem 0}
.meta{color:#6F6357;font-size:.9rem} .scroll{overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:1rem}
.card{border:1px solid #CABD91;border-radius:7px;padding:.6rem;background:#fff}
.card img{width:100%;height:auto} .card p{margin:.25rem 0;font-size:.82rem}
.ok{color:#08519C;font-weight:600} .err{color:#9F2B2C;font-weight:600}
"""


def _esc(value):
    return html.escape("" if value is None else str(value))


def _fmt(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _table(headers, rows, numeric=()):
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td class='num'>{_esc(cell)}</td>" if index in numeric else f"<td>{_esc(cell)}</td>"
            for index, cell in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _mean_std(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "—"
    if len(values) == 1:
        return _fmt(values.iloc[0])
    return f"{values.mean():.4f} ± {values.std():.4f}"


def _setup(run_path):
    for setup in ("federated", "standalone", "centralized"):
        if (run_path / f"{setup}_test_results.csv").exists():
            return setup
    raise FileNotFoundError(f"No *_test_results.csv in {run_path}")


def _read_optional(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _class_floors(predictions):
    """Majority-class accuracy and chance balanced accuracy on each dataset's test rows."""
    floors = {}
    if predictions.empty:
        return floors
    for dataset, group in predictions.groupby("dataset"):
        shares = group["ground_truth"].value_counts(normalize=True)
        names = group.filter(like="class_name_").iloc[0].dropna()
        majority = int(shares.idxmax())
        floors[dataset] = {
            "acc": float(shares.max()),
            "balanced_acc": 1.0 / (len(names) or len(shares)),
            "majority": names.get(f"class_name_{majority}", majority),
        }
    return floors


def _header(run_path, setup, config, results):
    fed = (config or {}).get("federated", {})
    scheme = results["evaluation_scheme"].iloc[0] if "evaluation_scheme" in results else "—"
    rows = [
        ("Run", run_path.name),
        ("Setup", setup),
        ("Datasets", ", ".join(sorted(results["dataset"].unique()))),
        ("Clientes", results["client_id"].nunique()),
        ("Folds", results["fold"].nunique()),
        ("Rodadas", fed.get("rounds", "—")),
        ("Avaliação", scheme),
        ("Orçamento local", (fed.get("local_training") or {}).get("mode", "—")),
    ]
    return _table(["Campo", "Valor"], rows)


def _overall_section(results):
    parts = []
    for task in ("seg", "cls"):
        subset = results[results["task"] == task]
        if subset.empty:
            continue
        metrics = [m for m in TASK_METRICS[task] if m in subset]
        # Mean of dataset means: every dataset counts once, whatever its client count.
        dataset_means = subset.groupby("dataset")[metrics].mean(numeric_only=True)
        rows = [(TASK_METRICS[task][m], _fmt(dataset_means[m].mean()), len(dataset_means))
                for m in metrics]
        parts.append(f"<h3>{TASK_LABELS[task]}</h3>" + _table(
            ["Métrica", "Média das médias por dataset", "Datasets"], rows, numeric=(1, 2)
        ))
    return "".join(parts)


def _dataset_section(results, floors):
    parts = []
    for task in ("seg", "cls"):
        subset = results[results["task"] == task]
        if subset.empty:
            continue
        metrics = [m for m in TASK_METRICS[task] if m in subset]
        headers = ["Dataset", "Observações"] + [TASK_METRICS[task][m] for m in metrics]
        rows = []
        for dataset, group in subset.groupby("dataset"):
            rows.append([dataset, len(group)] + [_mean_std(group[m]) for m in metrics])
        parts.append(f"<h3>{TASK_LABELS[task]}</h3>" + _table(
            headers, rows, numeric=tuple(range(1, len(headers)))
        ))
        if task == "cls" and floors:
            floor_rows = [(d, f["majority"], _fmt(f["acc"]), _fmt(f["balanced_acc"]))
                          for d, f in sorted(floors.items())]
            parts.append(
                "<p class='meta'>Pisos de referência no conjunto de teste: acurácia de um "
                "preditor constante na classe majoritária e acurácia balanceada do acaso.</p>"
                + _table(["Dataset", "Classe majoritária", "Acurácia (constante)",
                          "Acurácia balanceada (acaso)"], floor_rows, numeric=(2, 3))
            )
    parts.append(
        "<p class='meta'>Média ± desvio entre observações cliente × fold. Essas observações não "
        "são independentes; os valores são descritivos.</p>"
    )
    return "".join(parts)


def _client_section(results):
    parts = []
    for task in ("seg", "cls"):
        subset = results[results["task"] == task].sort_values(["dataset", "client_id", "fold"])
        if subset.empty:
            continue
        metrics = [m for m in TASK_METRICS[task] if m in subset]
        headers = ["Dataset", "Cliente", "Fold", "n teste"] + [TASK_METRICS[task][m] for m in metrics]
        rows = [
            [r["dataset"], r["client_id"], r["fold"], _fmt(r.get("n_test"))]
            + [_fmt(r[m]) for m in metrics]
            for r in subset.to_dict("records")
        ]
        parts.append(f"<h3>{TASK_LABELS[task]}</h3>" + _table(
            headers, rows, numeric=tuple(range(2, len(headers)))
        ))
    return "".join(parts)


def _load_candidates(run_path, setup, folds):
    frames, missing = [], []
    for fold in folds:
        frame = _read_optional(run_path / f"fold_{fold}" / CANDIDATES_FILE.format(setup=setup))
        if frame.empty:
            missing.append(fold)
        else:
            frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), missing


def _card(record):
    lines = [f"<p><b>Cliente:</b> {_esc(record['client_id'])} · <b>fold</b> {_esc(record['fold'])}"
             f" · <b>id</b> {_esc(record['patient_id'])}</p>"]
    if pd.notna(record.get("cls_ok")):
        ok = bool(record["cls_ok"])
        lines.append(
            f"<p><b>Classificação:</b> <span class='{'ok' if ok else 'err'}'>"
            f"{'✔ correta' if ok else '✘ errada'}</span> — real <b>{_esc(record['true_class'])}</b>,"
            f" prevista <b>{_esc(record['predicted_class'])}</b> "
            f"(confiança {_fmt(record['confidence'], 3)})</p>"
        )
    if pd.notna(record.get("seg_ok")):
        ok = bool(record["seg_ok"])
        if bool(record["target_empty"]):
            detail = ("alvo vazio, nada previsto" if int(record["pred_pixels"]) == 0 else
                      f"alvo vazio, {int(record['pred_pixels'])} px previstos (falso positivo)")
        else:
            detail = (f"Dice {_fmt(record['dice_positive'], 3)} · lesão "
                      f"{int(record['target_pixels'])} px, previstos {int(record['pred_pixels'])} px")
        lines.append(
            f"<p><b>Segmentação:</b> <span class='{'ok' if ok else 'err'}'>"
            f"{'✔ boa' if ok else '✘ ruim'}</span> — {detail}</p>"
        )
    return (f"<div class='card'><img loading='lazy' alt='exemplo {_esc(record['patient_id'])}' "
            f"src='{_esc(record['image_path'])}'>{''.join(lines)}</div>")


def _examples_section(candidates, missing):
    parts = [
        "<p class='meta'>Por dataset, até "
        f"{PER_QUADRANT} exemplos por quadrante, escolhidos de forma determinística: segmentações "
        "boas com maior Dice e ruins com menor Dice; depois a predição mais confiante (um erro "
        "confiante é o mais instrutivo). Segmentação boa = Dice ≥ 0,5 em alvo com lesão, ou nenhum "
        "pixel previsto em alvo vazio. Sobreposição: azul = acerto, vermelho = falso positivo, "
        "bege = lesão não detectada.</p>"
    ]
    if missing:
        parts.append(f"<div class='notice'>Exemplos indisponíveis para o(s) fold(s) "
                     f"{', '.join(map(str, missing))} (run anterior a este relatório, ou fold "
                     "reaproveitado de uma execução interrompida).</div>")
    if candidates.empty:
        return "".join(parts)
    for dataset, group in candidates.groupby("dataset", sort=True):
        parts.append(f"<h3>{_esc(dataset)}</h3>")
        chosen = rank_candidates(group, PER_QUADRANT)
        for quadrant in QUADRANT_ORDER:
            rows = chosen[chosen["quadrant"] == quadrant]
            if rows.empty:
                continue
            parts.append(f"<h4>{_esc(QUADRANT_LABELS[quadrant])}</h4><div class='grid'>"
                         + "".join(_card(r) for r in rows.to_dict("records")) + "</div>")
    return "".join(parts)


def build_run_report(run_path):
    run_path = Path(run_path)
    setup = _setup(run_path)
    results = pd.read_csv(run_path / f"{setup}_test_results.csv", low_memory=False)
    predictions = _read_optional(run_path / f"{setup}_cls_predictions.csv")
    config_path = run_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    candidates, missing = _load_candidates(run_path, setup, sorted(results["fold"].unique()))

    links = []
    if (run_path / "training_curves" / "dashboard.html").exists():
        links.append("<a href='training_curves/dashboard.html'>Curvas de treinamento</a>")
    if setup == "federated" and (run_path / "negative_transfer.html").exists():
        links.append("<a href='negative_transfer.html'>Transferência negativa</a>")

    body = (
        f"<h1>Relatório da run</h1><p class='meta'>{' · '.join(links)}</p>"
        + _header(run_path, setup, config, results)
        + "<h2>Métricas gerais</h2>" + _overall_section(results)
        + "<h2>Métricas por dataset</h2>" + _dataset_section(results, _class_floors(predictions))
        + "<h2>Métricas por cliente</h2>" + _client_section(results)
        + "<h2>Exemplos de acertos e erros</h2>" + _examples_section(candidates, missing)
    )
    document = (
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Relatório da run</title><style>{STYLE}</style></head><body>{body}</body></html>"
    )
    output = run_path / REPORT_FILE
    output.write_text(document, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path")
    args = parser.parse_args()
    print(build_run_report(args.run_path))


if __name__ == "__main__":
    main()
