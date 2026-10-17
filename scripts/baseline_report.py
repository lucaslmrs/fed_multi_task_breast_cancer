"""
Static HTML report: non-federated baselines (plano.md, B0) against federated runs.

    python -m scripts.baseline_report --baseline-root runs/baseline \
        --federated "fed seed/client=<run>" --federated "fed seed/dataset=<run>" \
        --reference "29/09=<run>" --out runs/baseline/report.html

The latest run of every (mode, dataset) under ``--baseline-root`` is used. Classification is
detailed with confusion matrices pooled over every client test slice of a dataset; segmentation
with a pixel confusion matrix (row-normalised, from the global sensitivity/specificity of each
client weighted by its test images) and the false-positive rate on empty targets.
One seed and one holdout: every number is descriptive.
"""

import argparse
import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plot_style import (ACCENT, CATEGORICAL_8, EMBED_DPI, MARKERS, PLOT_RC, colormap,
                                  style_axis)

DATASETS = ["Curated_BUSI", "ISIC_2018", "SIIM_ACR", "TCGA_LGG"]
SHORT = {"Curated_BUSI": "BUSI", "ISIC_2018": "ISIC", "SIIM_ACR": "SIIM", "TCGA_LGG": "TCGA"}
MODES = {"segmentation": "baseline seg", "classification": "baseline cls",
         "multitask": "baseline multitask"}
SEQUENTIAL = colormap("sequential")


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=EMBED_DPI, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _labelled(values):
    out = {}
    for value in values or []:
        label, path = value.split("=", 1)
        out[label] = Path(path)
    return out


def latest_baselines(root):
    """{(mode, dataset): run_dir} with the newest completed run of each pair."""
    found = {}
    for mode in MODES:
        for run in sorted((Path(root) / mode).glob("*_2*")):
            if (run / "baseline_test_results.csv").exists():
                dataset = run.name.rsplit("_", 2)[0]
                found[(mode, dataset)] = run
    return found


def load_setups(baselines, federated, references):
    """One results frame and one predictions frame per setup label."""
    results, predictions = {}, {}
    for mode, label in MODES.items():
        runs = [run for (m, _), run in baselines.items() if m == mode]
        if runs:
            results[label] = pd.concat(pd.read_csv(r / "baseline_test_results.csv") for r in runs)
            preds = [r / "baseline_cls_predictions.csv" for r in runs
                     if (r / "baseline_cls_predictions.csv").exists()]
            if preds:
                predictions[label] = pd.concat(pd.read_csv(p) for p in preds)
    for label, run in {**references, **federated}.items():
        results[label] = pd.read_csv(run / "federated_test_results.csv")
        if label in federated and (run / "federated_cls_predictions.csv").exists():
            predictions[label] = pd.read_csv(run / "federated_cls_predictions.csv")
    return results, predictions


def summary_table(results):
    rows = []
    for label, frame in results.items():
        for dataset in DATASETS:
            cls = frame[(frame.dataset == dataset) & (frame.task == "cls")]
            seg = frame[(frame.dataset == dataset) & (frame.task == "seg")]
            row = {"setup": label, "dataset": SHORT[dataset]}
            for column in ("balanced_acc", "macro_f1", "auc", "acc"):
                row[column] = cls[column].mean() if len(cls) and column in cls else np.nan
            for column in ("dice_positive", "iou_positive", "empty_fp_image_rate"):
                row[column] = seg[column].mean() if len(seg) and column in seg else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def table_html(summary, noise):
    metrics = [("balanced_acc", "cls: balanced acc", True), ("macro_f1", "cls: macro-F1", True),
               ("auc", "cls: AUC", True), ("dice_positive", "seg: Dice positivo", True),
               ("empty_fp_image_rate", "seg: FP em imagem vazia", False)]
    parts = []
    for column, title, higher in metrics:
        pivot = summary.pivot(index="setup", columns="dataset", values=column)
        pivot = pivot.reindex(columns=[SHORT[d] for d in DATASETS])
        pivot = pivot.reindex([s for s in summary.setup.unique() if s in pivot.index])
        best = pivot.max() if higher else pivot.min()
        head = "".join(f"<th>{c}</th>" for c in pivot.columns)
        body = []
        for setup, values in pivot.iterrows():
            cells = []
            for dataset, value in values.items():
                if pd.isna(value):
                    cells.append("<td class='na'>—</td>")
                    continue
                mark = " class='best'" if np.isclose(value, best[dataset]) else ""
                cells.append(f"<td{mark}>{value:.3f}</td>")
            body.append(f"<tr><th>{html.escape(setup)}</th>{''.join(cells)}</tr>")
        band = noise.get(column)
        note = ""
        if band is not None:
            note = "<tr class='noise'><th>ruído entre runs</th>" + "".join(
                f"<td>±{band.get(c, np.nan):.3f}</td>" if pd.notna(band.get(c, np.nan))
                else "<td>—</td>" for c in pivot.columns) + "</tr>"
        parts.append(f"<h3>{html.escape(title)}</h3><table><tr><th></th>{head}</tr>"
                     f"{''.join(body)}{note}</table>")
    return "\n".join(parts)


def fig_metrics(summary, noise, setups):
    panels = [("balanced_acc", "balanced accuracy (cls)"), ("macro_f1", "macro-F1 (cls)"),
              ("dice_positive", "Dice positivo (seg)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    colors = {setup: CATEGORICAL_8[i % len(CATEGORICAL_8)] for i, setup in enumerate(setups)}
    for ax, (column, label) in zip(axes, panels):
        for index, setup in enumerate(setups):
            values = summary[summary.setup == setup].set_index("dataset")[column] \
                .reindex([SHORT[d] for d in DATASETS])
            offset = (index - (len(setups) - 1) / 2) * 0.11
            ax.scatter(np.arange(len(DATASETS)) + offset, values.values, s=45,
                       marker=MARKERS[index % len(MARKERS)], color=colors[setup], label=setup,
                       zorder=3)
        ax.set_xticks(range(len(DATASETS)), [SHORT[d] for d in DATASETS])
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        style_axis(ax)
    axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


def confusion(frame, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, predicted in zip(frame.ground_truth.astype(int), frame.predicted.astype(int)):
        matrix[truth, predicted] += 1
    return matrix


def fig_confusions(predictions, dataset, setups):
    available = [s for s in setups if s in predictions
                 and (predictions[s].dataset == dataset).any()]
    if not available:
        return None
    sample = predictions[available[0]]
    sample = sample[sample.dataset == dataset]
    names_cols = sorted([c for c in sample.columns if c.startswith("class_name_")],
                        key=lambda c: int(c.rsplit("_", 1)[1]))
    names = [str(sample[c].dropna().iloc[0]) for c in names_cols if sample[c].notna().any()]
    k = len(names)
    size = 2.6 + 0.45 * k
    fig, axes = plt.subplots(1, len(available), figsize=(size * len(available), size),
                             constrained_layout=True, squeeze=False)
    for ax, setup in zip(axes[0], available):
        frame = predictions[setup]
        matrix = confusion(frame[frame.dataset == dataset], k)
        rows = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, rows, out=np.zeros_like(matrix, dtype=float), where=rows > 0)
        ax.imshow(normalized, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{normalized[i, j]:.2f}\n({matrix[i, j]})", ha="center",
                        va="center", fontsize=7 if k > 3 else 8,
                        color="white" if normalized[i, j] > 0.55 else "#1E1216")
        recall = np.diag(normalized)
        ax.set_title(f"{setup}\nbalanced acc = {recall[rows[:, 0] > 0].mean():.3f}", fontsize=9)
        tilt = k > 3 or max(len(name) for name in names) > 8
        ax.set_xticks(range(k), names, rotation=30 if tilt else 0, ha="right" if tilt else "center")
        ax.set_yticks(range(k), names)
        ax.set_xlabel("predito")
        ax.set_ylabel("verdadeiro")
    return fig


def pixel_confusion(results, dataset):
    seg = results[(results.dataset == dataset) & (results.task == "seg")]
    if seg.empty or not {"sensitivity", "specificity"} <= set(seg.columns):
        return None
    weights = seg["n_test"] if "n_test" in seg else pd.Series(1.0, index=seg.index)
    tpr = float(np.average(seg["sensitivity"].fillna(0), weights=weights))
    tnr = float(np.average(seg["specificity"].fillna(0), weights=weights))
    return np.array([[tnr, 1 - tnr], [1 - tpr, tpr]])


def fig_pixel_confusions(results, setups):
    seg_setups = [s for s in setups if s in results
                  and (results[s].task == "seg").any()]
    fig, axes = plt.subplots(len(DATASETS), len(seg_setups),
                             figsize=(2.4 * len(seg_setups), 2.5 * len(DATASETS)),
                             constrained_layout=True, squeeze=False)
    for row, dataset in enumerate(DATASETS):
        for column, setup in enumerate(seg_setups):
            ax = axes[row][column]
            matrix = pixel_confusion(results[setup], dataset)
            if matrix is None:
                ax.axis("off")
                continue
            ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if matrix[i, j] > 0.55 else "#1E1216")
            ax.set_xticks([0, 1], ["fundo", "lesão"])
            ax.set_yticks([0, 1], ["fundo", "lesão"])
            if row == 0:
                ax.set_title(setup, fontsize=9)
            if column == 0:
                ax.set_ylabel(f"{SHORT[dataset]}\nverdadeiro")
            if row == len(DATASETS) - 1:
                ax.set_xlabel("predito")
    return fig


def fig_curves(baselines):
    fig, axes = plt.subplots(len(MODES), len(DATASETS), figsize=(15, 8.4),
                             constrained_layout=True, squeeze=False)
    for row, mode in enumerate(MODES):
        for column, dataset in enumerate(DATASETS):
            ax = axes[row][column]
            run = baselines.get((mode, dataset))
            if run is None or not (run / "history.csv").exists():
                ax.axis("off")
                continue
            history = pd.read_csv(run / "history.csv")
            ax.plot(history.epoch, history.train_loss, color=CATEGORICAL_8[0], label="treino")
            ax.plot(history.epoch, history.val_loss, color=ACCENT, label="validação")
            best = history[history.best].epoch.max()
            ax.axvline(best, color="#6B5559", linestyle="--", linewidth=0.9)
            capped = len(history) >= 200 and len(history) - best < 51
            ax.set_title(f"{MODES[mode]} — {SHORT[dataset]}\nmelhor época {best}"
                         f"{' · teto de épocas' if capped else ''}", fontsize=9)
            ax.set_xlabel("época")
            style_axis(ax)
    axes[0][0].legend(fontsize=8)
    return fig


def noise_band(results, references, federated):
    """|reference - same-config federated run| per dataset and metric."""
    if not references or not federated:
        return {}
    reference = summary_table({"r": results[next(iter(references))]}).set_index("dataset")
    same = summary_table({"f": results[next(iter(federated))]}).set_index("dataset")
    return {column: (reference[column] - same[column]).abs().to_dict()
            for column in ("balanced_acc", "macro_f1", "auc", "dice_positive",
                           "empty_fp_image_rate")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline-root", default="runs/baseline")
    parser.add_argument("--federated", action="append", default=[], help="label=run_dir")
    parser.add_argument("--reference", action="append", default=[],
                        help="label=run_dir with the same config as the FIRST --federated run")
    parser.add_argument("--out", default="runs/baseline/report.html")
    args = parser.parse_args()

    baselines = latest_baselines(args.baseline_root)
    federated, references = _labelled(args.federated), _labelled(args.reference)
    results, predictions = load_setups(baselines, federated, references)
    setups = [*[MODES[m] for m in MODES if MODES[m] in results], *federated, *references]
    summary = summary_table({s: results[s] for s in setups})
    noise = noise_band(results, references, federated)
    summary.to_csv(Path(args.out).with_name("report_summary.csv"), index=False)

    sections = []
    with plt.rc_context(PLOT_RC):
        sections.append(("Métricas por dataset", table_html(summary, noise),
                         _png(fig_metrics(summary, noise, setups))))
        for dataset in DATASETS:
            figure = fig_confusions(predictions, dataset,
                                    [s for s in setups if s not in references])
            if figure is not None:
                sections.append((f"Matrizes de confusão — classificação {SHORT[dataset]}",
                                 "<p>Todas as fatias de teste do dataset agrupadas. Cada célula: "
                                 "fração da linha (classe verdadeira) e, entre parênteses, a "
                                 "contagem. A diagonal é o recall por classe; a média dela é a "
                                 "balanced accuracy.</p>", _png(figure)))
        sections.append(("Matrizes de confusão — segmentação por pixel",
                         "<p>Normalizada por linha a partir da sensibilidade e especificidade "
                         "globais de cada cliente, média ponderada pelo nº de imagens de teste. "
                         "Linha 'lesão' = sensibilidade; linha 'fundo' = especificidade. Como o "
                         "fundo domina os pixels, a especificidade é alta em todos; compare a "
                         "linha 'lesão'.</p>", _png(fig_pixel_confusions(results, setups))))
        sections.append(("Curvas de treino dos baselines",
                         "<p>Loss por época. Tracejado: melhor época (checkpoint usado no "
                         "teste). 'teto de épocas' indica que o treino chegou às 200 épocas "
                         "ainda melhorando, logo o baseline pode estar subestimado.</p>",
                         _png(fig_curves(baselines))))

    runs_list = "".join(
        f"<li><b>{html.escape(MODES[m])} — {SHORT.get(d, d)}</b>: <code>{html.escape(str(r))}</code></li>"
        for (m, d), r in sorted(baselines.items()))
    runs_list += "".join(f"<li><b>{html.escape(k)}</b>: <code>{html.escape(str(v))}</code></li>"
                         for k, v in {**federated, **references}.items())
    body = "\n".join(f"<section><h2>{html.escape(t)}</h2>{text}"
                     f"<img src='data:image/png;base64,{img}' alt='{html.escape(t)}'></section>"
                     for t, text, img in sections)
    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Baselines vs federado</title>
<style>body{{font-family:system-ui,sans-serif;color:#1E1216;max-width:1200px;margin:24px auto;
padding:0 16px;background:#FFF7F1}}h1,h2{{color:#7B2032}}h2{{font-size:1.15em;margin-top:34px}}
h3{{font-size:.95em;margin:18px 0 6px}}img{{max-width:100%;border:1px solid #E6D6D1}}
table{{border-collapse:collapse;font-size:.85em;margin-bottom:8px}}th,td{{padding:4px 10px;
border-bottom:1px solid #E6D6D1;text-align:right}}th:first-child{{text-align:left}}
td.best{{font-weight:700;color:#7B2032}}tr.noise td,tr.noise th{{color:#6B5559;font-style:italic}}
td.na{{color:#9B8385}}code{{font-size:.8em}}.note{{background:#F3E6E1;border-left:4px solid
#7B2032;padding:8px 12px}}</style></head><body>
<h1>Baselines não federados vs federado</h1>
<div class="note"><p><b>Como ler.</b> Todos os modelos são MTnnUNet avaliados nas mesmas fatias de
teste da partição congelada (holdout 70/30), com as mesmas métricas. Baselines: treino centralizado
com todos os clientes de um dataset, early stopping pela validação (até 200 épocas). Federado: 16
clientes, 200 rodadas fixas. <b>Uma seed e um holdout</b>: diferenças dentro da linha "ruído entre
runs" (dois runs federados com a mesma configuração) não devem ser lidas como efeito. Negrito =
melhor valor da coluna.</p></div>
{body}
<h2>Runs</h2><ul>{runs_list}</ul></body></html>"""
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
