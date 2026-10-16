"""
Static HTML report comparing two federated runs' gradient-conflict diagnostics (plano.md, E0).

    python -m scripts.gradient_conflict_report \
        --run client=<run_dir> --run dataset=<run_dir> \
        [--reference 29/09=<run_dir>] [--null client=<null_dir> --null dataset=<null_dir>] \
        --out <report.html>

Every figure is embedded as a PNG, so the HTML is a single self-contained file. Single seed and a
single holdout: every number is descriptive.
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.utils.plot_style import ACCENT_RED, CATEGORICAL_8, EMBED_DPI, MARKERS, PLOT_RC, style_axis

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#08519C", "#6BAED6", "#FCFAE1", "#CABD91", "#9F2B2C"])
DATASETS = ["Curated_BUSI", "ISIC_2018", "SIIM_ACR", "TCGA_LGG"]
SHORT = {"Curated_BUSI": "BUSI", "ISIC_2018": "ISIC", "SIIM_ACR": "SIIM", "TCGA_LGG": "TCGA"}


def _pairs(values):
    out = {}
    for value in values or []:
        name, path = value.split("=", 1)
        out[name] = Path(path)
    return out


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=EMBED_DPI, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def fig_m1_trajectory(m1s):
    fig, axes = plt.subplots(1, len(m1s), figsize=(5.2 * len(m1s), 3.6), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, (name, m1) in zip(axes[0], m1s.items()):
        total = m1[m1.block == "shared_total"]
        for index, dataset in enumerate(DATASETS):
            series = total[(total.pair_class == "intra_same_tasks") & (total.dataset_a == dataset)]
            series = series.groupby("round").cos.mean()
            if len(series):
                ax.plot(series.index, series.values, marker=MARKERS[index], markersize=3.5,
                        color=CATEGORICAL_8[index], label=SHORT[dataset])
        inter = total[total.pair_class == "inter"].groupby("round").cos.mean()
        ax.plot(inter.index, inter.values, color="#6F6357", linestyle="--", label="entre datasets")
        ax.axhline(0, color="#CABD91", linewidth=0.8)
        ax.set_title(f"seed por {name}")
        ax.set_xlabel("rodada")
        style_axis(ax)
    axes[0][0].set_ylabel("cosseno do gradiente (média dos pares)")
    axes[0][-1].legend(loc="upper right", fontsize=8)
    return fig


def fig_magnitude(m1s):
    classes = ["intra_same_tasks", "intra_diff_tasks", "inter"]
    labels = ["intra, mesmas tarefas", "intra, tarefas diferentes", "entre datasets"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    for offset, (index, (name, m1)) in zip((-0.12, 0.12), enumerate(m1s.items())):
        total = m1[m1.block == "shared_total"]
        per_round = total.groupby(["pair_class", "round"]).magnitude_similarity.mean()
        for position, cls in enumerate(classes):
            values = per_round.loc[cls].values if cls in per_round.index.get_level_values(0) else []
            ax.scatter(np.full(len(values), position + offset), values, s=10, alpha=0.35,
                       color=CATEGORICAL_8[index])
            ax.scatter([position + offset], [np.mean(values)], s=60, marker=MARKERS[index],
                       color=CATEGORICAL_8[index], edgecolor="white", zorder=3,
                       label=f"seed por {name}" if position == 0 else None)
    ax.set_xticks(range(len(classes)), labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("similaridade de magnitude Φ")
    ax.legend(fontsize=8, loc="lower left")
    style_axis(ax)
    return fig


def fig_m2_heatmaps(m2s):
    fig, axes = plt.subplots(2, len(m2s), figsize=(4.6 * len(m2s), 7.4), constrained_layout=True,
                             squeeze=False)
    matrices = {}
    for name, m2 in m2s.items():
        cross = m2[~m2.pair_class.isin(["self", "aggregate"])]
        per_round = cross.groupby(["source_dataset", "target_dataset", "split", "round"]).z.mean()
        mean = per_round.groupby(["source_dataset", "target_dataset", "split"]).mean() * 1e4
        for split in ("train", "val"):
            matrices[(name, split)] = mean.xs(split, level="split").unstack("target_dataset") \
                .reindex(index=DATASETS, columns=DATASETS)
    limit = max(np.nanmax(np.abs(m.values)) for m in matrices.values())
    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    for column, name in enumerate(m2s):
        for row, split in enumerate(("train", "val")):
            ax = axes[row][column]
            matrix = matrices[(name, split)]
            image = ax.imshow(matrix.values, cmap=DIVERGING, norm=norm)
            for i in range(len(DATASETS)):
                for j in range(len(DATASETS)):
                    value = matrix.values[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                                color="white" if abs(value) > 0.55 * limit else "#584A47")
            ax.set_xticks(range(len(DATASETS)), [SHORT[d] for d in DATASETS])
            ax.set_yticks(range(len(DATASETS)), [SHORT[d] for d in DATASETS])
            ax.set_xlabel("alvo (loss avaliada)")
            ax.set_ylabel("origem (update aplicado)")
            ax.set_title(f"seed por {name} — {split}")
    fig.colorbar(image, ax=axes, shrink=0.6, label="Z × 1e-4 (média sobre rodadas; < 0 = piora o alvo)")
    return fig


def fig_null(nulls):
    variants = ["delta", "random", "delta_scaled", "random_scaled"]
    labels = ["Δ real", "aleatório\nmesma norma", "Δ × wᵢ", "aleatório × wᵢ"]
    panels = [("inter_shared_task", "entre datasets"), ("self", "próprio cliente")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    width = 0.38
    for ax, (cls, title) in zip(axes, panels):
        for index, (name, null) in enumerate(nulls.items()):
            values = null[(null.pair_class == cls) & (null.split == "train")] \
                .groupby("variant").z.mean().reindex(variants) * 1e4
            positions = np.arange(len(variants)) + (index - 0.5) * width
            ax.bar(positions, values.values, width, color=CATEGORICAL_8[index],
                   label=f"seed por {name}")
        ax.axhline(0, color="#584A47", linewidth=0.8)
        ax.set_xticks(range(len(variants)), labels)
        ax.set_title(f"{title} — split de treino")
        ax.set_ylabel("Z × 1e-4")
        style_axis(ax)
    axes[0].legend(fontsize=8)
    return fig


def fig_performance(results, reference_name):
    metrics = [("cls", "balanced_acc", "balanced accuracy (cls)"),
               ("seg", "dice_positive", "Dice positivo (seg)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    names = list(results)
    for ax, (task, column, label) in zip(axes, metrics):
        means = {name: frame[frame.task == task].groupby("dataset")[column].mean()
                 .reindex(DATASETS) for name, frame in results.items()}
        if reference_name and reference_name in means and "client" in means:
            low = np.minimum(means[reference_name], means["client"])
            high = np.maximum(means[reference_name], means["client"])
            for position, (lo, hi) in enumerate(zip(low, high)):
                ax.add_patch(plt.Rectangle((position - 0.3, lo), 0.6, max(hi - lo, 1e-3),
                                           color="#EDE2B5", zorder=0))
        for index, name in enumerate(names):
            color = ACCENT_RED if name == "dataset" else CATEGORICAL_8[0 if name == "client" else 2]
            ax.scatter(np.arange(len(DATASETS)) + (index - 1) * 0.12, means[name].values,
                       marker=MARKERS[index], s=55, color=color, zorder=3, label=name)
        ax.set_xticks(range(len(DATASETS)), [SHORT[d] for d in DATASETS])
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        style_axis(ax)
    axes[0].legend(fontsize=8, title="run", title_fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, help="label=run_dir")
    parser.add_argument("--reference", action="append", help="label=run_dir (noise reference)")
    parser.add_argument("--null", action="append", help="label=null_control_dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs, references, nulls = _pairs(args.run), _pairs(args.reference), _pairs(args.null)

    m1s = {name: pd.read_csv(path / "conflict_m1_pairs.csv") for name, path in runs.items()}
    m2s = {name: pd.read_csv(path / "conflict_m2_lookahead.csv") for name, path in runs.items()}
    results = {name: pd.read_csv(path / "federated_test_results.csv")
               for name, path in {**references, **runs}.items()}
    null_frames = {name: pd.read_csv(path / "m2_null_control.csv") for name, path in nulls.items()}

    sections = []
    with plt.rc_context(PLOT_RC):
        sections.append(("M1 — cosseno do gradiente bruto dentro de cada dataset",
                         "Média dos pares de clientes com o mesmo dataset e as mesmas tarefas, por "
                         "rodada amostrada. Tracejado: média dos pares entre datasets.",
                         _png(fig_m1_trajectory(m1s))))
        sections.append(("M1 — similaridade de magnitude Φ",
                         "Pontos claros: média de cada rodada amostrada; marcador grande: média das "
                         "rodadas. Φ = 1 indica normas iguais; valores baixos indicam que um lado "
                         "domina o update.", _png(fig_magnitude(m1s))))
        sections.append(("M2 — afinidade lookahead dataset → dataset",
                         "Z = 1 − L_alvo(θ + Δ_origem) / L_alvo(θ), com o update real inteiro, só "
                         "entre clientes distintos. Média sobre as rodadas amostradas; escala "
                         "simétrica centrada em 0.", _png(fig_m2_heatmaps(m2s))))
        if null_frames:
            sections.append(("M2 — controle nulo de curvatura (estado final)",
                             "O update real é comparado a uma direção aleatória de mesma norma e às "
                             "duas versões escaladas pelo peso real de agregação wᵢ. Note as escalas "
                             "diferentes dos dois painéis.", _png(fig_null(null_frames))))
        reference_name = next(iter(references), None)
        sections.append(("Desempenho de teste por dataset",
                         "Faixa bege: diferença entre dois runs com a mesma configuração "
                         f"({reference_name} e client), isto é, o ruído GPU entre runs. "
                         "Uma seed, um holdout.", _png(fig_performance(results, reference_name))))

    body = "\n".join(
        f"<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>"
        f"<img src='data:image/png;base64,{image}' alt='{html.escape(title)}'></section>"
        for title, caption, image in sections)
    sources = "".join(f"<li><b>{html.escape(k)}</b>: <code>{html.escape(str(v))}</code></li>"
                      for k, v in {**references, **runs, **nulls}.items())
    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conflito de gradientes — E0</title>
<style>body{{font-family:system-ui,sans-serif;color:#584A47;max-width:1100px;margin:24px auto;
padding:0 16px;background:#fff}}h1{{color:#08519C}}h2{{color:#08519C;font-size:1.15em;
margin-top:32px}}img{{max-width:100%;border:1px solid #EDE2B5}}code{{font-size:.85em}}
.note{{background:#FCFAE1;border-left:4px solid #9F2B2C;padding:8px 12px}}</style></head><body>
<h1>Conflito de gradientes — resultados da E0</h1>
<p class="note">Uma seed e um holdout: todos os números são descritivos. Diferenças menores que o
ruído entre runs (faixa bege no último gráfico) não devem ser lidas como efeito.</p>
{body}
<h2>Fontes</h2><ul>{sources}</ul></body></html>"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
