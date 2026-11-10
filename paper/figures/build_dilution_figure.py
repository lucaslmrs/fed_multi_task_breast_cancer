"""Figura 2 do artigo ESANN: conflito × diluição (conceito) e a geometria da agregação por rodada.

    .venv/bin/python -m paper.figures.build_dilution_figure
Gera paper/figures/diluicao_por_rodada.png (300 dpi, fundo branco). Lê o aggregation_history.json
de cada braço federado; braços ainda sem histórico são omitidos.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plot_style import ACCENT, EXPORT_DPI, PRIMARY, REFERENCE_LINE, WINE_BLACK, \
    categorical_color, style_axis, with_plot_style

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("diluicao_por_rodada.png")
ARMS = [  # (rótulo, histórico, cor, traço)
    ("Fed-MTL (FedAvg)",
     "runs/e8_resume/20261108_155130_FEDERATED_MTResNet34UNet_resnet34_256_control", ACCENT, "-"),
    ("Fed-MTL + FedExP", "runs/e9_esann/resnet34_256_fedexp", PRIMARY, "-"),
    ("Fed-seg", "runs/e9_esann/resnet34_256_fed_seg", categorical_color(2), "--"),
    ("Fed-cls", "runs/e9_esann/resnet34_256_fed_cls", categorical_color(3), ":"),
]
SMOOTH = 10


def _series(run):
    path = ROOT / run / "fold_0" / "aggregation_history.json"
    if not path.exists():
        return None
    rows = [r for r in json.loads(path.read_text()) if r.get("stage") == "fit"
            and (r.get("negative_transfer") or {}).get("overall") is not None]
    rounds = np.array([r["round"] for r in rows])
    c = np.array([r["negative_transfer"]["overall"] for r in rows])
    ref = np.array([r["negative_transfer"]["orthogonal_reference"] for r in rows])
    eta = np.array([(r.get("server_optimizer") or {}).get("eta_g", 1.0) for r in rows])
    rho = eta * (1.0 - c)
    return rounds, c, ref, rho


def _smooth(y):
    k = np.ones(SMOOTH) / SMOOTH
    return np.convolve(np.pad(y, (SMOOTH // 2, SMOOTH - 1 - SMOOTH // 2), mode="edge"), k, "valid")


def _concept(ax):
    """Dois clientes: conflito (opostos) e diluição (ortogonais); a média em bordô."""
    ax.set_xlim(-1.25, 3.55)
    ax.set_ylim(-0.6, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    kw = dict(angles="xy", scale_units="xy", scale=1, width=0.018)
    # conflito
    ax.quiver([0, 0], [0, 0], [0.95, -0.95], [0.3, -0.3], color=REFERENCE_LINE, **kw)
    ax.quiver(0, 0, 0.0, 0.0, color=PRIMARY, **kw)
    ax.plot(0, 0, "o", color=PRIMARY, ms=4)
    ax.text(0, 1.15, "Conflito", ha="center", fontsize=8, color=WINE_BLACK)
    ax.text(0, -0.5, "média ≈ 0", ha="center", fontsize=7, color=PRIMARY)
    # diluição
    x0 = 2.1
    ax.quiver([x0, x0], [0, 0], [1.0, 0.0], [0.0, 1.0], color=REFERENCE_LINE, **kw)
    ax.quiver(x0, 0, 0.5, 0.5, color=PRIMARY, **kw)
    ax.text(x0 + 0.5, 1.15, "Diluição", ha="center", fontsize=8, color=WINE_BLACK)
    ax.text(x0 + 0.5, -0.5, "‖média‖ = 1/√2", ha="center", fontsize=7, color=PRIMARY)
    ax.set_title("(a) conflito × diluição", fontsize=8, loc="left", y=1.02)


@with_plot_style
def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})
    fig.patch.set_facecolor("white")
    _concept(axes[0])
    ax_c, ax_r = axes[1], axes[2]
    ref_drawn = False
    for label, run, color, ls in ARMS:
        data = _series(run)
        if data is None:
            continue
        rounds, c, ref, rho = data
        ax_c.plot(rounds, _smooth(c), color=color, ls=ls, lw=1.2, label=label)
        if not ref_drawn:
            ax_c.plot(rounds, _smooth(ref), color=REFERENCE_LINE, ls=(0, (1, 1)), lw=1.0,
                      label="Referência ortogonal C⊥")
            ref_drawn = True
        ax_r.plot(rounds, _smooth(rho), color=color, ls=ls, lw=1.2, label=label)
    ax_r.axhline(1.0, color=REFERENCE_LINE, lw=0.8, ls=(0, (1, 1)))
    ax_r.text(400, 1.03, "passo médio de um cliente", ha="right", va="bottom", fontsize=6,
              color=REFERENCE_LINE)
    for ax, title, ylab in ((ax_c, "(b) cancelamento C", "C"),
                            (ax_r, "(c) passo efetivo ρ", "ρ")):
        style_axis(ax)
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("rodada", fontsize=7)
        ax.set_ylabel(ylab, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(0, 400)
    ax_c.set_ylim(0.45, 0.8)
    ax_r.set_ylim(0, 1.9)
    ax_c.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
