"""Figura do modelo para o artigo ESANN: MTResNet34UNet com o conjunto compartilhado destacado.

    .venv/bin/python -m paper.figures.build_model_diagram
Gera paper/figures/modelo_mtresnet34unet.png (300 dpi, fundo branco).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from src.utils.plot_style import BURGUNDY, CREAM, EXPORT_DPI, REFERENCE_LINE, ROSE, WINE_BLACK, \
    with_plot_style

OUT = Path(__file__).with_name("modelo_mtresnet34unet.png")
SHARED, PERSONAL = BURGUNDY, ROSE
W, H = 2.5, 0.66


def box(ax, x, y, text, shared, w=W, h=H):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=SHARED if shared else PERSONAL,
                                edgecolor=WINE_BLACK, linewidth=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.2,
            color=CREAM if shared else WINE_BLACK)


def arrow(ax, a, b, style="-|>", color=WINE_BLACK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, color=color,
                                 linewidth=0.8, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


@with_plot_style
def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-1.4, 11.8)
    ax.set_ylim(-1.1, 7.3)
    ax.axis("off")

    xe, xd, xo = 1.4, 6.0, 9.2
    levels = ["H", "H/2", "H/4", "H/8", "H/16", "H/32"]
    ys = [6.3, 5.25, 4.2, 3.15, 2.1, 1.05]
    enc = [("Stem (encoder1)\n1 ou 3 canais", False),
           ("encoder2\nconv1 ResNet-34", True),
           ("encoder3\nmaxpool + layer1", True),
           ("encoder4\nlayer2", True),
           ("encoder5\nlayer3", True),
           ("Gargalo\nlayer4", True)]
    for (label, shared), y, lev in zip(enc, ys, levels):
        box(ax, xe, y, label, shared)
        ax.text(xe - W / 2 - 0.15, y, lev, ha="right", va="center", fontsize=6.5,
                color=REFERENCE_LINE)
    for a, b in zip(ys[1:-1], ys[2:]):
        arrow(ax, (xe, a - H / 2), (xe, b + H / 2))
    arrow(ax, (xe, ys[4] - H / 2), (xe, ys[5] + H / 2))
    ax.text(xe, 7.0, "Imagem de entrada", ha="center", fontsize=7, color=WINE_BLACK)
    arrow(ax, (xe, 6.85), (xe, ys[0] + H / 2))
    arrow(ax, (xe - W / 2 - 0.35, 6.9), (xe - W / 2, ys[1]), rad=0.35)  # entrada normalizada -> encoder2

    dec = ["decoder1", "decoder2", "decoder3", "decoder4", "decoder5"]
    for name, y in zip(dec, ys[:5]):
        box(ax, xd, y, f"{name}\n+ upsample", True)
    # gargalo -> decoder5, e subida do decoder
    arrow(ax, (xe + W / 2, ys[5]), (xd, ys[4] - H / 2), rad=0.0)
    for a, b in zip(ys[4:0:-1], ys[3::-1]):
        arrow(ax, (xd, a + H / 2), (xd, b - H / 2))
    # conexões de atalho (skip)
    for y in ys[:5]:
        arrow(ax, (xe + W / 2, y), (xd - W / 2, y), color=REFERENCE_LINE, ls=(0, (3, 2)))

    # saídas de segmentação (supervisão profunda)
    outs = [("output1 (H)", ys[0]), ("output2", ys[1]), ("output3", ys[2]), ("output4", ys[3])]
    for label, y in outs:
        box(ax, xo, y, label, False, w=1.6)
        arrow(ax, (xd + W / 2, y), (xo - 0.8, y))
    ax.text(xo, 6.95, "Segmentação\n(supervisão profunda)", ha="center", fontsize=7,
            color=WINE_BLACK)

    # ramo de classificação
    box(ax, xo, 1.35, "process_encoder_5\n+ process_decoder_5", True, w=2.6)
    box(ax, xo, 0.2, "Classificador", False, w=2.6)
    ax.text(xo + 1.4, 1.35, "+ up5", ha="left", va="center", fontsize=6.5, color=REFERENCE_LINE)
    arrow(ax, (xe + W / 2, ys[4] - 0.2), (xo - 1.3, 1.25), color=REFERENCE_LINE, rad=0.25)
    arrow(ax, (xd + W / 2, ys[4]), (xo - 1.3, 1.5), color=REFERENCE_LINE, rad=-0.2)
    arrow(ax, (xo, 1.35 - H / 2), (xo, 0.2 + H / 2))
    ax.text(xo, -0.5, "Classificação", ha="center", fontsize=7, color=WINE_BLACK)

    ax.legend(handles=[Patch(facecolor=SHARED, edgecolor=WINE_BLACK, label="Compartilhado (φ), agregado pelo servidor"),
                       Patch(facecolor=PERSONAL, edgecolor=WINE_BLACK, label="Personalizado (ψᵢ), fica no cliente")],
              loc="lower left", bbox_to_anchor=(-0.02, -0.06), fontsize=6.8, frameon=False)
    fig.savefig(OUT, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
