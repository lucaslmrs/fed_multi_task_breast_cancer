#!/usr/bin/env python3
"""Render the reference palettes bundled with the python-plotting skill."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUES_6 = ["#EFF3FF", "#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
EARTH_5 = ["#FCFAE1", "#EDE2B5", "#CABD91", "#6F6357", "#584A47"]
ACCENT_RED = "#9F2B2C"


def _draw_palette(ax: plt.Axes, title: str, colors: list[str]) -> None:
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", color="#584A47")
    ax.axis("off")
    for index, color in enumerate(colors):
        ax.add_patch(Rectangle((index, 0.25), 1, 0.55, facecolor=color, edgecolor="white"))
        ax.text(
            index + 0.5,
            0.12,
            color,
            ha="center",
            va="center",
            fontsize=9,
            color="#584A47",
        )


def render(output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    _draw_palette(axes[0], "Blues 6 · sequential", BLUES_6)
    _draw_palette(axes[1], "Earth 5 · sequential / neutral", EARTH_5)
    _draw_palette(axes[2], "Accent red · highlight", [ACCENT_RED])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="PNG file to create")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    render(args.output)


if __name__ == "__main__":
    main()
