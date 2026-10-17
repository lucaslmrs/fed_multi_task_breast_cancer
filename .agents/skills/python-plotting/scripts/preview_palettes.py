#!/usr/bin/env python3
"""Render the project's default (Amazon-inspired) palettes from src/utils/plot_style.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "utils").is_dir())
sys.path.insert(0, str(REPO_ROOT))

from src.utils import plot_style as ps  # noqa: E402


def _draw_palette(ax: plt.Axes, title: str, colors, surface="white", ink=ps.INK) -> None:
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.set_facecolor(surface)
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", color=ps.INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, color in enumerate(colors):
        ax.add_patch(Rectangle((index, 0.3), 1, 0.55, facecolor=color, edgecolor=surface, linewidth=2))
        ax.text(index + 0.5, 0.14, color, ha="center", va="center", fontsize=8, color=ink)


def render(output: Path) -> None:
    rows = [
        ("Categorical · light", ps.CATEGORICAL["light"], ps.SURFACE["light"], ps.TEXT["light"]),
        ("Categorical · dark", ps.CATEGORICAL["dark"], ps.SURFACE["dark"], ps.TEXT["dark"]),
        ("Sequential blue", ps.SEQUENTIAL_BLUE, "white", ps.INK),
        ("Sequential orange", ps.SEQUENTIAL_ORANGE, "white", ps.INK),
        ("Diverging (blue · neutral · orange)", ps.DIVERGING, "white", ps.INK),
        ("Status · light", tuple(ps.STATUS["light"].values()), "white", ps.INK),
        ("Brand anchors", (ps.SQUID_INK, ps.AMAZON_ORANGE, ps.NAV_INK), "white", ps.INK),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.6, 1.25 * len(rows)), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, (title, colors, surface, ink) in zip(axes, rows):
        _draw_palette(ax, title, colors, surface, ink)
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
