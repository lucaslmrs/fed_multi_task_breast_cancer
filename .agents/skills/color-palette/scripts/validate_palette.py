#!/usr/bin/env python3
"""Validate the project palette (color-palette skill): colour-vision safety and contrast, computed.

Checks (thresholds from the dataviz method; Delta E is Euclidean OKLab distance x100):

  categorical  - OKLCH lightness band per mode, chroma floor >= 0.10, adjacent-pair Delta E under
                 protanopia/deuteranopia (Machado et al. 2009, severity 1.0) >= 8 target / 6 floor,
                 unsimulated adjacent Delta E >= 15 (hard gate), WCAG contrast vs surface >= 3:1.
  sequential   - monotone OKLCH lightness, adjacent delta L >= 0.06, single hue (spread <= 40 deg).
  diverging    - each arm monotone toward a low-chroma midpoint; poles distinct under CVD.
  text         - WCAG contrast of text tokens vs their surface >= 4.5:1 (muted >= 4.5:1 too).

Usage (from the repository root):

    python .agents/skills/color-palette/scripts/validate_palette.py            # src/utils/plot_style.py
    python .agents/skills/color-palette/scripts/validate_palette.py --css      # + src/monitor/static/style.css
    python .agents/skills/color-palette/scripts/validate_palette.py \
        --categorical "#7B2032,#2B7EC9,..." --mode light --surface "#FFF7F1"

Exit 0 when nothing hard-FAILs, 1 otherwise. WARN rows (CVD floor band, sub-3:1 relief, a user
-mandated base colour outside the lightness band) obligate
secondary encoding (marker, dash, direct label) and are not failures.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from pathlib import Path

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
MARK_CONTRAST = 3.0
TEXT_CONTRAST = 4.5
ORDINAL_MIN_DL = 0.06

# Machado, Oliveira & Fernandes (2009) CVD transforms at severity 1.0, linear RGB.
MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


# ---- colour maths ----------------------------------------------------------------------------

def _srgb(hex_color):
    value = hex_color.strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        raise ValueError(f"not a #RRGGBB colour: {hex_color!r}")
    return [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear(hex_color):
    return [_to_linear(c) for c in _srgb(hex_color)]


def luminance(hex_color):
    r, g, b = _linear(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab_from_linear(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(hex_color):
    L, a, b = _oklab_from_linear(_linear(hex_color))
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def _simulate(hex_color, kind):
    rgb = _linear(hex_color)
    return [min(1.0, max(0.0, sum(m * c for m, c in zip(row, rgb)))) for row in MACHADO[kind]]


def delta_e(a, b, kind=None):
    la = _oklab_from_linear(_simulate(a, kind) if kind else _linear(a))
    lb = _oklab_from_linear(_simulate(b, kind) if kind else _linear(b))
    return 100 * math.dist(la, lb)


# ---- checks ----------------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.rows = []

    def add(self, scope, check, state, detail):
        self.rows.append((scope, check, state, detail))

    @property
    def failed(self):
        return any(state == "FAIL" for _, _, state, _ in self.rows)

    def print(self):
        width = max(len(s) for s, *_ in self.rows) if self.rows else 10
        for scope, check, state, detail in self.rows:
            print(f"{state:4}  {scope:<{width}}  {check:<22} {detail}")


def check_categorical(report, name, palette, mode, surface, pairs="adjacent", anchors=frozenset()):
    lo, hi = BAND[mode]
    off = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    # A base colour the user chose is kept even when it leaves the band; the separation and
    # contrast gates below still apply to it in full.
    state = "PASS" if not off else "WARN" if all(c.upper() in anchors for c, _ in off) else "FAIL"
    report.add(name, "lightness band", state,
               f"outside L {lo}-{hi}: {off}" + (" (user base colour)" if state == "WARN" else "")
               if off else f"all {len(palette)} inside L {lo}-{hi}")
    grey = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    report.add(name, "chroma floor", "FAIL" if grey else "PASS",
               f"below C {CHROMA_FLOOR}: {grey}" if grey else f"all C >= {CHROMA_FLOOR}")

    indices = range(len(palette))
    pairlist = (list(itertools.combinations(indices, 2)) if pairs == "all"
                else list(zip(indices, indices[1:])))
    worst = min(((delta_e(palette[i], palette[j], kind), kind, palette[i], palette[j])
                 for kind in ("protan", "deutan") for i, j in pairlist), default=None)
    if worst:
        d, kind, a, b = worst
        state = "PASS" if d >= CVD_TARGET else "WARN" if d >= CVD_FLOOR else "FAIL"
        report.add(name, f"CVD {pairs}", state, f"worst {d:.1f} ({kind}) {a} vs {b}")
        n = min((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in pairlist)
        report.add(name, f"normal vision {pairs}", "PASS" if n[0] >= NORMAL_FLOOR else "FAIL",
                   f"worst {n[0]:.1f} {n[1]} vs {n[2]}")

    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < MARK_CONTRAST]
    report.add(name, f"contrast vs {surface}", "WARN" if low else "PASS",
               f"below 3:1 (needs labels/table): {low}" if low
               else f"min {min(contrast(c, surface) for c in palette):.2f}:1")


def check_sequential(report, name, ramp, surface):
    Ls = [oklch(c)[0] for c in ramp]
    monotone = all(x < y for x, y in zip(Ls, Ls[1:])) or all(x > y for x, y in zip(Ls, Ls[1:]))
    report.add(name, "monotone lightness", "PASS" if monotone else "FAIL",
               " -> ".join(f"{L:.2f}" for L in Ls))
    gap = min(abs(x - y) for x, y in zip(Ls, Ls[1:]))
    report.add(name, "step size", "PASS" if gap >= ORDINAL_MIN_DL else "FAIL", f"min dL {gap:.3f}")
    hues = [oklch(c)[2] for c in ramp if oklch(c)[1] >= 0.03]
    spread = max(hues) - min(hues) if hues else 0
    report.add(name, "single hue", "PASS" if spread <= 40 else "FAIL", f"hue spread {spread:.0f} deg")
    darkest = min(ramp, key=lambda c: oklch(c)[0])
    report.add(name, "dark end contrast", "PASS" if contrast(darkest, surface) >= MARK_CONTRAST else "FAIL",
               f"{darkest} {contrast(darkest, surface):.2f}:1 vs {surface}")


def check_diverging(report, name, ramp):
    mid = len(ramp) // 2
    left, right = ramp[:mid + 1], ramp[mid:]
    ok_left = all(oklch(a)[0] < oklch(b)[0] for a, b in zip(left, left[1:]))
    ok_right = all(oklch(a)[0] > oklch(b)[0] for a, b in zip(right, right[1:]))
    report.add(name, "arms toward midpoint", "PASS" if ok_left and ok_right else "FAIL",
               " | ".join(f"{oklch(c)[0]:.2f}" for c in ramp))
    c_mid = oklch(ramp[mid])[1]
    report.add(name, "neutral midpoint", "PASS" if c_mid < 0.04 else "FAIL", f"{ramp[mid]} C {c_mid:.3f}")
    # The poles carry the sign, so they get 1.5x the categorical CVD target; below the target
    # itself the sign is unreadable for dichromats.
    worst = min((delta_e(ramp[0], ramp[-1], k), k) for k in ("protan", "deutan", "tritan"))
    state = "PASS" if worst[0] >= 1.5 * CVD_TARGET else "WARN" if worst[0] >= CVD_TARGET else "FAIL"
    report.add(name, "poles under CVD", state, f"min {worst[0]:.1f} ({worst[1]})")


def check_text(report, name, tokens, surface, minimum=TEXT_CONTRAST):
    for token, color in tokens.items():
        ratio = contrast(color, surface)
        report.add(name, f"text {token}", "PASS" if ratio >= minimum else "FAIL",
                   f"{color} {ratio:.2f}:1 vs {surface} (needs {minimum})")


# ---- sources ---------------------------------------------------------------------------------

def _repo_root():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "utils" / "plot_style.py").is_file():
            return parent
    raise SystemExit("run inside the repository: src/utils/plot_style.py not found")


def validate_plot_style(report):
    root = _repo_root()
    sys.path.insert(0, str(root))
    from src.utils import plot_style as ps  # noqa: E402  (repo import, resolved at runtime)

    for mode in ("light", "dark"):
        palette = ps.CATEGORICAL[mode]
        surface = ps.SURFACE[mode]
        anchors = {c.upper() for c in ps.ANCHORS}
        check_categorical(report, f"categorical/{mode}", palette, mode, surface, anchors=anchors)
        check_categorical(report, f"categorical[:3]/{mode}", palette[:3], mode, surface, pairs="all",
                          anchors=anchors)
        check_text(report, f"text/{mode}", {"ink": ps.TEXT[mode], "muted": ps.MUTED[mode]}, surface)
        check_text(report, f"status/{mode}", ps.STATUS[mode], surface)
    check_sequential(report, "sequential", list(ps.SEQUENTIAL), ps.SURFACE["light"])
    check_diverging(report, "diverging", list(ps.DIVERGING))


def _css_tokens(block):
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;", block))


def validate_css(report):
    text = (_repo_root() / "src" / "monitor" / "static" / "style.css").read_text(encoding="utf-8")
    light = _css_tokens(text.split("@media", 1)[0])
    dark = _css_tokens(text[text.index('[data-theme="dark"]'):])
    for mode, tokens in (("light", light), ("dark", dark)):
        surface = tokens["panel"]
        text_tokens = ("text", "muted", "link", "critical", "warning", "ok", "info", "pending")
        check_text(report, f"css/{mode}", {k: tokens[k] for k in text_tokens if k in tokens}, surface)
        for fg, bg in (("on-accent", "accent"), ("on-brand", "brand")):
            if fg in tokens and bg in tokens:
                check_text(report, f"css/{mode}", {f"{fg} on {bg}": tokens[fg]}, tokens[bg])
        series = [tokens[f"series-{i}"] for i in range(1, 9) if f"series-{i}" in tokens]
        anchors = {"#7B2032"}  # BURGUNDY, a user base colour (see plot_style.ANCHORS)
        check_categorical(report, f"css-series/{mode}", series, mode, surface, anchors=anchors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categorical", help="comma-separated hexes to validate instead of plot_style")
    parser.add_argument("--mode", choices=("light", "dark"), default="light")
    parser.add_argument("--surface", help="chart surface (default #FFF7F1 light, #1E1216 dark)")
    parser.add_argument("--pairs", choices=("adjacent", "all"), default="adjacent")
    parser.add_argument("--css", action="store_true", help="also validate src/monitor/static/style.css")
    args = parser.parse_args(argv)

    report = Report()
    if args.categorical:
        palette = [c.strip() for c in args.categorical.split(",") if c.strip()]
        surface = args.surface or ("#FFF7F1" if args.mode == "light" else "#1E1216")
        check_categorical(report, f"cli/{args.mode}", palette, args.mode, surface, args.pairs)
    else:
        validate_plot_style(report)
        if args.css:
            validate_css(report)
    report.print()
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
