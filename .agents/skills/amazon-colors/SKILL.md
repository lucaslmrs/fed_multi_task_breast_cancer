---
name: amazon-colors
description: The project's default colour system, Amazon-inspired (squid ink, Amazon orange and validated companion hues) for internal figures, HTML reports and the monitor app. Use when choosing, changing or reviewing any colour in Matplotlib/Seaborn/Plotly figures, src/utils/plot_style.py, report CSS or src/monitor/static, and to validate a palette's colour-blind safety and contrast with the bundled script.
---

# Amazon-inspired colours

This is the **default** colour system of the repository: every figure (`src/utils/plot_style.py`),
static HTML report and the monitor app (`src/monitor/static/`) wears it. It is *Amazon-inspired*:
it borrows the two public brand anchors and builds a validated data-viz system around them. Use it
for internal figures and tooling only. Never reproduce Amazon's logo, smile arrow, wordmark or any
trademark, and never present a figure as Amazon-branded or Amazon-endorsed.

The single source of truth for values is `src/utils/plot_style.py`; `src/monitor/static/style.css`
and the constants at the top of `src/monitor/static/app.js` mirror it. If they disagree,
`plot_style.py` wins and the others are bugs.

## Rule zero: compute, don't eyeball

Any change to a categorical, sequential, diverging, text or status colour must pass:

```bash
python .agents/skills/amazon-colors/scripts/validate_palette.py --css
```

It checks, in light **and** dark mode: OKLCH lightness band and chroma floor, adjacent-pair
separation under protanopia/deuteranopia (Machado 2009, OKLab Delta E x100: target >= 8, floor 6),
the normal-vision floor (>= 15, hard gate), WCAG >= 3:1 for marks and >= 4.5:1 for text/status,
monotone single-hue sequential ramps and a neutral diverging midpoint. Exit 1 on any FAIL. A WARN
obligates a secondary channel (marker, dash, direct label or table). Try a candidate without
editing the code with `--categorical "#..,#.." --mode dark [--pairs all]`.

## Palette

**Brand anchors.** `SQUID_INK #232F3E` (text, dark surfaces), `AMAZON_ORANGE #FF9900`
(highlight fills, callout rules, h1 underline, primary button with squid-ink text, 6.3:1), and
`NAV_INK #131921`. `#FF9900` is only 2.1:1 on white: **never** use it for lines, markers or text on
a light surface; use the mark-weight orange (`ACCENT`, slot 2) instead.

**Categorical, fixed order** (blue, orange, magenta, gold, teal, red, purple, green). Same hue
family per slot in both modes; assign in order, never skip, never re-rank, never generate a 9th:

| Slot | Light (on `#FFFFFF`) | Dark (on `#161D26`) |
|---|---|---|
| 1 `PRIMARY` | `#146EB4` | `#007ED4` |
| 2 `ACCENT` | `#B75F0B` | `#C9690C` |
| 3 | `#AE1A69` | `#CA3A80` |
| 4 | `#99740E` | `#A87F09` |
| 5 | `#0698A4` | `#0698A4` |
| 6 | `#B71A18` | `#D33C33` |
| 7 | `#6F41C1` | `#855BDC` |
| 8 | `#087736` | `#119245` |

Validated: worst adjacent CVD Delta E 14.0 light / 13.2 dark; normal vision 18.8 / 17.8; mark
contrast >= 3.48:1 / 3.58:1. For scatter or small multiples (any two marks can touch) only the
first **three** slots are validated all-pairs; beyond that facet or fold into "other".

**Sequential** (light to dark, one hue): `SEQUENTIAL_BLUE` (default magnitude scale, registered
as Matplotlib cmap `amazon_blue`) and `SEQUENTIAL_ORANGE` (`amazon_orange`). The light steps are
fills, not thin lines.

**Diverging** `DIVERGING = #005896, #599DDF, #EFF2F5, #FDA856, #B85F05` (blue, neutral grey
midpoint, orange; cmap `amazon_diverging`, `_r` reversed). Project convention for cosine/conflict
matrices: **conflict (negative) on the orange pole**, i.e. `colormap("diverging_r")` with
`TwoSlopeNorm(vcenter=0)`. Only use it when the centre is meaningful.

**Neutrals.** Light: text `#232F3E`, muted `#545B64`, border/grid `#D5D9D9`, tint `#F0F2F2`,
callout `#FFF4E5`, reference lines `#879596`, link `#007185`. Dark: page `#0F1419`, panel
`#161D26`, panel-2 `#232F3E`, text `#F2F4F6`, muted `#AAB7B8`, border `#37475A`, reference
`#5A6B7E`, link `#4DB8C9`.

**Status (reserved, always with icon or label, never a series colour):**

| Role | Light | Dark |
|---|---|---|
| ok | `#067D62` | `#4CC08F` |
| warning | `#8A5300` | `#FFB84D` |
| critical | `#B12704` | `#FF7A6B` |
| info | `#007185` | `#4DB8C9` |

## Semantic mapping (keep stable across files)

- Federated / proposed method: `PRIMARY`. Local-only baseline: `MUTED_INK` (it recedes).
  Centralized: `ACCENT`. Use `setup_color()`, never a literal.
- Loss and the one focal series: `ACCENT`. Dice, balanced accuracy, AUC: `PRIMARY`; IoU, macro-F1:
  slot 5. Use `metric_color()`.
- Segmentation overlays: hit `PRIMARY`, false positive `ACCENT`, missed lesion slot 3.
- Context traces (per-client curves behind a mean, round boundaries, orthogonal reference):
  `REFERENCE_LINE`, optionally with alpha.
- Folds: `fold_style()` (colour + dash + marker, so identity is never colour alone).

## Applying it

**Matplotlib/Seaborn.** Import from `src.utils.plot_style` and wrap builders in
`@with_plot_style` (scoped `rc_context`, never global `rcParams`). Use `PRIMARY`, `ACCENT`,
`categorical_color(i)`, `colormap("blue" | "orange" | "diverging" | "diverging_r")`,
`style_axis(ax)`. Never write a hex literal in plotting code; add a named token to
`plot_style.py` and validate it. The `python-plotting` skill covers chart form and integrity; this
skill owns the colours.

**Static HTML reports** (`run_report.py`, `analyze.py`, `training_curves.py`,
`negative_transfer.py`, `executive_report.py`, `scripts/*_report.py`): text `#232F3E`, h2/primary
`#146EB4`, h1 rule and `.notice` left border `#FF9900` on `#FFF4E5`, borders `#D5D9D9`, `th`
background `#F0F2F2`, links `#007185`, errors `#B12704`, ok `#067D62`.

**Monitor app.** Use CSS variables only (`var(--text)`, `var(--series-3)`, `var(--critical)`...).
Plotly traces get `seriesColor(i)` in `app.js`, which reads the theme's `--series-N`; heatmaps use
`DIVERGING_SCALE`. Exports are always light: `downloadLight` remaps the dark-theme series to
`LIGHT_SERIES`, so if you add a token that a trace can carry, add it to that map too. Dark mode is
a separately validated set, not an inversion.

## Changing the palette

1. Edit `plot_style.py`, then mirror the change in `style.css` (both dark blocks) and the
   constants in `app.js`.
2. Run the validator with `--css`, then `python -m unittest discover -v`.
3. Regenerate one affected figure/report and look at it (label collisions, overlays on grayscale
   ultrasound, legend legibility). The validator checks colour, not layout.
4. Update the tables above.
