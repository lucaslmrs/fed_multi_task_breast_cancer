"""Deterministic examples of hits and misses for the run report.

Each client contributes a few candidates per quadrant (classification correct/wrong x segmentation
good/bad); ``run_report`` later picks the final examples per dataset from these candidates with
the same ordering. Only candidates are rendered, so the cost is bounded per client.

Segmentation is "good" when a lesion target reaches ``GOOD_DICE`` and, for an empty target, when
nothing is predicted -- an empty mask is a legitimate target, so a false positive on it is an error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.utils.plot_style import ACCENT_RED, EMBED_DPI, with_plot_style  # noqa: E402

GOOD_DICE = 0.5
PER_QUADRANT = 2
CANDIDATES_FILE = "{setup}_example_candidates.csv"
EXAMPLES_DIR = Path("report") / "examples"

# Overlay colours: dark blue = hit, red = false positive, light earth = missed lesion.
TP_COLOR, FP_COLOR, FN_COLOR = "#08519C", ACCENT_RED, "#EDE2B5"

QUADRANT_LABELS = {
    "cls_ok__seg_ok": "Classificação correta · segmentação boa",
    "cls_ok__seg_bad": "Classificação correta · segmentação ruim",
    "cls_err__seg_ok": "Classificação errada · segmentação boa",
    "cls_err__seg_bad": "Classificação errada · segmentação ruim",
    "seg_ok": "Segmentação boa",
    "seg_bad": "Segmentação ruim",
    "cls_ok": "Classificação correta",
    "cls_err": "Classificação errada",
}
QUADRANT_ORDER = list(QUADRANT_LABELS)


def _seg_table(frame):
    frame = frame.copy()
    area = frame["image_pixels"].where(frame["image_pixels"] > 0, 1)
    frame["seg_ok"] = np.where(
        frame["target_empty"], frame["pred_pixels"] == 0, frame["dice_positive"] >= GOOD_DICE
    )
    # One score for both strata: Dice on lesions, the untouched fraction on empty targets.
    frame["seg_score"] = np.where(
        frame["target_empty"], 1.0 - frame["pred_pixels"] / area, frame["dice_positive"]
    )
    return frame


def _cls_table(preds, class_names):
    prob_columns = [f"prob_{index}" for index in range(len(class_names))]
    frame = preds[["patient_id", "ground_truth", "predicted", *prob_columns]].copy()
    frame["cls_ok"] = frame["ground_truth"] == frame["predicted"]
    frame["confidence"] = frame[prob_columns].max(axis=1)
    frame["true_class"] = frame["ground_truth"].map(lambda value: class_names[int(value)])
    frame["predicted_class"] = frame["predicted"].map(lambda value: class_names[int(value)])
    return frame.drop(columns=prob_columns)


def _quadrant(row, has_seg, has_cls):
    parts = []
    if has_cls:
        parts.append("cls_ok" if row["cls_ok"] else "cls_err")
    if has_seg:
        parts.append("seg_ok" if row["seg_ok"] else "seg_bad")
    return "__".join(parts)


def rank_candidates(frame, per_quadrant=PER_QUADRANT):
    """Top rows per quadrant: best segmentations first when good, worst first when bad; then
    the most confident prediction (a confident error is the instructive one); then patient id."""
    frame = frame.copy()
    has_seg = "seg_score" in frame and frame["seg_score"].notna().any()
    has_cls = "confidence" in frame and frame["confidence"].notna().any()
    frame["_seg_key"] = (
        np.where(frame["seg_ok"].astype(bool), -frame["seg_score"], frame["seg_score"])
        if has_seg else 0.0
    )
    frame["_cls_key"] = -frame["confidence"] if has_cls else 0.0
    frame["_order"] = frame["quadrant"].map(QUADRANT_ORDER.index)
    frame = frame.sort_values(
        ["_order", "_seg_key", "_cls_key", "patient_id"], kind="mergesort"
    )
    return (
        frame.groupby("quadrant", sort=False).head(per_quadrant)
        .drop(columns=["_seg_key", "_cls_key", "_order"])
        .reset_index(drop=True)
    )


def select_candidates(seg_frame=None, cls_preds=None, class_names=None,
                      per_quadrant=PER_QUADRANT):
    """Candidate rows for one client; joins the tasks on ``patient_id`` when both exist."""
    has_seg = seg_frame is not None and not seg_frame.empty
    has_cls = cls_preds is not None and not cls_preds.empty
    if not has_seg and not has_cls:
        return pd.DataFrame()
    if has_seg and has_cls:
        frame = _seg_table(seg_frame).merge(
            _cls_table(cls_preds, class_names), on="patient_id", how="inner"
        )
    elif has_seg:
        frame = _seg_table(seg_frame)
    else:
        frame = _cls_table(cls_preds, class_names)
    if frame.empty:
        return frame
    frame["quadrant"] = frame.apply(_quadrant, axis=1, has_seg=has_seg, has_cls=has_cls)
    return rank_candidates(frame, per_quadrant)


def _overlay(image, mask, pred):
    base = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)
    rgb = base.astype(np.float32) / 255.0
    for selector, colour in (
        ((mask > 0) & (pred > 0), TP_COLOR),
        ((mask == 0) & (pred > 0), FP_COLOR),
        ((mask > 0) & (pred == 0), FN_COLOR),
    ):
        rgb[selector] = 0.35 * rgb[selector] + 0.65 * np.asarray(matplotlib.colors.to_rgb(colour))
    return rgb


@with_plot_style
def render_example(sample, path, title=None):
    """Image | real mask | predicted mask | overlay; the image alone for a classification-only row."""
    image = sample["image"]
    cmap = "gray" if image.ndim == 2 else None
    has_mask = "mask" in sample
    panels = 4 if has_mask else 1
    fig, axes = plt.subplots(1, panels, figsize=(2.1 * panels, 2.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    axes[0].imshow(image, cmap=cmap)
    axes[0].set_title("Imagem", fontsize=9)
    if has_mask:
        axes[1].imshow(sample["mask"], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Máscara real", fontsize=9)
        axes[2].imshow(sample["pred"], cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Máscara prevista", fontsize=9)
        axes[3].imshow(_overlay(image, sample["mask"], sample["pred"]))
        axes[3].set_title("Sobreposição", fontsize=9)
        fig.legend(
            handles=[Patch(color=TP_COLOR, label="acerto"),
                     Patch(color=FP_COLOR, label="falso positivo"),
                     Patch(color=FN_COLOR, label="lesão não detectada")],
            loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.08),
        )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=9)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=EMBED_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def client_examples(run_path, setup, fold, client_id, dataset, task_outputs, class_names):
    """Select and render one client's candidates; returns rows for the fold candidates CSV.

    ``task_outputs`` maps task -> (preds, samples) as returned by
    ``unified_eval.evaluate(..., return_samples=True)``.
    """
    run_path = Path(run_path)
    seg_frame, cls_preds, images = None, None, {}
    if "cls" in task_outputs:
        cls_preds, samples = task_outputs["cls"]
        images.update(samples["images"])
    if "seg" in task_outputs:
        _, samples = task_outputs["seg"]
        seg_frame = samples["frame"]
        if seg_frame is not None and not seg_frame.empty:
            sizes = {pid: sample["mask"].size for pid, sample in samples["images"].items()}
            seg_frame = seg_frame.assign(image_pixels=seg_frame["patient_id"].map(sizes))
        images.update(samples["images"])  # seg samples also carry the masks

    candidates = select_candidates(seg_frame, cls_preds, class_names)
    if candidates.empty:
        return []
    rows = []
    for record in candidates.to_dict("records"):
        relative = (
            EXAMPLES_DIR / f"fold_{fold}" / str(client_id) / f"{record['patient_id']}.png"
        )
        render_example(images[record["patient_id"]], run_path / relative)
        rows.append({
            "setup": setup, "fold": fold, "dataset": dataset, "client_id": client_id,
            **record, "image_path": relative.as_posix(),
        })
    return rows


def write_fold_candidates(fold_dir, setup, rows):
    path = Path(fold_dir) / CANDIDATES_FILE.format(setup=setup)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
