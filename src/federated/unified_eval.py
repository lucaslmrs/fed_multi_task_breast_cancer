"""
Single shared evaluation used by ALL comparison setups (federated, local-only, centralized MTL),
so the numbers are computed identically and stay comparable. No prediction-refining is applied. Valid empty targets participate in segmentation;
positive overlap and false positives on empty targets are reported separately.

`evaluate(model, loader, task, num_classes, device, class_names=None)` returns:
    - metrics: flat dict of scalar metrics (NaN where undefined) plus optional class-name metadata
    - preds:   per-image predictions DataFrame for cls (gt + class probabilities) so `analyze.py`
               can compute pooled AUC / confusion matrices; None for seg.

Seg metrics: Dice, IoU/Jaccard, Sensitivity, Specificity, Precision (reused from metrics.py,
Hausdorff intentionally omitted). Cls metrics: accuracy, macro-F1, per-class precision/recall,
balanced accuracy, and One-vs-Rest macro AUC. Per-class metric columns keep their stable numeric
names (``precision_class_0`` etc.); optional ``class_names`` add ``class_name_0`` metadata so
mixed label spaces remain human-readable without breaking legacy consumers.
"""

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.utils.metrics import calculate_metrics, multiclass_classification_metrics, summarize_segmentation_strata
from src.utils.supervision import _supervised
from src.utils.training_runtime import PrecisionPolicy

# metrics.py key -> short column name kept for segmentation (Hausdorff/pixel-accuracy dropped)
_SEG_KEYS = {"DICE": "dice", "Jaccard index": "iou", "Sensitivity": "sensitivity",
             "Specificity": "specificity", "Precision": "precision"}


@torch.inference_mode()
def evaluate(model, loader, task, num_classes, device, class_names=None, precision=None,
             return_samples=False):
    """Evaluate one task slice.

    ``class_names`` is optional for backwards compatibility.  When supplied it must describe the
    complete classification label space in numeric-label order.  The names are emitted as metadata
    rather than embedded in metric keys, keeping CSV schemas stable across renamed classes.

    ``return_samples=True`` appends a third value, ``{"frame": per-image seg metrics or None,
    "images": {patient_id: {"image", "mask"?, "pred"?}}}``, used to render run examples. The
    aggregated metrics are identical either way; images are kept as uint8 for the whole slice.
    """
    class_names = _validate_class_names(class_names, num_classes)
    precision = precision or PrecisionPolicy("fp32", device)
    model.eval()
    n = len(loader.dataset)
    samples = {"frame": None, "images": {}} if return_samples else None
    if n == 0:
        out, preds = {"n_test": 0}, None
        if task == "cls":
            _add_class_name_metadata(out, class_names)
        elif task == "seg":
            out.update(summarize_segmentation_strata([]))
    elif task == "seg":
        out, preds = _evaluate_seg(model, loader, device, n, precision, samples), None
    elif task == "cls":
        out, preds = _evaluate_cls(
            model, loader, device, num_classes, n, class_names, precision, samples
        )
    else:
        raise ValueError(f"Unknown task {task!r}; expected 'seg' or 'cls'")
    return (out, preds, samples) if return_samples else (out, preds)


def _display_image(image):
    """C x H x W tensor -> min-max scaled uint8, H x W for one channel or H x W x 3."""
    array = image.float().cpu().numpy()
    low, high = float(array.min()), float(array.max())
    scaled = (array - low) / (high - low) if high > low else np.zeros_like(array)
    scaled = (scaled * 255).round().astype(np.uint8)
    return scaled[0] if scaled.shape[0] == 1 else np.moveaxis(scaled[:3], 0, -1)


def _validate_class_names(class_names, num_classes):
    if class_names is None:
        return None
    names = [str(name) for name in class_names]
    if len(names) != num_classes:
        raise ValueError(
            f"class_names has {len(names)} entries, but num_classes={num_classes}"
        )
    if len(set(names)) != len(names):
        raise ValueError("class_names entries must be unique")
    return names


def _add_class_name_metadata(target, class_names):
    if class_names is not None:
        target.update({f"class_name_{index}": name for index, name in enumerate(class_names)})


def _evaluate_seg(model, loader, device, n, precision, samples=None):
    acc = {short: [] for short in _SEG_KEYS.values()}
    strata = []
    rows = []
    for data in loader:
        with precision.autocast():
            _, outputs = model(precision.move(data["image"]))
        seg = outputs[-1] if isinstance(outputs, list) else outputs
        seg = (torch.sigmoid(seg.float()) > 0.5).float().cpu().numpy()
        mask = data["mask"].cpu().numpy()
        patient_ids = data["patient_id"].reshape(-1).cpu().tolist()
        keep = _supervised(data, "seg", "cpu", len(patient_ids)).tolist()
        for index, patient_id in enumerate(patient_ids):
            if not keep[index]:
                continue
            m = calculate_metrics(mask[index:index + 1], seg[index:index + 1], str(patient_id))
            strata.append(m)
            for key, short in _SEG_KEYS.items():
                acc[short].append(m[key])
            if samples is not None:
                target, predicted = mask[index, 0] > 0.5, seg[index, 0] > 0.5
                rows.append({
                    "patient_id": patient_id,
                    "dice": float(m["DICE"]),
                    "dice_positive": float(m.get("dice_positive", np.nan)),
                    "target_empty": bool(not target.any()),
                    "target_pixels": int(target.sum()),
                    "pred_pixels": int(predicted.sum()),
                })
                samples["images"][patient_id] = {
                    "image": _display_image(data["image"][index]),
                    "mask": target.astype(np.uint8),
                    "pred": predicted.astype(np.uint8),
                }
    out = {short: float(np.mean([x for x in v if np.isfinite(x)]))
           if any(np.isfinite(x) for x in v) else np.nan for short, v in acc.items()}
    out.update(summarize_segmentation_strata(strata))
    out["n_test"] = len(strata)
    if samples is not None:
        samples["frame"] = pd.DataFrame(rows)
    return out


def _evaluate_cls(model, loader, device, num_classes, n, class_names, precision, samples=None):
    patients, gt, pred, probs = [], [], [], []
    for data in loader:
        with precision.autocast():
            logits, _ = model(precision.move(data["image"]))
        pl = torch.mean(torch.stack(logits, dim=0), dim=0) if isinstance(logits, list) else logits
        keep = _supervised(data, "cls", device, len(data["image"]))
        pl = pl[keep].float()
        if num_classes == 2 and pl.shape[1] == 1:
            positive = pl.sigmoid()
            p = torch.cat([1 - positive, positive], dim=1).cpu().numpy()
        else:
            p = torch.softmax(pl, dim=1).cpu().numpy()
        label = data["label"][keep.cpu()].flatten().to(torch.int64).cpu().numpy()
        kept_patients = data["patient_id"][keep.cpu()].cpu().numpy().tolist()
        patients.extend(kept_patients)
        if samples is not None:
            for image, patient_id in zip(data["image"][keep.cpu()], kept_patients):
                samples["images"][patient_id] = {"image": _display_image(image)}
        gt.extend(label.tolist())
        pred.extend(p.argmax(axis=1).tolist())
        probs.extend(p.tolist())

    labels = list(range(num_classes))
    m = multiclass_classification_metrics(gt, pred, labels=labels) if gt else {}
    out = {
        "acc": float(m.get("accuracy", np.nan)),
        "macro_f1": float(m.get("f1_macro", np.nan)),
        "balanced_acc": float(balanced_accuracy_score(gt, pred)) if gt else np.nan,
        "auc": _safe_auc(gt, np.asarray(probs), labels) if gt else np.nan,
        "n_test": len(gt),
    }
    for c in range(num_classes):
        out[f"precision_class_{c}"] = float(m.get(f"precision_class_{c}", np.nan))
        out[f"recall_class_{c}"] = float(m.get(f"recall_class_{c}", np.nan))
    _add_class_name_metadata(out, class_names)

    preds = pd.DataFrame({"patient_id": patients, "ground_truth": gt, "predicted": pred})
    for c in range(num_classes):
        preds[f"prob_{c}"] = [p[c] for p in probs]
    if class_names is not None:
        for c, name in enumerate(class_names):
            preds[f"class_name_{c}"] = name
    return out, preds


def _safe_auc(gt, probs, labels):
    """OvR macro AUC; NaN when the (small per-client) slice lacks a class so it is undefined."""
    try:
        if len(labels) == 2:
            return float(roc_auc_score(gt, probs[:, 1], labels=labels))
        return float(roc_auc_score(gt, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError as e:
        logging.info(f"AUC undefined for this slice ({e}); recording NaN")
        return float("nan")
