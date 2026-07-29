"""
Single shared evaluation used by ALL comparison setups (federated, local-only, centralized MTL),
so the numbers are computed identically and stay comparable. No prediction-refining is applied
(refino is a separate ablation), and segmentation slices already exclude the `normal` class.

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

from src.utils.metrics import calculate_metrics, multiclass_classification_metrics

# metrics.py key -> short column name kept for segmentation (Hausdorff/pixel-accuracy dropped)
_SEG_KEYS = {"DICE": "dice", "Jaccard index": "iou", "Sensitivity": "sensitivity",
             "Specificity": "specificity", "Precision": "precision"}


@torch.inference_mode()
def evaluate(model, loader, task, num_classes, device, class_names=None):
    """Evaluate one task slice.

    ``class_names`` is optional for backwards compatibility.  When supplied it must describe the
    complete classification label space in numeric-label order.  The names are emitted as metadata
    rather than embedded in metric keys, keeping CSV schemas stable across renamed classes.
    """
    class_names = _validate_class_names(class_names, num_classes)
    model.eval()
    n = len(loader.dataset)
    if n == 0:
        out = {"n_test": 0}
        if task == "cls":
            _add_class_name_metadata(out, class_names)
        return out, None
    if task == "seg":
        return _evaluate_seg(model, loader, device, n), None
    if task != "cls":
        raise ValueError(f"Unknown task {task!r}; expected 'seg' or 'cls'")
    return _evaluate_cls(model, loader, device, num_classes, n, class_names)


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


def _evaluate_seg(model, loader, device, n):
    acc = {short: [] for short in _SEG_KEYS.values()}
    for data in loader:
        _, outputs = model(data["image"].to(device))
        seg = outputs[-1] if isinstance(outputs, list) else outputs
        seg = (torch.sigmoid(seg) > 0.5).float().cpu().numpy()
        mask = data["mask"].cpu().numpy()
        m = calculate_metrics(mask, seg, str(data["patient_id"].item()))
        for key, short in _SEG_KEYS.items():
            acc[short].append(m[key])
    out = {short: float(np.nanmean(v)) for short, v in acc.items()}
    out["n_test"] = n
    return out


def _evaluate_cls(model, loader, device, num_classes, n, class_names=None):
    patients, gt, pred, probs = [], [], [], []
    for data in loader:
        logits, _ = model(data["image"].to(device))
        pl = torch.mean(torch.stack(logits, dim=0), dim=0) if isinstance(logits, list) else logits
        p = torch.softmax(pl, dim=1).cpu().numpy()
        label = data["label"].flatten().to(torch.int64).cpu().numpy()
        patients.extend(data["patient_id"].cpu().numpy().tolist())
        gt.extend(label.tolist())
        pred.extend(p.argmax(axis=1).tolist())
        probs.extend(p.tolist())

    labels = list(range(num_classes))
    m = multiclass_classification_metrics(gt, pred, labels=labels)
    out = {
        "acc": float(m["accuracy"]),
        "macro_f1": float(m["f1_macro"]),
        "balanced_acc": float(balanced_accuracy_score(gt, pred)),
        "auc": _safe_auc(gt, np.asarray(probs), labels),
        "n_test": n,
    }
    for c in range(num_classes):
        out[f"precision_class_{c}"] = float(m.get(f"precision_class_{c}", np.nan))
        out[f"recall_class_{c}"] = float(m.get(f"recall_class_{c}", np.nan))
    _add_class_name_metadata(out, class_names)

    preds = pd.DataFrame({"patient_id": patients, "ground_truth": gt, "predicted": pred})
    preds[[f"prob_{c}" for c in range(num_classes)]] = probs
    if class_names is not None:
        for c, name in enumerate(class_names):
            preds[f"class_name_{c}"] = name
    return out, preds


def _safe_auc(gt, probs, labels):
    """OvR macro AUC; NaN when the (small per-client) slice lacks a class so it is undefined."""
    try:
        return float(roc_auc_score(gt, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError as e:
        logging.info(f"AUC undefined for this slice ({e}); recording NaN")
        return float("nan")
