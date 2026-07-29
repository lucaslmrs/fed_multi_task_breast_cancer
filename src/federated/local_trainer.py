"""
Framework-agnostic local training / evaluation for a single-task client.

A client owns ONE task (``seg`` or ``cls``) but runs the full MTnnUNet. Training optimises the
shared encoder AND the client's personalized head locally; only the encoder is later federated.
The segmentation loss handling (deep-supervision list, inverse weighting) mirrors the centralized
``training_multitask.py`` so federated and centralized runs stay comparable.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from src.utils.metrics import dice_score_from_tensor


def _seg_loss(criterion, masks, outputs, inversely_weighted):
    if isinstance(outputs, list):
        if inversely_weighted:
            return torch.sum(torch.stack(
                [criterion(s, masks) / (n + 1) for n, s in enumerate(reversed(outputs))]))
        return torch.sum(torch.stack([criterion(s, masks) for s in outputs]))
    return criterion(outputs, masks)


def _cls_loss(criterion, label, logits):
    if isinstance(logits, list):
        return torch.sum(torch.stack([criterion(c, label) for c in logits]))
    return criterion(logits, label)


def _prep_label(label, num_classes):
    if num_classes > 2:
        return F.one_hot(label.flatten().to(torch.int64), num_classes=num_classes).to(torch.float)
    return label


def _avg_logits(logits):
    """Collapse a deep-supervision list of class logits into a single tensor."""
    if isinstance(logits, list):
        return torch.mean(torch.stack(logits, dim=0), dim=0)
    return logits


def train_local(model, loader, optimizer, task, device, local_epochs, num_classes,
                seg_criterion=None, cls_criterion=None, inversely_weighted=True,
                training_mode="epochs", steps_per_round=10):
    """Run local SGD and return loss plus exact optimization/exposure telemetry.

    ``epochs`` preserves the legacy nested epoch/loader loop. In ``steps`` mode the loader must
    yield exactly the configured number of full batches for the currently selected round.
    """
    training_mode = str(training_mode).lower()
    if training_mode not in {"epochs", "steps"}:
        raise ValueError("training_mode must be 'epochs' or 'steps'")
    if isinstance(local_epochs, bool) or not isinstance(local_epochs, int) or local_epochs < 1:
        raise ValueError(f"local_epochs must be a positive integer, got {local_epochs!r}")
    if (
        isinstance(steps_per_round, bool)
        or not isinstance(steps_per_round, int)
        or steps_per_round < 1
    ):
        raise ValueError(f"steps_per_round must be a positive integer, got {steps_per_round!r}")
    model.train()
    running, n_batches, examples_processed = 0.0, 0, 0
    passes = local_epochs if training_mode == "epochs" else 1
    for _ in range(passes):
        for data in loader:
            inputs = data["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, outputs = model(inputs)
            if task == "seg":
                loss = _seg_loss(seg_criterion, data["mask"].to(device), outputs, inversely_weighted)
            else:
                label = _prep_label(data["label"].to(device), num_classes)
                loss = _cls_loss(cls_criterion, label, logits)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
            examples_processed += int(inputs.shape[0])

    if training_mode == "steps" and n_batches != steps_per_round:
        raise RuntimeError(
            f"steps mode expected {steps_per_round} batches, but loader yielded {n_batches}"
        )
    return {
        "loss": running / max(n_batches, 1),
        "optimizer_steps": n_batches,
        "examples_processed": examples_processed,
    }


@torch.inference_mode()
def evaluate_local(model, loader, task, device, num_classes,
                   seg_criterion=None, cls_criterion=None, inversely_weighted=True):
    """Evaluate the client's task. Returns {loss, metric, metric_name, n}."""
    model.eval()
    n_samples = len(loader.dataset)
    if n_samples == 0:
        return {"loss": float("nan"), "metric": float("nan"), "metric_name": "na", "n": 0}

    if task == "seg":
        total_loss, dice, n = 0.0, 0.0, 0
        for data in loader:
            logits, outputs = model(data["image"].to(device))
            masks = data["mask"].to(device)
            total_loss += _seg_loss(seg_criterion, masks, outputs, inversely_weighted).item()
            seg = outputs[-1] if isinstance(outputs, list) else outputs
            dice += float(dice_score_from_tensor(masks, torch.sigmoid(seg) > 0.5))
            n += 1
        return {"loss": total_loss / n, "metric": dice / n, "metric_name": "dice", "n": n_samples}

    # classification
    total_loss, n = 0.0, 0
    gt, pred = [], []
    for data in loader:
        label = _prep_label(data["label"].to(device), num_classes)
        logits, outputs = model(data["image"].to(device))
        total_loss += _cls_loss(cls_criterion, label, logits).item()
        n += 1
        pl = _avg_logits(logits)
        if num_classes > 2:
            probs = torch.softmax(pl, dim=1)
            gt.extend(torch.argmax(label, dim=1).detach().cpu().tolist())
            pred.extend(torch.argmax(probs, dim=1).detach().cpu().tolist())
        else:
            gt.extend(label.flatten().detach().cpu().tolist())
            pred.extend((torch.sigmoid(pl).flatten() > 0.5).double().detach().cpu().tolist())

    acc = accuracy_score(gt, pred)
    f1v = f1_score(gt, pred, labels=list(range(num_classes)), average="weighted", zero_division=0)
    return {"loss": total_loss / n, "metric": acc, "f1": f1v, "metric_name": "acc", "n": n_samples}
