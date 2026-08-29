"""
Framework-agnostic local training / evaluation for a federated client.

A client owns one OR MORE tasks (``seg``, ``cls``) and runs the full MTnnUNet. Training optimises
the shared encoder AND the client's personalized heads locally; only the encoder is later
federated. The segmentation loss handling (deep-supervision list, inverse weighting) mirrors the
centralized ``training_multitask.py`` so federated and centralized runs stay comparable.

With a single task the code path is exactly the historical one -- no masking, no task weight -- so
existing arms reproduce their numbers. With several tasks the objective becomes
``L = sum_t lambda_t * L_t`` evaluated over the samples each task actually supervises, read from the
per-sample ``has_mask``/``has_label`` flags. A task with no supervised sample in a batch is
OMITTED, never contributed as a zero: an all-zero mask is a legitimate target for BUSI's ``normal``
class, so a zero term and an absent term are not the same thing.
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


def resolve_tasks(task=None, tasks=None) -> tuple:
    """Normalise the legacy ``task`` argument and the multi-task ``tasks`` argument into a tuple."""
    if tasks is None:
        if task is None:
            raise ValueError("Either task or tasks must be provided")
        tasks = (task,)
    tasks = tuple(dict.fromkeys(tasks))
    if not tasks:
        raise ValueError("tasks must contain at least one task")
    unknown = [item for item in tasks if item not in {"seg", "cls"}]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}; expected 'seg' and/or 'cls'")
    return tasks


def resolve_task_lambdas(tasks, task_lambdas=None) -> dict:
    """Per-task loss weights, defaulting to a uniform split (1.0 for a single-task client)."""
    if task_lambdas is None:
        return {name: 1.0 / len(tasks) for name in tasks}
    missing = [name for name in tasks if name not in task_lambdas]
    if missing:
        raise ValueError(f"task_lambdas is missing weights for {missing}")
    return {name: float(task_lambdas[name]) for name in tasks}


_SUPERVISION_FLAG = {"seg": "has_mask", "cls": "has_label"}


def _supervised(data, task, device, batch_size):
    """Boolean selector of the samples in this batch that supervise ``task``.

    Falls back to "everything is supervised" when the flag is absent, which keeps loaders built by
    older callers (and hand-made test fixtures) working.
    """
    flag = data.get(_SUPERVISION_FLAG[task])
    if flag is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return flag.reshape(-1).to(device=device, dtype=torch.bool)


def _combined_loss(data, logits, outputs, tasks, lambdas, device, num_classes,
                   seg_criterion, cls_criterion, inversely_weighted):
    """``sum_t lambda_t * L_t`` over the supervised subset; returns (loss, tasks_counted)."""
    batch_size = int(data["image"].shape[0])
    terms, counted = [], []
    for task in tasks:
        keep = _supervised(data, task, device, batch_size)
        if not bool(keep.any()):
            continue
        if task == "seg":
            masks = data["mask"].to(device)[keep]
            selected = [item[keep] for item in outputs] if isinstance(outputs, list) else outputs[keep]
            term = _seg_loss(seg_criterion, masks, selected, inversely_weighted)
        else:
            label = _prep_label(data["label"].to(device)[keep], num_classes)
            selected = [item[keep] for item in logits] if isinstance(logits, list) else logits[keep]
            term = _cls_loss(cls_criterion, label, selected)
        terms.append(lambdas[task] * term)
        counted.append(task)
    if not terms:
        raise RuntimeError(
            "A batch carried no supervision for any owned task; the client partition is malformed"
        )
    return torch.sum(torch.stack(terms)), counted


def train_local(model, loader, optimizer, task, device, local_epochs, num_classes,
                seg_criterion=None, cls_criterion=None, inversely_weighted=True,
                training_mode="epochs", steps_per_round=10, tasks=None, task_lambdas=None):
    """Run local SGD and return loss plus exact optimization/exposure telemetry.

    ``epochs`` preserves the legacy nested epoch/loader loop. In ``steps`` mode the loader must
    yield exactly the configured number of full batches for the currently selected round.

    ``tasks``/``task_lambdas`` enable the multi-task objective. A client that declares several
    tasks and finishes a round having exercised only some of them is a silent degradation to a
    single-task client, so it is rejected.
    """
    owned = resolve_tasks(task, tasks)
    lambdas = resolve_task_lambdas(owned, task_lambdas)
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
    task_batches = {name: 0 for name in owned}
    passes = local_epochs if training_mode == "epochs" else 1
    for _ in range(passes):
        for data in loader:
            inputs = data["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, outputs = model(inputs)
            if len(owned) == 1:
                # Historical single-task path, kept verbatim so existing arms reproduce exactly.
                if owned[0] == "seg":
                    loss = _seg_loss(
                        seg_criterion, data["mask"].to(device), outputs, inversely_weighted
                    )
                else:
                    label = _prep_label(data["label"].to(device), num_classes)
                    loss = _cls_loss(cls_criterion, label, logits)
                counted = list(owned)
            else:
                loss, counted = _combined_loss(
                    data, logits, outputs, owned, lambdas, device, num_classes,
                    seg_criterion, cls_criterion, inversely_weighted,
                )
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
            examples_processed += int(inputs.shape[0])
            for name in counted:
                task_batches[name] += 1

    if training_mode == "steps" and n_batches != steps_per_round:
        raise RuntimeError(
            f"steps mode expected {steps_per_round} batches, but loader yielded {n_batches}"
        )
    starved = sorted(name for name, count in task_batches.items() if count == 0)
    if len(owned) > 1 and starved:
        raise RuntimeError(
            f"Client owns tasks {list(owned)} but task(s) {starved} received no supervised batch "
            "this round; it would silently degrade to a single-task client"
        )
    return {
        "loss": running / max(n_batches, 1),
        "optimizer_steps": n_batches,
        "examples_processed": examples_processed,
        "task_batches": task_batches,
    }


@torch.inference_mode()
def evaluate_local(model, loader, task, device, num_classes,
                   seg_criterion=None, cls_criterion=None, inversely_weighted=True,
                   tasks=None, task_lambdas=None):
    """Evaluate the client's task(s). Returns {loss, metric, metric_name, n}.

    A multi-task client reports the same combined objective it optimises, so the server's
    diagnostic early-stopping signal stays consistent with the local loss, plus one
    ``val_metric_<task>`` entry per owned task.
    """
    owned = resolve_tasks(task, tasks)
    model.eval()
    n_samples = len(loader.dataset)
    if n_samples == 0:
        return {"loss": float("nan"), "metric": float("nan"), "metric_name": "na", "n": 0}

    if len(owned) > 1:
        return _evaluate_multitask(
            model, loader, owned, resolve_task_lambdas(owned, task_lambdas), device, num_classes,
            seg_criterion, cls_criterion, inversely_weighted, n_samples,
        )
    task = owned[0]

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


@torch.inference_mode()
def _evaluate_multitask(model, loader, tasks, lambdas, device, num_classes,
                        seg_criterion, cls_criterion, inversely_weighted, n_samples):
    """Combined-objective evaluation with one metric per owned task.

    Dice is averaged over the mask-supervised samples only, and accuracy over the label-supervised
    ones, so BUSI's ``normal`` images count towards classification without ever entering Dice.
    """
    total_loss, n_batches = 0.0, 0
    dice_sum, dice_batches = 0.0, 0
    gt, pred = [], []
    for data in loader:
        logits, outputs = model(data["image"].to(device))
        loss, _ = _combined_loss(
            data, logits, outputs, tasks, lambdas, device, num_classes,
            seg_criterion, cls_criterion, inversely_weighted,
        )
        total_loss += loss.item()
        n_batches += 1

        if "seg" in tasks:
            keep = _supervised(data, "seg", device, int(data["image"].shape[0]))
            if bool(keep.any()):
                masks = data["mask"].to(device)[keep]
                seg = outputs[-1] if isinstance(outputs, list) else outputs
                dice_sum += float(dice_score_from_tensor(masks, torch.sigmoid(seg[keep]) > 0.5))
                dice_batches += 1
        if "cls" in tasks:
            keep = _supervised(data, "cls", device, int(data["image"].shape[0]))
            if bool(keep.any()):
                label = _prep_label(data["label"].to(device)[keep], num_classes)
                pl = _avg_logits(logits)[keep]
                if num_classes > 2:
                    gt.extend(torch.argmax(label, dim=1).detach().cpu().tolist())
                    pred.extend(torch.argmax(torch.softmax(pl, dim=1), dim=1).detach().cpu().tolist())
                else:
                    gt.extend(label.flatten().detach().cpu().tolist())
                    pred.extend((torch.sigmoid(pl).flatten() > 0.5).double().detach().cpu().tolist())

    per_task = {}
    if "seg" in tasks:
        per_task["seg"] = dice_sum / dice_batches if dice_batches else float("nan")
    if "cls" in tasks:
        per_task["cls"] = float(accuracy_score(gt, pred)) if gt else float("nan")

    observed = [value for value in per_task.values() if value == value]  # drop NaN
    result = {
        "loss": total_loss / max(n_batches, 1),
        "metric": float(sum(observed) / len(observed)) if observed else float("nan"),
        "metric_name": "multitask",
        "n": n_samples,
    }
    result.update({f"metric_{name}": value for name, value in per_task.items()})
    return result
