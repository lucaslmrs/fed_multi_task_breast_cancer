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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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
    """``sum_t lambda_t * L_t`` over supervised samples plus raw per-task terms."""
    batch_size = int(data["image"].shape[0])
    terms, counted, raw_terms = [], [], {}
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
        raw_terms[task] = term
    if not terms:
        raise RuntimeError(
            "A batch carried no supervision for any owned task; the client partition is malformed"
        )
    return torch.sum(torch.stack(terms)), counted, raw_terms


def _empty_task_stats(tasks):
    return {
        task: {
            "n_samples": 0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "predicted_positive": 0.0,
            "ground_truth_positive": 0.0,
            "ground_truth": [],
            "predicted": [],
        }
        for task in tasks
    }


def _update_task_stats(stats, data, logits, outputs, task, device, num_classes):
    """Accumulate metrics from the forward pass already used for optimization."""
    batch_size = int(data["image"].shape[0])
    keep = _supervised(data, task, device, batch_size)
    if not bool(keep.any()):
        return
    current = stats[task]
    current["n_samples"] += int(keep.sum().item())
    if task == "seg":
        masks = data["mask"].to(device)[keep].to(torch.bool)
        seg = outputs[-1] if isinstance(outputs, list) else outputs
        predicted = (torch.sigmoid(seg[keep]) > 0.5).to(torch.bool)
        current["tp"] += float(torch.logical_and(predicted, masks).sum().item())
        current["fp"] += float(torch.logical_and(predicted, torch.logical_not(masks)).sum().item())
        current["fn"] += float(torch.logical_and(torch.logical_not(predicted), masks).sum().item())
        current["predicted_positive"] += float(predicted.sum().item())
        current["ground_truth_positive"] += float(masks.sum().item())
        return

    label = _prep_label(data["label"].to(device)[keep], num_classes)
    averaged = _avg_logits(logits)[keep]
    if num_classes > 2:
        current["ground_truth"].extend(
            torch.argmax(label, dim=1).detach().cpu().tolist()
        )
        current["predicted"].extend(
            torch.argmax(torch.softmax(averaged, dim=1), dim=1).detach().cpu().tolist()
        )
    else:
        current["ground_truth"].extend(
            label.flatten().to(torch.int64).detach().cpu().tolist()
        )
        current["predicted"].extend(
            (torch.sigmoid(averaged).flatten() > 0.5).to(torch.int64).detach().cpu().tolist()
        )


def _finalize_task_metrics(stats, task, num_classes):
    current = stats[task]
    if not current["n_samples"]:
        return {}
    if task == "seg":
        tp, fp, fn = current["tp"], current["fp"], current["fn"]
        if current["ground_truth_positive"] == 0:
            dice = 1.0 if current["predicted_positive"] == 0 else 0.0
            iou = dice
        else:
            dice = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
            iou = tp / max(tp + fp + fn, 1.0)
        return {"dice": float(dice), "iou": float(iou)}

    ground_truth = current["ground_truth"]
    predicted = current["predicted"]
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(ground_truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(ground_truth, predicted)),
        "macro_f1": float(f1_score(
            ground_truth, predicted, labels=labels, average="macro", zero_division=0
        )),
    }


def _history_rows(tasks, task_stats, task_loss_sums, task_loss_counts,
                  combined_loss, combined_count, unit_type, unit_index,
                  optimizer_steps, examples_processed):
    rows = []
    if len(tasks) > 1:
        rows.append({
            "task": "combined", "metric_name": "loss",
            "value": combined_loss / max(combined_count, 1),
            "n_samples": sum(task_stats[name]["n_samples"] for name in tasks),
        })
    for task in tasks:
        if task_loss_counts[task]:
            rows.append({
                "task": task, "metric_name": "loss",
                "value": task_loss_sums[task] / task_loss_counts[task],
                "n_samples": task_stats[task]["n_samples"],
            })
        for metric_name, value in _finalize_task_metrics(
            task_stats, task, num_classes=task_stats[task].get("num_classes", 2)
        ).items():
            rows.append({
                "task": task, "metric_name": metric_name, "value": value,
                "n_samples": task_stats[task]["n_samples"],
            })
    for row in rows:
        row.update({
            "unit_type": unit_type,
            "unit_index": int(unit_index),
            "optimizer_steps": int(optimizer_steps),
            "examples_processed": int(examples_processed),
        })
    return rows


def train_local(model, loader, optimizer, task, device, local_epochs, num_classes,
                seg_criterion=None, cls_criterion=None, inversely_weighted=True,
                training_mode="epochs", steps_per_round=10, tasks=None, task_lambdas=None,
                epoch_end_callback=None, record_step_history=False):
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
    task_loss_sums = {name: 0.0 for name in owned}
    task_loss_counts = {name: 0 for name in owned}
    task_stats = _empty_task_stats(owned)
    for name in owned:
        task_stats[name]["num_classes"] = num_classes
    history, validation_history = [], []
    passes = local_epochs if training_mode == "epochs" else 1
    for pass_index in range(1, passes + 1):
        epoch_running, epoch_batches, epoch_examples = 0.0, 0, 0
        epoch_loss_sums = {name: 0.0 for name in owned}
        epoch_loss_counts = {name: 0 for name in owned}
        epoch_stats = _empty_task_stats(owned)
        for name in owned:
            epoch_stats[name]["num_classes"] = num_classes
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
                raw_terms = {owned[0]: loss}
            else:
                loss, counted, raw_terms = _combined_loss(
                    data, logits, outputs, owned, lambdas, device, num_classes,
                    seg_criterion, cls_criterion, inversely_weighted,
                )
            for name in counted:
                _update_task_stats(task_stats, data, logits, outputs, name, device, num_classes)
                _update_task_stats(epoch_stats, data, logits, outputs, name, device, num_classes)
                value = float(raw_terms[name].detach().item())
                task_loss_sums[name] += value
                task_loss_counts[name] += 1
                epoch_loss_sums[name] += value
                epoch_loss_counts[name] += 1
            if training_mode == "steps" and record_step_history:
                step_stats = _empty_task_stats(owned)
                for name in owned:
                    step_stats[name]["num_classes"] = num_classes
                for name in counted:
                    _update_task_stats(
                        step_stats, data, logits, outputs, name, device, num_classes
                    )
                history.extend(_history_rows(
                    owned,
                    step_stats,
                    {name: float(raw_terms[name].detach().item()) if name in raw_terms else 0.0
                     for name in owned},
                    {name: int(name in raw_terms) for name in owned},
                    float(loss.detach().item()), 1,
                    "step", n_batches + 1, 1, int(inputs.shape[0]),
                ))
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
            examples_processed += int(inputs.shape[0])
            epoch_running += loss.item()
            epoch_batches += 1
            epoch_examples += int(inputs.shape[0])
            for name in counted:
                task_batches[name] += 1

        if training_mode == "epochs":
            history.extend(_history_rows(
                owned, epoch_stats, epoch_loss_sums, epoch_loss_counts,
                epoch_running, epoch_batches, "epoch", pass_index,
                epoch_batches, epoch_examples,
            ))
            if epoch_end_callback is not None:
                validation_history.append({
                    "unit_type": "epoch",
                    "unit_index": pass_index,
                    "result": epoch_end_callback(pass_index),
                })
                model.train()

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
        "task_losses": {
            name: task_loss_sums[name] / max(task_loss_counts[name], 1) for name in owned
        },
        "task_metrics": {
            name: _finalize_task_metrics(task_stats, name, num_classes) for name in owned
        },
        "history": history,
        "validation_history": validation_history,
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
        stats = _empty_task_stats(("seg",))
        stats["seg"]["num_classes"] = num_classes
        for data in loader:
            logits, outputs = model(data["image"].to(device))
            masks = data["mask"].to(device)
            total_loss += _seg_loss(seg_criterion, masks, outputs, inversely_weighted).item()
            seg = outputs[-1] if isinstance(outputs, list) else outputs
            dice += float(dice_score_from_tensor(masks, torch.sigmoid(seg) > 0.5))
            _update_task_stats(stats, data, logits, outputs, "seg", device, num_classes)
            n += 1
        metrics = _finalize_task_metrics(stats, "seg", num_classes)
        # Preserve the historical batch-mean Dice as the legacy ``metric`` field. The telemetry
        # metrics use one consistent micro accumulation, which also provides IoU without a pass.
        return {
            "loss": total_loss / n, "metric": dice / n, "metric_name": "dice",
            "n": n_samples, "metrics": {"seg": metrics}, "task_losses": {"seg": total_loss / n},
        }

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
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_accuracy_score(gt, pred)),
        "macro_f1": float(f1_score(
            gt, pred, labels=list(range(num_classes)), average="macro", zero_division=0
        )),
    }
    return {
        "loss": total_loss / n, "metric": acc, "f1": f1v, "metric_name": "acc",
        "n": n_samples, "metrics": {"cls": metrics}, "task_losses": {"cls": total_loss / n},
    }


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
    stats = _empty_task_stats(tasks)
    for name in tasks:
        stats[name]["num_classes"] = num_classes
    task_loss_sums = {name: 0.0 for name in tasks}
    task_loss_counts = {name: 0 for name in tasks}
    for data in loader:
        logits, outputs = model(data["image"].to(device))
        loss, _, raw_terms = _combined_loss(
            data, logits, outputs, tasks, lambdas, device, num_classes,
            seg_criterion, cls_criterion, inversely_weighted,
        )
        total_loss += loss.item()
        n_batches += 1
        for name, term in raw_terms.items():
            task_loss_sums[name] += float(term.item())
            task_loss_counts[name] += 1
            _update_task_stats(stats, data, logits, outputs, name, device, num_classes)

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
    metrics = {name: _finalize_task_metrics(stats, name, num_classes) for name in tasks}
    result = {
        "loss": total_loss / max(n_batches, 1),
        "metric": float(sum(observed) / len(observed)) if observed else float("nan"),
        "metric_name": "multitask",
        "n": n_samples,
        "metrics": metrics,
        "task_losses": {
            name: task_loss_sums[name] / max(task_loss_counts[name], 1) for name in tasks
        },
    }
    result.update({f"metric_{name}": value for name, value in per_task.items()})
    return result
