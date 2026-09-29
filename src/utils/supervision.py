"""Shared sample-level supervision and multitask objective (logits in, masked losses out)."""
import torch
import torch.nn.functional as F
from src.utils.training_runtime import move_to_device


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
    return move_to_device(flag.reshape(-1), device).to(dtype=torch.bool)


def _combined_loss(data, logits, outputs, tasks, lambdas, device, num_classes,
                   seg_criterion, cls_criterion, inversely_weighted):
    """``sum_t lambda_t * L_t`` over supervised samples plus raw per-task terms."""
    batch_size = int(data["image"].shape[0])
    supervision = torch.stack([_supervised(data, t, device, batch_size) for t in tasks])
    if not bool(supervision.any(dim=0).all()):
        raise ValueError("Each image must supervise at least one requested task")
    terms, counted, raw_terms = [], [], {}
    for task in tasks:
        keep = _supervised(data, task, device, batch_size)
        if not bool(keep.any()):
            continue
        if task == "seg":
            masks = move_to_device(data["mask"], device)[keep]
            selected = [item[keep] for item in outputs] if isinstance(outputs, list) else outputs[keep]
            term = _seg_loss(seg_criterion, masks, selected, inversely_weighted)
        else:
            label = _prep_label(move_to_device(data["label"], device)[keep], num_classes)
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




class WeightedBinaryClassificationLoss(torch.nn.Module):
    """Weight both binary classes by training-only frequencies, including negative examples."""
    def __init__(self, weights):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.shape != (2,) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Expected two finite non-negative binary class weights')
        self.register_buffer('weights', weights)

    def forward(self, logits, target):
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return (losses * self.weights[target.long()]).mean()
