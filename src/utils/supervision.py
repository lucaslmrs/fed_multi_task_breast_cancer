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


TASK_WEIGHTING_MODES = ("fixed", "uncertainty")
LOG_VARS_ATTR = "task_log_vars"


def task_weighting_mode(config) -> str:
    """``loss.task_weighting``: ``fixed`` (default, lambdas only) or ``uncertainty`` (Kendall)."""
    mode = str(config.get("loss", {}).get("task_weighting", "fixed")).lower()
    if mode not in TASK_WEIGHTING_MODES:
        raise ValueError(f"loss.task_weighting must be one of {TASK_WEIGHTING_MODES}, got {mode!r}")
    return mode


def attach_task_uncertainty(model, tasks):
    """Give ``model`` one learnable ``s_t = log sigma_t^2`` per task, initialised at 0.

    The parameters live under their own state_dict prefix, outside the shared trunk prefixes, so
    they are personalized: persisted with the client and never aggregated. Call before building
    the optimizer. A single-task client has nothing to balance and is left untouched.
    """
    tasks = list(tasks)
    if len(tasks) < 2:
        return model
    device = next(model.parameters()).device
    setattr(model, LOG_VARS_ATTR, torch.nn.ParameterDict(
        {name: torch.nn.Parameter(torch.zeros((), device=device)) for name in tasks}))
    return model


def attach_saved_task_uncertainty(model, state_dict):
    """Recreate the log-variance slots a saved state carries, so it loads into a fresh model.

    Evaluation never uses them; it only needs the state_dict keys to match. Reading the task names
    from the state itself keeps this exact for any task set and a no-op for fixed-weighting runs.
    """
    prefix = f"{LOG_VARS_ATTR}."
    tasks = [key[len(prefix):] for key in state_dict if key.startswith(prefix)]
    if tasks and not hasattr(model, LOG_VARS_ATTR):
        device = next(model.parameters()).device
        setattr(model, LOG_VARS_ATTR, torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.zeros((), device=device)) for name in tasks}))
    return model


def uncertainty_weighted_loss(raw_terms, lambdas, log_vars):
    """Kendall et al. 2018 (Eq. 10, classification form): ``sum_t lambda_t e^{-s_t} L_t + s_t/2``.

    Only the tasks present in ``raw_terms`` contribute: a task without a supervised sample in the
    batch is omitted, its ``s_t/2`` regulariser included, never added as zero.
    """
    terms = []
    for task, term in raw_terms.items():
        s = log_vars[task].float()
        terms.append(lambdas[task] * torch.exp(-s) * term.float() + 0.5 * s)
    return torch.sum(torch.stack(terms))


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
