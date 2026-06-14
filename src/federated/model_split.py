"""
Splitting an MTnnUNet into the FEDERATED (shared) block and the PERSONALIZED (local) block.

FedPer scheme (approved):
    - Shared / federated via FedAvg: the encoder backbone + bottleneck.
    - Personalized / local (never aggregated, persisted per client to disk): everything else
      (segmentation decoder, upsamples, outputs, and the classification head). Because each
      client owns a single task, a seg client uses upsample5/decoder5 through the segmentation
      path while a cls client uses them through the classifier path -- never both at once.

The helpers below convert the shared block to/from a list of numpy arrays (the exchange format
Flower expects) keeping a stable key ordering, and read/write the personalized block as a
state_dict for on-disk persistence.
"""

from collections import OrderedDict

import numpy as np
import torch

# Parameter-name prefixes that make up the shared (federated) backbone of MTnnUNet.
SHARED_PREFIXES = ("encoder1", "encoder2", "encoder3", "encoder4", "encoder5", "bottleneck")


def is_shared(param_name: str) -> bool:
    """True if a state_dict key belongs to the shared (federated) backbone."""
    return param_name.startswith(SHARED_PREFIXES)


def shared_keys(model: torch.nn.Module) -> list:
    """Stable, ordered list of the shared state_dict keys (used by both get/set)."""
    return [k for k in model.state_dict().keys() if is_shared(k)]


def get_shared_state(model: torch.nn.Module) -> list:
    """Return the shared backbone weights as a list of numpy arrays (Flower exchange format)."""
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in shared_keys(model)]


def set_shared_state(model: torch.nn.Module, arrays: list) -> None:
    """Load a list of numpy arrays (same order as ``get_shared_state``) into the backbone."""
    keys = shared_keys(model)
    if len(keys) != len(arrays):
        raise ValueError(f"Expected {len(keys)} shared tensors, received {len(arrays)}")

    sd = model.state_dict()
    update = OrderedDict()
    for k, arr in zip(keys, arrays):
        ref = sd[k]
        update[k] = torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    model.load_state_dict(update, strict=False)  # only the shared subset is provided


def get_personalized_state(model: torch.nn.Module) -> OrderedDict:
    """The complement of the shared block: the personalized head/decoder weights."""
    return OrderedDict((k, v.detach().cpu().clone())
                       for k, v in model.state_dict().items() if not is_shared(k))


def set_personalized_state(model: torch.nn.Module, state: OrderedDict) -> None:
    """Load a previously persisted personalized state_dict back into the model."""
    model.load_state_dict(state, strict=False)  # only the personalized subset is provided


if __name__ == "__main__":
    # Self-test: round-trip the shared block and confirm the shared/personalized split is exhaustive.
    from src.models.multitask.MTnnUNet import MTnnUNet

    model = MTnnUNet(sequences=1, regions=1, n_classes=3)
    total = len(model.state_dict())
    n_shared = len(shared_keys(model))
    n_personal = len(get_personalized_state(model))
    assert n_shared + n_personal == total, "shared + personalized must cover every tensor"
    assert n_shared > 0 and n_personal > 0

    # mutate shared weights, push them in, pull them back -> must match
    arrays = [a + 1.0 for a in get_shared_state(model)]
    set_shared_state(model, arrays)
    reloaded = get_shared_state(model)
    assert all(np.allclose(a, b) for a, b in zip(arrays, reloaded))

    print(f"OK: {total} tensors total -> {n_shared} shared + {n_personal} personalized; round-trip matches")
