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

# Parameter-name prefixes that make up the shared (federated) trunk of MTnnUNet.
# ``encoder1`` is a dataset/client-specific stem when modalities have different channel counts.
TRUNK_PREFIXES = ("encoder2", "encoder3", "encoder4", "encoder5", "bottleneck")
STEM_PREFIX = "encoder1"


def shared_prefixes(share_stem: bool = True) -> tuple:
    """Return the prefixes federated for the selected experiment mode.

    ``share_stem=True`` is the legacy single-dataset FedPer split. Multi-modal runs use
    ``False`` so inputs with different channel counts still expose an identical shared state.
    """
    return (STEM_PREFIX, *TRUNK_PREFIXES) if share_stem else TRUNK_PREFIXES


def is_shared(param_name: str, share_stem: bool = True) -> bool:
    """True if a state_dict key belongs to the shared (federated) trunk."""
    return param_name.startswith(shared_prefixes(share_stem))


def shared_keys(model: torch.nn.Module, share_stem: bool = True) -> list:
    """Stable, ordered list of the shared state_dict keys (used by both get/set)."""
    return [k for k in model.state_dict().keys() if is_shared(k, share_stem)]


def get_shared_state(model: torch.nn.Module, share_stem: bool = True) -> list:
    """Return the shared backbone weights as a list of numpy arrays (Flower exchange format)."""
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in shared_keys(model, share_stem)]


def set_shared_state(model: torch.nn.Module, arrays: list, share_stem: bool = True) -> None:
    """Load a list of numpy arrays (same order as ``get_shared_state``) into the backbone."""
    keys = shared_keys(model, share_stem)
    if len(keys) != len(arrays):
        raise ValueError(f"Expected {len(keys)} shared tensors, received {len(arrays)}")

    sd = model.state_dict()
    update = OrderedDict()
    for k, arr in zip(keys, arrays):
        ref = sd[k]
        if tuple(arr.shape) != tuple(ref.shape):
            raise ValueError(
                f"Shared tensor '{k}' has shape {tuple(ref.shape)}, received {tuple(arr.shape)}"
            )
        update[k] = torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    model.load_state_dict(update, strict=False)  # only the shared subset is provided


def get_personalized_state(model: torch.nn.Module, share_stem: bool = True) -> OrderedDict:
    """The complement of the shared block: the personalized head/decoder weights."""
    return OrderedDict((k, v.detach().cpu().clone())
                       for k, v in model.state_dict().items() if not is_shared(k, share_stem))


def set_personalized_state(model: torch.nn.Module, state: OrderedDict, share_stem: bool = True) -> None:
    """Load a previously persisted personalized state_dict back into the model."""
    expected = set(model.state_dict()) - set(shared_keys(model, share_stem))
    unexpected = set(state) - expected
    if unexpected:
        raise ValueError(f"Personalized state contains shared/unknown keys: {sorted(unexpected)[:3]}")
    model.load_state_dict(state, strict=False)  # only the personalized subset is provided


if __name__ == "__main__":
    # Self-test: round-trip the shared block and confirm the shared/personalized split is exhaustive.
    from src.models.multitask.MTnnUNet import MTnnUNet

    gray = MTnnUNet(sequences=1, regions=1, n_classes=3)
    rgb = MTnnUNet(sequences=3, regions=1, n_classes=7)
    total = len(gray.state_dict())

    for share_stem in (True, False):
        n_shared = len(shared_keys(gray, share_stem))
        n_personal = len(get_personalized_state(gray, share_stem))
        assert n_shared + n_personal == total, "shared + personalized must cover every tensor"
        assert n_shared > 0 and n_personal > 0

        arrays = [a + 1.0 for a in get_shared_state(gray, share_stem)]
        set_shared_state(gray, arrays, share_stem)
        assert all(np.allclose(a, b) for a, b in zip(arrays, get_shared_state(gray, share_stem)))

    gray_shapes = [a.shape for a in get_shared_state(gray, share_stem=False)]
    rgb_shapes = [a.shape for a in get_shared_state(rgb, share_stem=False)]
    assert gray_shapes == rgb_shapes, "multi-modal shared trunk shapes must match"
    assert [a.shape for a in get_shared_state(gray)] != [a.shape for a in get_shared_state(rgb)]

    print("OK: legacy and modality-specific-stem splits are exhaustive; shared trunk shapes match")
