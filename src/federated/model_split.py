"""
Splitting a multi-task model into the FEDERATED (shared) block and the PERSONALIZED (local) block.

FedPer scheme: the shared block is averaged with FedAvg; everything else is personalized (never
aggregated, persisted per client to disk). Which top-level modules are shared is the ``sharing``
spec every helper below takes, resolved from the config by ``src.federated.config.shared_modules``:

    - ``bool`` -- the legacy spec, ``federated.share_stem`` with no ``federated.shared_modules``.
      State_dict keys are matched by PREFIX against ``encoder2..5`` + ``bottleneck`` (plus the
      ``encoder1`` stem when true). This is byte-identical to every historical run, including the
      prefix quirk that also captures Multi_BTSUNet's ``bottleneck2``.
    - ``tuple`` of names -- an explicit ``federated.shared_modules`` list. A key is shared iff its
      first component (``name.split(".", 1)[0]``) is EXACTLY one of the names, so ``enc`` never
      captures ``enc2``. E6 uses it to federate the decoders, upsamples and ``process_*`` too.

The personalized remainder always includes whatever the spec leaves out: on MTnnUNet the modality
stem (``encoder1``, unless shared), the output heads and the classifier, plus Kendall's
``task_log_vars`` when attached. ``validate_sharing`` refuses an empty shared set, non-float
shared tensors and, for an explicit list, names that are not top-level modules owning a tensor.
``assert_pretrained_shared`` refuses a spec that leaves a pretrained module (E7) personalized.

Shared BUFFERS travel with the shared block too: MTResNet34UNet's FrozenBatchNorm2d statistics are
float32 buffers inside shared modules, identical on every client, so FedAvg returns them unchanged.

The helpers convert the shared block to/from a list of numpy arrays (the exchange format Flower
expects) keeping a stable key ordering, and read/write the personalized block as a state_dict for
on-disk persistence.
"""

from collections import OrderedDict

import numpy as np
import torch

# Top-level modules of MTnnUNet's shared trunk under the legacy spec. ``encoder1`` is the
# dataset/client-specific stem when modalities have different channel counts.
TRUNK_MODULES = ("encoder2", "encoder3", "encoder4", "encoder5", "bottleneck")
STEM_MODULE = "encoder1"
# Historical names: under the legacy spec these are matched as key prefixes.
TRUNK_PREFIXES = TRUNK_MODULES
STEM_PREFIX = STEM_MODULE


def _spec(sharing):
    """Normalize a sharing spec: ``bool`` (legacy prefix rule) or ``tuple`` (exact names)."""
    if isinstance(sharing, str):
        raise TypeError(
            "sharing must be a bool (legacy share_stem) or a tuple of top-level module names; "
            "resolve it with src.federated.config.shared_modules"
        )
    if isinstance(sharing, (list, tuple)):
        return tuple(sharing)
    return bool(sharing)


def shared_prefixes(sharing=True) -> tuple:
    """Return what the selected spec federates.

    Legacy ``True`` is the single-dataset FedPer split (stem shared); ``False`` keeps the stem
    personalized so inputs with different channel counts still expose an identical shared state.
    Both are matched as key prefixes. An explicit tuple is returned as is: exact top-level names.
    """
    spec = _spec(sharing)
    if isinstance(spec, tuple):
        return spec
    return (STEM_MODULE, *TRUNK_MODULES) if spec else TRUNK_MODULES


def is_shared(param_name: str, sharing=True) -> bool:
    """True if a state_dict key belongs to the shared (federated) block."""
    spec = _spec(sharing)
    if isinstance(spec, tuple):
        return str(param_name).split(".", 1)[0] in spec
    return str(param_name).startswith(shared_prefixes(spec))


def shared_keys(model: torch.nn.Module, sharing=True) -> list:
    """Stable, ordered list of the shared state_dict keys (used by both get/set)."""
    spec = _spec(sharing)
    return [k for k in model.state_dict().keys() if is_shared(k, spec)]


def resolved_shared_modules(model: torch.nn.Module, sharing=True) -> list:
    """The top-level modules that are actually shared, in state_dict order."""
    return list(dict.fromkeys(str(key).split(".", 1)[0] for key in shared_keys(model, sharing)))


def validate_sharing(model: torch.nn.Module, sharing=True) -> list:
    """Fail before Flower if the spec cannot describe a usable shared block; return its keys.

    An empty shared set used to pass silently (MTUNetPlusPlus matches no legacy prefix), leaving
    FedAvg nothing to average. Shared tensors must be floating point because FedAvg averages them.
    An explicit list must name top-level children that own at least one state_dict tensor, which
    rejects typos (``decoder6``) and parameter-free modules (MTnnUNet's ``downsample``).
    """
    spec = _spec(sharing)
    state = model.state_dict()
    architecture = type(model).__name__
    if isinstance(spec, tuple):
        children = dict(model.named_children())
        owners = list(dict.fromkeys(str(key).split(".", 1)[0] for key in state))
        unknown = [name for name in spec if name not in children]
        if unknown:
            raise ValueError(
                f"federated.shared_modules names {unknown}, which are not top-level modules of "
                f"{architecture}; modules owning tensors: {owners}"
            )
        empty = [name for name in spec if name not in owners]
        if empty:
            raise ValueError(
                f"federated.shared_modules names {empty}, which own no parameter or buffer in "
                f"{architecture}; there is nothing to federate in them"
            )
    keys = shared_keys(model, spec)
    if not keys:
        raise ValueError(
            f"{architecture} exposes no shared tensor under sharing={spec!r} (matched against "
            f"{shared_prefixes(spec)}); check federated.shared_modules/share_stem"
        )
    non_float = [key for key in keys if not state[key].is_floating_point()]
    if non_float:
        raise ValueError(
            f"Shared tensors must be floating point for FedAvg; {architecture} shares "
            f"{non_float[:3]} ({len(non_float)} non-float)"
        )
    return keys


def assert_pretrained_shared(model: torch.nn.Module, sharing=True) -> None:
    """Refuse a spec that leaves any of ``model.pretrained_modules`` personalized.

    Pretrained weights reach the clients only through the server's initial global parameters: a
    client builds its model with random weights and is never handed the checkpoint. A pretrained
    module left personalized would therefore start from random weights on every client. Models
    without ``pretrained_modules`` pass trivially.
    """
    pretrained = tuple(getattr(model, "pretrained_modules", ()) or ())
    if not pretrained:
        return
    spec = _spec(sharing)
    personal = [key for key in model.state_dict()
                if str(key).split(".", 1)[0] in pretrained and not is_shared(key, spec)]
    if personal:
        blocks = list(dict.fromkeys(str(key).split(".", 1)[0] for key in personal))
        raise ValueError(
            f"{type(model).__name__} keeps pretrained modules {blocks} personalized under "
            f"sharing={spec!r}; clients receive pretrained weights only through the global "
            "initial parameters, so list every one of them in federated.shared_modules"
        )


def get_shared_state(model: torch.nn.Module, sharing=True) -> list:
    """Return the shared weights as a list of numpy arrays (Flower exchange format)."""
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in shared_keys(model, sharing)]


def set_shared_state(model: torch.nn.Module, arrays: list, sharing=True) -> None:
    """Load a list of numpy arrays (same order as ``get_shared_state``) into the shared block."""
    keys = shared_keys(model, sharing)
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


def get_personalized_state(model: torch.nn.Module, sharing=True) -> OrderedDict:
    """The complement of the shared block: the personalized stem/head/decoder weights."""
    spec = _spec(sharing)
    return OrderedDict((k, v.detach().cpu().clone())
                       for k, v in model.state_dict().items() if not is_shared(k, spec))


def set_personalized_state(model: torch.nn.Module, state: OrderedDict, sharing=True) -> None:
    """Load a previously persisted personalized state_dict back into the model."""
    expected = set(model.state_dict()) - set(shared_keys(model, sharing))
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
    wide = (*TRUNK_MODULES, "decoder5", "decoder4", "decoder3", "decoder2", "decoder1",
            "upsample5", "upsample4", "upsample3", "upsample2", "upsample1",
            "process_encoder_5", "process_decoder_5")

    for sharing in (True, False, wide):
        validate_sharing(gray, sharing)
        n_shared = len(shared_keys(gray, sharing))
        n_personal = len(get_personalized_state(gray, sharing))
        assert n_shared + n_personal == total, "shared + personalized must cover every tensor"
        assert n_shared > 0 and n_personal > 0

        arrays = [a + 1.0 for a in get_shared_state(gray, sharing)]
        set_shared_state(gray, arrays, sharing)
        assert all(np.allclose(a, b) for a, b in zip(arrays, get_shared_state(gray, sharing)))

    for sharing in (False, wide):
        gray_shapes = [a.shape for a in get_shared_state(gray, sharing=sharing)]
        rgb_shapes = [a.shape for a in get_shared_state(rgb, sharing=sharing)]
        assert gray_shapes == rgb_shapes, "multi-modal shared shapes must match"
    assert [a.shape for a in get_shared_state(gray)] != [a.shape for a in get_shared_state(rgb)]
    assert resolved_shared_modules(gray, wide) == list(wide)

    print("OK: legacy, modality-specific-stem and explicit splits are exhaustive; "
          "multi-modal shared shapes match")
