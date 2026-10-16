"""Online gradient-conflict diagnostics: M1 (gradient geometry) and M2 (lookahead affinity).

Both are observational. They run on sampled rounds only, inside ``_preserve_rng_state`` on the
client, and never step the model, so a run produces the same training trajectory with them on or
off. Sources and the validation of both methods: ``docs/GRADIENT_CONFLICT_METHODS.md`` and
``plano.md`` (stage E0).

M1 (PCGrad's definitions). Before local training, a client differentiates its own local objective
w.r.t. the shared trunk it RECEIVED, over the round's probe batches. Two halves of the batches give
``g_A`` and ``g_B``; ``g = (g_A + g_B) / 2``. Exact per-block norms and the exact ``g_A·g_B`` are
stored together with a CountSketch of ``g_A`` and ``g_B`` (shared seed, so sketches of different
clients are comparable). Offline this yields, per pair and block:

    cos_ij          from sketched inner products and exact norms
    phi_ij          gradient magnitude similarity 2|g_i||g_j| / (|g_i|^2 + |g_j|^2)
    ceiling_i       cos(g_A, g_B), Spearman-Brown corrected to the full average: 2r / (1 + r)
    cos_normalized  cos_ij / sqrt(ceiling_i * ceiling_j), the cosine between the noise-free parts

M2 (TAG's lookahead affinity with the real round update). After aggregation the server persists
the sent trunk and every client's real round update; in the evaluate phase client j scores

    Z(i -> j) = 1 - L_j(theta + Delta_i) / L_j(theta)

with its PRE-round personalized parameters fixed, on a train and a validation split. ``source`` is
``__aggregate__`` for the aggregated trunk. Z(train) and Z(val) negative together point at
interference; Z(train) >= 0 with Z(val) < 0 points at overfitting instead.
"""

import itertools
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.federated.model_split import (get_personalized_state, set_personalized_state,
                                       set_shared_state, shared_keys)
from src.utils.training_runtime import runtime_config

CONFLICT_DIR = "conflict"
DELTAS_FILE = "deltas.npz"
SKETCH_SEED = 20261009
AGGREGATE_SOURCE = "__aggregate__"
TOTAL_BLOCK = "shared_total"
M1_FILE = "conflict_m1_pairs.csv"
M2_FILE = "conflict_m2_lookahead.csv"
SUMMARY_FILE = "conflict_summary.csv"
MISSING_STATUS = "n/a: run without gradient-conflict diagnostics"


# ---- configuration ---------------------------------------------------------------------------
def conflict_config(config):
    return runtime_config(config)["diagnostics"]["gradient_conflict"]


def is_enabled(config):
    return bool(conflict_config(config)["enabled"]) and not config["federated"].get(
        "standalone", False)


def lookahead_enabled(config):
    return is_enabled(config) and bool(conflict_config(config)["lookahead"]["enabled"])


def is_sampled_round(config, server_round):
    """Rounds 1, 1 + n, 1 + 2n, ... and always the last one."""
    if not is_enabled(config) or int(server_round) < 1:
        return False
    every = int(conflict_config(config)["every_n_rounds"])
    return (int(server_round) - 1) % every == 0 or int(server_round) == int(
        config["federated"]["rounds"])


def round_dir(fold_dir, server_round):
    return Path(fold_dir) / CONFLICT_DIR / f"round_{int(server_round):04d}"


def _block(key):
    return str(key).split(".")[0]


# ---- CountSketch -----------------------------------------------------------------------------
class CountSketch:
    """Unbiased inner-product sketch: bucket h(k) and sign s(k) per coordinate, fixed seed."""

    def __init__(self, size, dim, seed):
        generator = torch.Generator().manual_seed(int(seed))
        self.size, self.dim = int(size), int(dim)
        self.index = torch.randint(0, self.dim, (self.size,), generator=generator)
        self.sign = torch.randint(0, 2, (self.size,), generator=generator).to(torch.float64) * 2 - 1

    def __call__(self, vector):
        vector = torch.as_tensor(vector).detach().to("cpu", torch.float64).reshape(-1)
        if vector.numel() != self.size:
            raise ValueError(f"sketch expects {self.size} coordinates, got {vector.numel()}")
        return torch.zeros(self.dim, dtype=torch.float64).index_add_(
            0, self.index, vector * self.sign)


_SKETCHES = {}


def block_sketch(block, size, total_size, sketch_dim):
    """Per-block sketch whose width is proportional to the block, identical in every process."""
    dim = max(256, int(round(sketch_dim * size / total_size)))
    key = (block, size, dim)
    if key not in _SKETCHES:
        _SKETCHES[key] = CountSketch(size, dim, SKETCH_SEED + zlib.crc32(block.encode()))
    return _SKETCHES[key]


def observational(function, label):
    """Run a diagnostic so that its failure is logged, never propagated into training."""
    try:
        return function()
    except Exception:
        logging.warning("[gradient conflict] %s failed; training continues", label, exc_info=True)
        return None


# ---- client side: M1 probes ------------------------------------------------------------------
class BatchList(list):
    """A list of batches that ``evaluate_local`` consumes like a DataLoader."""

    @property
    def dataset(self):
        return range(sum(int(batch["image"].shape[0]) for batch in self))


def trunk_gradient_halves(client, batches):
    """Mean trunk gradient of the client's local objective over two halves of ``batches``."""
    from src.federated.local_trainer import local_objective
    from src.utils.supervision import resolve_task_lambdas

    model = client.model
    keys = shared_keys(model, client.share_stem)
    params = dict(model.named_parameters())
    lambdas = resolve_task_lambdas(client.tasks, client.task_lambdas)
    half = len(batches) // 2
    sums = [{key: torch.zeros_like(params[key], dtype=torch.float32) for key in keys}
            for _ in range(2)]
    model.train()
    for position, data in enumerate(batches):
        model.zero_grad(set_to_none=True)
        inputs = client.precision.move(data["image"])
        loss, *_ = local_objective(
            model, data, inputs, client.tasks, lambdas, client.device, client.num_classes,
            client.seg_criterion, client.cls_criterion, client.inversely_weighted,
            client.precision,
        )
        client.precision.ensure_finite(loss, "gradient-conflict probe")
        loss.backward()
        target = sums[0 if position < half else 1]
        for key in keys:
            target[key] += params[key].grad.detach().float()
    model.zero_grad(set_to_none=True)
    counts = (half, len(batches) - half)
    return keys, [{key: value / count for key, value in group.items()}
                  for group, count in zip(sums, counts)]


def collect_fit_probes(client, server_round):
    """Persist M1 probes and, for M2, the pre-round personalized state. Never steps the model."""
    settings = conflict_config(client.config)
    directory = round_dir(client.state_dir.parent, server_round)
    directory.mkdir(parents=True, exist_ok=True)
    if lookahead_enabled(client.config):
        torch.save(get_personalized_state(client.model, client.share_stem),
                   directory / f"{client.client_id}_pre.pt")
    # In steps mode the round yields ``steps_per_round`` batches, which caps the probe count.
    batches = list(itertools.islice(iter(client.train_loader), settings["probe_batches"]))
    if len(batches) < 2:
        logging.warning("[%s] M1 skipped at round %s: %s probe batch(es); two halves need >= 2",
                        client.client_id, server_round, len(batches))
        return
    keys, (half_a, half_b) = trunk_gradient_halves(client, batches)

    blocks = list(dict.fromkeys(_block(key) for key in keys))
    sizes = {block: sum(half_a[k].numel() for k in keys if _block(k) == block) for block in blocks}
    total = sum(sizes.values())
    payload = {"blocks": np.asarray(blocks)}
    for block in blocks:
        a = torch.cat([half_a[k].reshape(-1) for k in keys if _block(k) == block]).double().cpu()
        b = torch.cat([half_b[k].reshape(-1) for k in keys if _block(k) == block]).double().cpu()
        sketch = block_sketch(block, sizes[block], total, settings["sketch_dim"])
        g = (a + b) / 2
        payload[f"{block}::sketch_a"] = sketch(a).numpy()
        payload[f"{block}::sketch_b"] = sketch(b).numpy()
        payload[f"{block}::exact"] = np.asarray(
            [float(a @ a), float(b @ b), float(a @ b), float(g @ g)], dtype=np.float64)
    np.savez(directory / f"{client.client_id}_m1.npz", **payload)


# ---- server side: deltas for M2 --------------------------------------------------------------
def write_round_deltas(directory, sent_arrays, updates):
    """Persist the sent trunk and every client's real round update for the evaluate phase."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"client_ids": np.asarray([update["client_id"] for update in updates])}
    for index, array in enumerate(sent_arrays):
        payload[f"theta::{index}"] = array
    for position, update in enumerate(updates):
        for index, (array, sent) in enumerate(zip(update["arrays"], sent_arrays)):
            payload[f"delta::{position}::{index}"] = array - sent
    np.savez(directory / DELTAS_FILE, **payload)


def cleanup_round(directory):
    """Drop the large per-round transients; the CSV/sketch results stay."""
    directory = Path(directory)
    for path in [directory / DELTAS_FILE, *directory.glob("*_pre.pt")]:
        if path.exists():
            path.unlink()


# ---- client side: M2 lookahead ---------------------------------------------------------------
def _loss(client, personalized, trunk, batches):
    from src.federated import local_trainer

    set_personalized_state(client.model, personalized, share_stem=client.share_stem)
    set_shared_state(client.model, trunk, share_stem=client.share_stem)
    result = local_trainer.evaluate_local(
        client.model, batches, client.task, client.device, client.num_classes,
        client.seg_criterion, client.cls_criterion, client.inversely_weighted,
        tasks=client.tasks, task_lambdas=client.task_lambdas, precision=client.precision,
    )
    return float(result["loss"])


def collect_lookahead(client, aggregated_parameters, server_round):
    """Score every client's real update, and the aggregate, on this client's train/val batches."""
    settings = conflict_config(client.config)["lookahead"]
    directory = round_dir(client.state_dir.parent, server_round)
    deltas_path = directory / DELTAS_FILE
    pre_path = directory / f"{client.client_id}_pre.pt"
    if not deltas_path.exists() or not pre_path.exists():
        logging.warning("[%s] lookahead skipped at round %s: transient files missing",
                        client.client_id, server_round)
        return
    personalized = torch.load(pre_path, map_location=client.device)
    with np.load(deltas_path) as stored:
        client_ids = [str(value) for value in stored["client_ids"]]
        n_arrays = sum(1 for key in stored.files if key.startswith("theta::"))
        theta = [stored[f"theta::{index}"] for index in range(n_arrays)]
        deltas = [[stored[f"delta::{position}::{index}"] for index in range(n_arrays)]
                  for position in range(len(client_ids))]

    sampler = getattr(client.train_loader, "batch_sampler", None)
    if hasattr(sampler, "set_round"):
        sampler.set_round(max(int(server_round), 1))
    splits = {
        "train": BatchList(itertools.islice(iter(client.train_loader), settings["train_batches"])),
        "val": BatchList(itertools.islice(iter(client.val_loader), settings["val_batches"])),
    }
    sources = [(source, [t + d for t, d in zip(theta, delta)])
               for source, delta in zip(client_ids, deltas)]
    # Copy: on CPU, arrays from ``get_shared_state`` alias the live model weights.
    sources.append((AGGREGATE_SOURCE, [np.array(a, copy=True) for a in aggregated_parameters]))
    rows = []
    for split, batches in splits.items():
        if not len(batches):
            continue
        base = _loss(client, personalized, theta, batches)
        for source, trunk in sources:
            value = _loss(client, personalized, trunk, batches)
            rows.append({"round": int(server_round), "source": source,
                         "target": client.client_id, "split": split, "base_loss": base,
                         "loss": value, "z": 1.0 - value / base if base > 0 else float("nan")})
    pd.DataFrame(rows).to_csv(directory / f"{client.client_id}_m2.csv", index=False)


# ---- offline consolidation -------------------------------------------------------------------
def _client_metadata(fold_dir):
    meta = {}
    for path in Path(fold_dir).glob("client_*/metadata.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        meta[str(data["client_id"])] = (str(data["dataset"]), tuple(data.get("tasks") or
                                                                   [data["task"]]))
    return meta


def pair_class(meta, a, b):
    if a == b:
        return "self"
    if a == AGGREGATE_SOURCE or b == AGGREGATE_SOURCE:
        return "aggregate"
    (dataset_a, tasks_a), (dataset_b, tasks_b) = meta[a], meta[b]
    if dataset_a == dataset_b:
        return "intra_same_tasks" if tasks_a == tasks_b else "intra_diff_tasks"
    return "inter"


def spearman_brown(r):
    return 2 * r / (1 + r) if np.isfinite(r) and r > 0 else float("nan")


def m1_pairs(directory, meta, fold, server_round):
    """Pairwise M1 rows (per block and shared_total) from one round's probe files."""
    probes = {}
    for path in sorted(Path(directory).glob("*_m1.npz")):
        with np.load(path) as stored:
            blocks = [str(block) for block in stored["blocks"]]
            probes[path.name[:-len("_m1.npz")]] = {
                block: (stored[f"{block}::sketch_a"], stored[f"{block}::sketch_b"],
                        stored[f"{block}::exact"]) for block in blocks}
    rows = []
    for a, b in itertools.combinations(sorted(probes), 2):
        totals = np.zeros(3)  # g_a.g_b estimate, |g_a|^2, |g_b|^2
        for block in probes[a]:
            sa_a, sb_a, ex_a = probes[a][block]
            sa_b, sb_b, ex_b = probes[b][block]
            dot = float(((sa_a + sb_a) / 2) @ ((sa_b + sb_b) / 2))
            totals += [dot, ex_a[3], ex_b[3]]
            rows.append(_m1_row(fold, server_round, block, a, b, meta, dot, ex_a, ex_b))
        exact_a = sum(probes[a][block][2] for block in probes[a])
        exact_b = sum(probes[b][block][2] for block in probes[b])
        rows.append(_m1_row(fold, server_round, TOTAL_BLOCK, a, b, meta, totals[0],
                            exact_a, exact_b))
    return rows


def _m1_row(fold, server_round, block, a, b, meta, dot, exact_a, exact_b):
    norm_a, norm_b = np.sqrt(exact_a[3]), np.sqrt(exact_b[3])
    cos = dot / (norm_a * norm_b) if norm_a * norm_b > 0 else float("nan")
    ceiling = []
    for exact in (exact_a, exact_b):
        denominator = np.sqrt(exact[0] * exact[1])
        ceiling.append(spearman_brown(exact[2] / denominator) if denominator > 0 else float("nan"))
    product = ceiling[0] * ceiling[1]
    return {
        "fold": fold, "round": server_round, "block": block, "client_a": a, "client_b": b,
        "dataset_a": meta.get(a, ("?",))[0], "dataset_b": meta.get(b, ("?",))[0],
        "pair_class": pair_class(meta, a, b) if a in meta and b in meta else "unknown",
        "cos": cos,
        "cos_normalized": cos / np.sqrt(product) if np.isfinite(product) and product > 0
        else float("nan"),
        "magnitude_similarity": 2 * norm_a * norm_b / (norm_a ** 2 + norm_b ** 2)
        if norm_a + norm_b > 0 else float("nan"),
        "ceiling_a": ceiling[0], "ceiling_b": ceiling[1],
    }


def collect_run(run_path):
    """All M1 pair rows and M2 lookahead rows of a run, or two empty frames."""
    m1, m2 = [], []
    for fold_dir in sorted(Path(run_path).glob("fold_*")):
        fold = int(fold_dir.name.split("_")[1])
        meta = _client_metadata(fold_dir)
        for directory in sorted((fold_dir / CONFLICT_DIR).glob("round_*")):
            server_round = int(directory.name.split("_")[1])
            m1.extend(m1_pairs(directory, meta, fold, server_round))
            for path in sorted(directory.glob("*_m2.csv")):
                frame = pd.read_csv(path)
                frame.insert(0, "fold", fold)
                frame["pair_class"] = [pair_class(meta, s, t) if (s in meta or s == AGGREGATE_SOURCE)
                                       and t in meta else "unknown"
                                       for s, t in zip(frame.source, frame.target)]
                frame["source_dataset"] = [AGGREGATE_SOURCE if s == AGGREGATE_SOURCE
                                           else meta.get(s, ("?",))[0] for s in frame.source]
                frame["target_dataset"] = [meta.get(t, ("?",))[0] for t in frame.target]
                m2.append(frame)
    return pd.DataFrame(m1), (pd.concat(m2, ignore_index=True) if m2 else pd.DataFrame())


def _over_rounds(frame, keys, value):
    """Mean over pairs inside a round, then mean/sd over the sampled rounds (TAG's Z-hat)."""
    per_round = frame.groupby(keys + ["fold", "round"])[value].mean().reset_index()
    summary = per_round.groupby(keys)[value].agg(["mean", "std", "count"]).reset_index()
    return summary.rename(columns={"count": "rounds"})


def summarize(m1, m2):
    parts = []
    if len(m1):
        total = m1[m1.block == TOTAL_BLOCK].copy()
        total["dataset_pair"] = [" | ".join(sorted(pair))
                                 for pair in zip(total.dataset_a, total.dataset_b)]
        for value in ("cos", "cos_normalized", "magnitude_similarity"):
            for keys, scope in ((["pair_class"], "pair_class"), (["dataset_pair"], "dataset_pair")):
                summary = _over_rounds(total, keys, value)
                summary.insert(0, "metric", f"m1_{value}")
                summary.insert(1, "scope", scope)
                summary = summary.rename(columns={keys[0]: "group"})
                parts.append(summary)
        blocks = _over_rounds(m1, ["block"], "cos")
        blocks.insert(0, "metric", "m1_cos")
        blocks.insert(1, "scope", "block")
        parts.append(blocks.rename(columns={"block": "group"}))
    if len(m2):
        for split in ("train", "val"):
            frame = m2[m2.split == split].copy()
            by_class = _over_rounds(frame, ["pair_class"], "z")
            by_class.insert(0, "metric", f"m2_z_{split}")
            by_class.insert(1, "scope", "pair_class")
            parts.append(by_class.rename(columns={"pair_class": "group"}))
            cross = frame[~frame.pair_class.isin(["self", "aggregate"])].copy()
            cross["group"] = cross.source_dataset + " -> " + cross.target_dataset
            by_dataset = _over_rounds(cross, ["group"], "z")
            by_dataset.insert(0, "metric", f"m2_z_{split}")
            by_dataset.insert(1, "scope", "dataset_pair")
            parts.append(by_dataset)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def report_run(run_path):
    """Write the consolidated CSVs beside the run and log the main tables."""
    run_path = Path(run_path)
    m1, m2 = collect_run(run_path)
    if not len(m1) and not len(m2):
        logging.info("[gradient conflict] %s", MISSING_STATUS)
        return None
    m1.to_csv(run_path / M1_FILE, index=False)
    m2.to_csv(run_path / M2_FILE, index=False)
    summary = summarize(m1, m2)
    summary.to_csv(run_path / SUMMARY_FILE, index=False)
    view = summary[summary.scope == "pair_class"].pivot_table(
        index="group", columns="metric", values="mean")
    logging.info("[gradient conflict] mean over sampled rounds (pair class):\n%s",
                 view.to_string(float_format=lambda v: f"{v:.5f}"))
    return summary
