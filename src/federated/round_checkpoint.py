"""Round-level checkpoints so a crashed federated run resumes instead of restarting.

A checkpoint is taken at the end of a round R (after ``aggregate_evaluate``) in which every client
fit and every client evaluation succeeded. At that point each client's durable ``state.pt`` holds
its post-fit round-R state and the strategy holds the round-R global shared state, which is exactly
what round R+1 starts from. The checkpoint stores:

    fold_<k>/round_checkpoint/
        server.pt                 round, global shared arrays, aggregation history, best-round info
        clients/<client_id>/state.pt   the client's round-R state (a hard link when possible)

Client states are hard-linked rather than copied: ``FedPerClient._save_state`` writes through a
temporary file and ``os.replace``, so a later round gives ``state.pt`` a new inode and the linked
checkpoint keeps the round-R bytes without doubling the disk usage.

Resuming restores every client's ``state.pt`` from the checkpoint, drops history rows of later
rounds, and restarts Flower at round R+1 with the round-R global as initial parameters. Every
per-round random stream is addressed by the round number (``stable_client_seed``, the round batch
sampler), so a resumed run reproduces the uninterrupted one up to the order in which Flower returns
client results.
"""

import csv
import logging
import os
import shutil
from pathlib import Path

import torch

CHECKPOINT_DIR = "round_checkpoint"
SERVER_FILE = "server.pt"


def checkpoint_dir(fold_dir):
    return Path(fold_dir) / CHECKPOINT_DIR


def _client_state(fold_dir, client_id):
    return Path(fold_dir) / f"client_{client_id}" / "state.pt"


def _state_round(path):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return int(state.get("round", -1))


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def save(fold_dir, server_round, global_arrays, aggregation_history, best_mean_val, best_round,
         client_ids):
    """Write the round-R checkpoint atomically; refuse if any client state is not from round R."""
    fold_dir = Path(fold_dir)
    for client_id in client_ids:
        path = _client_state(fold_dir, client_id)
        if not path.exists() or _state_round(path) != int(server_round):
            raise RuntimeError(
                f"Client {client_id} state is not from round {server_round}; checkpoint skipped"
            )
    final = checkpoint_dir(fold_dir)
    staging = final.with_name(f"{CHECKPOINT_DIR}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "clients").mkdir(parents=True)
    for client_id in client_ids:
        target = staging / "clients" / str(client_id)
        target.mkdir()
        _link_or_copy(_client_state(fold_dir, client_id), target / "state.pt")
    torch.save(
        {
            "round": int(server_round),
            "client_ids": [str(client_id) for client_id in client_ids],
            "arrays": [torch.as_tensor(array).cpu() for array in global_arrays],
            "aggregation_history": list(aggregation_history),
            "best_mean_val": float(best_mean_val),
            "best_round": int(best_round),
        },
        staging / SERVER_FILE,
    )
    previous = final.with_name(f"{CHECKPOINT_DIR}.old")
    if previous.exists():
        shutil.rmtree(previous)
    if final.exists():
        final.rename(previous)
    staging.rename(final)
    if previous.exists():
        shutil.rmtree(previous)
    logging.info("[round %s] round checkpoint written", server_round)


def load(fold_dir, client_ids, total_rounds):
    """Return the checkpoint payload, or None when there is no usable one."""
    path = checkpoint_dir(fold_dir) / SERVER_FILE
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = sorted(str(client_id) for client_id in client_ids)
    if sorted(payload["client_ids"]) != expected:
        raise ValueError(
            f"Checkpoint in {path.parent} covers clients {sorted(payload['client_ids'])}, "
            f"but the run roster is {expected}"
        )
    server_round = int(payload["round"])
    if not 1 <= server_round < int(total_rounds):
        return None
    for client_id in expected:
        state = checkpoint_dir(fold_dir) / "clients" / client_id / "state.pt"
        if not state.exists() or _state_round(state) != server_round:
            raise ValueError(f"Checkpoint state of client {client_id} is missing or inconsistent")
    payload["arrays"] = [tensor.numpy() for tensor in payload["arrays"]]
    return payload


def discard(fold_dir):
    """Remove the resume point once the fold's final state is durable."""
    for name in (CHECKPOINT_DIR, f"{CHECKPOINT_DIR}.tmp", f"{CHECKPOINT_DIR}.old"):
        path = Path(fold_dir) / name
        if path.exists():
            shutil.rmtree(path)


def _truncate_history(path, last_round):
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        kept = [row for row in reader if int(row["round"]) <= int(last_round)]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(path)


def restore_clients(fold_dir, payload):
    """Put every client back at the checkpointed round before Flower restarts."""
    fold_dir = Path(fold_dir)
    server_round = int(payload["round"])
    for client_id in payload["client_ids"]:
        client_dir = fold_dir / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        source = checkpoint_dir(fold_dir) / "clients" / client_id / "state.pt"
        temporary = client_dir / "state.pt.restore"
        if temporary.exists():
            temporary.unlink()
        _link_or_copy(source, temporary)
        os.replace(temporary, client_dir / "state.pt")
        _truncate_history(client_dir / "training_history.csv", server_round)
