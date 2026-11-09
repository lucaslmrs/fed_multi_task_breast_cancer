"""End-to-end check of round-level resume: an uninterrupted run vs. one killed and resumed.

Both runs use the same bounded CPU configuration (one client per dataset/task, a few samples, a
few rounds, a checkpoint every ``--every`` rounds). Run B is started in its own process group and
SIGKILLed while a round after the first checkpoint is in progress -- the same abrupt death a lost
GPU context causes -- and then resumed with ``resume=True``. The check passes when B's final global
trunk, client states and test metrics match A's, and B's histories have no duplicated rounds.

    python -m scripts.smoke_resume --config studies/e8_layer_mixing/resnet34_256_layer_mixing.yaml
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.training_federated import run


def _bounded_config(config, rounds, every, samples):
    config = yaml.safe_load(yaml.safe_dump(config))
    config["federated"].update({
        "rounds": rounds,
        "device": "cpu",
        "max_samples_per_split": samples,
        "max_clients_per_dataset_task": 1,
        "max_folds": 1,
    })
    local_training = config["federated"].setdefault("local_training", {})
    if local_training.get("mode") == "steps":
        local_training["steps_per_round"] = 1
    config["training"]["precision"] = "fp32"
    runtime = config.setdefault("runtime", {})
    runtime.setdefault("federated", {}).update({
        "ray_num_cpus": 1,
        "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
        "checkpoint_every_rounds": every,
    })
    runtime["telemetry"] = {"enabled": True, "gpu_interval_seconds": 1.0}
    runtime.setdefault("dataloader", {})["pin_memory"] = False
    return config


def _state_round(path):
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)["round"])
    except Exception:
        return -1


def _kill_mid_run(config_path, run_path, kill_after_round, timeout):
    """Start the run, wait for the checkpoint and a client past it, then SIGKILL everything."""
    log = Path(run_path).with_suffix(".killed.log").open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", "src.training_federated", "--config", config_path,
         "--run-path", str(run_path)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    fold = Path(run_path) / "fold_0"
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError("run B finished before it could be interrupted")
            server = fold / "round_checkpoint" / "server.pt"
            ahead = [p for p in fold.glob("client_*/state.pt")
                     if _state_round(p) > kill_after_round]
            if server.exists() and ahead:
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("run B never reached the interruption point")
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        subprocess.run([str(Path(sys.executable).parent / "ray"), "stop", "--force"],
                       capture_output=True)
    checkpoint = torch.load(fold / "round_checkpoint" / "server.pt", weights_only=False)
    states = {p.parent.name: _state_round(p) for p in fold.glob("client_*/state.pt")}
    print(f"KILLED checkpoint_round={checkpoint['round']} client_rounds={states}", flush=True)
    return int(checkpoint["round"])


def _compare(a, b, rounds):
    a, b = Path(a), Path(b)
    ga = torch.load(a / "fold_0/global_shared.pt", weights_only=False)
    gb = torch.load(b / "fold_0/global_shared.pt", weights_only=False)
    assert ga["round"] == gb["round"] == rounds, (ga["round"], gb["round"])
    trunk = max(float((x - y).abs().max()) for x, y in zip(ga["arrays"], gb["arrays"]))

    client_gap = 0.0
    for path in sorted((a / "fold_0").glob("client_*/state.pt")):
        sa = torch.load(path, weights_only=False)
        sb = torch.load(b / path.relative_to(a), weights_only=False)
        assert sa["round"] == sb["round"] == rounds
        key = "model" if "model" in sa else "personalized"
        for name, tensor in sa[key].items():
            if tensor.is_floating_point():
                client_gap = max(client_gap, float((tensor - sb[key][name]).abs().max()))

    setup = "federated" if (a / "federated_test_results.csv").exists() else "standalone"
    ra = pd.read_csv(a / f"{setup}_test_results.csv").sort_values(["client_id", "task"])
    rb = pd.read_csv(b / f"{setup}_test_results.csv").sort_values(["client_id", "task"])
    numeric = [c for c in ra.columns if pd.api.types.is_numeric_dtype(ra[c])]
    metric_gap = float(np.nanmax(np.abs(ra[numeric].to_numpy(float) - rb[numeric].to_numpy(float))))

    history_b = json.loads((b / "fold_0/aggregation_history.json").read_text())
    pairs = [(row["round"], row.get("stage")) for row in history_b]
    assert len(pairs) == len(set(pairs)), "duplicated rounds in B's aggregation history"
    assert sorted({r for r, _ in pairs}) == list(range(1, rounds + 1)), "missing rounds in B"
    history_a = json.loads((a / "fold_0/aggregation_history.json").read_text())
    assert len(history_a) == len(history_b)
    for path in (b / "fold_0").glob("client_*/training_history.csv"):
        frame = pd.read_csv(path)
        keys = ["round", "phase", "split", "unit_type", "unit_index", "task", "metric_name"]
        assert not frame.duplicated(keys).any(), f"duplicated history rows in {path}"
        assert sorted(frame["round"].unique()) == list(range(1, rounds + 1)), path
    assert not (b / "fold_0/round_checkpoint").exists(), "checkpoint not discarded after the fold"
    print(f"MAX_DIFF trunk={trunk:.3e} client_states={client_gap:.3e} test_metrics={metric_gap:.3e}", flush=True)
    return trunk, client_gap, metric_gap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--standalone", action="store_true")
    # Two uninterrupted CPU runs of the same config already differ by ~1e-4 in test metrics
    # (float summation order), so the check is "within run-to-run noise", not bit-exact.
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = _bounded_config(yaml.safe_load(stream), args.rounds, args.every, args.samples)
    config["federated"]["standalone"] = bool(args.standalone)
    with tempfile.TemporaryDirectory(prefix="smoke_resume_") as tmp:
        config_path = Path(tmp) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_a, run_b = Path(tmp) / "A", Path(tmp) / "B"
        run(str(config_path), run_path=run_a)
        checkpoint_round = _kill_mid_run(str(config_path), run_b, args.every, args.timeout)
        run(str(config_path), run_path=run_b, resume=True)
        log = (run_b / "execution.log").read_text(encoding="utf-8")
        assert f"resuming from the round-{checkpoint_round} checkpoint" in log, "did not resume"
        gaps = _compare(run_a, run_b, args.rounds)
    if max(gaps) > args.tolerance:
        raise SystemExit(f"RESUME_MISMATCH max diff {max(gaps):.3e} > {args.tolerance}")
    print(f"RESUME_OK checkpoint_round={checkpoint_round} rounds={args.rounds}")


if __name__ == "__main__":
    main()
