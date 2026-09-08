"""Paired GPU acceptance benchmark for the accelerated federated runtime.

The command never creates or edits a federated partition.  It runs two bounded copies of the
same frozen seed/client/sample protocol: FP32 with one client per GPU, then BF16 with two clients
per GPU.  Large checkpoints live in a temporary directory and are removed unless ``--keep-runs``
is supplied; the compact acceptance report is kept under ``runs/benchmarks``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.experiments.study_runner import build_execution_plan, load_manifest
from src.training_federated import run


PRIMARY_METRIC = {"seg": "dice", "cls": "balanced_acc"}
PAIR_KEYS = ["dataset", "fold", "client_id", "task"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _benchmark_config(base: dict, *, precision: str, num_gpus: float, args) -> dict:
    config = copy.deepcopy(base)
    config["training"]["precision"] = precision
    config["training"]["cuda_benchmark"] = precision == "bf16"
    config["federated"].update(
        rounds=args.rounds,
        standalone=False,
        device="cuda",
        max_folds=1,
        max_samples_per_split=args.max_samples,
        max_clients_per_dataset_task=args.clients_per_dataset_task,
    )
    config["federated"].setdefault("local_training", {}).update(
        mode="steps", steps_per_round=args.steps_per_round
    )
    runtime = config.setdefault("runtime", {})
    runtime["inference_batch_size"] = args.inference_batch_size
    runtime["federated"] = {
        "ray_num_cpus": args.ray_num_cpus,
        "client_resources": {"num_cpus": args.client_cpus, "num_gpus": num_gpus},
    }
    runtime["telemetry"] = {"enabled": False, "gpu_interval_seconds": 1.0}
    return config


def _run_arm(config: dict, root: Path, name: str) -> tuple[float, Path]:
    config_path = root / f"{name}.yaml"
    run_path = root / name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.perf_counter()
    run(config_path, run_path=run_path)
    elapsed = time.perf_counter() - started
    return elapsed, run_path


def _metric_pairs(fp32_path: Path, bf16_path: Path) -> tuple[pd.DataFrame, float]:
    left = pd.read_csv(fp32_path / "federated_test_results.csv")
    right = pd.read_csv(bf16_path / "federated_test_results.csv")
    rows = []
    for task, metric in PRIMARY_METRIC.items():
        ltask = left[left["task"] == task][PAIR_KEYS + [metric]].rename(
            columns={metric: "fp32"}
        )
        rtask = right[right["task"] == task][PAIR_KEYS + [metric]].rename(
            columns={metric: "bf16"}
        )
        paired = ltask.merge(rtask, on=PAIR_KEYS, how="outer", validate="one_to_one")
        if paired[["fp32", "bf16"]].isna().any().any():
            raise RuntimeError(f"Unpaired or non-finite {task}/{metric} benchmark results")
        paired["metric"] = metric
        paired["absolute_difference"] = (paired["fp32"] - paired["bf16"]).abs()
        rows.append(paired)
    combined = pd.concat(rows, ignore_index=True)
    if not all(math.isfinite(float(value)) for value in combined[["fp32", "bf16"]].to_numpy().ravel()):
        raise RuntimeError("Benchmark produced NaN or Inf in a primary metric")
    return combined, float(combined["absolute_difference"].mean())


def _write_report(output: Path, report: dict, pairs: pd.DataFrame) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    pairs.to_csv(output / "metric_pairs.csv", index=False)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="studies/example_multi_dataset.yaml"
    )
    parser.add_argument("--seed", type=int, default=1993)
    # Five rounds amortize the fixed Ray/model startup while keeping the gate short enough
    # for routine acceptance runs. Two rounds measure startup more than training throughput.
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--steps-per-round", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--clients-per-dataset-task", type=int, default=1)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--ray-num-cpus", type=int, default=8)
    parser.add_argument("--client-cpus", type=int, default=2)
    parser.add_argument("--min-speedup", type=float, default=0.25)
    parser.add_argument("--max-metric-delta", type=float, default=0.02)
    parser.add_argument("--output-root", default="runs/benchmarks")
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("Benchmark requires a CUDA GPU with native BF16 support")
    for field in ("rounds", "steps_per_round", "max_samples", "clients_per_dataset_task"):
        if getattr(args, field) < 1:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive")

    manifest = load_manifest(args.manifest)
    primary = next(arm for arm in manifest["arms"] if arm["arm_id"] == "primary")
    plan, _, _ = build_execution_plan(manifest, [args.seed], [primary])
    base = plan[0]["config"]
    partition_path = Path(base["federated"]["partition_file"])
    if not partition_path.is_file():
        raise FileNotFoundError(
            f"Frozen partition is missing: {partition_path}. The benchmark will not regenerate it."
        )

    # The frozen master is authoritative. This check also makes a stale base config actionable
    # without changing a single row of the partition.
    roster = pd.read_csv(
        partition_path, usecols=["dataset", "task", "client_id"], low_memory=False
    ).drop_duplicates()
    expected = {
        dataset: {
            task: int(group["client_id"].nunique())
            for task, group in dataset_rows.groupby("task")
        }
        for dataset, dataset_rows in roster.groupby("dataset")
    }
    base["federated"]["n_clients"] = expected

    fp32_config = _benchmark_config(base, precision="fp32", num_gpus=1.0, args=args)
    bf16_config = _benchmark_config(base, precision="bf16", num_gpus=0.5, args=args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output_root) / f"{stamp}_paired_fp32_bf16"

    if args.keep_runs:
        work_root = output.parent / f"{output.name}_runs"
        work_root.mkdir(parents=True, exist_ok=False)
        context = None
    else:
        context = tempfile.TemporaryDirectory(prefix="fed_runtime_benchmark_", dir="/tmp")
        work_root = Path(context.name)

    try:
        fp32_seconds, fp32_path = _run_arm(fp32_config, work_root, "fp32_sequential")
        bf16_seconds, bf16_path = _run_arm(bf16_config, work_root, "bf16_dual_client")
        pairs, mean_delta = _metric_pairs(fp32_path, bf16_path)
        speedup = 1.0 - (bf16_seconds / fp32_seconds)
        finite = all(
            math.isfinite(value) for value in (fp32_seconds, bf16_seconds, speedup, mean_delta)
        )
        passed = finite and speedup >= args.min_speedup and mean_delta <= args.max_metric_delta
        report = {
            "created_at_utc": _utc_now(),
            "passed": passed,
            "partition_file": str(partition_path),
            "seed": args.seed,
            "rounds": args.rounds,
            "steps_per_round": args.steps_per_round,
            "max_samples_per_split": args.max_samples,
            "clients_per_dataset_task": args.clients_per_dataset_task,
            "fp32_seconds": fp32_seconds,
            "bf16_seconds": bf16_seconds,
            "speedup_fraction": speedup,
            "minimum_speedup_fraction": args.min_speedup,
            "mean_absolute_primary_metric_difference": mean_delta,
            "maximum_metric_difference": args.max_metric_delta,
            "finite": finite,
            "temporary_runs_removed": not args.keep_runs,
        }
        _write_report(output, report, pairs)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not passed:
            raise RuntimeError(
                "Runtime acceptance failed: require speedup >= "
                f"{args.min_speedup:.0%} and mean primary-metric delta <= "
                f"{args.max_metric_delta:.4f}"
            )
        return output
    finally:
        if context is not None:
            context.cleanup()


if __name__ == "__main__":
    main()
