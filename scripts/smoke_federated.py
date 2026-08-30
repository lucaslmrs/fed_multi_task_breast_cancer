"""Run a bounded real-data, multi-dataset Flower smoke test on CPU.

The script keeps ``src/config.yaml`` untouched. It exercises one client for every
dataset/task pair, both aggregation rounds, per-client persistence, and final unified evaluation
on fold zero. Training/validation/test loaders are capped to keep the check suitable for CI or a
developer workstation. ``--setup both`` also validates the paired local-only arm and builds the
cross-setup analysis report.
"""

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from src.experiments.analyze import run as analyze_runs
from src.dataset.federated_partition import build_multi_dataset_partition
from src.training_federated import run


def _verify_run(run_path, setup, datasets):
    """Fail the smoke command if a required final artifact is absent or malformed."""
    run_path = Path(run_path)
    results = run_path / f"{setup}_test_results.csv"
    predictions = run_path / f"{setup}_cls_predictions.csv"
    if not results.exists() or not predictions.exists():
        raise RuntimeError(f"Incomplete {setup} smoke artifacts in {run_path}")
    frame = pd.read_csv(results)
    if set(frame["dataset"]) != set(datasets):
        raise RuntimeError(
            f"{setup} results contain datasets {sorted(frame['dataset'].unique())}, "
            f"expected {sorted(datasets)}"
        )
    if setup == "federated" and not (run_path / "fold_0" / "global_shared.pt").exists():
        raise RuntimeError("Federated smoke did not persist fold_0/global_shared.pt")
    curves = run_path / "training_curves"
    for artifact in ("history.csv", "dashboard.html"):
        if not (curves / artifact).exists():
            raise RuntimeError(f"{setup} smoke did not persist training_curves/{artifact}")
    history = pd.read_csv(curves / "history.csv")
    if history.empty or set(history["dataset"]) != set(datasets):
        raise RuntimeError(f"{setup} smoke training history is empty or dataset-incomplete")
    for level in ("overview", "clients", "datasets"):
        if not list((curves / "plots" / level).rglob("*.png")):
            raise RuntimeError(f"{setup} smoke produced no {level} training plot")
    return results, predictions


def _verify_multitask(run_path, setup):
    """Check what only a live run can show: two result rows per client and both loss terms."""
    run_path = Path(run_path)
    frame = pd.read_csv(run_path / f"{setup}_test_results.csv")
    busi = frame[frame["dataset"] == "Curated_BUSI"]
    per_client = busi.groupby("client_id")["task"].nunique()
    if per_client.empty or not (per_client == 2).all():
        raise RuntimeError(
            f"Multi-task BUSI clients must emit one row per task, got {per_client.to_dict()}"
        )

    for metadata_file in (run_path / "fold_0").glob("client_Curated_BUSI_mt_*/metadata.yaml"):
        metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("client_topology") != "multi_task":
            raise RuntimeError(f"{metadata_file} is not recorded as a multi-task client")
        if sorted(metadata.get("tasks", [])) != ["cls", "seg"]:
            raise RuntimeError(f"{metadata_file} does not own both tasks")


def _verify_gradient_conflict(run_path):
    """The cosine matrix cannot be recovered after the run, so assert it was actually written."""
    history = json.loads((Path(run_path) / "fold_0" / "aggregation_history.json").read_text())
    fits = [entry for entry in history if entry.get("stage") == "fit"]
    if not fits:
        raise RuntimeError("No fit aggregation was recorded")
    for entry in fits:
        conflict = entry.get("gradient_conflict")
        if not conflict or "shared_total" not in conflict.get("cosine", {}):
            raise RuntimeError(f"Round {entry['round']} recorded no gradient-conflict matrix")
        matrix = conflict["cosine"]["shared_total"]
        size = len(conflict["clients"])
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise RuntimeError("Gradient-conflict matrix is not square over the client roster")
        for index, row in enumerate(matrix):
            if row[index] is not None and abs(row[index] - 1.0) > 1e-4:
                raise RuntimeError(f"Cosine diagonal is {row[index]}, expected 1.0")


def _verify_paired_controls(federated_path, standalone_path):
    """Check that the two smoke arms differ only by federation, not budget or RNG policy."""
    fed_config = yaml.safe_load((Path(federated_path) / "config.yaml").read_text())
    local_config = yaml.safe_load((Path(standalone_path) / "config.yaml").read_text())
    fed_config["federated"].pop("standalone", None)
    local_config["federated"].pop("standalone", None)
    if fed_config != local_config:
        raise RuntimeError("Paired smoke configs differ by more than federated.standalone")

    fed_metadata = sorted((Path(federated_path) / "fold_0").glob("client_*/metadata.yaml"))
    if not fed_metadata:
        raise RuntimeError("Paired smoke produced no client metadata")
    for fed_file in fed_metadata:
        local_file = Path(standalone_path) / "fold_0" / fed_file.parent.name / "metadata.yaml"
        fed_values = yaml.safe_load(fed_file.read_text())
        local_values = yaml.safe_load(local_file.read_text())
        for field in ("initial_seed", "round_seed_policy", "class_names", "class_weights"):
            if fed_values.get(field) != local_values.get(field):
                raise RuntimeError(
                    f"Paired control mismatch for {fed_file.parent.name}: {field}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--samples", type=int, default=2, help="samples per client/split")
    parser.add_argument(
        "--setup",
        choices=("federated", "standalone", "both"),
        default="federated",
        help="arm to exercise; 'both' runs the paired controlled comparison",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="exercise CV=1 with a temporary 70/30 master instead of the configured partition",
    )
    parser.add_argument(
        "--topology",
        choices=("single_task", "multi_task"),
        default="single_task",
        help="BUSI client topology; multi_task builds its own temporary master",
    )
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    temporary_partition_dir = None
    if args.topology == "multi_task":
        # A multi-task BUSI client owns both tasks over the same images, so it needs its own
        # master. Four clients keep the per-round step budget equal to the single-task topology.
        config["datasets"]["Curated_BUSI"]["client_topology"] = "multi_task"
        config["datasets"]["Curated_BUSI"]["oversampling"] = {"seg": False, "cls": False}
        config["federated"]["n_clients"]["Curated_BUSI"] = {"seg": 4, "cls": 4}
    if args.holdout or args.topology == "multi_task":
        temporary_partition_dir = tempfile.TemporaryDirectory(prefix="fed_smoke_partition_")
        if args.holdout:
            config["training"]["CV"] = 1
            config["training"].setdefault("holdout_test_size", 0.30)
        partition_path = Path(temporary_partition_dir.name) / "federated_mapping.csv"
        config["federated"]["partition_file"] = str(partition_path)
        build_multi_dataset_partition(config, output_path=str(partition_path))
    setups = ("federated", "standalone") if args.setup == "both" else (args.setup,)
    artifacts = {}
    run_paths = {}
    try:
        for setup in setups:
            arm_config = yaml.safe_load(yaml.safe_dump(config))
            arm_config["federated"].update({
                "rounds": 2,
                "local_epochs": 1,
                "standalone": setup == "standalone",
                "device": "cpu",
                "ray_num_cpus": 1,
                "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
                "max_samples_per_split": args.samples,
                "max_clients_per_dataset_task": 1,
                "max_folds": 1,
            })
            local_training = arm_config["federated"].setdefault("local_training", {})
            if local_training.get("mode") == "steps":
                local_training["steps_per_round"] = 1
            for dataset in arm_config.get("datasets", {}).values():
                dataset["batch_size"] = 1

            temporary = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", prefix="fed_multi_smoke_", delete=False, encoding="utf-8"
            )
            try:
                with temporary:
                    yaml.safe_dump(arm_config, temporary, sort_keys=False)
                run_path = run(temporary.name)
                run_paths[setup] = Path(run_path)
                artifacts[setup] = _verify_run(
                    run_path, setup, arm_config["federated"]["datasets"]
                )
                if args.topology == "multi_task":
                    _verify_multitask(run_path, setup)
                if setup == "federated":
                    _verify_gradient_conflict(run_path)
                print(f"SMOKE_OK setup={setup} topology={args.topology} run_path={run_path}")
            finally:
                Path(temporary.name).unlink(missing_ok=True)

        if args.setup == "both":
            _verify_paired_controls(run_paths["federated"], run_paths["standalone"])
            analysis_dir = run_paths["federated"] / "paired_analysis"
            analyze_runs(
                [artifacts["federated"][0], artifacts["standalone"][0]],
                [artifacts["federated"][1], artifacts["standalone"][1]],
                analysis_dir,
            )
            print(f"SMOKE_ANALYSIS_OK out={analysis_dir}")
    finally:
        if temporary_partition_dir is not None:
            temporary_partition_dir.cleanup()


if __name__ == "__main__":
    main()
