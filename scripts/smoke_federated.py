"""Run a bounded real-data, multi-dataset Flower smoke test on CPU.

The script keeps ``src/config.yaml`` untouched. It exercises one client for every
dataset/task pair, both aggregation rounds, per-client persistence, and final unified evaluation
on fold zero. Training/validation/test loaders are capped to keep the check suitable for CI or a
developer workstation. ``--setup both`` also validates the paired local-only arm and builds the
cross-setup analysis report. A pretrained architecture (MTResNet34UNet) is also checked for an
ImageNet encoder whose FrozenBN buffers never moved (``_verify_frozen_pretrained``).
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.experiments.analyze import run as analyze_runs
from src.dataset.federated_partition import build_multi_dataset_partition
from src.training_federated import run
from src.utils.experiment_init import PRETRAINED_ARCHITECTURES


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


def _verify_conflict_diagnostics(run_path):
    """Both online methods wrote results for every round and dropped their transients."""
    from src.federated import conflict_diagnostics as cd

    run_path = Path(run_path)
    for name in (cd.M1_FILE, cd.M2_FILE, cd.SUMMARY_FILE):
        if not (run_path / name).exists():
            raise RuntimeError(f"Gradient-conflict diagnostics did not write {name}")
    m2 = pd.read_csv(run_path / cd.M2_FILE)
    if set(m2["round"]) != {1, 2} or set(m2["split"]) != {"train", "val"}:
        raise RuntimeError("M2 lookahead does not cover both rounds and both splits")
    self_rows = m2[m2.source == m2.target]
    if self_rows.empty or not np.isfinite(self_rows.z).all():
        raise RuntimeError("M2 lookahead has no finite self-affinity rows")
    leftovers = list((run_path / "fold_0" / cd.CONFLICT_DIR).glob(f"round_*/{cd.DELTAS_FILE}"))
    leftovers += list((run_path / "fold_0" / cd.CONFLICT_DIR).glob("round_*/*_pre.pt"))
    if leftovers:
        raise RuntimeError(f"Transient lookahead files were not removed: {leftovers[:2]}")


def _verify_m1_coverage(run_path):
    """Every sampled round holds an M1 probe from EVERY client, all over the same shared blocks.

    A failing probe is swallowed by ``observational()`` on purpose (training must continue), and
    Ray worker logs never reach ``execution.log``, so before E6 a crashing probe -- a single-task
    client under wide sharing, a shared buffer -- still let the smoke pass. The files are the only
    reliable evidence. The expected block set is that of the keys in ``global_shared.pt``; a shared
    block owning only buffers would have no gradient to probe (none exists today).
    """
    import torch

    from src.federated import conflict_diagnostics as cd

    run_path = Path(run_path)
    fold_dir = run_path / "fold_0"
    config = yaml.safe_load((run_path / "config.yaml").read_text(encoding="utf-8"))
    clients = sorted(
        str(yaml.safe_load(path.read_text(encoding="utf-8"))["client_id"])
        for path in fold_dir.glob("client_*/metadata.yaml")
    )
    if not clients:
        raise RuntimeError("M1 coverage: fold_0 has no client metadata")
    expected_rounds = [
        server_round for server_round in range(1, int(config["federated"]["rounds"]) + 1)
        if cd.is_sampled_round(config, server_round)
    ]
    round_dirs = sorted((fold_dir / cd.CONFLICT_DIR).glob("round_*"))
    found_rounds = [int(directory.name.split("_")[1]) for directory in round_dirs]
    if found_rounds != expected_rounds:
        raise RuntimeError(
            f"M1 coverage: conflict rounds {found_rounds}, expected sampled rounds {expected_rounds}"
        )
    shared = torch.load(fold_dir / "global_shared.pt", map_location="cpu")
    expected_blocks = list(dict.fromkeys(cd._block(key) for key in shared["keys"]))
    for directory in round_dirs:
        probes = sorted(path.name[:-len("_m1.npz")] for path in directory.glob("*_m1.npz"))
        if probes != clients:
            raise RuntimeError(
                f"M1 coverage: {directory.name} has probes {probes}, expected one per client "
                f"{clients}; missing {sorted(set(clients) - set(probes))}"
            )
        for client_id in clients:
            with np.load(directory / f"{client_id}_m1.npz") as stored:
                blocks = [str(block) for block in stored["blocks"]]
                if blocks != expected_blocks:
                    raise RuntimeError(
                        f"M1 coverage: {directory.name}/{client_id} probes blocks {blocks}, "
                        f"expected the shared blocks {expected_blocks}"
                    )
                exact = sum(stored[f"{block}::exact"] for block in blocks)
            if not np.isfinite(exact).all() or exact[3] <= 0:
                raise RuntimeError(
                    f"M1 coverage: {directory.name}/{client_id} has a non-finite or zero "
                    f"shared_total gradient ({exact.tolist()})"
                )
    if cd.lookahead_enabled(config):
        _verify_m2_coverage(round_dirs, clients)
    print(f"M1_COVERAGE_OK rounds={expected_rounds} clients={len(clients)} "
          f"blocks={len(expected_blocks)}")


def _verify_m2_coverage(round_dirs, clients):
    """Every sampled round holds a lookahead CSV from EVERY client, covering both splits.

    Like M1, a failing ``collect_lookahead`` is swallowed by ``observational()``; the consolidated
    run-level M2 file still passes when one client is missing, so the per-client files are checked.
    """
    for directory in round_dirs:
        targets = sorted(path.name[:-len("_m2.csv")] for path in directory.glob("*_m2.csv"))
        if targets != clients:
            raise RuntimeError(
                f"M2 coverage: {directory.name} has lookahead files {targets}, expected one per "
                f"client {clients}; missing {sorted(set(clients) - set(targets))}"
            )
        for client_id in clients:
            rows = pd.read_csv(directory / f"{client_id}_m2.csv")
            if set(rows["split"]) != {"train", "val"} or set(rows["target"]) != {client_id}:
                raise RuntimeError(
                    f"M2 coverage: {directory.name}/{client_id} lacks a split or has foreign targets"
                )
            if not np.isfinite(rows["z"]).all():
                raise RuntimeError(f"M2 coverage: {directory.name}/{client_id} has non-finite z")
    print(f"M2_COVERAGE_OK rounds={len(round_dirs)} clients={len(clients)}")


def _verify_frozen_pretrained(run_path, max_conv1_drift=0.05):
    """The ImageNet encoder reached the trunk, and its FrozenBN buffers never moved.

    Reads every ``fold_*/global_shared.pt`` of a federated run (or every client's full
    ``state.pt`` of a local-only run). Each FrozenBatchNorm2d buffer must equal the mapped ImageNet
    tensor EXACTLY: nothing trains it and FedAvg of identical arrays is the identity. The relative
    L2 distance of ``encoder2.conv1.weight`` from ImageNet must stay below ``max_conv1_drift``: a
    random initialization sits near 1.4, so a smoke (2 rounds) proves the weights came from the
    checkpoint. After a full run the drift is legitimately larger; pass ``max_conv1_drift=None`` to
    check the buffers only and just print it.
    """
    import torch

    from src.models.multitask.MTResNet34UNet import MTResNet34UNet, load_imagenet_state
    from src.utils.experiment_init import PRETRAINED_ARCHITECTURES

    run_path = Path(run_path)
    config = yaml.safe_load((run_path / "config.yaml").read_text(encoding="utf-8"))
    architecture = config["model"]["architecture"]
    if architecture not in PRETRAINED_ARCHITECTURES:
        raise RuntimeError(f"{run_path} trains {architecture}, which has no pretrained encoder")
    imagenet, _, _ = load_imagenet_state()
    parameters = dict(MTResNet34UNet(sequences=1, regions=1).named_parameters())
    buffer_keys = [key for key in imagenet if key not in parameters]
    conv1 = "encoder2.conv1.weight"

    states = {}
    for path in sorted(run_path.glob("fold_*/global_shared.pt")):
        stored = torch.load(path, map_location="cpu")
        states[f"{path.parent.name} round {stored.get('round')}"] = dict(
            zip(stored["keys"], stored["arrays"]))
    if not states:  # local-only: no global trunk, every client keeps its full model
        for path in sorted(run_path.glob("fold_*/client_*/state.pt")):
            stored = torch.load(path, map_location="cpu")
            states[f"{path.parent.parent.name}/{path.parent.name} round {stored.get('round')}"] = (
                stored["model"])
    if not states:
        raise RuntimeError(f"Frozen/pretrained check: no global_shared.pt or state.pt in {run_path}")

    for label, state in states.items():
        missing = [key for key in [*buffer_keys, conv1] if key not in state]
        if missing:
            raise RuntimeError(f"Frozen/pretrained check: {label} lacks {missing[:3]}")
        moved = [key for key in buffer_keys
                 if not torch.equal(torch.as_tensor(state[key]), imagenet[key])]
        if moved:
            raise RuntimeError(
                f"Frozen/pretrained check: {label} has {len(moved)} FrozenBN buffers that differ "
                f"from ImageNet, e.g. {moved[:3]}"
            )
        reference = imagenet[conv1].double()
        drift = float((torch.as_tensor(state[conv1]).double() - reference).norm() / reference.norm())
        if max_conv1_drift is not None and not drift < max_conv1_drift:
            raise RuntimeError(
                f"Frozen/pretrained check: {label} {conv1} is {drift:.4f} (relative L2) from "
                f"ImageNet, expected < {max_conv1_drift}"
            )
        print(f"FROZEN_PRETRAINED_OK {label} buffers={len(buffer_keys)} "
              f"conv1_rel_l2={drift:.6f}")


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
    parser.add_argument(
        "--conflict-diagnostics",
        action="store_true",
        help="also exercise the online M1/M2 gradient-conflict diagnostics on every round",
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
                "max_samples_per_split": args.samples,
                "max_clients_per_dataset_task": 1,
                "max_folds": 1,
            })
            arm_config["training"]["precision"] = "fp32"
            arm_config.setdefault("runtime", {}).setdefault("federated", {}).update({
                "ray_num_cpus": 1,
                "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
            })
            arm_config["runtime"]["telemetry"] = {
                "enabled": False,
                "gpu_interval_seconds": 1.0,
            }
            local_training = arm_config["federated"].setdefault("local_training", {})
            if local_training.get("mode") == "steps":
                # M1 splits the round's batches into two halves, so it needs at least two.
                local_training["steps_per_round"] = 2 if args.conflict_diagnostics else 1
            if args.conflict_diagnostics:
                arm_config["runtime"]["diagnostics"] = {"gradient_conflict": {
                    "enabled": True, "every_n_rounds": 1, "probe_batches": 2,
                    "sketch_dim": 4096,
                    "lookahead": {"enabled": True, "val_batches": 2, "train_batches": 2},
                }}
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
                    if args.conflict_diagnostics:
                        _verify_conflict_diagnostics(run_path)
                        _verify_m1_coverage(run_path)
                if arm_config["model"]["architecture"] in PRETRAINED_ARCHITECTURES:
                    _verify_frozen_pretrained(run_path)
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
