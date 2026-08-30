"""Reproducible sequential runner for the multi-dataset comparison study.

The runner never edits the base configuration.  It resolves one immutable YAML per
``seed x arm``, builds one paired master partition per seed, executes arms sequentially, and
keeps a restartable run index.  A completed arm is reused; an incomplete arm is preserved and is
only moved aside when ``--retry-incomplete`` is explicitly supplied.

Examples::

    python -m src.experiments.study_runner --dry-run
    python -m src.experiments.study_runner --smoke --seed-profile operational
    python -m src.experiments.study_runner --arms primary local_steps_ce
    python -m src.experiments.study_runner --seed-profile final
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


DEFAULT_MANIFEST = "studies/multi_dataset_balance_v1.yaml"
SUPPORTED_SCHEMA_VERSION = 1
SETUPS = {"federated", "standalone"}

# Overrides of these values would break pairing because the runner intentionally creates only
# one frozen partition per seed and shares it among all arms.
_PARTITION_EXACT_PATHS = {
    "training.seed",
    "training.CV",
    "training.holdout_test_size",
    "data.root",
    "data.dataset",
    "data.variant",
    "federated.datasets",
    "federated.partition_file",
    "federated.n_clients",
    "federated.dirichlet_alpha",
    "federated.val_size",
}
_PARTITION_DATASET_FIELDS = {
    "root",
    "variant",
    "fold_strategy",
    "cls_source_split",
    "seg_exclude_classes",
    "client_topology",
}
DEFAULT_PARTITION_VARIANT = "base"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge mappings while replacing lists and scalar leaves."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _flatten_paths(value: Any, prefix: str = "") -> Iterable[str]:
    if not isinstance(value, dict):
        if prefix:
            yield prefix
        return
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            yield from _flatten_paths(child, path)
        else:
            yield path


def _validate_arm_overrides(
    overrides: dict, arm_id: str, partition_variant: str | None = None
) -> None:
    """Reject partition-defining overrides unless the arm declares its own partition variant.

    An arm that changes the client topology or the roster necessarily needs its own partition, so
    it opts in explicitly with ``partition_variant``. Every other arm keeps the original guard:
    silently diverging partitions would break the pairing the whole comparison rests on.
    """
    if partition_variant is not None:
        return
    for path in _flatten_paths(overrides):
        if path in _PARTITION_EXACT_PATHS or any(
            path.startswith(exact + ".") for exact in _PARTITION_EXACT_PATHS
        ):
            raise ValueError(
                f"Arm '{arm_id}' overrides partition-defining field '{path}'. "
                "Partitions must remain paired across arms."
            )
        parts = path.split(".")
        if len(parts) >= 3 and parts[0] == "datasets" and parts[2] in _PARTITION_DATASET_FIELDS:
            raise ValueError(
                f"Arm '{arm_id}' overrides partition-defining field '{path}'. "
                "Move this choice to the base config."
            )


def load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Study manifest '{manifest_path}' must contain a mapping")
    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported study schema_version={manifest.get('schema_version')}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    for field in (
        "study_id",
        "base_config",
        "seed_profiles",
        "default_seed_profile",
        "output_root",
        "partition_template",
        "arms",
    ):
        if field not in manifest:
            raise ValueError(f"Study manifest is missing required field '{field}'")
    if manifest["default_seed_profile"] not in manifest["seed_profiles"]:
        raise ValueError("default_seed_profile is not present in seed_profiles")
    if not manifest["arms"]:
        raise ValueError("Study manifest must define at least one arm")

    arm_ids = []
    for arm in manifest["arms"]:
        missing = {"arm_id", "method_id", "setup"}.difference(arm)
        if missing:
            raise ValueError(f"Study arm is missing fields: {sorted(missing)}")
        if arm["setup"] not in SETUPS:
            raise ValueError(f"Arm '{arm['arm_id']}' has unsupported setup '{arm['setup']}'")
        variant = arm.get("partition_variant")
        if variant is not None and not str(variant).strip():
            raise ValueError(f"Arm '{arm['arm_id']}' has an empty partition_variant")
        _validate_arm_overrides(arm.get("overrides", {}), arm["arm_id"], variant)
        arm_ids.append(str(arm["arm_id"]))
    if len(arm_ids) != len(set(arm_ids)):
        raise ValueError("Study arm_id values must be unique")

    manifest["_path"] = str(manifest_path)
    return manifest


def _project_path(value: str | Path, manifest: dict, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        cwd_candidate = (Path.cwd() / path).resolve()
        manifest_candidate = (Path(manifest["_path"]).parent / path).resolve()
        if must_exist and not cwd_candidate.exists() and manifest_candidate.exists():
            resolved = manifest_candidate
        else:
            resolved = cwd_candidate
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Required study path '{resolved}' does not exist")
    return resolved


def _format_path(
    template: str, study_id: str, seed: int | None = None, variant: str | None = None
) -> str:
    values = {"study_id": study_id}
    if seed is not None:
        values["seed"] = seed
    if variant is not None:
        values["variant"] = variant
    return template.format(**values)


def _arm_partition_variant(arm: dict) -> str:
    return str(arm.get("partition_variant") or DEFAULT_PARTITION_VARIANT)


def _partition_path_for(manifest: dict, study_id: str, seed: int, variant: str) -> Path:
    """Resolve the master partition of one ``(seed, partition_variant)`` pair.

    The default variant keeps the historical path byte-identical -- the completed arms record it
    inside their resolved config, and their config hash is what marks them reusable. A non-default
    variant is nested one directory deeper instead of changing the template, so adding a topology
    can never invalidate work that is already on disk.
    """
    base = _project_path(
        _format_path(manifest["partition_template"], study_id, seed), manifest
    )
    if variant == DEFAULT_PARTITION_VARIANT:
        return base
    return base.parent / variant / base.name


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return _sha256_bytes(payload)


def _partition_signature(config: dict) -> dict:
    """Select exactly the configuration consumed while constructing a master partition."""
    fed = config["federated"]
    active = list(fed.get("datasets") or [config["data"]["dataset"]])
    dataset_fields = {}
    for name in active:
        entry = config.get("datasets", {}).get(name, {})
        dataset_fields[name] = {
            key: copy.deepcopy(entry.get(key))
            for key in sorted(_PARTITION_DATASET_FIELDS)
            if key in entry
        }
    training_signature = {
        key: config["training"].get(key) for key in ("seed", "CV")
    }
    if config["training"].get("CV") == 1:
        training_signature["holdout_test_size"] = config["training"].get(
            "holdout_test_size", 0.30
        )
    return {
        "training": training_signature,
        "data": {
            key: config["data"].get(key)
            for key in ("root", "dataset", "variant")
        },
        "datasets": dataset_fields,
        "federated": {
            key: copy.deepcopy(fed.get(key))
            for key in ("datasets", "n_clients", "dirichlet_alpha", "val_size")
        },
    }


def _config_signature(config: dict) -> dict:
    """Hash only effective settings; holdout size is inert for CV with two or more folds."""
    signature = copy.deepcopy(config)
    # Curve settings are observational and must not invalidate completed scientific arms.
    signature.get("federated", {}).pop("training_telemetry", None)
    if signature.get("training", {}).get("CV", 0) > 1:
        signature["training"].pop("holdout_test_size", None)
    return signature


def _load_base_config(manifest: dict) -> tuple[dict, Path]:
    base_path = _project_path(manifest["base_config"], manifest, must_exist=True)
    with base_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Base config '{base_path}' must contain a mapping")
    return config, base_path


def _select_seeds(manifest: dict, profile: str | None, seeds: list[int] | None) -> list[int]:
    if seeds:
        selected = [int(seed) for seed in seeds]
    else:
        profile = profile or manifest["default_seed_profile"]
        if profile not in manifest["seed_profiles"]:
            raise ValueError(
                f"Unknown seed profile '{profile}'; choose from "
                f"{sorted(manifest['seed_profiles'])}"
            )
        selected = [int(seed) for seed in manifest["seed_profiles"][profile]]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selected seeds must be a non-empty unique list")
    return selected


def _select_arms(manifest: dict, requested: list[str] | None) -> list[dict]:
    arms = {str(arm["arm_id"]): arm for arm in manifest["arms"]}
    if not requested:
        return list(manifest["arms"])
    missing = set(requested).difference(arms)
    if missing:
        raise ValueError(f"Unknown arms: {sorted(missing)}; choose from {sorted(arms)}")
    return [arms[arm_id] for arm_id in requested]


def resolve_arm_config(
    base: dict,
    manifest: dict,
    arm: dict,
    seed: int,
    partition_path: Path,
    *,
    smoke: bool = False,
) -> dict:
    config = _deep_merge(base, arm.get("overrides", {}))
    config["training"]["seed"] = int(seed)
    config["federated"]["partition_file"] = str(partition_path)
    config["federated"]["standalone"] = arm["setup"] == "standalone"
    config["experiment"] = {
        "study_id": manifest["study_id"],
        "arm_id": arm["arm_id"],
        "method_id": arm["method_id"],
        "seed": int(seed),
        "setup": arm["setup"],
    }
    if smoke:
        config["federated"].update(
            {
                "rounds": 2,
                "device": "cpu",
                "ray_num_cpus": 1,
                "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
                "max_samples_per_split": 2,
                "max_clients_per_dataset_task": 1,
                "max_folds": 1,
            }
        )
        local_training = config["federated"].setdefault("local_training", {})
        if local_training.get("mode") == "steps":
            local_training["steps_per_round"] = 1
        for dataset in config.get("datasets", {}).values():
            dataset["batch_size"] = 1
        config["experiment"]["smoke"] = True
    return config


def build_execution_plan(
    manifest: dict,
    seeds: list[int],
    arms: list[dict],
    *,
    smoke: bool = False,
) -> tuple[list[dict], Path, Path]:
    base, base_path = _load_base_config(manifest)
    study_id = manifest["study_id"]
    output_root = _project_path(_format_path(manifest["output_root"], study_id), manifest)
    if smoke:
        output_root = output_root / "smoke"
    rows = []
    for seed in seeds:
        for arm in arms:
            variant = _arm_partition_variant(arm)
            partition_path = _partition_path_for(manifest, study_id, seed, variant)
            config = resolve_arm_config(
                base, manifest, arm, seed, partition_path, smoke=smoke
            )
            config_path = (
                output_root / "resolved" / f"seed_{seed}" / f"{arm['arm_id']}.yaml"
            )
            run_path = output_root / "runs" / f"seed_{seed}" / str(arm["arm_id"])
            rows.append(
                {
                    "study_id": study_id,
                    "arm_id": arm["arm_id"],
                    "method_id": arm["method_id"],
                    "seed": int(seed),
                    "setup": arm["setup"],
                    "partition_variant": variant,
                    "base_config": str(base_path),
                    "resolved_config": str(config_path),
                    "partition_path": str(partition_path),
                    "run_path": str(run_path),
                    "config": config,
                    "config_sha256": _stable_hash(_config_signature(config)),
                    "status": "planned",
                }
            )
    return rows, output_root, base_path


def _atomic_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)
    temporary.replace(path)


def _public_plan_frame(rows: list[dict]) -> pd.DataFrame:
    public = [{key: value for key, value in row.items() if key != "config"} for row in rows]
    return pd.DataFrame(public)


def _write_plan(rows: list[dict], output_root: Path, manifest: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        _atomic_yaml(Path(row["resolved_config"]), row["config"])
    _public_plan_frame(rows).to_csv(output_root / "execution_plan.csv", index=False)
    manifest_snapshot = {key: value for key, value in manifest.items() if key != "_path"}
    _atomic_yaml(output_root / "study_manifest.yaml", manifest_snapshot)


def _partition_metadata_path(partition_path: Path) -> Path:
    return partition_path.with_name(partition_path.stem + "_metadata.yaml")


def _ensure_partition(
    seed_rows: list[dict], *, rebuild: bool = False
) -> tuple[Path, str]:
    representative = seed_rows[0]
    config = representative["config"]
    partition_path = Path(representative["partition_path"])
    metadata_path = _partition_metadata_path(partition_path)
    signature = _partition_signature(config)
    signature_hash = _stable_hash(signature)

    if partition_path.exists() and metadata_path.exists() and not rebuild:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        if metadata.get("partition_signature_sha256") != signature_hash:
            raise RuntimeError(
                f"Existing partition metadata does not match seed configuration: "
                f"{partition_path}. Use --rebuild-partitions to replace it."
            )
        actual_hash = _sha256_file(partition_path)
        if metadata.get("csv_sha256") != actual_hash:
            raise RuntimeError(f"Partition checksum mismatch for '{partition_path}'")
        logging.info("Reusing paired partition %s", partition_path)
        return partition_path, actual_hash

    if partition_path.exists() and not rebuild:
        raise RuntimeError(
            f"Partition '{partition_path}' exists without matching study metadata. "
            "Use --rebuild-partitions to replace it."
        )

    from src.dataset.federated_partition import build_multi_dataset_partition

    partition_path.parent.mkdir(parents=True, exist_ok=True)
    build_multi_dataset_partition(config, output_path=str(partition_path))
    csv_hash = _sha256_file(partition_path)
    _atomic_yaml(
        metadata_path,
        {
            "study_id": representative["study_id"],
            "seed": int(representative["seed"]),
            "partition_variant": str(representative.get("partition_variant", "base")),
            "created_at_utc": _utc_now(),
            "partition_signature": signature,
            "partition_signature_sha256": signature_hash,
            "csv_sha256": csv_hash,
            "rows": int(len(pd.read_csv(partition_path, usecols=["fold"]))),
        },
    )
    return partition_path, csv_hash


def _required_artifacts(run_path: Path, setup: str) -> list[Path]:
    return [
        run_path / "config.yaml",
        run_path / f"{setup}_test_results.csv",
        run_path / f"{setup}_cls_predictions.csv",
        run_path / "study_run.yaml",
    ]


def _completed_run_matches(row: dict) -> bool:
    run_path = Path(row["run_path"])
    if not all(path.exists() for path in _required_artifacts(run_path, row["setup"])):
        return False
    metadata = yaml.safe_load((run_path / "study_run.yaml").read_text(encoding="utf-8")) or {}
    return (
        metadata.get("study_id") == row["study_id"]
        and metadata.get("arm_id") == row["arm_id"]
        and metadata.get("method_id") == row["method_id"]
        and int(metadata.get("seed", -1)) == int(row["seed"])
        and metadata.get("setup") == row["setup"]
        and metadata.get("config_sha256") == row["config_sha256"]
    )


def _preserve_incomplete(run_path: Path) -> Path:
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    preserved = run_path.with_name(f"{run_path.name}.incomplete_{suffix}")
    if preserved.exists():
        raise RuntimeError(f"Cannot preserve incomplete run; target exists: {preserved}")
    run_path.rename(preserved)
    return preserved


def _annotate_csv(path: Path, metadata: dict) -> None:
    frame = pd.read_csv(path)
    ordered = ("study_id", "arm_id", "method_id", "seed")
    for column in ordered:
        frame[column] = metadata[column]
    front = list(ordered)
    frame = frame[front + [column for column in frame.columns if column not in front]]
    frame.to_csv(path, index=False)


def _finalize_run(
    generated_path: Path,
    row: dict,
    partition_sha256: str,
) -> Path:
    target = Path(row["run_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if generated_path.resolve() != target.resolve():
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite existing run target '{target}'")
        shutil.move(str(generated_path), str(target))
    elif not target.exists():
        raise RuntimeError(f"Training returned a missing run target '{target}'")
    metadata = {
        "study_id": row["study_id"],
        "arm_id": row["arm_id"],
        "method_id": row["method_id"],
        "seed": int(row["seed"]),
        "setup": row["setup"],
        "completed_at_utc": _utc_now(),
        "resolved_config": row["resolved_config"],
        "config_sha256": row["config_sha256"],
        "partition_path": row["partition_path"],
        "partition_sha256": partition_sha256,
    }
    for csv_path in (
        target / f"{row['setup']}_test_results.csv",
        target / f"{row['setup']}_cls_predictions.csv",
    ):
        if csv_path.exists():
            _annotate_csv(csv_path, metadata)
    _atomic_yaml(target / "study_run.yaml", metadata)
    missing = [path for path in _required_artifacts(target, row["setup"]) if not path.exists()]
    if missing:
        raise RuntimeError(f"Arm completed with missing artifacts: {missing}")
    return target


def _save_run_index(rows: list[dict], output_root: Path) -> None:
    _public_plan_frame(rows).to_csv(output_root / "run_index.csv", index=False)


def _collect_analysis_inputs(rows: list[dict]) -> tuple[list[Path], list[Path]]:
    results, predictions = [], []
    for row in rows:
        if row["status"] not in {"complete", "reused"}:
            continue
        run_path = Path(row["run_path"])
        result = run_path / f"{row['setup']}_test_results.csv"
        prediction = run_path / f"{row['setup']}_cls_predictions.csv"
        if result.exists():
            results.append(result)
        if prediction.exists():
            predictions.append(prediction)
    return results, predictions


def execute_study(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    seed_profile: str | None = None,
    seeds: list[int] | None = None,
    arm_ids: list[str] | None = None,
    dry_run: bool = False,
    smoke: bool = False,
    analyze_only: bool = False,
    skip_analysis: bool = False,
    retry_incomplete: bool = False,
    rebuild_partitions: bool = False,
    continue_on_error: bool = False,
) -> Path:
    manifest = load_manifest(manifest_path)
    selected_seeds = _select_seeds(manifest, seed_profile, seeds)
    selected_arms = _select_arms(manifest, arm_ids)
    rows, output_root, _ = build_execution_plan(
        manifest, selected_seeds, selected_arms, smoke=smoke
    )
    _write_plan(rows, output_root, manifest)
    logging.info("Resolved %d arm executions under %s", len(rows), output_root)
    if dry_run:
        logging.info("Dry run complete; no partitions or training runs were started")
        return output_root

    failures = []
    # One partition per (seed, partition_variant): arms sharing a variant stay exactly paired,
    # while an arm that changes the client topology gets its own frozen master.
    partition_groups = list(dict.fromkeys(
        (row["seed"], row["partition_variant"]) for row in rows
        if row["seed"] in set(selected_seeds)
    ))
    for seed, variant in partition_groups:
        seed_rows = [
            row for row in rows
            if row["seed"] == seed and row["partition_variant"] == variant
        ]
        partition_sha256 = ""
        if not analyze_only:
            try:
                _, partition_sha256 = _ensure_partition(
                    seed_rows, rebuild=rebuild_partitions
                )
            except Exception as exc:
                for row in seed_rows:
                    row["status"] = "failed_partition"
                    row["error"] = str(exc)
                _save_run_index(rows, output_root)
                if not continue_on_error:
                    raise
                failures.append(f"seed={seed} variant={variant} partition: {exc}")
                continue

        for row in seed_rows:
            target = Path(row["run_path"])
            if _completed_run_matches(row):
                row["status"] = "reused"
                logging.info("Reusing completed arm seed=%s arm=%s", seed, row["arm_id"])
                continue
            if analyze_only:
                row["status"] = "missing"
                continue
            if target.exists():
                if not retry_incomplete:
                    raise RuntimeError(
                        f"Incomplete run exists at '{target}'. Re-run with "
                        "--retry-incomplete to resume its durable folds."
                    )
                logging.warning("Resuming incomplete arm at %s", target)

            row["status"] = "running"
            row["started_at_utc"] = _utc_now()
            _save_run_index(rows, output_root)
            try:
                from src.training_federated import run as train_federated

                generated = Path(
                    train_federated(
                        row["resolved_config"],
                        run_path=target,
                        resume=retry_incomplete,
                    )
                )
                _finalize_run(generated, row, partition_sha256)
                row["status"] = "complete"
                row["completed_at_utc"] = _utc_now()
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                failures.append(f"seed={seed} arm={row['arm_id']}: {exc}")
                _save_run_index(rows, output_root)
                if not continue_on_error:
                    raise
            _save_run_index(rows, output_root)

    _save_run_index(rows, output_root)
    if not skip_analysis:
        result_paths, prediction_paths = _collect_analysis_inputs(rows)
        if result_paths:
            from src.experiments.analyze import run as analyze_runs

            analyze_runs(
                result_paths,
                prediction_paths,
                output_root / "analysis",
                manifest=manifest_path,
            )
        else:
            logging.warning("No complete result files are available for analysis")

    if failures:
        raise RuntimeError("Study finished with failures:\n- " + "\n- ".join(failures))
    return output_root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed-profile", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--rebuild-partitions", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.analyze_only:
        parser.error("--dry-run and --analyze-only are mutually exclusive")
    output = execute_study(
        args.manifest,
        seed_profile=args.seed_profile,
        seeds=args.seeds,
        arm_ids=args.arms,
        dry_run=args.dry_run,
        smoke=args.smoke,
        analyze_only=args.analyze_only,
        skip_analysis=args.skip_analysis,
        retry_incomplete=args.retry_incomplete,
        rebuild_partitions=args.rebuild_partitions,
        continue_on_error=args.continue_on_error,
    )
    print(f"STUDY_OK output={output}")


if __name__ == "__main__":
    main()
