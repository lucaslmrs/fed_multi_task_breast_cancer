"""Check that a candidate federated master is a reference master read at another resolution.

E4 reads TCGA-LGG and SIIM-ACR from a ``processed_256`` variant, and a master copies every
mapping column (paths, ``dim1``/``dim2``, mask statistics), so the E4 run needs its own master.
It is comparable with the E3 runs only if it is the SAME partition: same rows, same clients, same
splits, same images, same labels. Only the variant prefix of the paths and the mask-derived
numbers of the rescaled datasets may change. Every check below is a problem when it fails, and any
problem exits with status 1:

* identical columns and row count;
* identical ``dataset, fold, client_id, task, split, id`` in the same row order;
* the rows of every dataset NOT rescaled are identical, byte for byte as CSV text;
* in a rescaled dataset, ``class``, ``id``, ``lesion_id`` and ``official_split`` are identical
  (so 0 class flips), and ``img_path``/``mask_path`` are identical after swapping the
  ``paths.processed_dir`` prefix of the source variant for the target one;
* in a rescaled dataset, ``dim1 == dim2 == size`` and ``tumor_pixels > 0`` is unchanged row by
  row (so an empty mask stays empty and a lesion never vanishes);
* the rescaled image and mask files exist (``--no-file-check`` skips this);
* the reference's ``*_metadata.yaml`` records the reference CSV's actual sha256.

``--write-metadata STUDY_ID SEED --config RUN_CONFIG`` then writes ``<candidate>_metadata.yaml``
in the schema of ``study_runner._ensure_partition`` (partition signature, its hash, the CSV
sha256, the row count), plus the path of the equivalence report, so the candidate is a frozen
study master like the others. The metadata is never overwritten.

Example (E4):
    python -m scripts.partition_equivalence \\
        --reference data/federated_studies/e3_loss_ablation/seed_1993/federated_mapping.csv \\
        --candidate data/federated_studies/e4_preprocessing/seed_1993/federated_mapping.csv \\
        --rescaled TCGA_LGG SIIM_ACR \\
        --write-metadata e4_preprocessing 1993 --config studies/e4_preprocessing/resolution_256.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from src.dataset import paths
from src.experiments.study_runner import (
    DEFAULT_PARTITION_VARIANT,
    _atomic_yaml,
    _partition_metadata_path,
    _partition_signature,
    _sha256_file,
    _stable_hash,
    _utc_now,
)

KEY_COLUMNS = ["dataset", "fold", "client_id", "task", "split", "id"]
IDENTITY_COLUMNS = ["class", "id", "lesion_id", "official_split"]
PATH_COLUMNS = ["img_path", "mask_path"]
_VARIANT_SIZE = re.compile(r"^processed_(\d+)$")


def read_master(path: str | Path) -> pd.DataFrame:
    """Every cell as its CSV text, so "identical" means identical on disk and NaN == NaN ("")."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _numeric(column: pd.Series) -> pd.Series:
    return pd.to_numeric(column.replace("", float("nan")), errors="coerce")


def _first_rows(mask: pd.Series, limit: int = 3) -> list:
    return [int(index) for index in mask[mask].index[:limit]]


def _variant_prefix(root: str, dataset: str, variant: str) -> str:
    return paths.processed_dir({"root": root, "dataset": dataset, "variant": variant}).as_posix() + "/"


def _positive_images(frame: pd.DataFrame) -> int:
    return int(frame.loc[_numeric(frame["tumor_pixels"]) > 0, "id"].nunique())


def compare_masters(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    rescaled: Iterable[str],
    *,
    root: str = "data",
    from_variant: str = "processed_128",
    to_variant: str = "processed_256",
    size: int = 256,
    check_files: bool = True,
) -> dict:
    """Return ``{"problems": [...], "datasets": {...}}``; an empty problem list means equivalent."""
    rescaled = list(rescaled)
    problems: list[str] = []
    report = {"rows": int(len(candidate)), "problems": problems, "datasets": {}}

    if list(reference.columns) != list(candidate.columns):
        problems.append(f"columns differ: reference {list(reference.columns)}, "
                        f"candidate {list(candidate.columns)}")
        return report
    if len(reference) != len(candidate):
        problems.append(f"row count differs: reference {len(reference)}, candidate {len(candidate)}")
        return report
    missing = [column for column in KEY_COLUMNS + ["tumor_pixels", "dim1", "dim2"]
               if column not in reference.columns]
    if missing:
        problems.append(f"masters lack required columns {missing}")
        return report
    reference = reference.reset_index(drop=True)
    candidate = candidate.reset_index(drop=True)

    key_differs = (reference[KEY_COLUMNS] != candidate[KEY_COLUMNS]).any(axis=1)
    if key_differs.any():
        problems.append(f"{int(key_differs.sum())} rows differ in {KEY_COLUMNS} (row order, "
                        f"clients or splits changed), e.g. rows {_first_rows(key_differs)}")

    unknown = sorted(set(rescaled) - set(reference["dataset"]))
    if unknown:
        problems.append(f"--rescaled names datasets absent from the reference: {unknown}")

    for dataset in sorted(set(reference["dataset"]) - set(rescaled)):
        rows = reference["dataset"] == dataset
        differs = (reference[rows] != candidate[rows]).any(axis=1)
        report["datasets"][dataset] = {"rescaled": False, "rows": int(rows.sum()),
                                       "rows_differing": int(differs.sum())}
        if differs.any():
            problems.append(f"{dataset}: {int(differs.sum())} rows differ although it is not "
                            f"rescaled, e.g. rows {_first_rows(differs)}")

    prefixes = {}
    for dataset in [name for name in rescaled if name not in unknown]:
        rows = reference["dataset"] == dataset
        ref, cand = reference[rows], candidate[rows]
        summary = {"rescaled": True, "rows": int(rows.sum()),
                   "images": int(cand["id"].nunique()),
                   "positive_images_reference": _positive_images(ref),
                   "positive_images_candidate": _positive_images(cand)}
        report["datasets"][dataset] = summary

        for column in [name for name in IDENTITY_COLUMNS if name in reference.columns]:
            differs = ref[column] != cand[column]
            summary[f"{column}_differing"] = int(differs.sum())
            if differs.any():
                problems.append(f"{dataset}: {int(differs.sum())} rows change '{column}', "
                                f"e.g. rows {_first_rows(differs)}")

        source = _variant_prefix(root, dataset, from_variant)
        target = _variant_prefix(root, dataset, to_variant)
        prefixes[dataset] = {"from": source, "to": target}
        for column in [name for name in PATH_COLUMNS if name in reference.columns]:
            present = ref[column] != ""
            outside = present & ~ref[column].str.startswith(source)
            if outside.any():
                problems.append(f"{dataset}: {int(outside.sum())} reference '{column}' values are "
                                f"not under '{source}', e.g. rows {_first_rows(outside)}")
            expected = ref[column].where(~present, target + ref[column].str[len(source):])
            differs = expected != cand[column]
            summary[f"{column}_differing"] = int(differs.sum())
            if differs.any():
                problems.append(f"{dataset}: {int(differs.sum())} '{column}' values are not the "
                                f"reference path under '{target}', e.g. rows {_first_rows(differs)}")
            if check_files:
                absent = [value for value in cand.loc[cand[column] != "", column].unique()
                          if not Path(value).exists()]
                summary[f"{column}_missing_files"] = len(absent)
                if absent:
                    problems.append(f"{dataset}: {len(absent)} '{column}' files do not exist, "
                                    f"e.g. {absent[:2]}")

        for column in ("dim1", "dim2"):
            wrong = _numeric(cand[column]) != size
            if wrong.any():
                problems.append(f"{dataset}: {int(wrong.sum())} rows have {column} != {size}, "
                                f"e.g. rows {_first_rows(wrong)}")

        ref_pixels, cand_pixels = _numeric(ref["tumor_pixels"]), _numeric(cand["tumor_pixels"])
        flips = ((ref_pixels > 0) != (cand_pixels > 0)) | (ref_pixels.isna() != cand_pixels.isna())
        summary["positive_flips"] = int(flips.sum())
        if flips.any():
            problems.append(f"{dataset}: {int(flips.sum())} rows change 'tumor_pixels > 0' (a "
                            f"lesion vanished or appeared), e.g. rows {_first_rows(flips)}")

    report["path_prefixes"] = prefixes
    return report


def check_reference_metadata(reference_path: Path, reference_sha256: str) -> list[str]:
    metadata_path = _partition_metadata_path(reference_path)
    if not metadata_path.exists():
        return [f"reference metadata '{metadata_path}' does not exist"]
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    if metadata.get("csv_sha256") != reference_sha256:
        return [f"reference sha256 {reference_sha256} does not match its metadata "
                f"({metadata.get('csv_sha256')}); the reference master changed"]
    return []


def candidate_metadata(config: dict, study_id: str, seed: int, candidate_sha256: str, rows: int,
                       report_path: Path) -> dict:
    """Metadata in the ``study_runner._ensure_partition`` schema, plus the equivalence report."""
    signature = _partition_signature(config)
    return {
        "study_id": study_id,
        "seed": int(seed),
        "partition_variant": DEFAULT_PARTITION_VARIANT,
        "created_at_utc": _utc_now(),
        "partition_signature": signature,
        "partition_signature_sha256": _stable_hash(signature),
        "csv_sha256": candidate_sha256,
        "rows": int(rows),
        "equivalence_report": str(report_path),
    }


def check_run_config(config: dict, candidate_path: Path, seed: int, rescaled: list[str],
                     to_variant: str) -> list[str]:
    """The run config must be the one the candidate was built from (path, seed, variants)."""
    problems = []
    configured = config.get("federated", {}).get("partition_file")
    if not configured or Path(configured).resolve() != candidate_path.resolve():
        problems.append(f"config federated.partition_file '{configured}' is not the candidate")
    if int(config.get("training", {}).get("seed", -1)) != int(seed):
        problems.append(f"config training.seed {config.get('training', {}).get('seed')} != {seed}")
    for dataset in rescaled:
        variant = config.get("datasets", {}).get(dataset, {}).get("variant")
        if variant != to_variant:
            problems.append(f"config datasets.{dataset}.variant is '{variant}', not '{to_variant}'")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="frozen master read at --from-variant")
    parser.add_argument("--candidate", required=True, help="new master read at --to-variant")
    parser.add_argument("--rescaled", nargs="+", required=True, metavar="DATASET",
                        help="datasets whose variant changed; every other dataset must be identical")
    parser.add_argument("--root", default="data", help="data.root of both masters (default: data)")
    parser.add_argument("--from-variant", default="processed_128")
    parser.add_argument("--to-variant", default="processed_256")
    parser.add_argument("--size", type=int, default=None,
                        help="expected dim1/dim2 (default: the N of --to-variant processed_N)")
    parser.add_argument("--no-file-check", action="store_true",
                        help="do not check that the rescaled images and masks exist")
    parser.add_argument("--write-metadata", nargs=2, metavar=("STUDY_ID", "SEED"),
                        help="write <candidate>_metadata.yaml when every check passes")
    parser.add_argument("--config", help="run config the candidate was built from "
                                         "(required by --write-metadata)")
    parser.add_argument("--report", help="where to write the YAML report (default with "
                                         "--write-metadata: <candidate>_equivalence.yaml)")
    args = parser.parse_args(argv)

    size = args.size
    if size is None:
        match = _VARIANT_SIZE.match(args.to_variant)
        if match is None:
            parser.error("--size is required when --to-variant is not processed_<N>")
        size = int(match.group(1))
    if args.write_metadata and not args.config:
        parser.error("--write-metadata needs --config (the run config the candidate came from)")

    reference_path, candidate_path = Path(args.reference), Path(args.candidate)
    for path in (reference_path, candidate_path):
        if not path.exists():
            print(f"[FAIL] '{path}' does not exist")
            return 1
    reference_sha256 = _sha256_file(reference_path)
    candidate_sha256 = _sha256_file(candidate_path)

    report = compare_masters(
        read_master(reference_path), read_master(candidate_path), args.rescaled,
        root=args.root, from_variant=args.from_variant, to_variant=args.to_variant, size=size,
        check_files=not args.no_file_check,
    )
    problems = report["problems"]
    problems.extend(check_reference_metadata(reference_path, reference_sha256))
    report.update(reference=str(reference_path), reference_sha256=reference_sha256,
                  candidate=str(candidate_path), candidate_sha256=candidate_sha256,
                  rescaled=list(args.rescaled), from_variant=args.from_variant,
                  to_variant=args.to_variant, size=size, checked_at_utc=_utc_now())

    config = None
    if args.write_metadata:
        with open(args.config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        problems.extend(check_run_config(config, candidate_path, int(args.write_metadata[1]),
                                         list(args.rescaled), args.to_variant))
        if _partition_metadata_path(candidate_path).exists():
            problems.append(f"'{_partition_metadata_path(candidate_path)}' already exists; "
                            f"a master's metadata is written once")

    print(f"reference {reference_path} sha256={reference_sha256}")
    print(f"candidate {candidate_path} sha256={candidate_sha256} rows={report['rows']}")
    for dataset, summary in sorted(report["datasets"].items()):
        print(f"  {dataset}: " + ", ".join(f"{key}={value}" for key, value in summary.items()))

    report_path = Path(args.report) if args.report else None
    if report_path is None and args.write_metadata:
        report_path = candidate_path.with_name(candidate_path.stem + "_equivalence.yaml")
    report["status"] = "fail" if problems else "ok"
    if report_path is not None:
        _atomic_yaml(report_path, report)
        print(f"report {report_path}")

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1

    if args.write_metadata:
        study_id, seed = args.write_metadata[0], int(args.write_metadata[1])
        metadata_path = _partition_metadata_path(candidate_path)
        _atomic_yaml(metadata_path, candidate_metadata(
            config, study_id, seed, candidate_sha256, report["rows"], report_path))
        print(f"metadata {metadata_path}")
    print("EQUIVALENCE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
