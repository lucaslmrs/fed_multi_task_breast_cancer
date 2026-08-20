"""Shared outer evaluation splits for holdout and cross-validation.

``training.CV == 1`` means one deterministic holdout split.  Values greater than one retain the
historical scikit-learn cross-validation implementations so existing frozen partitions remain
reproducible.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


DEFAULT_HOLDOUT_TEST_SIZE = 0.30
HOLDOUT = "holdout"
CROSS_VALIDATION = "cross_validation"
UNKNOWN_SINGLE_SPLIT = "unknown_single_split"


def evaluation_settings(training_config: dict) -> tuple[int, str, float | None]:
    """Validate and resolve the configured evaluation design."""
    n_splits = training_config.get("CV")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError(f"training.CV must be an integer >= 1, got {n_splits!r}")

    if n_splits > 1:
        return n_splits, CROSS_VALIDATION, None

    test_size = training_config.get("holdout_test_size", DEFAULT_HOLDOUT_TEST_SIZE)
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise ValueError(
            "training.holdout_test_size must be a number strictly between 0 and 1"
        )
    test_size = float(test_size)
    if not math.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError(
            f"training.holdout_test_size must be strictly between 0 and 1, got {test_size}"
        )
    return n_splits, HOLDOUT, test_size


def evaluation_metadata(training_config: dict) -> dict:
    """Metadata written beside metrics so analyses never guess a one-split design."""
    n_splits, scheme, test_size = evaluation_settings(training_config)
    return {
        "evaluation_scheme": scheme,
        "n_splits": n_splits,
        "holdout_test_size": test_size,
    }


def outer_split_indices(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    strategy: str,
    holdout_test_size: float = DEFAULT_HOLDOUT_TEST_SIZE,
    label_col: str = "class",
    group_col: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return positional train/test indices for the requested outer evaluation design.

    Strategies are ``stratified`` (row labels), ``random`` (unlabelled rows), and
    ``stratified_group`` (stratified group IDs mapped back to rows).
    """
    _, scheme, resolved_test_size = evaluation_settings(
        {"CV": n_splits, "holdout_test_size": holdout_test_size}
    )
    if strategy not in {"stratified", "random", "stratified_group"}:
        raise ValueError(f"Unsupported outer split strategy '{strategy}'")

    if scheme == CROSS_VALIDATION:
        if strategy == "stratified":
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            return list(splitter.split(frame, frame[label_col]))
        if strategy == "random":
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            return list(splitter.split(frame))
        if group_col is None:
            raise ValueError("stratified_group requires group_col")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=int(seed)
        )
        return list(
            splitter.split(frame, frame[label_col], groups=frame[group_col])
        )

    positions = np.arange(len(frame))
    if strategy == "random":
        train_ix, test_ix = train_test_split(
            positions,
            test_size=resolved_test_size,
            random_state=int(seed),
            shuffle=True,
        )
        return [(np.asarray(train_ix), np.asarray(test_ix))]

    if strategy == "stratified":
        train_ix, test_ix = train_test_split(
            positions,
            test_size=resolved_test_size,
            random_state=int(seed),
            shuffle=True,
            stratify=frame[label_col],
        )
        return [(np.asarray(train_ix), np.asarray(test_ix))]

    if group_col is None:
        raise ValueError("stratified_group requires group_col")
    if group_col not in frame.columns or frame[group_col].isna().any():
        raise ValueError(f"Grouped holdout requires non-null column '{group_col}'")
    conflicts = frame.groupby(group_col, sort=False)[label_col].nunique(dropna=False)
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.tolist()[:5]
        raise ValueError(f"Groups span multiple classes in '{group_col}': {bad}")

    groups = (
        frame.groupby(group_col, sort=False, as_index=False)
        .agg(**{label_col: (label_col, "first")})
    )
    train_groups, test_groups = train_test_split(
        groups[group_col],
        test_size=resolved_test_size,
        random_state=int(seed),
        shuffle=True,
        stratify=groups[label_col],
    )
    train_mask = frame[group_col].isin(set(train_groups))
    test_mask = frame[group_col].isin(set(test_groups))
    return [(positions[train_mask.to_numpy()], positions[test_mask.to_numpy()])]


def validate_master_splits(master: pd.DataFrame, training_config: dict) -> None:
    """Reject a frozen master whose outer split count does not match the run config."""
    n_splits, scheme, _ = evaluation_settings(training_config)
    if "fold" not in master.columns:
        raise ValueError("Federated partition is missing required column 'fold'")
    actual = sorted(pd.to_numeric(master["fold"], errors="raise").astype(int).unique())
    expected = list(range(n_splits))
    if actual != expected:
        raise ValueError(
            f"Federated partition has fold IDs {actual}, but training.CV={n_splits} "
            f"requires {expected} ({scheme}). Regenerate the intended partition explicitly."
        )
    if "split" in master.columns:
        missing = {"train", "test"}.difference(set(master["split"].astype(str)))
        if missing:
            raise ValueError(f"Federated partition is missing required splits: {sorted(missing)}")
