"""
Cross-setup analysis for the comparison experiment.

Reads the per-setup results CSVs (federated / standalone / centralized), all produced on the SAME
frozen partition, and computes:
  - summary_per_task_setup.csv : mean +/- std of every metric per (task, setup) across client x fold
  - federated_vs_local_wilcoxon.csv : paired Wilcoxon (federated vs standalone) + mean/median delta
  - per_client_deltas.csv : federated - local per (fold, client) for the primary metric
  - pooled_auc.csv : OvR macro AUC pooled across clients (per setup x fold, and overall)

Usage: edit the FED / STD / CEN run-directory constants below and run

    python -m src.experiments.analyze

or override them on the CLI with --results / --preds / --out.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

# --- Edit these to point at each setup's run directory before running (CLI flags override them) ---
FED = "runs/<FED>"  # federated run dir
STD = "runs/<STD>"  # standalone (local-only) run dir
CEN = "runs/<CEN>"  # centralized run dir

RESULTS = [
    f"{FED}/federated_test_results.csv",
    f"{STD}/standalone_test_results.csv",
    f"{CEN}/centralized_test_results.csv",
]
PREDS = [
    f"{FED}/federated_cls_predictions.csv",
    f"{STD}/standalone_cls_predictions.csv",
    f"{CEN}/centralized_cls_predictions.csv",
]
OUT = "runs/comparison"
# --------------------------------------------------------------------------------------------------

SEG_METRICS = ["dice", "iou", "sensitivity", "specificity", "precision"]
CLS_METRICS = ["acc", "macro_f1", "balanced_acc", "auc",
               "precision_class_0", "recall_class_0", "precision_class_1", "recall_class_1",
               "precision_class_2", "recall_class_2"]
PRIMARY = {"seg": "dice", "cls": "acc"}


def _metrics_for(task):
    return SEG_METRICS if task == "seg" else CLS_METRICS


def summary_table(df):
    rows = []
    for (setup, task), g in df.groupby(["setup", "task"]):
        for m in _metrics_for(task):
            if m in g:
                rows.append({"setup": setup, "task": task, "metric": m,
                             "mean": g[m].mean(), "std": g[m].std(), "n": int(g[m].notna().sum())})
    return pd.DataFrame(rows)


def paired_deltas(df, a="federated", b="standalone"):
    """Paired Wilcoxon of setup `a` vs `b` across (fold, client_id), per task and metric."""
    key = ["fold", "client_id", "task"]
    da, db = df[df.setup == a].set_index(key), df[df.setup == b].set_index(key)
    common = da.index.intersection(db.index)
    rows = []
    for task in sorted(df.task.unique()):
        idx = [i for i in common if i[2] == task]
        for m in _metrics_for(task):
            if m not in da or m not in db:
                continue
            va, vb = da.loc[idx, m].to_numpy(float), db.loc[idx, m].to_numpy(float)
            keep = ~(np.isnan(va) | np.isnan(vb))
            va, vb = va[keep], vb[keep]
            if len(va) == 0:
                continue
            diff = va - vb
            try:
                stat, p = wilcoxon(va, vb)
            except ValueError:  # all-zero diffs or too few pairs
                stat, p = np.nan, np.nan
            rows.append({"task": task, "metric": m, "n_pairs": len(va),
                         f"{a}_mean": va.mean(), f"{b}_mean": vb.mean(),
                         "mean_delta": diff.mean(), "median_delta": float(np.median(diff)),
                         "wilcoxon_stat": stat, "wilcoxon_p": p})
    return pd.DataFrame(rows)


def per_client_deltas(df, a="federated", b="standalone"):
    """federated - local per (fold, client_id) for each task's primary metric."""
    key = ["fold", "client_id", "task"]
    da, db = df[df.setup == a].set_index(key), df[df.setup == b].set_index(key)
    rows = []
    for idx in da.index.intersection(db.index):
        fold, client_id, task = idx
        m = PRIMARY[task]
        rows.append({"fold": fold, "client_id": client_id, "task": task, "metric": m,
                     a: da.loc[idx, m], b: db.loc[idx, m], "delta": da.loc[idx, m] - db.loc[idx, m]})
    return pd.DataFrame(rows)


def pooled_auc(pred_paths):
    if not pred_paths:
        return pd.DataFrame()
    preds = pd.concat([pd.read_csv(p) for p in pred_paths], ignore_index=True)
    prob_cols = sorted(c for c in preds.columns if c.startswith("prob_"))
    labels = list(range(len(prob_cols)))

    def auc(g):
        try:
            return roc_auc_score(g["ground_truth"], g[prob_cols].to_numpy(),
                                 multi_class="ovr", average="macro", labels=labels)
        except ValueError:
            return np.nan

    rows = [{"setup": s, "fold": f, "auc_pooled": auc(g)} for (s, f), g in preds.groupby(["setup", "fold"])]
    rows += [{"setup": s, "fold": "all", "auc_pooled": auc(g)} for s, g in preds.groupby("setup")]
    return pd.DataFrame(rows)


def run(results, preds, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in results], ignore_index=True)

    summ = summary_table(df)
    summ.to_csv(out / "summary_per_task_setup.csv", index=False)

    if {"federated", "standalone"}.issubset(set(df.setup.unique())):
        paired_deltas(df).to_csv(out / "federated_vs_local_wilcoxon.csv", index=False)
        per_client_deltas(df).to_csv(out / "per_client_deltas.csv", index=False)

    pauc = pooled_auc(preds)
    if not pauc.empty:
        pauc.to_csv(out / "pooled_auc.csv", index=False)

    logging.info(f"\nSummary (mean per task x setup):\n"
                 f"{summ.pivot_table(index=['task', 'metric'], columns='setup', values='mean').round(4)}")
    logging.info(f"Wrote analysis tables to {out}")


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=RESULTS, help="per-setup *_test_results.csv files")
    ap.add_argument("--preds", nargs="*", default=PREDS, help="per-setup *_cls_predictions.csv files (for pooled AUC)")
    ap.add_argument("--out", default=OUT, help="output directory for the analysis tables")
    args = ap.parse_args()
    run(args.results, args.preds, args.out)


if __name__ == "__main__":
    main()
