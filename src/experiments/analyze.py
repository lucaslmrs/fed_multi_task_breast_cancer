"""
Cross-setup analysis for the comparison experiment.

Reads the per-setup results CSVs (federated / standalone / centralized), all produced on the SAME
frozen partition, and computes:
  - summary_per_task_setup.csv : mean +/- std of every metric per (task, setup) across client x fold
  - federated_vs_local_wilcoxon.csv : paired Wilcoxon (federated vs standalone) + mean/median delta
  - per_client_deltas.csv : federated - local per (fold, client) for the primary metric
  - pooled_auc.csv : OvR macro AUC pooled across clients (per setup x fold, and overall)
  - report.html : self-contained report with charts (metrics by setup, per-client deltas, AUC)

Usage: edit the FED / STD / CEN run-directory constants below and run

    python -m src.experiments.analyze

or override them on the CLI with --results / --preds / --out.
"""

import argparse
import base64
import io
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend selection)

# --- Edit these to point at each setup's run directory before running (CLI flags override them) ---
FED = "runs/20260622_192540_FEDERATED_MTnnUNet_2seg_2cls"  # federated run dir
STD = "runs/20260622_195528_STANDALONE_MTnnUNet_2seg_2cls"  # standalone (local-only) run dir
CEN = "runs/20260622_190616_CENTRALIZED_MTnnUNet"  # centralized run dir

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


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _metric_boxplot(df, task, metrics):
    """Boxplots of each metric's distribution across client×fold, grouped by setup."""
    sub = df[df.task == task]
    setups = sorted(sub.setup.unique())
    width = 0.8 / max(len(setups), 1)
    colors = plt.cm.Set2.colors
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, s in enumerate(setups):
        ss = sub[sub.setup == s]
        for j, m in enumerate(metrics):
            vals = ss[m].dropna().to_numpy()
            if len(vals) == 0:
                continue
            pos = j + i * width - 0.4 + width / 2
            bp = ax.boxplot(vals, positions=[pos], widths=width * 0.9,
                            patch_artist=True, manage_ticks=False)
            bp["boxes"][0].set(facecolor=colors[i % len(colors)], alpha=0.85)
        ax.plot([], [], color=colors[i % len(colors)], lw=6, label=s)  # legend proxy
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{task} — distribution across client×fold")
    ax.legend(fontsize=8)
    return _fig_to_b64(fig)


def _delta_bars(per_client, task):
    """federated − local per client×fold (green = federation helps, red = hurts)."""
    sub = per_client[per_client.task == task].copy()
    sub["label"] = sub["client_id"] + " f" + sub["fold"].astype(str)
    colors = ["#2a9d8f" if d >= 0 else "#e76f51" for d in sub["delta"]]
    fig, ax = plt.subplots(figsize=(max(6, len(sub) * 0.5), 4))
    ax.bar(sub["label"], sub["delta"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=60)
    ax.set_title(f"{task} — federated − local ({PRIMARY[task]}) per client×fold")
    return _fig_to_b64(fig)


def _auc_bars(pauc):
    overall = pauc[pauc.fold == "all"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(overall["setup"], overall["auc_pooled"], color="#264653")
    ax.set_ylim(0, 1)
    ax.set_title("Pooled OvR-macro AUC (all folds)")
    return _fig_to_b64(fig)


def build_html(df, summ, deltas, per_client, pauc, out):
    charts = [(f"{task} metrics by setup",
               _metric_boxplot(df, task, SEG_METRICS if task == "seg" else ["acc", "macro_f1", "balanced_acc", "auc"]))
              for task in sorted(df.task.unique())]
    if not per_client.empty:
        charts += [(f"{task} per-client delta", _delta_bars(per_client, task))
                   for task in sorted(per_client.task.unique())]
    if not pauc.empty:
        charts.append(("Pooled AUC", _auc_bars(pauc)))

    figs = "".join(f'<h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{b64}"/>'
                   for title, b64 in charts)
    tables = ""
    if not deltas.empty:
        tables += "<h2>Federated vs Local — paired Wilcoxon</h2>" + deltas.round(4).to_html(index=False)
    tables += "<h2>Summary (mean ± std per task × setup)</h2>" + summ.round(4).to_html(index=False)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Comparison report</title><style>
body{{font-family:system-ui,Arial,sans-serif;margin:2rem auto;max-width:1000px;color:#222}}
h1{{border-bottom:2px solid #264653}} h2{{margin-top:2rem;color:#264653}}
img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
table{{border-collapse:collapse;font-size:.82rem}} th,td{{border:1px solid #ccc;padding:4px 8px}}
th{{background:#f4f4f4}}</style></head><body>
<h1>Federated vs Local-only vs Centralized</h1>
{figs}{tables}
</body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")


def run(results, preds, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in results], ignore_index=True)

    summ = summary_table(df)
    summ.to_csv(out / "summary_per_task_setup.csv", index=False)

    deltas, per_client = pd.DataFrame(), pd.DataFrame()
    if {"federated", "standalone"}.issubset(set(df.setup.unique())):
        deltas = paired_deltas(df)
        deltas.to_csv(out / "federated_vs_local_wilcoxon.csv", index=False)
        per_client = per_client_deltas(df)
        per_client.to_csv(out / "per_client_deltas.csv", index=False)

    pauc = pooled_auc(preds)
    if not pauc.empty:
        pauc.to_csv(out / "pooled_auc.csv", index=False)

    build_html(df, summ, deltas, per_client, pauc, out)

    logging.info(f"\nSummary (mean per task x setup):\n"
                 f"{summ.pivot_table(index=['task', 'metric'], columns='setup', values='mean').round(4)}")
    logging.info(f"Wrote analysis tables + report.html to {out}")


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
