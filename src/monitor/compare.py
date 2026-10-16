"""Line several runs up: final metrics side by side, mean curves, comparability warnings.

The comparison is descriptive. It never computes a p-value: the study's pairs are not
independent replicates (AGENTS.md), and the app is not the place for inference.
"""

from collections import defaultdict

import numpy as np

from src.monitor import readers

DESCRIPTIVE_NOTE = ("Comparação descritiva de runs de uma seed: as diferenças não são testadas e "
                    "não devem ser lidas como significativas.")


def warnings_for(entries):
    """Comparability warnings for catalog entries (dicts from ``catalog.describe_run``)."""
    warnings = []
    federated = [e for e in entries if e.get("kind") != "baseline"]
    modes = {e.get("budget_mode") for e in federated if e.get("budget_mode")}
    if len(modes) > 1:
        warnings.append({
            "code": "budget_mode", "severity": "critical",
            "message": "local_training.mode difere (" + ", ".join(sorted(modes)) + "): a "
                       "comparação mede o budget, não a federação (AGENTS.md)."})
    masters = {e.get("master") for e in entries if e.get("master")}
    if len(masters) > 1:
        warnings.append({
            "code": "master", "severity": "high",
            "message": "Os runs leem masters congelados diferentes: as fatias de teste diferem, "
                       "então compare só agregados por dataset/tarefa, nunca deltas por cliente "
                       "(regra D12)."})
    topologies = defaultdict(set)
    for entry in federated:
        for dataset, topology in (entry.get("topology") or {}).items():
            topologies[dataset].add(topology)
    mixed = sorted(d for d, t in topologies.items() if len(t) > 1)
    if mixed:
        warnings.append({
            "code": "topology", "severity": "high",
            "message": "Topologia de clientes difere em " + ", ".join(mixed) + ": a comparação de "
                       "topologia é agregada por dataset/tarefa, não um delta pareado por "
                       "cliente."})
    unfinished = [e["name"] for e in entries if e.get("status") != "complete"]
    if unfinished:
        warnings.append({
            "code": "unfinished", "severity": "info",
            "message": "Não concluídos (ainda sem métricas finais): " + ", ".join(unfinished) + "."})
    return warnings


def mean_curves(payload):
    """Mean over every fold x client of each ``dataset -> split/task/metric`` curve."""
    if not payload.get("available"):
        return {}
    buckets = defaultdict(lambda: defaultdict(list))
    for series in payload["series"]:
        for split in ("train", "val"):
            for key, xy in series[split].items():
                for x, y in zip(xy["x"], xy["y"]):
                    if y is not None:
                        buckets[(series["dataset"], f"{split}/{key}")][x].append(y)
    result = defaultdict(dict)
    for (dataset, key), points in buckets.items():
        xs = sorted(points)
        result[dataset][key] = {"x": xs, "y": [float(np.mean(points[x])) for x in xs]}
    return dict(result)


def compare(run_dirs, entries):
    """``run_dirs`` and their catalog ``entries`` in the same order; the first is the reference."""
    runs, reference = [], None
    for run_dir, entry in zip(run_dirs, entries):
        final = readers.final_metrics(run_dir)
        summary = {(r["dataset"], r["task"], r["metric"]): r
                   for r in (final.get("summary") or [])}
        if reference is None:
            reference = summary
        rows = []
        for key, row in summary.items():
            ref = reference.get(key)
            delta = None
            if ref is not None and summary is not reference:
                delta = row["mean"] - ref["mean"]
            rows.append({**row, "delta_vs_reference": delta})
        runs.append({"entry": entry, "final": rows,
                     "final_available": bool(final.get("available")),
                     "curves": mean_curves(readers.curves(run_dir))})
    return readers.clean({"reference": entries[0]["path"] if entries else "", "runs": runs,
                          "warnings": warnings_for(entries), "note": DESCRIPTIVE_NOTE})
