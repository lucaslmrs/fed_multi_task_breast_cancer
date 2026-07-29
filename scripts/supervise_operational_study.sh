#!/usr/bin/env bash
# Keep the long operational study recoverable across transient GPU/Ray failures.
set -euo pipefail

STUDY_HOME="/home/lucas/fed_multi_task_breast_cancer"
STUDY_DIR="$STUDY_HOME/runs/studies/multi_dataset_balance_v1"
SUPERVISOR_LOG="$STUDY_DIR/supervisor.log"

cd "$STUDY_HOME"

all_arms_complete() {
  .venv/bin/python - <<'PY'
import pandas as pd

index = pd.read_csv("runs/studies/multi_dataset_balance_v1/run_index.csv")
complete = {"complete", "reused"}
raise SystemExit(0 if len(index) and index.status.isin(complete).all() else 1)
PY
}

runner_is_active() {
  pgrep -f '^\.venv/bin/python -u -m src\.experiments\.study_runner' >/dev/null
}

while ! all_arms_complete; do
  if runner_is_active; then
    sleep 60
    continue
  fi

  if ! nvidia-smi >/dev/null 2>&1; then
    printf '[%s] GPU indisponível; aguardando 5 min antes de nova tentativa\n' "$(date -Is)" >> "$SUPERVISOR_LOG"
    sleep 300
    continue
  fi

  printf '[%s] retomando braços/folds incompletos\n' "$(date -Is)" >> "$SUPERVISOR_LOG"
  .venv/bin/python -u -m src.experiments.study_runner --seed-profile operational --retry-incomplete \
    >> "$SUPERVISOR_LOG" 2>&1 || true
  sleep 30
done

# The runner writes the complete analysis before it exits. Generate the final executive handoff and
# repeat the automated verification after the last successful completion.
.venv/bin/python -m src.experiments.executive_report \
  --output RELATORIO_EXECUTIVO_FEDERACAO_MULTI_DATASET.html >> "$SUPERVISOR_LOG" 2>&1
.venv/bin/python -m pytest -q >> "$SUPERVISOR_LOG" 2>&1
printf '[%s] estudo, relatório e testes finais concluídos\n' "$(date -Is)" >> "$SUPERVISOR_LOG"
