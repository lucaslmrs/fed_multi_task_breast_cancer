#!/usr/bin/env bash
# Run one federated arm to completion, resuming from the last round checkpoint after a crash.
#
#   scripts/run_with_resume.sh <config.yaml> <run_dir> [max_attempts=20]
#
# A dead Ray actor (e.g. a lost CUDA context under WSL2) does not end the Python process: Flower
# keeps refusing rounds. The watchdog kills the run as soon as a round is refused, stops Ray, and
# restarts it with --resume, which continues from round_checkpoint/ instead of round 1.
set -u
config="$1"; run_dir="$2"; max_attempts="${3:-20}"
cd "$(dirname "$0")/.." || exit 1
log_dir="$(dirname "$run_dir")"; name="$(basename "$run_dir")"
mkdir -p "$log_dir"
status="$log_dir/$name.status"

for attempt in $(seq 1 "$max_attempts"); do
  resume=(); [ -d "$run_dir" ] && resume=(--resume)
  log="$log_dir/$name.attempt$attempt.log"
  echo "$(date '+%F %T') START attempt=$attempt ${resume[*]}" >> "$status"
  .venv/bin/python -m src.training_federated --config "$config" --run-path "$run_dir" \
    "${resume[@]}" > "$log" 2>&1 &
  pid=$!
  failed=0
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q -E "refusing to aggregate|refusing validation aggregate|ActorPool is empty" "$log"; then
      failed=1; kill -9 "$pid"; .venv/bin/ray stop --force >/dev/null 2>&1; break
    fi
    sleep 30
  done
  wait "$pid"; rc=$?
  [ "$failed" = 1 ] && rc=ACTOR_DIED
  echo "$(date '+%F %T') END attempt=$attempt exit=$rc" >> "$status"
  [ "$rc" = 0 ] && { echo "$(date '+%F %T') DONE" >> "$status"; exit 0; }
  sleep 30
done
echo "$(date '+%F %T') GAVE_UP" >> "$status"
exit 1
