#!/usr/bin/env bash
# Sequential benchmark of reconstruction variants on the det3 1500-event subset.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=/tmp/wft_logs

# wait for the baseline run to release its workers
while ! grep -q "wrote" "$LOG/bench_base1500.log" 2>/dev/null; do sleep 30; done

for V in "$@"; do
  echo "[$(date +%H:%M:%S)] variant $V"
  $PY mx_june_wft/bench/run_bench.py sat_det3 --variant "$V" --subset 1500 \
      --jobs 6 > "$LOG/bench_${V}1500.log" 2>&1
  tail -3 "$LOG/bench_${V}1500.log"
done
echo "[$(date +%H:%M:%S)] all variants done"
