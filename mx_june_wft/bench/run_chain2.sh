#!/usr/bin/env bash
# Chain 2: production candidates, speed, and Y-kernel diagnostics.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=/tmp/wft_logs
D=/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft

while ! grep -q "all variants done" "$LOG/bench_variants.log" 2>/dev/null; do sleep 30; done

run () {  # run <log-tag> <args...>
  local TAG=$1; shift
  echo "[$(date +%H:%M:%S)] $TAG"
  $PY mx_june_wft/bench/run_bench.py sat_det3 "$@" --jobs 6 \
      > "$LOG/bench_${TAG}.log" 2>&1
  grep -hE "within5" "$LOG/bench_${TAG}.log" | head -2
}

# Production candidates (full subset)
run candA_mf5v --variant mf5 --bundle "$D/calib_bundle_mf5v" --subset 1500
run candB_w0   --variant mf5 --bundle "$D/calib_bundle_w0"   --subset 1500
# Speed + structure (full subset)
run fast   --variant fast   --subset 1500
run mf10   --variant mf10   --subset 1500
run iter2  --variant iter2  --subset 1500
# Y-kernel diagnostics (small subset, ensemble medians only)
for V in c1p10 kyp10 ayp15 aym15 ktau120; do
  run "$V" --variant "$V" --subset 800
done
echo "[$(date +%H:%M:%S)] chain 2 done"
