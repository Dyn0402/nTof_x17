#!/usr/bin/env bash
# Y-plane knob scan around the converged lp2 optimum (sigma_Y trade-off).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=/tmp/wft_logs; mkdir -p "$LOG"
D=/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft

run () {  # run <tag> <patch-json>
  echo "[$(date +%H:%M:%S)] $1"
  $PY mx_june_wft/bench/run_bench.py sat_det3 --variant prod \
      --bundle "$D/calib_bundle_lp2" --subset 800 --jobs 8 \
      --tag "$1" --patch "$2" > "$LOG/scan_$1.log" 2>&1
  grep -hE "within5" "$LOG/scan_$1.log" | head -1
}

run lp2base   '{}'
run ktau12    '{"kTauY": 1.2}'
run ktau24    '{"kTauY": 2.4}'
run ktau30    '{"kTauY": 3.0}'
run ssy40     '{"sigma_sY": 40.0}'
run ssy80     '{"sigma_sY": 80.0}'
run sp0y25    '{"sigma_p0Y": 0.25}'
run sp0y15    '{"sigma_p0Y": 0.15, "DpY": 0.008}'
echo "[$(date +%H:%M:%S)] scan done"
