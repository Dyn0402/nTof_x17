#!/usr/bin/env bash
# latency_scan.sh -- at a FIXED window length, where should the frame sit?
#
# The window scan showed that cropping to 26 samples -- which removes only
# PRE-SIGNAL samples and keeps the whole tail -- already costs angle
# compression (-0.10 -> -0.33 deg on Y). The leading edge is not dead weight:
# it is what constrains t0, the mesh arrival time. So the frame position is a
# free parameter worth optimising at fixed readout cost.
#
# Scans the crop start at n=20 (run_79's length). start=6 is the measured
# run_79 framing; smaller start = the signal sits LATER in the frame = a
# HIGHER DREAM latency.
#
#   bash mx_june_wft/bench/latency_scan.sh [JOBS] [SUBSET]
set -u
JOBS=${1:-2}
SUBSET=${2:-1200}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY="$REPO/.venv/bin/python"
BUNDLE=/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/calib_bundle_lp2
for S in 3 4 5 7 8; do
  echo "=== start $S, n 20 ==="
  nice -n 15 "$PY" "$REPO/mx_june_wft/bench/run_bench.py" sat_det3 \
      --bundle "$BUNDLE" --variant prod --subset "$SUBSET" --jobs "$JOBS" \
      --tag "s${S}n20" --crop "${S}:20" 2>&1 | grep -v '^  [0-9]'
done
echo "ALL DONE"
