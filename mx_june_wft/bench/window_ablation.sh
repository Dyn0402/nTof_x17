#!/usr/bin/env bash
# window_ablation.sh -- what does the n_TOF readout window cost the fit?
#
# Crops the det3 bench windows to the run_79 framing (start = +6 samples,
# measured by framing_compare.py: the bench prompt sits at sample 6.4, the beam
# prompt at 0.2, identically on all four chambers and both planes) and scans
# the kept length. Every point is the SAME 1200 events (seed 42), so the
# comparison is paired.
#
#   n = 20  is run_79 as recorded
#   n < 20  emulates the wetter/slower chambers, whose column is longer
#   n > 20  is what asking for more samples would buy
#
#   bash mx_june_wft/bench/window_ablation.sh [JOBS] [SUBSET]
set -u
JOBS=${1:-3}
SUBSET=${2:-1200}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY="$REPO/.venv/bin/python"
BUNDLE=/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/calib_bundle_lp2
START=6

run () {  # run <tag> <extra args...>
  local tag=$1; shift
  echo "=== $tag ==="
  nice -n 15 "$PY" "$REPO/mx_june_wft/bench/run_bench.py" sat_det3 \
      --bundle "$BUNDLE" --variant prod --subset "$SUBSET" --jobs "$JOBS" \
      --tag "$tag" "$@" 2>&1 | grep -v '^  [0-9]'
}

run full32                                    # reference: the whole bench window
for N in 20 26 16 24 18 22 14; do
  run "w${N}" --crop "${START}:${N}"
done
# does shortening the charge basis help when the window is short?
run w20_k15 --crop "${START}:20" --k-bins 15
run w20_k12 --crop "${START}:20" --k-bins 12
echo "ALL DONE"
