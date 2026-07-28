#!/usr/bin/env bash
# Fleet driver: run the waveform-first chain on every golden key, in order of
# how much is already known about the detector. Sequential on purpose -- each
# reco already saturates the box, and running two at once thrashes the disk.
#
#   mx_june_wft/run_fleet.sh [--jobs N]
#
# det3 must have passed its gate before this is worth starting.
set -u
cd "$(dirname "$0")/.."
JOBS=${2:-12}
LOG=/tmp/wft_logs; mkdir -p "$LOG"
A=/home/dylan/x17/cosmic_bench/Analysis

stage () {  # name, then the run_chain.sh args
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] ===== $name ====="
  bash mx_june_wft/run_chain.sh "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  tail -n 25 "$LOG/$name.log"
  echo "[$(date +%H:%M:%S)] $name exit=$rc"
  return 0                      # keep going; a bad detector must not stop the fleet
}

# --- detectors with a validated R&D calibration -----------------------------
stage det2 o22_long_det2 --jobs "$JOBS" \
      --legacy "$A/mx17_det2_det3_overnight_6-22-26/long_run/mx17_2/waveform_first" \
      --hyper-file hyper_det2.json
# NB: the det2 bundle was fitted on the 6-22 long_run and is used here on the
# longer_run subrun of the SAME run (same HV/gas, different duration) so the
# numbers compare with rerun_baseline.json. check_conditions flags the subrun.

stage det4 g_det4 --jobs "$JOBS" \
      --legacy "$A/mx17_det4_day_6-24-26/long_run/mx17_4/waveform_first" \
      --hyper-file hyper_det4.json

# --- detectors that need their own calibration first ------------------------
# det6/det7 are the low-gain pair: expect the CALIBRATION, not the fit, to be
# the hard part. Seeded from det3's kernel (same production batch) so the
# optimiser starts somewhere physical.
DET3_BUNDLE="$A/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/calib_bundle"
stage det6 g_det6_long --jobs "$JOBS" --calibrate --seed-bundle "$DET3_BUNDLE"
stage det7 g_det7_long --jobs "$JOBS" --calibrate --seed-bundle "$DET3_BUNDLE"

echo "[$(date +%H:%M:%S)] fleet done"
.venv/bin/python mx_june_wft/digest.py sat_det3 o22_long_det2 g_det4 g_det6_long g_det7_long \
    --out mx_june_wft/FLEET_DIGEST.md || true
