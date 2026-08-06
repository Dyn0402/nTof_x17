#!/usr/bin/env bash
# det3 golden-run chain v3: RC-ladder kernel (calib_bundle_lp2) + prod config.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=/tmp/wft_logs; mkdir -p "$LOG"
D=/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft

export WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1 WFT_CHI2DOF_BAD=250
$PY -m wft.cli reco sat_det3 --jobs 8 --matched-only --bundle "$D/calib_bundle_lp2" \
    > "$LOG/det3_reco_v3.log" 2>&1 || { echo "RECO FAILED"; exit 1; }
echo "[$(date +%H:%M:%S)] reco v3 done"

run () { echo "[$(date +%H:%M:%S)] $1"; shift; "$@" || echo "STAGE FAILED: $*"; }
run alignment  $PY mx_june_wft/01_alignment.py sat_det3        > "$LOG/det3_v3_align.log" 2>&1
run eff-nocut  $PY mx_june_wft/02_efficiency.py sat_det3 --max-dropped -1 > "$LOG/det3_v3_eff.log" 2>&1
run eff-cut    $PY mx_june_wft/02_efficiency.py sat_det3       >> "$LOG/det3_v3_eff.log" 2>&1
run eff-hits   $PY mx_june_wft/02_efficiency.py sat_det3 --source hits >> "$LOG/det3_v3_eff.log" 2>&1
run angles     $PY mx_june_wft/03_angles.py sat_det3           > "$LOG/det3_v3_angles.log" 2>&1
run maps       $PY mx_june_wft/04_maps.py sat_det3             > "$LOG/det3_v3_maps.log" 2>&1
run w0check    $PY mx_june_wft/bench/set_w0.py sat_det3        > "$LOG/det3_v3_w0.log" 2>&1
echo "[$(date +%H:%M:%S)] det3 v3 chain complete"
