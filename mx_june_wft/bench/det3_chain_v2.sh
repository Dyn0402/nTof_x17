#!/usr/bin/env bash
# det3 golden-run analysis chain on the v2 (mf3 + w0/kw + prescan) reco.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=/tmp/wft_logs

while ! grep -q "wrote .*events.parquet" "$LOG/det3_reco_v2.log" 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] reco done, running chain"

run () { echo "[$(date +%H:%M:%S)] $1"; shift; "$@" || echo "STAGE FAILED: $*"; }

run alignment  $PY mx_june_wft/01_alignment.py sat_det3        > "$LOG/det3_v2_align.log" 2>&1
run eff-nocut  $PY mx_june_wft/02_efficiency.py sat_det3 --max-dropped -1 > "$LOG/det3_v2_eff.log" 2>&1
run eff-cut    $PY mx_june_wft/02_efficiency.py sat_det3       >> "$LOG/det3_v2_eff.log" 2>&1
run eff-hits   $PY mx_june_wft/02_efficiency.py sat_det3 --source hits >> "$LOG/det3_v2_eff.log" 2>&1
run angles     $PY mx_june_wft/03_angles.py sat_det3           > "$LOG/det3_v2_angles.log" 2>&1
run maps       $PY mx_june_wft/04_maps.py sat_det3             > "$LOG/det3_v2_maps.log" 2>&1
run digest     $PY mx_june_wft/digest.py sat_det3              > "$LOG/det3_v2_digest.log" 2>&1
echo "[$(date +%H:%M:%S)] det3 v2 chain complete"
