#!/bin/bash
# finalize_fe.sh <src arm> <dst arm> [nobow] -- round 2 of the FE re-finalisation (set_w0kw.py, the pass's finalize step):
# from a reconstruction made WITH calib_bundle_<src> (gate arm <src>): FC's one sigma0 = 0.30 mm for both views, FB's
# static bow block (unless 'nobow'), then w0/kw (after the bow shape) and tan_floor from 03_angles' per-event table, the
# column-time tolerance tc_tol and the charge balance per event class from the table.  The t0 pin and the dead list come
# with <src>.  Reads the local mirror of pull_fe.sh; writes $OUT/<det>/calib_bundle_<dst>.
set -e
SRC=$1; DST=$2; MODE=${3:-bow}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
M=${M:-$HOME/x17/cosmic_bench/model_audit}
T=${T:-$M/fe_unify2/gate_tree}
OUT=${OUT:-$M/fe_unify2/bundles}
PY=${PY:-/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python}
SIGMA0=${SIGMA0:-0.30,0.30}
declare -A ROW=( [det2]=mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2 [det3]=mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3
                 [det4]=mx17_det4_day_6-24-26/long_run/mx17_4 [det6]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_6
                 [det7]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_7 )
for d in det2 det3 det4 det6 det7; do
  W=$T/${ROW[$d]}/wft
  BOW=""; [ "$MODE" = "bow" ] && BOW="--bow-json $REPO/rerun/model_audit/followups/fb_bow_maps/results/bow_block_$d.json"
  echo "== $d"
  "$PY" "$REPO/rerun/w12/set_w0kw.py" "$d" --bundle "$W/calib_bundle_$SRC" --angles "$W/$SRC/angles" \
      --table "$W/$SRC/events.parquet" --sigma0 "$SIGMA0" $BOW --out "$OUT/$d/calib_bundle_$DST"
done
