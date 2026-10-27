#!/bin/bash
# finalize_fe.sh <src arm> <dst arm> -- round 2 of the FE re-finalisation (set_w0kw.py --table, the pass's finalize step):
# from a reconstruction made WITH calib_bundle_<src> (the gate arm <src>), measure w0/kw and tan_floor (03_angles per
# event), the column-time tolerance tc_tol and the charge balance per event class (the table), and write
# calib_bundle_<dst>.  Reads the local mirror of pull_fe.sh; writes $OUT/<det>/calib_bundle_<dst>.
set -e
SRC=$1; DST=$2
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
M=${M:-$HOME/x17/cosmic_bench/model_audit}
T=${T:-$M/fe_unify2/gate_tree}
OUT=${OUT:-$M/fe_unify2/bundles}
PY=${PY:-/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python}
declare -A ROW=( [det2]=mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2 [det3]=mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3
                 [det4]=mx17_det4_day_6-24-26/long_run/mx17_4 [det6]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_6
                 [det7]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_7 )
for d in det2 det3 det4 det6 det7; do
  W=$T/${ROW[$d]}/wft
  echo "== $d"
  "$PY" "$REPO/rerun/w12/set_w0kw.py" "$d" --bundle "$W/calib_bundle_$SRC" --angles "$W/$SRC/angles" \
      --table "$W/$SRC/events.parquet" --out "$OUT/$d/calib_bundle_$DST"
done
