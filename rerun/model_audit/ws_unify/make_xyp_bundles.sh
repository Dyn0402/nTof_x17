#!/bin/bash
# make_xyp_bundles.sh <gate_tree> <gate_full> -- the fleet gate's unified arm runs on each golden bundle with dt_xy and
# xy_pairing measured from the production gate table (set_w0kw.py --pairing-only; w0/kw untouched), which is what the
# finalize step (set_w0kw.py --table) gives every bundle of the pass.  Writes <row>/wft/calib_bundle_xyp beside the
# gate's calib_bundle.
set -e
TREE=$1; FULL=$2
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
declare -A ROW=( [det2]=mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2 [det3]=mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3
                 [det4]=mx17_det4_day_6-24-26/long_run/mx17_4 [det6]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_6
                 [det7]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_7 )
for d in det2 det3 det4 det6 det7; do
  W=$TREE/${ROW[$d]}/wft
  echo "== $d"
  $PY "$REPO/rerun/w12/set_w0kw.py" "$d" --bundle "$W/calib_bundle" --table "$FULL/$d/prod/events.parquet" \
      --pairing-only --out "$W/calib_bundle_xyp" | grep -E "dt_xy|rsig|wrote"
done
