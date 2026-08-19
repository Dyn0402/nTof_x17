#!/usr/bin/env bash
# 21_r06_gate.sh -- score the c2-slaved (r06) calibration against the frozen
# one on a golden key, the same way the t0-prior gate was scored: same code,
# same accounting, two arms, nothing overwritten.
#
# The r06 bundle ships with STALE w0/kw (they were measured from a reco under
# the old kernel), so the arm is built in three steps:
#   1. reco with r06                        -> events_r06.parquet
#   2. bench/set_w0.py --write              -> w0/kw measured from THAT reco
#   3. bench/apply_w0.py --write            -> re-apply them post-hoc
# Step 3 needs no second reco: w0/kw only map the fitted w to an angle.
#
#   bash mx_june_wft/21_r06_gate.sh sat_det3 <path-to-wft-dir>
set -euo pipefail
KEY="$1"; W="$2"
PY="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== $KEY: w0/kw from the r06 reco"
"$PY" "$HERE/bench/set_w0.py"   "$KEY" --bundle "$W/calib_bundle_r06" \
                                --events events_r06.parquet --write
echo "=== $KEY: re-apply w0/kw to the r06 table"
"$PY" "$HERE/bench/apply_w0.py" "$KEY" --bundle "$W/calib_bundle_r06" \
                                --events events_r06.parquet --write
echo "=== $KEY: alignment (r06 arm)"
"$PY" "$HERE/01_alignment.py" "$KEY" --table "$W/events_r06.parquet" \
                              --out "$W/alignment_r06"
echo "=== $KEY: angles (r06 arm)"
"$PY" "$HERE/03_angles.py" "$KEY" --table "$W/events_r06.parquet" \
                           --alignment "$W/alignment_r06/alignment.json" \
                           --out "$W/angles_r06"
echo "=== $KEY: done -- compare $W/angles/angular_resolution.json (frozen)"
echo "                    with $W/angles_r06/angular_resolution.json (r06)"
