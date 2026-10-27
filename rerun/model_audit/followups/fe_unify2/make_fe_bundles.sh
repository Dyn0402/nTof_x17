#!/bin/bash
# make_fe_bundles.sh -- round 1 of the FE re-finalisation: each golden row's promoted bundle edited to the decided
# content, w0/kw kept (they are re-measured in round 2 from a reconstruction made with these bundles):
#   fe1   fleet sigma0 (X 0.29, Y 0.33; FC default, per-event profile removed), the dead list under the one rule
#         (build_masks.py), t0 pinned at the measured trigger latency (wft.calibrate.measure_latency on the ws_t0 caches)
#   fe1g  fe1 + the 20 ns charge grid (depth_subdiv 3)
# Output: $OUT/<det>/calib_bundle_fe1{,g}
set -e
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
M=${M:-$HOME/x17/cosmic_bench/model_audit}
OUT=${OUT:-$M/fe_unify2/bundles}
PY=${PY:-/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python}
SIGMA0=${SIGMA0:-0.29,0.33}
for det in det2 det3 det4 det6 det7; do
  echo "== $det"
  "$PY" "$REPO/rerun/w12/set_w0kw.py" "$det" --bundle "$M/$det/calib_bundle" --keep-w0kw \
      --sigma0 "$SIGMA0" --dead-json "$HERE/masks/bench_$det.json" \
      --latency-cache "$M/ws_t0/caches/$det.pkl" --pin-t0 --out "$OUT/$det/calib_bundle_fe1"
  "$PY" "$REPO/rerun/w12/set_w0kw.py" "$det" --bundle "$OUT/$det/calib_bundle_fe1" --keep-w0kw \
      --depth-subdiv 3 --out "$OUT/$det/calib_bundle_fe1g"
done
