#!/bin/bash
# build_latency_bundles.sh -- each golden row's promoted bundle + t0_abs from the model-free trigger latency, pinned
# (set_w0kw.py --latency-cache --pin-t0; w0/kw re-measured from the same finalize angles, i.e. unchanged).
# Output: $M/ws_t0/bundles/<det>  (the held-out --t0-fix arm reads t0_abs from there).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
M=${M:-/home/dylan/x17/cosmic_bench/model_audit}
PY=${PY:-/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python}
for det in det2 det3 det4 det6 det7; do
    "$PY" "$REPO/rerun/w12/set_w0kw.py" "$det" --bundle "$M/$det/calib_bundle" \
        --angles "$M/$det/final_physmask/angles" --out "$M/ws_t0/bundles/$det" \
        --latency-cache "$M/ws_t0/caches/$det.pkl" --pin-t0 | grep -E 'latency|t0_abs'
done
