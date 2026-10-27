#!/bin/bash
# run_scan.sh <det> <C lo:hi:step> <n> [tag]  — t0_scan.py on one golden row's held-out events, locally (4 workers).
# Inputs: the w12/it1 calibration caches (copied from Analysis_rr1 on EOS) and the promoted bundles, under $M.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
M=${M:-/home/dylan/x17/cosmic_bench/model_audit}
PY=${PY:-/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python}
det=$1; C=$2; n=$3; tag=${4:-scan}
export WFT_RC_T0_INTERP=${WFT_RC_T0_INTERP:-1}
"$PY" "$HERE/t0_scan.py" --cache "$M/ws_t0/caches/$det.pkl" --bundle "$M/$det/calib_bundle" \
    --out "$M/ws_t0/${det}_${tag}.npz" --C "$C" --n "$n" --jobs "${JOBS:-4}"
