#!/bin/bash
# run_gain.sh: gain_fit on det4 (det3 footprint + beam D_rc) and det3, then resid_map on the gained bundles
V=~/x17/cosmic_bench/vladder; O=~/x17/cosmic_bench/det4dig; H=$(dirname $0); PY=$H/../../.venv/bin/python
$PY $H/gain_fit.py --cache $V/det4/cache.pkl --bundle $V/det4/bundle_s3beam --out-bundle $V/det4/bundle_s3beam_g --iters 3
$H/run_resid.sh det4_s3beam_g $V/det4/cache.pkl $V/det4/bundle_s3beam_g
$PY $H/gain_fit.py --cache $V/det3/cache.pkl --bundle $V/det3/bundle --out-bundle $V/det3/bundle_g --iters 3
$H/run_resid.sh det3_rc_g $V/det3/cache.pkl $V/det3/bundle_g
