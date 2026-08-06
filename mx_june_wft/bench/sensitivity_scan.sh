#!/usr/bin/env bash
# sensitivity_scan.sh -- which calibration constants actually have to be
# re-measured in situ at n_TOF?
#
# One-at-a-time +-25 % perturbation of every model hyper of the det3 production
# bundle (calib_bundle_lp2), scored against M3 on the same 800 events. A
# constant whose +-25 % perturbation moves sigma_theta / bias / within-5mm by
# less than the statistical error does not need an in-situ measurement -- the
# bench value can be carried to run_79.
#
# NOTE ON v_drift: v is NOT a hyper and does not enter the fit at all. It only
# converts the fitted transverse speed w into an angle
# (tan = (w*1e3 - w0) / (kw * v)), so a v error is a pure multiplicative angle
# scale that can be corrected after the fact without re-reconstructing. It is
# checked separately in transfer_ablation.sh.
#
#   bash mx_june_wft/bench/sensitivity_scan.sh [JOBS] [SUBSET]
set -u
JOBS=${1:-3}
SUBSET=${2:-800}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY="$REPO/.venv/bin/python"
BUNDLE=/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/calib_bundle_lp2

run () {  # run <tag> <json patch>
  echo "=== $1  $2 ==="
  nice -n 15 "$PY" "$REPO/mx_june_wft/bench/run_bench.py" sat_det3 \
      --bundle "$BUNDLE" --variant prod --subset "$SUBSET" --jobs "$JOBS" \
      --tag "$1" --patch "$2" 2>&1 | grep -v '^  [0-9]'
}

# base (calib_bundle_lp2): c1 0.050869  c2 0.057975  kY 2.875398  tau_s 145.524
#                          sigma_s 12.0714  sigma_p0 0.408716  Dp 0.0134157
#                          kTauY 1.78
run sens_base    '{}'
run sens_c1_m25  '{"c1": 0.038152}'
run sens_c1_p25  '{"c1": 0.063586}'
run sens_c2_m25  '{"c2": 0.043481}'
run sens_c2_p25  '{"c2": 0.072469}'
run sens_kY_m25  '{"kY": 2.156549}'
run sens_kY_p25  '{"kY": 3.594248}'
run sens_tau_m25 '{"tau_s": 109.1427}'
run sens_tau_p25 '{"tau_s": 181.9045}'
run sens_ss_m25  '{"sigma_s": 9.053539}'
run sens_ss_p25  '{"sigma_s": 15.089231}'
run sens_sp0_m25 '{"sigma_p0": 0.306537}'
run sens_sp0_p25 '{"sigma_p0": 0.510895}'
run sens_Dp_m25  '{"Dp": 0.010062}'
run sens_Dp_p25  '{"Dp": 0.016770}'
run sens_kty_m25 '{"kTauY": 1.335}'
run sens_kty_p25 '{"kTauY": 2.225}'
echo "ALL DONE"
