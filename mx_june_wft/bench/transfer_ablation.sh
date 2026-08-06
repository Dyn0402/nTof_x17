#!/usr/bin/env bash
# transfer_ablation.sh -- does a calibration bundle survive being used on data
# it was not fitted on?
#
# WHY THIS IS THE QUESTION FOR n_TOF. At n_TOF there is no M3 reference, so the
# ref-pinned hyper fit cannot be redone there; the plan (TRACK_PLAN_08) is to
# FREEZE the bench kernel + template and re-measure only the gas-dependent
# constants in situ. This measures what that freezing costs, on bench data
# where M3 truth exists.
#
# A note that shrinks the problem: v_drift does NOT enter the fit. It appears
# only in tan = (w*1e3 - w0) / (kw * v), i.e. as a multiplicative angle scale
# applied after the fact (plus a weak effect through the fit's seed). So
# "transfer the bundle and re-measure v" is really "transfer the KERNEL and the
# TEMPLATE", and a v error is correctable without re-reconstructing. The
# vscale runs at the end check that claim instead of assuming it.
#
#   bash mx_june_wft/bench/transfer_ablation.sh [JOBS] [SUBSET]
set -u
JOBS=${1:-3}
SUBSET=${2:-800}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY="$REPO/.venv/bin/python"
MK="$REPO/mx_june_wft/bench/make_bundle_variant.py"
A=/home/dylan/x17/cosmic_bench/Analysis
B_SAT3=$A/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/calib_bundle_lp2
B_O22_3=$A/mx17_det2_det3_overnight_6-22-26/long_run/mx17_3/wft/calib_bundle_lp
B_O22_2=$A/mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2/wft/calib_bundle_lp
DERIV=$REPO/mx_june_wft/bench/derived_bundles
mkdir -p "$DERIV"

# derived bundles: kernel+template from one run, angle constants from the other
"$PY" "$MK" --src "$B_SAT3"  --out "$DERIV/sat3_kernel_o22_w0" \
    --w0kw-from "$B_O22_3" --note 'transfer ablation: det3 6-27 kernel, 6-22 w0/kw'
"$PY" "$MK" --src "$B_O22_3" --out "$DERIV/o22_kernel_sat3_w0" \
    --w0kw-from "$B_SAT3"  --note 'transfer ablation: det3 6-22 kernel, 6-27 w0/kw'
"$PY" "$MK" --src "$B_SAT3"  --out "$DERIV/sat3_kernel_det2_w0" \
    --w0kw-from "$B_O22_2" --note 'transfer ablation: det3 kernel on det2, det2 w0/kw'
"$PY" "$MK" --src "$B_SAT3"  --out "$DERIV/sat3_v_p10" --v 40.26 \
    --note 'v +10 %: is v only an angle scale?'
"$PY" "$MK" --src "$B_SAT3"  --out "$DERIV/sat3_v_m10" --v 32.94 \
    --note 'v -10 %'

run () {  # run <run_key> <tag> <bundle>
  echo "=== $1 <- $2 ==="
  nice -n 15 "$PY" "$REPO/mx_june_wft/bench/run_bench.py" "$1" \
      --bundle "$3" --variant prod --subset "$SUBSET" --jobs "$JOBS" \
      --tag "$2" 2>&1 | grep -v '^  [0-9]'
}

# --- same chamber (det3), different run 5 days apart -----------------------
# 6-22 long_run vs 6-27 saturday scan: same detector, different slot (z 232 vs
# 702), and the gas dried over that week (H2O ~3 % -> ~1 %), so this is the
# closest bench analogue of "same chamber, different gas conditions".
run g_det3   xfer_own          "$B_O22_3"
run g_det3   xfer_sat3kernel   "$DERIV/sat3_kernel_o22_w0"
run g_det3   xfer_sat3full     "$B_SAT3"
run sat_det3 xfer_own          "$B_SAT3"
run sat_det3 xfer_o22kernel    "$DERIV/o22_kernel_sat3_w0"

# --- different chamber: the hard case, an upper bound on the damage --------
run o22_long_det2 xfer_own        "$B_O22_2"
run o22_long_det2 xfer_det3kernel "$DERIV/sat3_kernel_det2_w0"

# --- is v really just an angle scale? --------------------------------------
run sat_det3 vscale_p10 "$DERIV/sat3_v_p10"
run sat_det3 vscale_m10 "$DERIV/sat3_v_m10"
echo "ALL DONE"
