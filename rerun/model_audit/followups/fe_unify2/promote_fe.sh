#!/bin/bash
# promote_fe.sh [--apply] -- PREPARED, NOT RUN (FE 2026-10-27).  On lxplus: put the FE-finalised golden bundles
# (Analysis_fe/<row>/wft/calib_bundle_fe: t0 pinned at the latency, one sigma0 0.30, one-rule dead list, FB bow block,
# w0/kw after the bow, tan_floor, tc_tol, lq per class) in place of the production calib_bundle of each golden row.
# The current production bundle is kept beside it as calib_bundle_pre_fe.  Dry run (prints) unless --apply.
# After promotion: python rerun/promote_phys.py --fill-conditions --apply (det7's empty run/sub_run), then the pass's
# vrefit --keep-v / finalize (set_w0kw.py --table) for every non-golden row reading these bundles.
set -e
FE=/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe
A=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1
RUN=echo; [ "$1" = "--apply" ] && RUN=
for r in mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2 mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3 \
         mx17_det4_day_6-24-26/long_run/mx17_4 mx17_det6_det7_overnight_6-26-26/long_run/mx17_6 \
         mx17_det6_det7_overnight_6-26-26/long_run/mx17_7; do
  $RUN mv "$A/$r/wft/calib_bundle" "$A/$r/wft/calib_bundle_pre_fe"
  $RUN cp -r "$FE/$r/wft/calib_bundle_fe" "$A/$r/wft/calib_bundle"
  $RUN cp "$A/$r/wft/calib_bundle_pre_fe/PROMOTED.json" "$A/$r/wft/calib_bundle/PROMOTED_pre_fe.json"
done
# n_TOF: dead lists under the one rule (no t0 pin, no bow: wft_beam refuses both).  Per arm and epoch, e.g. run_145 A:
#   python rerun/w12/set_w0kw.py A --bundle <is2 calib_bundle_prelim of arm A> --keep-w0kw \
#       --dead-json rerun/model_audit/followups/fe_unify2/masks/ntof_A_run_145.json --out <...>/calib_bundle_prelim_dead
