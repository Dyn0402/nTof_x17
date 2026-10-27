#!/bin/bash
# upload_fe.sh <arm> [<arm> ...] -- put the local FE bundles (make_fe_bundles.sh / finalize_fe.sh) into the FE gate tree
# on EOS beside the production alignment of gate3.  Run on the desktop/laptop; goes through the literal `lxplus` alias.
set -e
M=${M:-$HOME/x17/cosmic_bench/model_audit}
SRC=${SRC:-$M/fe_unify2/bundles}
ROOT=${ROOT:-/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe}
AUDIT=/eos/user/d/dneff/x17rerun/Analysis_audit
declare -A ROW=( [det2]=mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2 [det3]=mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3
                 [det4]=mx17_det4_day_6-24-26/long_run/mx17_4 [det6]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_6
                 [det7]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_7 )
TMP=$(mktemp -d)
for d in det2 det3 det4 det6 det7; do
  for arm in "$@"; do
    mkdir -p "$TMP/${ROW[$d]}/wft"
    cp -r "$SRC/$d/calib_bundle_$arm" "$TMP/${ROW[$d]}/wft/"
  done
done
tar -C "$TMP" -czf "$TMP.tgz" .
scp -q "$TMP.tgz" lxplus:/tmp/fe_bundles_$$.tgz
ssh lxplus "set -e; mkdir -p $ROOT; tar -C $ROOT -xzf /tmp/fe_bundles_$$.tgz; rm /tmp/fe_bundles_$$.tgz
  for r in ${ROW[det2]} ${ROW[det3]} ${ROW[det4]} ${ROW[det6]} ${ROW[det7]}; do
    if [ ! -d $ROOT/\$r/wft/prod_alignment ]; then mkdir -p $ROOT/\$r/wft; cp -r $AUDIT/\$r/wft/prod_alignment $ROOT/\$r/wft/; fi
  done; ls $ROOT/${ROW[det3]}/wft"
rm -rf "$TMP" "$TMP.tgz"
