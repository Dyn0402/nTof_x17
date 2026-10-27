#!/bin/bash
# pull_fe.sh <arm> [<arm> ...] -- copy the FE gate outputs (budget, angles, tables) from EOS into the local mirror
#   $L/gate_tree/<row>/wft/<arm>/...   (gate_compare.py --root; the gate3 'prod'/'unified' arms are linked in once)
set -e
M=${M:-$HOME/x17/cosmic_bench/model_audit}
L=${L:-$M/fe_unify2}
ROOT=${ROOT:-/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe}
ROWS="mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2 mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3 mx17_det4_day_6-24-26/long_run/mx17_4 mx17_det6_det7_overnight_6-26-26/long_run/mx17_6 mx17_det6_det7_overnight_6-26-26/long_run/mx17_7"
for r in $ROWS; do
  mkdir -p "$L/gate_tree/$r/wft"
  for base in prod unified; do
    [ -e "$L/gate_tree/$r/wft/$base" ] || ln -s "$M/gate_tree/$r/wft/$base" "$L/gate_tree/$r/wft/$base"
  done
done
ssh lxplus "cd $ROOT && tar -czf /tmp/fe_pull_\$USER.tgz \$(for r in $ROWS; do for a in $*; do ls -d \$r/wft/\$a 2>/dev/null; ls -d \$r/wft/calib_bundle_\$a 2>/dev/null; done; done)"
scp -q lxplus:/tmp/fe_pull_dneff.tgz /tmp/fe_pull.tgz
tar -C "$L/gate_tree" -xzf /tmp/fe_pull.tgz
rm /tmp/fe_pull.tgz
du -sh "$L/gate_tree"
