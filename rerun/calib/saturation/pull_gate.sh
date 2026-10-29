#!/bin/bash
# pull_gate.sh <local root> <arm> [det ...] -- copy the gate products of one arm from Analysis_fe on EOS
L=$1; ARM=$2; shift 2
R=/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe
declare -A ROW=([det2]=mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2
  [det3]=mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3
  [det4]=mx17_det4_day_6-24-26/long_run/mx17_4 [det6]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_6
  [det7]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_7)
for d in "$@"; do
  r=${ROW[$d]}; D="$L/$r/wft/$ARM"; mkdir -p "$D/budget" "$D/angles"
  scp -q "lxplus:$R/$r/wft/$ARM/events.parquet" "lxplus:$R/$r/wft/$ARM/events.meta.json" "lxplus:$R/$r/wft/$ARM/events.candidates.parquet" "$D/" || echo "missing events $d $ARM"
  scp -q "lxplus:$R/$r/wft/$ARM/budget/budget_$d.json" "lxplus:$R/$r/wft/$ARM/budget/rays_$d.parquet" "$D/budget/" || echo "missing budget $d $ARM"
  scp -q "lxplus:$R/$r/wft/$ARM/angles/angular_resolution.json" "$D/angles/" || echo "missing angles $d $ARM"
done
