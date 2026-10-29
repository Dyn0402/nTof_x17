#!/bin/bash
# pull.sh -- copy the lx_sparks.py outputs (Analysis_fe/<row>/wft/gd_sparks) to ~/x17/cosmic_bench/model_audit/gd_sparks/<det>/
set -e
L=${L:-$HOME/x17/cosmic_bench/model_audit/gd_sparks}
ROOT=/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe
declare -A ROW=( [det2]=mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2 [det3]=mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3
                 [det4]=mx17_det4_day_6-24-26/long_run/mx17_4 [det6]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_6
                 [det7]=mx17_det6_det7_overnight_6-26-26/long_run/mx17_7 )
for d in ${@:-det2 det3 det4 det6 det7}; do
  mkdir -p $L/$d
  ssh lxplus "aklog; test -d $ROOT/${ROW[$d]}/wft/gd_sparks && tar -C $ROOT/${ROW[$d]}/wft/gd_sparks -czf /tmp/gd_$d.tgz ." && \
    scp -q lxplus:/tmp/gd_$d.tgz /tmp/gd_$d.tgz && tar -C $L/$d -xzf /tmp/gd_$d.tgz && rm /tmp/gd_$d.tgz && echo "pulled $d" || echo "no $d yet"
done
du -sh $L
