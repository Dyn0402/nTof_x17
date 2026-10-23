#!/bin/bash
# judge_all.sh <rays-variants spec, e.g. "base=DIR" or "base=DIR,out5=DIR"> <tag for output> -- run judge.py on every golden row.
SPEC=$1; TAG=$2
B=/eos/experiment/ntof/data/x17/cosmic_bench/june_tests
O=/eos/user/d/dneff/x17rerun/m3_audit/judge_$TAG; mkdir -p $O
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
cd ~/x17rerun/judge
while read run sub tags dets; do
  for d in ${dets//,/ }; do
    $PY rerun/m3_audit/judge.py --run-config $B/$run/run_config.json --det $d --sub $B/$run/$sub --tags $tags --rays "$SPEC" --out $O/${d}__${run}__${sub}.json
  done
done <<L
mx17_det6_det7_overnight_6-26-26 long_run 260626_06H03_000,260626_06H03_005 mx17_6,mx17_7
mx17_det3_saturday_scan_6-27-26 long_run_resist_490V_drift_1000V 260627_22H49_000,260627_22H49_001,260627_22H49_002,260627_22H49_003 mx17_3
mx17_det3_p2_det1_overnight_6-27-26 long_run_p2_det1_sanity_check 260628_01H34_000,260628_01H34_004 mx17_3
mx17_det4_day_6-24-26 long_run 260624_12H22_000,260624_12H22_003 mx17_4
mx17_det2_det3_overnight_6-22-26 longer_run 260622_20H20_000,260622_20H20_001 mx17_2,mx17_3
L
