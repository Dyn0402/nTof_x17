#!/usr/bin/env bash
# Submit files for the real-EAR2-gun Geant4 runs (capsule_pose, 2026-10-14):
# per finished ROOT file, the DriftGas steps (A, C) and the gap/wall reduction,
# into full_sim/angle_scale/{steps,neutrons}_<tag>/{s,r}NNN.parquet.
# Same scripts and naming as steps_nose / neutrons_nose, so run_digi and
# capsule_pose.library read them unchanged.  Run on lxplus:
#   bash make_ear2_subs.sh neutrons_thermal_ear2_L19p5 ear2_L19p5 && condor_submit steps_ear2_L19p5.sub && condor_submit reduce_ear2_L19p5.sub
set -euo pipefail
SIM=/eos/experiment/ntof/data/x17/full_sim/$1
TAG=$2
OUT=/eos/experiment/ntof/data/x17/full_sim/angle_scale
mkdir -p "$OUT/steps_$TAG" "$OUT/neutrons_$TAG" ~/condor/mx17_digi_steps/logs_$TAG ~/condor/mx17_angle_scale/logs_$TAG
for kind in steps reduce; do
  if [ $kind = steps ]; then
    D=~/condor/mx17_digi_steps; EXE=$D/run_steps.sh; P=s; ODIR=$OUT/steps_$TAG; FLAV=longlunch; MEM=8000
  else
    D=~/condor/mx17_angle_scale; EXE=$D/run_reduce.sh; P=r; ODIR=$OUT/neutrons_$TAG; FLAV=workday; MEM=16000   # the EAR2-gun files peak at ~12 GB
  fi
  {
    echo "executable = $EXE"
    echo 'arguments = $(inp) $(outp)'
    echo "output = $D/logs_$TAG/\$(tag).out"
    echo "error = $D/logs_$TAG/\$(tag).err"
    echo "log = $D/logs_$TAG/$kind.log"
    echo "+JobFlavour = \"$FLAV\""
    echo 'request_cpus = 1'
    echo "request_memory = $MEM"
    echo 'requirements = (OpSysAndVer =?= "AlmaLinux9")'
    echo 'queue inp,outp,tag from ('
    for f in "$SIM"/neutron_bg_job*_t0.root; do
      n=$(basename "$f" _t0.root); n=${n#neutron_bg_job}
      echo "$f, $ODIR/$P$n.parquet, $P$n"
    done
    echo ')'
  } > "${kind}_$TAG.sub"
  echo "wrote ${kind}_$TAG.sub ($(grep -c _t0.root "${kind}_$TAG.sub") files)"
done
