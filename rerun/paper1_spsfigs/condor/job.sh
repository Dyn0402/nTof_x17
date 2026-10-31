#!/bin/bash
# job.sh kernel|vbox <registry key>: one SPS figure input on the final cache builder, into
# /eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_spsfigs/<kind>/<key>/  (code: ~/x17rerun/paper1_spsfigs/code)
KIND=$1; K=$2
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
C=$HOME/x17rerun/paper1_spsfigs/code
export WFT_REPO=$C MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
W=${_CONDOR_SCRATCH_DIR:-/tmp/dneff_spsfigs_$K}; mkdir -p $W/out
O=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_spsfigs/$KIND/$K; mkdir -p $O
if [ "$KIND" = kernel ]; then
  $PY $C/rerun/paper1_spsfigs/kernel_job.py $K --out $W/out/kernel_$K.npz || exit 1
  cp $W/out/kernel_$K.npz $O/
else
  $PY $C/rerun/sps_followup/vbox.py --out-dir $W/out $K || exit 1
  cp $W/out/*.json $O/
fi
rm -rf $W/out
