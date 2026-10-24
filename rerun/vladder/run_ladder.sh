#!/bin/bash
# run_ladder.sh <code_dir> <det> <rel> <mode> [bundle: timeD (default) | r06]
#   ladder_fit.py on <rel>/wft/w12/it0/cache.pkl with the it0 arm-D bundle or the r06 production bundle -> EOS
set -eu
CODE=$1; DET=$2; REL=$3; MODE=$4; BSEL=${5:-timeD}
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
A=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/$REL/wft/w12/it0
O=/eos/user/d/dneff/vladder; mkdir -p $O
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/vl_$$; mkdir -p $W; cd $W
cp $A/cache.pkl cache.pkl
if [ "$BSEL" = r06 ]; then
  cp /eos/user/d/dneff/plane_ratio/inputs/$DET/bundle.tgz . && mkdir -p src && tar xzf bundle.tgz -C src
  B=$(ls -d $W/src/*/); TAG=${DET}_r06_${MODE}
else
  cp -r $A/timeD/bundle bundle; B=bundle; TAG=${DET}_${MODE}
fi
WFT_REPO=$CODE $PY $HOME/x17rerun/vladder/ladder_fit.py --cache cache.pkl --bundle $B --mode $MODE --out $O/$TAG.json \
    --jobs ${OMP_NUM_THREADS:-8} --n 400
cd /; rm -rf $W
