#!/bin/bash
# run_sf.sh <code_dir> <det> <rel> <start> <n> <fitters> [bundle sub (timeD)] : run_fitters.py on <rel>/wft/w12/it0 -> EOS
set -eu
CODE=$1; DET=$2; REL=$3; START=$4; N=$5; FIT=$6; BSUB=${7:-timeD}
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
A=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/$REL/wft/w12/it0
O=/eos/user/d/dneff/seedfit; mkdir -p $O
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/sf_$$; mkdir -p $W; cd $W
cp $A/cache.pkl cache.pkl; cp -r $A/$BSUB/bundle bundle
WFT_REPO=$CODE $PY $HOME/x17rerun/seedfit/run_fitters.py --cache cache.pkl --bundle bundle --fitters ${FIT//./,} \
    --start $START --n $N --out $O/${DET}_${BSUB}_${START}.json --jobs ${OMP_NUM_THREADS:-8}
cd /; rm -rf $W
