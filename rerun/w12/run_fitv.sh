#!/bin/bash
# run_fitv.sh <code_dir> <rel> <src bundle (absolute EOS path)> <K> <out subdir under <rel>/wft>
#   S1b v refit (RERUN_PLAN W7): fit v only (recal.py, cold objective, reference pinned to THIS row's waveform
#   alignment, kernel pinned = the chamber's golden bundle), on <rel>/wft/vrefit/cache.pkl; bundle + held-out eval.
set -eu
CODE=$1; REL=$2; SRC=$3; K=$4; SUBD=$5
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
export WFT_COL_GUARD=${WFT_COL_GUARD:-0.01}
A=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/$REL/wft
O=$A/$SUBD; mkdir -p $O
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/fv_$$; mkdir -p $W; cd $W
cp $A/vrefit/cache.pkl cache.pkl
cp -r $SRC src_bundle
$PY - <<P
import json; b=json.load(open('src_bundle/bundle.json')); b['n_depth_bins']=$K; json.dump(b,open('src_bundle/bundle.json','w'),indent=1)
json.dump(dict(arm='golden',hyper=b['hyper']),open('seed.json','w'))
P
NJ=${OMP_NUM_THREADS:-8}
$PY $CODE/sps_beam_test_26/analysis/plane_ratio/recal.py --cache cache.pkl --bundle src_bundle --arm vrefit --free v \
    --seed-json seed.json --objective cold --depth-bins $K --out arm.json --jobs $NJ --maxiter 200
$PY $CODE/sps_beam_test_26/analysis/plane_ratio/make_arm_bundle.py --src src_bundle --arm-json arm.json --cache cache.pkl --out bundle
$PY $CODE/rerun/w11/heldout_eval.py --cache cache.pkl --bundle bundle --depth-bins $K --out eval.json --jobs $NJ
cp arm.json eval.json $O/ && rm -rf $O/bundle && cp -r bundle $O/
echo "[$(date -u +%T)] done -> $O"
cd /; rm -rf $W
