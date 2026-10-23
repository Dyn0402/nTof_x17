#!/bin/bash
# run_fit2.sh <code_dir> <det> <rel> <iterdir name> <seed arm.json path, relative to <rel>/wft> <pinned|profiled> [K]
#   W12 iteration fit: recal.py (cold objective, v free with sigma_p0_x, sigma_p0_y, rc_D_y) on <rel>/wft/w12/<it>/cache.pkl,
#   seeded from a previous arm; then a bundle from it (the NEXT iteration's bundle) and the held-out evaluation.
#   Source bundle (templates, gains, masks) = the r06 production bundle, as in W11.
set -eu
CODE=$1; DET=$2; REL=$3; IT=$4; SEED=$5; MODE=$6; K=${7:-24}
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
export WFT_COL_GUARD=${WFT_COL_GUARD:-0.01}
A=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/$REL/wft
O=$A/w12/$IT/fit_$MODE; mkdir -p $O
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/w12_$$}; mkdir -p $W; cd $W
cp $A/w12/$IT/cache.pkl cache.pkl
cp /eos/user/d/dneff/plane_ratio/inputs/$DET/bundle.tgz . && mkdir -p src && tar xzf bundle.tgz -C src
B=$(ls -d $W/src/*/)
$PY -c "import json;a=json.load(open('$A/$SEED'));json.dump(dict(arm='seed',hyper=a['hyper']),open('seed.json','w'))"
V0=$($PY -c "import json;print(json.load(open('$A/$SEED'))['v'])")
X=""; [ "$MODE" = profiled ] && X="--p0-profile"
NJ=${OMP_NUM_THREADS:-8}
echo "[$(date -u +%T)] $(hostname) $DET $IT $MODE K=$K seed=$SEED v0=$V0"
# the source bundle's v is the starting v of the fit: put the seed arm's v in it
$PY - <<P
import json,glob; f=glob.glob('$B/bundle.json')[0]; b=json.load(open(f)); b['v_drift']=float($V0); b['n_depth_bins']=$K
json.dump(b,open(f,'w'),indent=1)
P
$PY $CODE/sps_beam_test_26/analysis/plane_ratio/recal.py --cache cache.pkl --bundle $B --arm w12 \
    --free sigma_p0_x,sigma_p0_y,rc_D_y,v --seed-json seed.json --objective cold $X --depth-bins $K \
    --out arm.json --jobs $NJ --maxiter 800
$PY $CODE/sps_beam_test_26/analysis/plane_ratio/make_arm_bundle.py --src $B --arm-json arm.json --cache cache.pkl --out bundle
$PY $CODE/rerun/w11/heldout_eval.py --cache cache.pkl --bundle bundle --depth-bins $K --out eval.json --jobs $NJ
cp arm.json eval.json $O/ && rm -rf $O/bundle && cp -r bundle $O/
echo "[$(date -u +%T)] done -> $O"
cd /; rm -rf $W
