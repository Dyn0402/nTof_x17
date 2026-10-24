#!/bin/bash
# run_fit.sh <code_dir> <det> <run>/<subrun>/<mx17_N> <cm0|cm5> <tag> <pinned|profiled> <free,list> [K]
#   W11: RC-kernel fit (recal.py, cold objective, seed rerun/precheck_rcv/seed_<det>.json, source bundle = the r06
#   production bundle) on one W11 cache, then a bundle from the arm and the held-out evaluation of it.
#   Outputs -> /eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/<rel>/wft/w11/<tag>_<cm>/
set -eu
CODE=$1; DET=$2; REL=$3; CM=$4; TAG=$5; MODE=$6; FREE=$7; K=${8:-}
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
A=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/$REL/wft/w11
O=$A/${TAG}_$CM; mkdir -p $O
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/w11_$$; mkdir -p $W; cd $W
cp $A/cache_$CM.pkl cache.pkl
cp /eos/user/d/dneff/plane_ratio/inputs/$DET/bundle.tgz . && mkdir -p src && tar xzf bundle.tgz -C src
B=$(ls -d $W/src/*/)
X=""; [ "$MODE" = profiled ] && X="--p0-profile"; [ -n "$K" ] && X="$X --depth-bins $K"
NJ=${OMP_NUM_THREADS:-8}
echo "[$(date -u +%T)] $(hostname) $DET $REL $CM $TAG $MODE free=$FREE K=$K"
$PY $CODE/sps_beam_test_26/analysis/plane_ratio/recal.py --cache cache.pkl --bundle $B --arm $TAG --free $FREE \
    --seed-json $CODE/rerun/precheck_rcv/seed_$DET.json --objective cold $X --out arm.json --jobs $NJ --maxiter 800
$PY $CODE/sps_beam_test_26/analysis/plane_ratio/make_arm_bundle.py --src $B --arm-json arm.json --cache cache.pkl \
    --out bundle_arm
KB=""; [ -n "$K" ] && KB="--depth-bins $K"
$PY $CODE/rerun/w11/heldout_eval.py --cache cache.pkl --bundle bundle_arm $KB --out eval.json --jobs $NJ
cp arm.json eval.json $O/ && cp -r bundle_arm $O/
echo "[$(date -u +%T)] done -> $O"
cd /; rm -rf $W
