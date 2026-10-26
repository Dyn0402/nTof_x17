#!/bin/bash
# run_gate.sh <code_dir> <det> <arm>: one mask-gate arm (resid_map on held-out [180, 1500)) on condor -> EOS
set -eu
CODE=$1; DET=$2; ARM=$3
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
G=/eos/user/d/dneff/x17rerun/maskgate
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/mg_$$; mkdir -p $W; cd $W
cp $G/in/$DET/cache.pkl . && cp -r $G/in/$DET/$ARM bundle
N=$($PY -c "import pickle;print(min(1320,len(pickle.load(open('cache.pkl','rb')))-180))")
WFT_REPO=$CODE $PY $CODE/rerun/det4/resid_map.py --cache cache.pkl --bundle bundle --start 180 --n $N \
    --jobs ${OMP_NUM_THREADS:-8} --out $G/out/rmF_G${DET}_${ARM#bundle_}.npz
cd /; rm -rf $W
