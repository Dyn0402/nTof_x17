#!/bin/bash
# run_vscan.sh <det> <cache> <arm tag> <depth bins or -> <vs>  -- v_scan.py on condor (rerun pre-check C8)
set -eu
DET=$1; CACHE=$2; ARM=$3; KB=$4; VS=$5
EOSD=/eos/user/d/dneff/plane_ratio
PY=/eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python
fetch() { for i in 1 2 3 4; do
    xrdcp -s -f "root://eosuser.cern.ch/$1" "$2" 2>/dev/null && return 0
    cp "$1" "$2" 2>/dev/null && return 0; sleep 20; done; return 1; }
tar xzf payload.tar.gz
fetch $EOSD/inputs/$DET/$CACHE.pkl cache.pkl
fetch $EOSD/inputs/$DET/bundle.tgz bundle.tgz && mkdir -p bundle && tar xzf bundle.tgz -C bundle
B=$(ls -d bundle/*/)
KARG=""; [ "$KB" != "-" ] && KARG="--depth-bins $KB"
env -u PYTHONPATH -u PYTHONHOME $PY nTof_x17/rerun/precheck_vscan/v_scan.py --cache cache.pkl --bundle "$B" \
   --arm arm_${DET}_${ARM}.json --vs "$VS" --n 60 --jobs "${OMP_NUM_THREADS:-8}" $KARG --out vscan_${DET}_${ARM}_k${KB}.json
rm -rf cache.pkl bundle bundle.tgz nTof_x17 payload.tar.gz
