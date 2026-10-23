#!/bin/bash
# run_guard.sh <det> -- guard_bench.py on the det's bench big cache (rerun pre-check W2)
set -eu
DET=$1
EOSD=/eos/user/d/dneff/plane_ratio
PY=/eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python
fetch() { for i in 1 2 3 4; do
    xrdcp -s -f "root://eosuser.cern.ch/$1" "$2" 2>/dev/null && return 0
    cp "$1" "$2" 2>/dev/null && return 0; sleep 20; done; return 1; }
tar xzf payload.tar.gz
fetch $EOSD/inputs/$DET/big_cache.pkl cache.pkl
fetch $EOSD/inputs/$DET/bundle.tgz bundle.tgz && mkdir -p bundle && tar xzf bundle.tgz -C bundle
B=$(ls -d bundle/*/)
env -u PYTHONPATH -u PYTHONHOME $PY nTof_x17/rerun/precheck_guard/guard_bench.py --cache cache.pkl --bundle "$B" \
   --out guard_$DET.json --jobs "${OMP_NUM_THREADS:-8}" --max 1500
rm -rf cache.pkl bundle bundle.tgz nTof_x17 payload.tar.gz
