#!/bin/bash
# run_rcv2.sh <det> <cache> <tag> <extra recal arg or -> <free,comma,list>  -- RC-kernel calibration with v free, seeded from seed_<det>.json
# (rerun pre-check P1, 2026-10-23).  Rerun environment: the EOS lock venv, not an LCG python.
set -eu
DET=$1; CACHE=$2; TAG=$3; EXTRA=$4; FREE=$5
[ "$EXTRA" = "-" ] && EXTRA=""
EOSD=/eos/user/d/dneff/plane_ratio
PY=/eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python
fetch() { for i in 1 2 3 4; do
    xrdcp -s -f "root://eosuser.cern.ch/$1" "$2" 2>/dev/null && return 0
    cp "$1" "$2" 2>/dev/null && return 0; sleep 20; done; return 1; }
echo "[$(date -u +%T)] $(hostname) ${DET}_$TAG free=$FREE"
tar xzf payload.tar.gz
fetch $EOSD/inputs/$DET/$CACHE.pkl cache.pkl
fetch $EOSD/inputs/$DET/bundle.tgz bundle.tgz && mkdir -p bundle && tar xzf bundle.tgz -C bundle
B=$(ls -d bundle/*/)
cd nTof_x17
time env -u PYTHONPATH -u PYTHONHOME $PY sps_beam_test_26/analysis/plane_ratio/recal.py --cache ../cache.pkl --bundle "../$B" \
   --arm "$TAG" --free "$FREE" --seed-json "../seed_${DET}.json" --objective cold \
   $EXTRA --out "../arm_${DET}_$TAG.json" --jobs "${OMP_NUM_THREADS:-8}" --maxiter 800
cd ..; rm -rf cache.pkl bundle bundle.tgz nTof_x17 payload.tar.gz
echo "[$(date -u +%T)] done"
