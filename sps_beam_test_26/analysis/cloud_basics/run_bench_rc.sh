#!/bin/bash
# run_bench_rc.sh <det> <train cache: calib_cache|big_cache>
# Held-out paired bench of all RC arms of one chamber (rcm = §12 no-refit,
# rcf3/rcf4x/rcfD = condor 4410535 refits) against production on the same events.
# Payload = wft/ (with build_matrix_rc) + plane_ratio/, built from mx17-paper-status.
set -eu
DET=$1; TRAIN=$2
EOSD=/eos/user/d/dneff/plane_ratio
fetch() {
  for i in 1 2 3 4; do
    xrdcp -s -f "root://eosuser.cern.ch/$1" "$2" 2>/dev/null && return 0
    cp "$1" "$2" 2>/dev/null && return 0
    echo "fetch $1 failed (try $i)" >&2; sleep 20
  done; return 1; }
echo "[$(date -u +%T)] $(hostname) bench_rc $DET"
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh; set -u
tar xzf payload_bench_rc.tar.gz
fetch $EOSD/inputs/$DET/big_cache.pkl big.pkl
fetch $EOSD/inputs/$DET/$TRAIN.pkl train.pkl
fetch $EOSD/inputs/$DET/bundle.tgz bundle.tgz
mkdir -p bundle && tar xzf bundle.tgz -C bundle
B=$(ls -d bundle/*/)
time python sps_beam_test_26/analysis/plane_ratio/plane_bench.py --cache big.pkl \
  --train-cache train.pkl --bundle "$B" \
  --arms arm_${DET}_rcm.json arm_${DET}_rcf3.json arm_${DET}_rcf4x.json arm_${DET}_rcfD.json \
  --max-held 2000 --jobs "${OMP_NUM_THREADS:-8}" --nboot 1000 --out "bench_rc_$DET.json"
rm -rf big.pkl train.pkl bundle bundle.tgz wft sps_beam_test_26 payload_bench_rc.tar.gz
echo "[$(date -u +%T)] done"
