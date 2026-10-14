#!/bin/bash
# run_bench_lor.sh <det> <train cache: calib_cache|big_cache>
# Held-out paired bench of the X footprint-tail arms (FINDINGS §23) against production:
# rcm (physical kernel), rcmlor (+ X pseudo-Voigt tail), rcmlor2 (+ tail and fitted X prompt width).
# Payload = wft/ (with lor_frac_<plane>) + plane_ratio/, built from mx17-paper-status.
set -eu
DET=$1; TRAIN=$2
EOSD=/eos/user/d/dneff/plane_ratio
fetch() {
  for i in 1 2 3 4; do
    xrdcp -s -f "root://eosuser.cern.ch/$1" "$2" 2>/dev/null && return 0
    cp "$1" "$2" 2>/dev/null && return 0
    echo "fetch $1 failed (try $i)" >&2; sleep 20
  done; return 1; }
echo "[$(date -u +%T)] $(hostname) bench_lor $DET"
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh; set -u
tar xzf payload_bench_lor.tar.gz
fetch $EOSD/inputs/$DET/big_cache.pkl big.pkl
fetch $EOSD/inputs/$DET/$TRAIN.pkl train.pkl
fetch $EOSD/inputs/$DET/bundle.tgz bundle.tgz
mkdir -p bundle && tar xzf bundle.tgz -C bundle
B=$(ls -d bundle/*/)
time python sps_beam_test_26/analysis/plane_ratio/plane_bench.py --cache big.pkl \
  --train-cache train.pkl --bundle "$B" \
  --arms arm_${DET}_rcm.json arm_${DET}_rcmlor.json arm_${DET}_rcmlor2.json \
  --max-held 2000 --jobs "${OMP_NUM_THREADS:-8}" --nboot 1000 --out "bench_lor_$DET.json"
rm -rf big.pkl train.pkl bundle bundle.tgz wft sps_beam_test_26 payload_bench_lor.tar.gz
echo "[$(date -u +%T)] done"
