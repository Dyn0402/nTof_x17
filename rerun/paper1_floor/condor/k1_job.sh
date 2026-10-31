#!/bin/bash
# K1 test (rerun/paper1_floor/K1_TEST_2026-11-01.md): one FD-toy ladder job with a per-event shape scatter injected on
# top of the full toy (rungs K1G/T/W/S/L, optional trailing w = K1-aware weights).  Unpack code + pass-1 bundles, run,
# copy to EOS, clean the scratch.   k1_job.sh <view> <K1_AMP> <K1_WT> <ladder args...>
set -e
VIEW=$1; AMP=$2; WT=$3; shift 3
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
tar xzf fd_code.tgz
export FD_AUDIT=$PWD/bundles FD_BASIS=pass FD_PLACE=ig FD_INPUTS=$PWD/toy_inputs_pass1.json FD_VIEW=$VIEW
export K1_AMP=$AMP K1_WT=$WT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=rerun/model_audit/followups/fd_floor_realism
mkdir -p out
python3 $D/ladder.py job "$@" --out $PWD/out
EOS=root://eosexperiment.cern.ch//eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/k1_floor/raw
for f in out/*; do xrdcp -f "$f" "$EOS/$(basename "$f")"; done
rm -rf out rerun wft bundles fd_code.tgz toy_inputs_pass1.json
