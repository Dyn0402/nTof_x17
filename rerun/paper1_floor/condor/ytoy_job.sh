#!/bin/bash
# The FD toy ladder fitted as the Y view (FD_VIEW=y): unpack the pass-1 code + bundles, run one ladder job, copy the output
# to EOS, clean the scratch.   ytoy_job.sh <ladder args...>
set -e
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
tar xzf fd_code.tgz
export FD_AUDIT=$PWD/bundles FD_BASIS=pass FD_PLACE=ig FD_INPUTS=$PWD/toy_inputs_pass1.json FD_VIEW=y
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=rerun/model_audit/followups/fd_floor_realism
mkdir -p out
python3 $D/ladder.py job "$@" --out $PWD/out
EOS=root://eosexperiment.cern.ch//eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_floor/ytoy_raw
for f in out/*; do xrdcp -f "$f" "$EOS/$(basename "$f")"; done
rm -rf out rerun wft bundles fd_code.tgz toy_inputs_pass1.json
