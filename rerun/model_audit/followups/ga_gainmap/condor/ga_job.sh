#!/bin/bash
# GA condor job: unpack code + bundles, run one ga_toy job, copy the output to EOS, clean the scratch.
set -e
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
tar xzf ga_code.tgz
export FD_AUDIT=$PWD/bundles OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p out
python3 rerun/model_audit/followups/ga_gainmap/ga_toy.py job "$@" --out $PWD/out
EOS=root://eosuser.cern.ch//eos/user/d/dneff/x17rerun/followups/ga/toy
for f in out/*; do xrdcp -f "$f" "$EOS/$(basename "$f")"; done
rm -rf out rerun wft bundles ga_code.tgz
