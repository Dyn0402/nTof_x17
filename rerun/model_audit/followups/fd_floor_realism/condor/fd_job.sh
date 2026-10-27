#!/bin/bash
# FD condor job: unpack the code + bundles, run one scan_a or ladder job, copy the output to EOS, clean the scratch.
#   fd_job.sh <scan_a|ladder> <args...>
set -e
KIND=$1; shift
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
tar xzf fd_code.tgz
export FD_AUDIT=$PWD/bundles
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=rerun/model_audit/followups/fd_floor_realism
mkdir -p out
python3 $D/$KIND.py job "$@" --out $PWD/out
EOS=root://eosuser.cern.ch//eos/user/d/dneff/x17rerun/followups/fd/$KIND
for f in out/*; do xrdcp -f "$f" "$EOS/$(basename "$f")"; done
rm -rf out rerun wft bundles fd_code.tgz
