#!/bin/bash
# seed_diff_job.sh <registry key>: seed_diff.py for one pass-1 SPS row; a small json to EOS.
K=$1
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=/eos/user/d/dneff/x17rerun/sps_audit/code PYTHONDONTWRITEBYTECODE=1
O=/eos/user/d/dneff/x17rerun/sps_audit/seed_diff; mkdir -p $O
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $WFT_REPO/rerun/sps_audit/seed_diff.py $K --out $O/$K.json
