#!/bin/bash
# res_check_job.sh <code dir> <run> <subrun> <group> <feu3 dir> <nsamp>: rerun/sps_junk/res_check.py -> a json on EOS.
CODE=$1; shift
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE PYTHONDONTWRITEBYTECODE=1
O=/eos/user/d/dneff/x17rerun/sps_junk/res_check; mkdir -p $O
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $HOME/x17rerun/sps_junk/res_check.py "$@" $O/$1_$2_$3.json
