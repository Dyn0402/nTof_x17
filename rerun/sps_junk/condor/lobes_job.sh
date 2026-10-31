#!/bin/bash
# lobes_job.sh <code dir> <registry key>: rerun/sps_junk/lobes.py for one SPS row -> an npz on EOS.
CODE=$1; K=$2
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE PYTHONDONTWRITEBYTECODE=1
O=/eos/user/d/dneff/x17rerun/sps_junk/lobes; mkdir -p $O
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $HOME/x17rerun/sps_junk/lobes.py $K $O/$K.npz
