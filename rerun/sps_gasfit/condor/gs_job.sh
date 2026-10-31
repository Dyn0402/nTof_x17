#!/bin/bash
# gs_job.sh <code_dir> <out_dir> <key> -- one gas-fit stack node (rerun/sps_gasfit), from the frozen code archive with the lock venv.
CODE=$1; OUT=$2; KEY=$3
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 WFT_REPO=$CODE
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/sps_gasfit/gas_stacks.py --out-dir $OUT $KEY
