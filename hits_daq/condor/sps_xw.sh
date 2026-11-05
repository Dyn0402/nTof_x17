#!/bin/bash
# sps_xw.sh <code_dir> <registry key> <out url> -- one SPS det4 row's Arm D windows (sps_extract.py), as sps_stage.sh
CODE=$1; KEY=$2; OUT=$3; shift 3
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') WFT_REPO=$CODE MPLBACKEND=Agg OMP_NUM_THREADS=1
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/spsxw_$$}; mkdir -p $W
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/papers/paper2/hits_daq/sps_extract.py $KEY --out $OUT --work $W "$@"
rc=$?; rm -f $W/sps_cache_*.pkl $W/*.pkl; exit $rc
