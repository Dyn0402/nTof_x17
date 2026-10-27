#!/bin/bash
# sps_stage.sh <code_dir> <registry key> <out dir on EOS> [extra sps_reco.py args] -- condor entry point for one SPS row (B3)
CODE=$1; KEY=$2; OUT=$3; shift 3
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') WFT_REPO=$CODE MPLBACKEND=Agg
# no WFT_* settings: the column guard and the pivot start are part of wft (wft/diag.py)
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/sps_$$}; mkdir -p $W
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/model_audit/sps_reco.py $KEY --out $OUT --work $W "$@"
rc=$?; rm -f $W/sps_cache_*.pkl; exit $rc
