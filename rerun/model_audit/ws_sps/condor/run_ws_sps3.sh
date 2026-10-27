#!/bin/bash
# run_ws_sps3.sh <base> <code subdir> <script> <args...>  -- like run_ws_sps2.sh but passes the arguments through verbatim
B=$1; C=$2; S=$3; shift 3
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WS_SPS_DATA=$B/data WS_SPS_BUNDLE=$B/data/calib_bundle MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $B/$C
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python rerun/model_audit/ws_sps/$S "$@"
