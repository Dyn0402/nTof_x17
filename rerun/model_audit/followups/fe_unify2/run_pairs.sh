#!/bin/bash
# run_pairs.sh <code_dir> <variant> [extra intra_bench build args] -- FE: n_TOF close-pair efficiency (overlay bench,
# sept26_prelim_analysis/intra_bench.py) on run_145 stat090_0000, A and C, inputs/outputs on EOS
CODE=$1; VAR=$2; shift 2
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_BEAM_BASE=/eos/experiment/ntof/data/x17/july_beam/runs/ X17_RUNS=/eos/experiment/ntof/data/x17/july_beam/runs
export X17_ROOT=/eos/user/d/dneff/x17rerun/followups/fe X17_SEPT26_OUT=/eos/user/d/dneff/x17rerun/followups/fe/pairs
export MPLBACKEND=Agg
cd $CODE
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
WO=""; [ "$1" != "-" ] && WO="--worker-opt $1"
$PY -m sept26_prelim_analysis.intra_bench build --jobs 16 --variant $VAR $WO && \
$PY -m sept26_prelim_analysis.intra_bench derive --variant $VAR
