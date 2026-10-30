#!/bin/bash
# vt_job.sh <code_dir> <script> <args...> -- one v(t) node (rerun/vt_all), run from the frozen code archive with the lock venv.
CODE=$1; S=$2; shift 2
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/vt_all/$S "$@"
