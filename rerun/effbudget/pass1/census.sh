#!/bin/bash
# census.sh <code_dir> <band_census.py args...> -- condor entry point (as rerun/stage_job.sh: lock venv, LCG_110)
CODE=$1; shift
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export MPLBACKEND=Agg
cd ${_CONDOR_SCRATCH_DIR:-/tmp}
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/effbudget/pass1/band_census.py "$@"
