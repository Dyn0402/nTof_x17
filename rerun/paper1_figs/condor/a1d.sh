#!/bin/bash
# a1d.sh <code_dir> <a1_displays.py args...> -- condor entry point (lock venv, LCG_110, as rerun/effbudget/pass1/census.sh)
CODE=$1; shift
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export MPLBACKEND=Agg
cd ${_CONDOR_SCRATCH_DIR:-/tmp}
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/paper1_figs/a1_displays.py "$@"
