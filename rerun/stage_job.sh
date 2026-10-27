#!/bin/bash
# stage_job.sh <code_dir> <stage_job.py args...> -- condor entry point for one re-run stage (W10).
# <code_dir> is a `git archive` of the frozen commit on AFS (~/x17rerun/code_<commit>); the lock venv on EOS runs it.
CODE=$1; shift
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//')
export WFT_RND_DIR=/eos/user/d/dneff/x17rerun/regression/wft_model_sat_det3
export MPLBACKEND=Agg
# the reconstruction takes no settings: the column guard and the pivot start are part of wft since 2026-10-27
# (a study passes WFT_DIAG and stage_job.py --diag; wft/diag.py)
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/stage_$$}; mkdir -p $W
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/stage_job.py --work $W "$@"
rc=$?
# condor copies whatever is left in the scratch dir back to AFS: leave nothing large behind
rm -rf $W/bench $W/Analysis
exit $rc
