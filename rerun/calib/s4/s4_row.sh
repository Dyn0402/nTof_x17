#!/bin/bash
# s4_row.sh <code_dir> <key> <bundle> -- S4: FA's model-free column duration (colsum.py) of every trigger of a golden row,
# with the PASS-1 bundle (its templates and dead masks).  -> /eos/user/d/dneff/x17rerun/s4_inputs/Analysis/<row>/fa/colsum.npz
# Same job as fa_row.sh colsum; only the output tree differs (never Analysis_pass1).  Leaves nothing large in scratch.
CODE=$1; KEY=$2; BUNDLE=$3
EOSA=/eos/user/d/dneff/x17rerun/s4_inputs/Analysis
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') MPLBACKEND=Agg
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/s4_$$}; mkdir -p $W
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/stage_job.py --work $W --eos-analysis $EOSA"
FA=rerun/model_audit/followups/fa_gas_v_tilt
$PY $KEY --parts run_config,m3,decoded --script $FA/colsum.py -- --bundle $BUNDLE --out {out}/fa/colsum.npz --jobs 8
rc=$?
rm -rf $W/bench $W/Analysis
exit $rc
