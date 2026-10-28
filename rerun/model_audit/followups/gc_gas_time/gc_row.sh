#!/bin/bash
# gc_row.sh <code_dir> <key> <bundle> -- GC condor job: model-free column duration (FA colsum.py) of every trigger of one
# bench row, staged via rerun/stage_job.py (rr1 inputs: decoded_root_fa80c06).  Output -> <EOSA>/<row>/gc/colsum.npz.
# Same as fa_row.sh mode colsum, but writes the GC tree.  Leaves nothing large in the scratch dir.
CODE=$1; KEY=$2; BUNDLE=$3
EOSA=/eos/user/d/dneff/x17rerun/followups/gc/Analysis
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') MPLBACKEND=Agg
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/gc_$$}; mkdir -p $W
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/stage_job.py --work $W --eos-analysis $EOSA"
FA=rerun/model_audit/followups/fa_gas_v_tilt
$PY $KEY --parts run_config,m3,decoded --script $FA/colsum.py -- --bundle $BUNDLE --out {out}/gc/colsum.npz --jobs 8
rc=$?
rm -rf $W/bench $W/Analysis
exit $rc
