#!/bin/bash
# pg_stack.sh <code_dir> <key> <row> -- paper1_gas: the S4 head-on stack (rerun/calib/s4/s4_stacks.py) of one bench row on the
# pass-1 alignment, decode and bundle, for the time-resolved bench gas fit (rerun/paper1_gas/README.md).  Analysis_pass1 read only.
# -> /eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_gas/Analysis/<row>/s4/{headon,incl}.pkl, s4_stacks.json
CODE=$1; KEY=$2; ROW=$3
EOSA=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_gas/Analysis
P1=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/$ROW/wft
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') MPLBACKEND=Agg
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/pg_$$}; mkdir -p $W
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/stage_job.py --work $W --eos-analysis $EOSA"
$PY $KEY --parts run_config,m3,decoded --script rerun/calib/s4/s4_stacks.py -- \
    --alignment $P1/alignment/alignment.json --bundle $P1/calib_bundle --out {out}/s4 --workers 4
rc=$?
rm -rf $W/bench $W/Analysis
exit $rc
