#!/bin/bash
# ycache_job.sh <code_dir> <key> <row> -- rerun/det4_gap/bench_ycache.py for one bench row (pass-1 alignment, decode),
# through rerun/stage_job.py.  -> /eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/det4_gap/bench/Analysis/<row>/gap/ycache.pkl
CODE=$1; KEY=$2; ROW=$3
EOSA=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/det4_gap/bench/Analysis
P1=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/$ROW/wft
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') RERUN_CODE_DIR=$CODE MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/ycache_$$}; mkdir -p $W
/afs/cern.ch/user/d/dneff/bin/lockpy $CODE/rerun/stage_job.py --work $W --eos-analysis $EOSA \
    $KEY --parts run_config,m3,decoded --script $HOME/x17rerun/det4_gap/bench_ycache.py -- \
    --alignment $P1/alignment/alignment.json --out {out}/gap
rc=$?
rm -rf $W/bench $W/Analysis
exit $rc
