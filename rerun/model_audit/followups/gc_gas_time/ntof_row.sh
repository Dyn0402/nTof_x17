#!/bin/bash
# ntof_row.sh <run> <subrun> -- GC condor job: n_TOF column duration vs time, four arms, one sub-run (ntof_colsum.py).
RUN=$1; SR=$2
OUT=/eos/user/d/dneff/x17rerun/followups/gc/ntof/${RUN}__${SR}.npz
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=/eos/user/d/dneff/x17rerun/followups/fa/code_2ce1dcc
mkdir -p $(dirname $OUT)
cd ${_CONDOR_SCRATCH_DIR:-/tmp}
python3 $HOME/x17rerun/gc/ntof_colsum.py $RUN $SR \
    --tmpl-dir $HOME/x17rerun/gc/tmpl --out $OUT --jobs 4
