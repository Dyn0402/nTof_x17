#!/bin/bash
# sps_row.sh <run> <subrun> <dataset> -- GC condor job: det4 H4 column duration vs time for one sub-run (sps_colsum.py).
RUN=$1; SR=$2; DS=$3
OUT=/eos/user/d/dneff/x17rerun/followups/gc/sps/${RUN}__${SR}.npz
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=/eos/user/d/dneff/x17rerun/followups/fa/code_2ce1dcc
mkdir -p $(dirname $OUT)
python3 $HOME/x17rerun/gc/sps_colsum.py $RUN $SR \
    --dataset $DS --tmpl $HOME/x17rerun/gc/det4_tmpl.npz --out $OUT --max-events 60000 2>&1 | grep -v 'so far'
