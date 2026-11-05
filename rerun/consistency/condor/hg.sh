#!/bin/bash
# hg.sh <code dir> <key>: rerun/paper1_ops/hits_gain.py on one row -> EOS x17rerun/consistency/hits_gain
# (the consistency audit's fixed-HV gain check on the 6-23 night; rerun/consistency/README.md)
CODE=$1; KEY=$2
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
OUT=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/consistency/hits_gain
mkdir -p $OUT
/afs/cern.ch/user/d/dneff/bin/lockpy $CODE/rerun/paper1_ops/hits_gain.py \
    --key $KEY --registry $CODE/rerun/pass_registry.csv --out-dir $OUT
