#!/bin/bash
# tel_gain.sh <code dir> <tag> <run> <subrun> <keys>: rerun/paper1_ops/tel_gain.py on one SPS sub-run -> EOS paper1_ops/tel_gain
CODE=$1; TAG=$2; RUN=$3; SUB=$4; KEYS=$5
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
W=${_CONDOR_SCRATCH_DIR:-/tmp/dneff_telgain_$TAG}; mkdir -p $W
O=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_ops/tel_gain; mkdir -p $O
/afs/cern.ch/user/d/dneff/bin/lockpy $CODE/rerun/paper1_ops/tel_gain.py \
    --run $RUN --subrun $SUB --keys $KEYS --out $W/$TAG.parquet || exit 1
cp $W/$TAG.parquet $O/ || exit 1
rm -f $W/*.parquet
