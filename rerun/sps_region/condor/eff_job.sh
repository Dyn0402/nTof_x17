#!/bin/bash
# eff_job.sh <code dir> <run> <subrun> <group> <key>: eff_job.py for one run-63 FEU group; parquet to EOS, scratch cleaned.
CODE=$1; RUN=$2; SUB=$3; G=$4; K=$5
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=/eos/user/d/dneff/offload/x17rerun_code/code_48244778 PYTHONDONTWRITEBYTECODE=1
O=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/sps_region/eff; mkdir -p $O
W=${_CONDOR_SCRATCH_DIR:-/tmp/dneff_eff_$G}; mkdir -p $W
/afs/cern.ch/user/d/dneff/bin/lockpy $CODE/eff_job.py --run $RUN --subrun $SUB \
    --group $G --key $K --out $W/E_${SUB}_$G.parquet || exit 1
cp $W/E_${SUB}_$G.parquet $O/ && rm -f $W/*.parquet
