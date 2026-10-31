#!/bin/bash
# spark.sh <code dir> <key>: rerun/paper1_ops/spark_timeline.py on one bench pass-1 row -> EOS paper1_ops/spark
CODE=$1; KEY=$2
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
O=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_ops/spark
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/paper1_ops/spark_timeline.py \
    --key $KEY --registry $CODE/rerun/pass_registry.csv --out-dir $O
