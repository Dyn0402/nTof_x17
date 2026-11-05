#!/bin/bash
# pg_fit.sh <code_dir> <out_dir> -- the time-resolved bench timing stacks: pg_specs.py, then s4f_fit.py on every spec
# (rerun/paper1_gas/README.md).  Reads the pg_stack.sh caches and the pass bundles through FUSE; writes <out_dir>.
CODE=$1; OUT=$2
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export MPLBACKEND=Agg XW_MIN_EV=50
PY="/afs/cern.ch/user/d/dneff/bin/lockpy"
mkdir -p $OUT
$PY $CODE/rerun/paper1_gas/pg_specs.py $OUT || exit 1
for s in $OUT/spec_T*.json; do
  $PY $CODE/rerun/calib/s4/s4f_fit.py --spec $s --out ${s/spec_/fit_} --nboot 20 --workers 8 || exit 2
done
