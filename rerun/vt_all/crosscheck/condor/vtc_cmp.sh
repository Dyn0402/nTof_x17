#!/bin/bash
# vtc_cmp.sh <code dir> <builder.py> <sps key> <out.pkl> -- one side of the sps_build_cache byte-compare (vtc_cachecmp.py
# build).  Reads the decoded FEU files on EOS; writes one pickle to the vt_check EOS tree; nothing left in scratch.
CODE=$1; B=$2; KEY=$3; OUT=$4; shift 4
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
mkdir -p $(dirname $OUT)
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/vt_all/crosscheck/vtc_cachecmp.py \
    build $B $KEY $OUT "$@"
