#!/bin/bash
# vtc_meas.sh <code dir> <key> <row> <inputs root> -- vtc_measure.py on one row: colsum.npz and s4/headon.pkl from
# <inputs root>/<row>/ (S4's s4_inputs/Analysis for the golden rows, vt_check/Analysis for the others), the survey's
# block JSON from vt_all/bench.  -> /eos/user/d/dneff/x17rerun/vt_check/measure/<key>.json
CODE=$1; KEY=$2; ROW=$3; IN=$4
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export MPLBACKEND=Agg
O=/eos/user/d/dneff/x17rerun/vt_check/measure; mkdir -p $O
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/vt_all/crosscheck/vtc_measure.py \
    $KEY --colsum $IN/$ROW/fa/colsum.npz --headon $IN/$ROW/s4/headon.pkl \
    --survey /eos/user/d/dneff/x17rerun/vt_all/bench/$KEY.json --out $O/$KEY.json --nblk 4 --nboot 20 --workers 8
