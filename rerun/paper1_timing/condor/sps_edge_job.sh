#!/bin/bash
# sps_edge_job.sh <my code dir> <registry key> -- sps_edge.py for one flat SPS row, the SPS chain from the frozen 48244778
# archive (the final oscillator rule, rerun/sps_junk); an npz of strip sums to EOS.
CODE=$1; K=$2
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=/eos/user/d/dneff/offload/x17rerun_code/code_48244778 PYTHONDONTWRITEBYTECODE=1
O=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_timing/sps; mkdir -p $O
[ -s $O/$K.npz ] && { echo "done $K"; exit 0; }
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/paper1_timing/sps_edge.py $K --out $O/$K.npz
