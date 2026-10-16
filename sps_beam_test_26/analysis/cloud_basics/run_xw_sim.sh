#!/bin/bash
# run_xw_sim.sh <set> <arm> <seed0> <nrep> -- one chunk of xw_sim.py replicas (lxplus condor, ~/cloud_basics_condor/xwsim)
set -eu
echo "[$(date -u +%T)] $(hostname) xw_sim $*"
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh; set -u
tar xzf payload_xw_sim.tar.gz
export OMP_NUM_THREADS=1
python3 xw_sim.py run "$1" "$2" "$3" "$4" --inputs inputs.pkl --out "sim_$1_$2_$3.json"
rm -rf payload_xw_sim.tar.gz inputs.pkl *.py __pycache__
echo "[$(date -u +%T)] done"
