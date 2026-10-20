#!/usr/bin/env bash
# 3He capsule pair job (B4C deck comparison): $1=ipc_fraction $2=multipole $3=nevents $4=seed $5=outfile
set -eo pipefail
set +u
source /afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_b4c/scripts/setup_lxplus.sh
set -u
EXE=/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_b4c/build/mx17_full_sim
LIB=/eos/experiment/ntof/data/x17/full_sim/libs_nose/gas_vertex_nose.csv
cd "${_CONDOR_SCRATCH_DIR:-/tmp}"
"$EXE" -t 1 -n "$3" -o "$5" -s "$4" -g ArIso --energy 20.58 --ipc "$1" --ipc-multipole "$2" \
    --pair-vertex-lib "$LIB" --pair-vertex-vol He3Gas
echo "done $(date)"
