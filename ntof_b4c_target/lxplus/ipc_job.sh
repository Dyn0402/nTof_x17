#!/usr/bin/env bash
# B4C IPC pair job: $1=radius_mm $2=E_MeV $3=multipole $4=nevents $5=seed $6=outfile (EOS)
set -eo pipefail
set +u
source /afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_b4c/scripts/setup_lxplus.sh
set -u
EXE=/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_b4c/build/mx17_full_sim
LIB=/eos/user/d/dneff/scratch/b4c_sim/lib/b4c_slab_vertices.csv
cd "${_CONDOR_SCRATCH_DIR:-/tmp}"
"$EXE" -t 1 -n "$4" -o "$6" -s "$5" -g ArIso --slab B4C:2.5 --slab-radius "$1" \
    --energy "$2" --ipc 1.0 --ipc-multipole "$3" --pair-vertex-lib "$LIB" --pair-vertex-vol Slab
echo "done $(date)"
