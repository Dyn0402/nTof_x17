#!/usr/bin/env bash
# B4C July target sim job: $1=mode (neutron|casc) $2=radius_mm $3=nevents $4=seed $5=outfile (EOS)
set -eo pipefail
set +u
source /afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_b4c/scripts/setup_lxplus.sh
set -u
MODE=$1; R=$2; N=$3; SEED=$4; OUT=$5
EXE=/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_b4c/build/mx17_full_sim
D=/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_v2/data
LIB=/eos/user/d/dneff/scratch/b4c_sim/lib/b4c_slab_vertices.csv
CASC="7.007+4.444:54.2,4.711+6.740:17.4,4.711+2.297+4.444:8.3,2.533+8.917:13.0,2.533+4.475+4.444:0.8,11.448:4.64"
cd "${_CONDOR_SCRATCH_DIR:-/tmp}"
if [[ $MODE == neutron ]]; then
  printf "/process/had/particle_hp/use_photo_evaporation true\n/run/beamOn %s\n" "$N" > run.mac
  "$EXE" -t 1 -o "$OUT" -s "$SEED" -g ArIso --slab B4C:2.5 --slab-radius "$R" \
      --neutron $D/Final_evaluated_flux_Phase4_EAR2.root $D/ear2_profile_july2021_L19.3.root \
      --emin 2.2e-3 --emax 2 run.mac
else
  "$EXE" -t 1 -n "$N" -o "$OUT" -s "$SEED" -g ArIso --slab B4C:2.5 --slab-radius "$R" \
      --gamma-lines "$CASC" --pair-vertex-lib "$LIB" --pair-vertex-vol Slab
fi
rm -f run.mac
echo "done $(date)"
