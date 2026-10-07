#!/usr/bin/env bash
# one Geant4 step file through the digitiser + production reco on one arm; result to EOS,
# nothing left in the scratch dir (condor copies leftovers back to AFS).
#   run_digi_job.sh <arm A|C> <steps.parquet on EOS> <seed> <out.parquet on EOS> [n]
set -eo pipefail
ARM="$1"; STEPS="$2"; SEED="$3"; OUTF="$4"; N="${5:-100000}"
JD=/afs/cern.ch/user/d/dneff/condor/mx17_g4_digi
S=${_CONDOR_SCRATCH_DIR:-/tmp/$USER.digi.$$}; mkdir -p "$S"; cd "$S"
tar xzf $JD/code.tar.gz
export WFT_BEAM_BASE=/eos/experiment/ntof/data/x17/july_beam/runs/
export G4DIGI_OUT="$S/out" G4DIGI_BUNDLES=$JD/bundles G4DIGI_IS2_TRACKS=$JD/is2_t0.parquet
export G4DIGI_REDUCED=/eos/experiment/ntof/data/x17/full_sim/angle_scale/neutrons_nose
export PYTHONPATH="$S"
G4ARM=2; [ "$ARM" = C ] && G4ARM=3
( set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python3 ntof_cosmics/g4_digi/run_digi.py g4 --arm "$ARM" --bundle is2_$ARM --steps "$STEPS" \
      --g4-arm $G4ARM --n "$N" --jobs ${DIGI_JOBS:-2} --seed "$SEED" --label part )
cp "$S/out/part.parquet" "$OUTF"
cd /; rm -rf "$S"/*
echo done
