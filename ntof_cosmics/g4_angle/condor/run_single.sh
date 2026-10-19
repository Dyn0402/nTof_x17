#!/usr/bin/env bash
# one single-particle config: simulate into job scratch, reduce, copy the parquet to EOS
set -eo pipefail
PART="$1"; E="$2"; PHI="$3"; N="$4"; SEED="$5"; TAG="$6"; OUT="$7"
G4=/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant
S=${_CONDOR_SCRATCH_DIR:-/tmp/$USER.$$}; mkdir -p $S; cd $S
( set +u; source $G4/scripts/setup_lxplus.sh >/dev/null 2>&1
  $G4/build/mx17_full_sim -n "$N" -g ArIso -s "$SEED" --single "$PART" "$E" 90 "$PHI" -o "$S/$TAG" )
( set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python3 /afs/cern.ch/user/d/dneff/condor/mx17_angle_scale/reduce_gap_wall.py "$S/${TAG}_t0.root" "$S/$TAG.parquet" --arms 2 )
cp "$S/$TAG.parquet" "$OUT/$TAG.parquet"
# keep the step-level ROOT on EOS, and leave nothing large in the scratch dir:
# condor copies leftovers back to the AFS submit dir (2026-10-07 quota incident)
mkdir -p "$OUT/../single_root" && cp "$S/${TAG}_t0.root" "$OUT/../single_root/"
rm -f "$S"/*.root "$S"/*.parquet
echo done
