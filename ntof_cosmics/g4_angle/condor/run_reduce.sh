#!/usr/bin/env bash
set -eo pipefail
IN="$1"; OUT="$2"
( set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python3 /afs/cern.ch/user/d/dneff/condor/mx17_angle_scale/reduce_gap_wall.py "$IN" "${_CONDOR_SCRATCH_DIR:-/tmp}/r.parquet" )
cp "${_CONDOR_SCRATCH_DIR:-/tmp}/r.parquet" "$OUT"
rm -f "${_CONDOR_SCRATCH_DIR:-/tmp}/r.parquet"   # nothing left for condor to copy back to AFS
