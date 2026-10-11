#!/usr/bin/env bash
# one nose-campaign file -> its prompt DriftGas steps (arms A, C) on EOS; nothing left in scratch
set -eo pipefail
IN="$1"; OUT="$2"
S=${_CONDOR_SCRATCH_DIR:-/tmp/$USER.$$}
( set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python3 /afs/cern.ch/user/d/dneff/condor/mx17_digi_steps/extract_steps.py "$IN" "$S/s.parquet" --arms 2 3 )
cp "$S/s.parquet" "$OUT"
rm -f "$S/s.parquet"
