#!/bin/bash
# promote_rows.sh <stage dir> <code> <backup dir>: the staged SPS rows (a subset) replace theirs in Analysis_pass1/sps,
# after a verified backup of exactly those rows (rerun/sps_audit/condor/promote_sps.sh, for a subset).
#   2026-10-31: bash promote_rows.sh /eos/user/d/dneff/x17rerun/sps_junk/stage 1b142fbf \
#               /eos/user/d/dneff/x17rerun/pass1_backup_2026-10-31/sps_before_coupled
# 1. every staged row must have events.parquet + events.meta.json on the expected code, else stop;
# 2. back up each such row's current Analysis_pass1/sps/<key>/ (all files) to <backup dir>/<key>/, verify by md5;
# 3. copy the staged files in, verify by md5.
set -e
A=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps
S=${1:?stage dir}; CODE=${2:?code}; B=${3:?backup dir}
keys=()
for d in $S/*/; do
  k=$(basename $d)
  [ -s $d/events.parquet ] && [ -s $d/events.meta.json ] || { echo "MISSING $k"; exit 1; }
  grep -q "\"code\": \"$CODE\"" $d/events.meta.json || { echo "CODE $k"; exit 1; }
  [ -d $A/$k ] || { echo "NOT A PASS ROW $k"; exit 1; }
  keys+=($k)
done
mkdir -p $B
for k in "${keys[@]}"; do
  mkdir -p $B/$k
  cp -p $A/$k/* $B/$k/
  for f in $A/$k/*; do [ "$(md5sum < $f)" = "$(md5sum < $B/$k/$(basename $f))" ] || { echo "BACKUP MISMATCH $f"; exit 1; }; done
done
echo "backup verified: ${#keys[@]} rows in $B"
for k in "${keys[@]}"; do
  for f in events.parquet events.meta.json; do
    cp $S/$k/$f $A/$k/$f
    [ "$(md5sum < $S/$k/$f)" = "$(md5sum < $A/$k/$f)" ] || { echo "COPY MISMATCH $k/$f"; exit 1; }
  done
done
echo "promoted ${#keys[@]} rows: ${keys[*]}"
