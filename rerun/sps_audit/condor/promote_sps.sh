#!/bin/bash
# promote_sps.sh <stage dir> <code> <backup tag>: a staged SPS re-run replaces Analysis_pass1/sps, after a verified backup.
#   2026-10-31: bash promote_sps.sh /eos/user/d/dneff/x17rerun/sps_final_stage/sps c3e6a04c sps_before_seedfix   (pass 1 -> seed fix)
#               bash promote_sps.sh /eos/user/d/dneff/x17rerun/sps_final2_stage/sps 3ac0e79 sps_before_blanking (-> final)
# 1. every staged row must have events.parquet + events.meta.json on the expected code, else stop;
# 2. back up every current Analysis_pass1/sps/<key>/ (all files) to pass1_backup_2026-10-30/<backup tag>/, verify by md5;
# 3. copy the staged files in, verify by md5.
set -e
A=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps
S=${1:?stage dir}; CODE=${2:?code}; B=/eos/user/d/dneff/x17rerun/pass1_backup_2026-10-30/${3:?backup tag}
n=0
for d in $S/*/; do
  k=$(basename $d)
  [ -s $d/events.parquet ] && [ -s $d/events.meta.json ] || { echo "MISSING $k"; exit 1; }
  grep -q "\"code\": \"$CODE\"" $d/events.meta.json || { echo "CODE $k"; exit 1; }
  n=$((n+1))
done
[ $n -eq 71 ] || { echo "only $n staged rows"; exit 1; }
mkdir -p $B
for d in $A/*/; do
  k=$(basename $d); mkdir -p $B/$k
  cp -p $d/* $B/$k/
  for f in $d/*; do [ "$(md5sum < $f)" = "$(md5sum < $B/$k/$(basename $f))" ] || { echo "BACKUP MISMATCH $f"; exit 1; }; done
done
echo "backup verified: $(ls $B | wc -l) rows"
for d in $S/*/; do
  k=$(basename $d)
  for f in events.parquet events.meta.json; do
    cp $d/$f $A/$k/$f
    [ "$(md5sum < $d/$f)" = "$(md5sum < $A/$k/$f)" ] || { echo "COPY MISMATCH $k/$f"; exit 1; }
  done
done
echo "promoted $n rows"
