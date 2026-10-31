#!/bin/bash
# restore_v2_rays.sh -- undo the 2026-10-31 EOS move's symlink loop on the M3 v2 rays (run on lxplus).
#
# The move dereferenced the 12 symlinks in x17rerun/m3_audit/prod (they point at june_tests/<run>/<subrun>/
# m3_tracking_root_v2/*_rays.root, the real files), deleted the TARGETS and left a link at each target pointing back
# at m3_audit/prod: a loop, no bytes.  The originals are in the n_TOF EOS recycle bin.  For each one: remove the loop
# link, restore by its recycle key, check the restored size against the recycle record.
export EOS_MGM_URL=root://eosexperiment.cern.ch
DAY=${1:-2026/10/31}
TMP=$(mktemp)
eos recycle ls -m "$DAY" 2>/dev/null | grep "m3_tracking_root_v2" | grep "cosmic_bench/june_tests" > "$TMP"
while read -r line; do
  key=$(echo "$line" | grep -o "restore-key=[^ ]*" | cut -d= -f2)
  path=$(echo "$line" | grep -o "restore-path=/[^ ]*" | cut -d= -f2)
  size=$(echo "$line" | grep -o " size=[0-9]*" | cut -d= -f2)
  if [ -L "$path" ]; then
    rm "$path" && eos recycle restore "$key" > /dev/null 2>&1
    got=$(stat -c %s "$path" 2> /dev/null)
    if [ ! -L "$path" ] && [ "$got" = "$size" ]; then echo "OK $size $path"; else echo "FAIL want=$size got=$got $path"; fi
  else
    echo "SKIP not-a-link $path"
  fi
done < "$TMP"
rm -f "$TMP"
