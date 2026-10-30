#!/bin/bash
# fetch.sh [dest] -- pull the small pass-1 inputs of the det6/det7 6-26 rows (events table + meta, 03_angles per-event,
# bundle json, alignment json; ~23 MB) from EOS through `ssh lxplus`.  Read-only on Analysis_pass1.
set -euo pipefail
D=${1:-${VT_DATA:-${WORK_DATA:-$HOME/x17}/vt}}
mkdir -p "$D"; cd "$D"
ssh lxplus 'cd /eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1 && tar cf - $(for r in mx17_det6_det7_overnight_6-26-26 mx17_det6_det7_hv_scan_6-26-26; do for s in $r/*/; do for d in mx17_6 mx17_7; do p=$s$d/wft; [ -f $p/events.parquet ] && echo $p/events.parquet $p/events.meta.json $p/angles/per_event.npz $p/angles/angular_resolution.json $(ls $p/calib_bundle/*.json 2>/dev/null) $(ls $p/alignment/*.json 2>/dev/null); done; done; done) 2>/dev/null' | tar xf -
echo "pulled into $D"
