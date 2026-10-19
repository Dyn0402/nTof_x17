#!/bin/bash
# Copy finished slims from a condor output tree to the DREAM tree on EOS.
#
#   ./publish_to_eos.sh out_224572     # one job's output
#   ./publish_to_eos.sh out_*          # the whole campaign
#
# Run on lxplus, not on a worker: writing to EOS needs the interactive token.
# Never --delete: the DREAM tree holds data this pipeline does not own.
#
# Sub-runs that straddle two n_TOF runs get one file from each job, with the
# n_TOF run in the filename, so they coexist in the same ntof_hits/ directory.
set -euo pipefail
DEST=${X17_EOS_JULY:-/eos/experiment/ntof/data/x17/july_beam}
[ $# -gt 0 ] || set -- out

ok=0; fail=0
for SRC in "$@"; do
  [ -d "$SRC/runs" ] || { echo "skip $SRC -- no runs/ inside"; continue; }
  n=$(find "$SRC/runs" -name 'ntof_hits_*.root' | wc -l)
  echo "== $SRC: $n slim file(s) -> $DEST"
  rsync -a "$SRC/runs/" "$DEST/runs/"
  while read -r f; do
    rel=${f#"$SRC"/}
    if [ -f "$DEST/$rel" ] && [ "$(stat -c%s "$f")" = "$(stat -c%s "$DEST/$rel")" ]; then
      ok=$((ok+1))
    else
      echo "  FAIL $rel"; fail=$((fail+1))
    fi
  done < <(find "$SRC/runs" -name 'ntof_hits_*.root')
done

echo
echo "verified $ok file(s), $fail failure(s)"
[ "$fail" -eq 0 ]
