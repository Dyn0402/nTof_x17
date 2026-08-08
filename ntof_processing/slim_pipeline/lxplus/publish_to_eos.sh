#!/bin/bash
# Copy finished slims from a condor output tree to the DREAM tree on EOS.
#
#   ./publish_to_eos.sh out            # from ~/x17slim on lxplus, after a job
#
# Run on lxplus, not on a worker: writing to EOS needs the interactive token.
# Never --delete: the DREAM tree holds data this pipeline does not own.
set -euo pipefail
SRC=${1:-out}
DEST=${X17_EOS_JULY:-/eos/experiment/ntof/data/x17/july_beam}

[ -d "$SRC/runs" ] || { echo "no $SRC/runs -- wrong directory?"; exit 2; }
n=$(find "$SRC/runs" -name 'ntof_hits_*.root' | wc -l)
echo "publishing $n slim file(s) from $SRC -> $DEST"
rsync -av "$SRC/runs/" "$DEST/runs/"
echo
echo "verifying:"
find "$SRC/runs" -name 'ntof_hits_*.root' | while read -r f; do
  rel=${f#"$SRC"/}
  if [ -f "$DEST/$rel" ] && [ "$(stat -c%s "$f")" = "$(stat -c%s "$DEST/$rel")" ]; then
    echo "  ok   $rel"
  else
    echo "  FAIL $rel"; exit 1
  fi
done
