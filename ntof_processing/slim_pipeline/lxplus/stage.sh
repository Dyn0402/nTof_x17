#!/bin/bash
# Push everything the slim job needs to ~/x17slim on lxplus.
#
#   ./ntof_processing/slim_pipeline/lxplus/stage.sh          # default target
#   ./ntof_processing/slim_pipeline/lxplus/stage.sh lxplus:x17slim
#
# Only source is staged -- no data. The job reads DREAM off EOS FUSE and copies
# the n_TOF run to node scratch itself.
set -euo pipefail
TARGET=${1:-lxplus:x17slim}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SSH='ssh -o ControlPath=none'

echo "staging $REPO -> $TARGET"
$SSH "${TARGET%%:*}" "mkdir -p ${TARGET#*:}/log"

# The packages the job imports, plus the one calibration file it reads.
rsync -a --delete -e "$SSH" \
  --include='*/' \
  --include='*.py' --include='*.json' --include='*.txt' --include='*.csv' \
  --exclude='*' \
  --exclude='__pycache__/' --exclude='cache/' --exclude='cache_pulse_match/' \
  "$REPO/ntof_processing/" "$TARGET/ntof_processing/"
for pkg in ntof_dream_merge ntof_july_analysis common; do
  rsync -a -e "$SSH" \
    --include='*/' --include='*.py' --include='*.json' --exclude='*' \
    --exclude='__pycache__/' --exclude='cache/' --exclude='cache_pulse_match/' \
    --exclude='match_study/data/' \
    "$REPO/$pkg/" "$TARGET/$pkg/"
done
$SSH "${TARGET%%:*}" "mkdir -p ${TARGET#*:}/mx_july_beam_qa/calib"
rsync -a -e "$SSH" "$REPO/mx_july_beam_qa/calib/" "$TARGET/mx_july_beam_qa/calib/"
rsync -a -e "$SSH" "$REPO/ntof_processing/slim_pipeline/lxplus/"{slim.sub,slim_wrapper.sh,publish_to_eos.sh} \
  "$TARGET/"
$SSH "${TARGET%%:*}" "chmod +x ${TARGET#*:}/slim_wrapper.sh ${TARGET#*:}/publish_to_eos.sh"

echo "staged. now:"
echo "  ssh -K lxplus"
echo "  cd ${TARGET#*:} && myschedd bump && condor_submit slim.sub run=224572"
