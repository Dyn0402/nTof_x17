#!/bin/bash
# Push the code (no data) to lxplus and lay out the submit directory.
#
#   ./stage.sh [remote]        default remote: lxplus:~/x17wf
#
# `transfer_input_files = pkg` with NO trailing slash.  With the slash condor
# transfers the CONTENTS into the scratch root and flattens the package, and the
# import then fails on the worker after the raw copy has already run.
set -eu

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
REMOTE=${1:-lxplus:x17wf}
HOST=${REMOTE%%:*}
DIR=${REMOTE#*:}

echo "staging $REPO -> $REMOTE"
ssh -K "$HOST" "mkdir -p $DIR/logs $DIR/ntof_processing $DIR/ntof_dream_merge $DIR/common"

# the two packages the worker imports, and nothing else
rsync -a --delete -e "ssh -K" \
    --include='*/' --include='*.py' --exclude='*' \
    "$REPO/ntof_processing/" "$HOST:$DIR/ntof_processing/"
rsync -a --delete -e "ssh -K" \
    --include='*/' --include='*.py' --exclude='*' \
    "$REPO/ntof_dream_merge/" "$HOST:$DIR/ntof_dream_merge/"
# ntof_io imports common.beam_july_paths at module level -- without it the job
# dies on the FIRST tflash read, after the raw list is already built.
rsync -a --delete -e "ssh -K" \
    --include='*/' --include='*.py' --exclude='*' \
    "$REPO/common/" "$HOST:$DIR/common/"
rsync -a -e "ssh -K" \
    "$REPO/ntof_processing/waveform_pull/lxplus/"{wrapper.sh,pull.sub,recall.sh,raw_inventory.sh,campaign.sh} \
    "$HOST:$DIR/"
ssh -K "$HOST" "chmod +x $DIR/*.sh"

cat <<EOF

staged. On lxplus:

  ssh -K lxplus                     # -K is mandatory: no token, no EOS
  cd $DIR
  myschedd bump

  # The whole campaign, paced against the tape staging lifetime. Recalled files
  # go offline again in WELL under a day, so recall and read are coupled: the
  # driver stages a batch, submits it, and pre-stages the next while that reads.
  ./campaign.sh plan runs.txt              # classify: disk / tape / no-slim
  nohup ./campaign.sh run > campaign.log 2>&1 &
  ./campaign.sh status                     # any time, from any session

  # or, by hand, for one batch:
  ./recall.sh request batch.txt && ./recall.sh check batch.txt
  condor_submit pull.sub -a 'runs=batch.txt'

  # what actually landed, read from the products and not from the logs:
  python -m ntof_processing.waveform_pull.fleet_report
EOF
