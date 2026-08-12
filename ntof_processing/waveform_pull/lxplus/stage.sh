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
    "$REPO/ntof_processing/waveform_pull/lxplus/"{wrapper.sh,pull.sub,recall.sh,raw_inventory.sh} \
    "$HOST:$DIR/"
ssh -K "$HOST" "chmod +x $DIR/*.sh"

cat <<EOF

staged. On lxplus:

  ssh -K lxplus                     # -K is mandatory: no token, no EOS
  cd $DIR
  ./raw_inventory.sh runs.txt > inventory.csv          # who needs a recall
  awk -F, 'NR>1 && \$2==0 {print \$1}' inventory.csv > recall.txt
  ./recall.sh request recall.txt                      # fire the tape recalls
  awk -F, 'NR>1 && \$2>0  {print \$1}' inventory.csv > ondisk.txt
  myschedd bump
  condor_submit pull.sub -a 'runs=ondisk.txt'         # start on what is here
EOF
