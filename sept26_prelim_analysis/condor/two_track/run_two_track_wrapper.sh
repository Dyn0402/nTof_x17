#!/bin/bash
# Condor executable for one two-track validation job.
#   run_two_track_wrapper.sh <kind> <arm> <tag> <outname> [intra_bench args...]
# Pushes <outname>.tar.gz to EOS from the job (the access point has no EOS
# mount; see ../EOS_WRITE_TEST.md) and hands condor back only a done marker.
set -e
OUTNAME=$4

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

tar xzf code.tar.gz          # -> code/
export PYTHONPATH=$PWD/code:$PYTHONPATH

python3 run_two_track_job.py "$@"

EOSOUT=${EOS_TT_OUT:-/eos/user/d/dneff/x17/two_track_limit/results}
XH=root://eosuser.cern.ch/
xrdcp -f "${OUTNAME}.tar.gz" "$XH$EOSOUT/${OUTNAME}.tar.gz" \
  || { echo "[wrapper] FATAL: xrdcp of ${OUTNAME}.tar.gz failed"; exit 5; }
{ echo "outname=$OUTNAME"
  echo "bytes=$(stat -c %s "${OUTNAME}.tar.gz")"
  echo "sha256=$(sha256sum "${OUTNAME}.tar.gz" | cut -d" " -f1)"
  echo "host=$(hostname) date=$(date -Is)"
} > "done_${OUTNAME}.txt"
echo "[wrapper] pushed ${OUTNAME}.tar.gz to EOS"
