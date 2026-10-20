#!/bin/bash
# Condor payload: emulated trigger candidates in one run's flash_tail products against
# the DREAM triggers recorded in its slims (trigger_cand.py), to EOS.
#   ./trig_job.sh <run>
set -eo pipefail
RUN=${1:?run}
SRC=/eos/experiment/ntof/data/x17/flash_tail/$RUN
J=/eos/experiment/ntof/data/x17/july_beam/runs
DEST=root://eosuser.cern.ch//eos/experiment/ntof/data/x17/flash_tail_trigger
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
klist -s || { echo "no Kerberos credential"; exit 3; }
slims=$(ls $J/*/*/ntof_hits/ntof_hits_*_${RUN}.root 2>/dev/null || true)
[ -n "$slims" ] || { echo "no slim for $RUN"; exit 0; }
python -m ntof_scint_recovery.trigger_cand $RUN --products $(ls $SRC/flash_*.root) --slims $slims --out trigcand_$RUN.npz
xrdcp --force --silent trigcand_$RUN.npz $DEST/trigcand_$RUN.npz
rm -f trigcand_$RUN.npz
echo "=== done $RUN"
