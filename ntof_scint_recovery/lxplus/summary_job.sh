#!/bin/bash
# Condor payload: summarise one run's flash_pull products into a small npz on EOS.
#   ./summary_job.sh <run>
set -eo pipefail
RUN=${1:?run}
SRC=/eos/experiment/ntof/data/x17/flash_tail/$RUN
DEST=root://eosuser.cern.ch//eos/experiment/ntof/data/x17/flash_tail_summary
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
klist -s || { echo "no Kerberos credential"; exit 3; }
files=$(ls $SRC/flash_*.root)
python -m ntof_scint_recovery.flash_summary $files --out summary_$RUN.npz
xrdcp --force --silent summary_$RUN.npz $DEST/summary_$RUN.npz
rm -f summary_$RUN.npz
echo "=== done $RUN"
