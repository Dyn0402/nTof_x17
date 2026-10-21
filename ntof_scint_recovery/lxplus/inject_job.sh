#!/bin/bash
# Condor payload: pulse injection into the flash tails of one run (inject.py, the first
# flash_tail file of the run, every bunch), to EOS.
#   ./inject_job.sh <run>
set -eo pipefail
RUN=${1:?run}
SRC=/eos/experiment/ntof/data/x17/flash_tail/$RUN
DEST=root://eosuser.cern.ch//eos/experiment/ntof/data/x17/flash_tail_inject
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
klist -s || { echo "no Kerberos credential"; exit 3; }
f=$(ls $SRC/flash_*.root | head -1)
python -m ntof_scint_recovery.inject $f --seed $RUN --out inject_$RUN.npz
xrdcp --force --silent inject_$RUN.npz $DEST/inject_$RUN.npz
rm -f inject_$RUN.npz
echo "=== done $RUN"
