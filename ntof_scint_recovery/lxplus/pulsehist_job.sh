#!/bin/bash
# Condor payload: summarise one run: pulse histograms (log t x amp, log t x width) per channel and class, to EOS.
#   ./summary_job.sh <run>
set -eo pipefail
RUN=${1:?run}
SRC=/eos/experiment/ntof/data/x17/flash_tail/$RUN
DEST=root://eosuser.cern.ch//eos/experiment/ntof/data/x17/flash_tail_pulsehist_v3
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
klist -s || { echo "no Kerberos credential"; exit 3; }
files=$(ls $SRC/flash_*.root)
python -m ntof_scint_recovery.pulses $files --hist --out pulsehist_$RUN.npz
xrdcp --force --silent pulsehist_$RUN.npz $DEST/pulsehist_$RUN.npz
rm -f pulsehist_$RUN.npz
echo "=== done $RUN"
