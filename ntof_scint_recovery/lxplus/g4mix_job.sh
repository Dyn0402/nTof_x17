#!/bin/bash
# Condor payload: Geant4 particle mix / light against time since the flash for one
# neutrons_fullrange file (g4_mix.py), to EOS.
#   ./g4mix_job.sh <job index>
set -eo pipefail
I=${1:?index}
SRC=/eos/experiment/ntof/data/x17/full_sim/neutrons_fullrange/neutron_bg_job$(printf %03d $I)_t0.root
DEST=root://eosuser.cern.ch//eos/experiment/ntof/data/x17/flash_tail_g4mix
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
klist -s || { echo "no Kerberos credential"; exit 3; }
python -m ntof_scint_recovery.g4_mix $SRC --out g4mix_$I.npz
xrdcp --force --silent g4mix_$I.npz $DEST/g4mix_$I.npz
rm -f g4mix_$I.npz
echo "=== done $I"
