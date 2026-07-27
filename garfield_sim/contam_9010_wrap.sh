#!/bin/bash
# HTCondor wrapper: source LCG_107 (Garfield++ via ROOT dictionary) and run the
# 90/10 contamination Magboltz suite. Writes the JSON into ./results, then copies
# it to the scratch root so condor's transfer_output_files can bring it back.
set -e
echo "[wrap] host=$(hostname) start=$(date)"
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
source "$LCG"
echo "[wrap] python=$(which python3)  root=$(root-config --version 2>/dev/null)"
python3 -u mm_drift_9010_contam_cern.py
cp results/drift_9010_contam_cern.json ./drift_9010_contam_cern.json
echo "[wrap] done=$(date) json_bytes=$(stat -c %s ./drift_9010_contam_cern.json)"
