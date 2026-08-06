#!/bin/bash
# HTCondor wrapper: set up the pinned Garfield++ (setup_garfield.sh) and run the
# 90/10 contamination Magboltz suite. Writes the JSON into ./results, then copies
# it to the scratch root so condor's transfer_output_files can bring it back.
set -e
echo "[wrap] host=$(hostname) start=$(date)"
source "$(dirname "${BASH_SOURCE[0]}")/setup_garfield.sh"
python3 -u mm_drift_9010_contam_cern.py
cp results/drift_9010_contam_cern.json ./drift_9010_contam_cern.json
echo "[wrap] done=$(date) json_bytes=$(stat -c %s ./drift_9010_contam_cern.json)"
