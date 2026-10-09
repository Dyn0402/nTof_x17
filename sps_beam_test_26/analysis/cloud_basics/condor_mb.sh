#!/bin/bash
# condor wrapper: unpack the pinned Garfield, run one Magboltz mixture
set -e
source ./setup_garfield.sh
python3 magboltz_drift.py "$1" $2
mv results/magboltz_*.json . 
rm -rf garfield* results
