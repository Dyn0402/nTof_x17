#!/bin/bash
# condor wrapper: one Magboltz mixture at ONE field, high statistics
set -e
source ./setup_garfield.sh
python3 magboltz_drift.py "$1" "E=$2" "c=$3"
mv results/magboltz_*.json .
rm -rf garfield* results
