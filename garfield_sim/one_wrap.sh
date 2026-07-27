#!/bin/bash
set -e
echo "[wrap] host=$(hostname) mix=$1 start=$(date)"
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
python3 -u mm_one_mixture.py "$1"
echo "[wrap] end=$(date)"
