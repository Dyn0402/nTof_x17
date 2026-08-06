#!/bin/bash
# mm_gasgen_fast_wrap.sh — as mm_gasgen_wrap.sh but with an output-label override.
# Usage: mm_gasgen_fast_wrap.sh <GAS_LABEL> <PRESSURE_LABEL> <NCOLL> <OUT_LABEL>
set -e

echo "[wrap] host=$(hostname) gas=$1 pressure=$2 ncoll=$3 out=$4 start=$(date)"

V=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt
source "$V/setup.sh"
source "$V/share/Garfield/setupGarfield.sh"
echo "[wrap] ROOT $(root-config --version)  python $(python3 --version)"

python3 -u mm_gasgen_one.py --gas-label "$1" --pressure-label "$2" \
        --ncoll "$3" --out-label "$4"

echo "[wrap] end=$(date)"
