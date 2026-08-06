#!/bin/bash
# mm_gasgen_quick_wrap.sh — gas table with an explicit cost budget.
# Usage: mm_gasgen_quick_wrap.sh <GAS_LABEL> <PRES> <NCOLL> <EMIN> <EMAX> <NPTS> <OUT_LABEL>
set -e

echo "[wrap] host=$(hostname) gas=$1 pres=$2 ncoll=$3 grid=$4-$5/$6 out=$7 start=$(date)"

V=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt
source "$V/setup.sh"
source "$V/share/Garfield/setupGarfield.sh"
echo "[wrap] ROOT $(root-config --version)  python $(python3 --version)"

python3 -u mm_gasgen_one.py --gas-label "$1" --pressure-label "$2" \
        --ncoll "$3" --emin "$4" --emax "$5" --npts "$6" --out-label "$7"

echo "[wrap] end=$(date)"
