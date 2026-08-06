#!/bin/bash
# mm_gasgen_quick_wrap.sh — gas table with an explicit cost budget.
# Usage: mm_gasgen_quick_wrap.sh <GAS_LABEL> <PRES> <NCOLL> <EMIN> <EMAX> <NPTS> <OUT_LABEL>
set -e

echo "[wrap] host=$(hostname) gas=$1 pres=$2 ncoll=$3 grid=$4-$5/$6 out=$7 start=$(date)"

source "$(dirname "${BASH_SOURCE[0]}")/setup_garfield.sh"

python3 -u mm_gasgen_one.py --gas-label "$1" --pressure-label "$2" \
        --ncoll "$3" --emin "$4" --emax "$5" --npts "$6" --out-label "$7"

echo "[wrap] end=$(date)"
