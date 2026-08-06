#!/bin/bash
# mm_gasgen_fast_wrap.sh — as mm_gasgen_wrap.sh but with an output-label override.
# Usage: mm_gasgen_fast_wrap.sh <GAS_LABEL> <PRESSURE_LABEL> <NCOLL> <OUT_LABEL>
set -e

echo "[wrap] host=$(hostname) gas=$1 pressure=$2 ncoll=$3 out=$4 start=$(date)"

source "$(dirname "${BASH_SOURCE[0]}")/setup_garfield.sh"

python3 -u mm_gasgen_one.py --gas-label "$1" --pressure-label "$2" \
        --ncoll "$3" --out-label "$4"

echo "[wrap] end=$(date)"
