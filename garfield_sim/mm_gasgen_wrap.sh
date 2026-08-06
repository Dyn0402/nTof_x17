#!/bin/bash
# mm_gasgen_wrap.sh — HTCondor wrapper for mm_gasgen_one.py
# Usage (from the JDL): mm_gasgen_wrap.sh <GAS_LABEL> <PRESSURE_LABEL> [NCOLL]
set -e

echo "[wrap] host=$(hostname) gas=$1 pressure=$2 ncoll=${3:-10} start=$(date)"

source "$(dirname "${BASH_SOURCE[0]}")/setup_garfield.sh"

python3 -u mm_gasgen_one.py --gas-label "$1" --pressure-label "$2" --ncoll "${3:-10}"

echo "[wrap] end=$(date)"
