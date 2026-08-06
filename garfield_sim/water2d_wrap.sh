#!/bin/bash
set -e
echo "[wrap] host=$(hostname) tag=$1 start=$(date)"
source "$(dirname "${BASH_SOURCE[0]}")/setup_garfield.sh"
python3 -u mm_water2d_one.py "$1"
echo "[wrap] end=$(date)"
