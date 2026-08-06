#!/bin/bash
set -e
echo "[wrap] host=$(hostname) mix=$1 start=$(date)"
source "$(dirname "${BASH_SOURCE[0]}")/setup_garfield.sh"
python3 -u mm_one_mixture.py "$1"
echo "[wrap] end=$(date)"
