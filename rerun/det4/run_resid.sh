#!/bin/bash
# run_resid.sh <tag> <cache> <bundle> [n]: resid_map on the laptop's vladder caches -> ~/x17/cosmic_bench/det4dig/
O=~/x17/cosmic_bench/det4dig; mkdir -p $O
$(dirname $0)/../../.venv/bin/python $(dirname $0)/resid_map.py --cache $2 --bundle $3 --n ${4:-400} --out $O/rm_$1.npz
