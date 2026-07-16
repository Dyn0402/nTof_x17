#!/bin/bash
# Local read pass on a run file: extract the binary hit cache if missing, then
# run the 6 read scripts in dependency order (01 -> 02 -> 03 -> 06 -> 07 -> 09).
# Usage: ./run_readpass.sh ~/x17/beam_july/data/runNNNNNN.root
set -e
cd "$(dirname "$0")"
RUN=${1:?usage: run_readpass.sh <run.root>}
PY=../.venv/bin/python

STEM=$(basename "$RUN" .root)
if [ ! -f "$(dirname "$RUN")/hitcache/$STEM/meta.json" ]; then
    echo "== extracting hit cache =="
    [ -x fastread/extract_hits ] || make -C fastread
    fastread/extract_hits "$RUN"
fi

total0=$(date +%s)
for s in 01_signal_qa 02_coincidence_scan 03_time_offsets \
         06_wall_geometry_test 07_mip_amplitude 09_late_inclusive; do
    echo "== $s =="
    t0=$(date +%s)
    $PY $s.py "$RUN"
    echo "== $s: $(($(date +%s) - t0)) s =="
done
echo "read pass total: $(($(date +%s) - total0)) s"
