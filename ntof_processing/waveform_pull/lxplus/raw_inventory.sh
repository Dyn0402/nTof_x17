#!/bin/bash
# Where is each run's raw stream1: EOS disk, CTA tape, or nowhere?
#
# The EOS disk staging area holds raw for ~2 WEEKS after the run; after that only
# the tape copy survives and reading it needs a recall.  So this answer changes
# every day and must be re-run, not remembered.
#
#   ./raw_inventory.sh runs.txt > inventory.csv
#
# Columns: run, files on disk, files on tape, tape bytes.  disk_files == 0 means
# a recall is needed; disk_files < cta_files means the disk copy is expiring and
# is ALREADY partial -- treat it as gone, because a short file list is exactly
# how 224526 came to be processed at 13 % coverage.
set -u

XRD=${X17_CTA_XRD:-root://eosctapublicdisk.cern.ch/}
CTA=${X17_CTA_BASE:-/eos/ctapublicdisk/archive/ntof/2026/EAR2/X17_measurement}
DISK=${X17_NTOF_RAW:-/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement}
LIST=${1:?usage: raw_inventory.sh <file of run numbers>}

echo "run,disk_files,cta_files,cta_bytes"
while read -r R; do
    [ -z "$R" ] && continue
    d=$(ls "$DISK/$R/stream1" 2>/dev/null | grep -c 's1\.raw')
    # $4 is the size in `xrdfs ls -l` output.  $5 is the DATE, and summing that
    # instead gives a plausible-looking number that is pure nonsense.
    read -r c b < <(xrdfs "$XRD" ls -l "$CTA/$R/stream1" 2>/dev/null |
        awk '/_s1\.raw/ {n++; s+=$4} END {print n+0, s+0}')
    echo "$R,$d,$c,$b"
done < "$LIST"
