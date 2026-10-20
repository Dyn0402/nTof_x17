#!/bin/bash
# Drive the flash pull: disk runs at once, tape runs in recall-then-submit batches.
#
#   ./flash_campaign.sh plan              # plan.csv + runs_disk.txt + tape batches
#   ./flash_campaign.sh disk              # submit every run whose files are all on disk
#   nohup ./flash_campaign.sh tape > tape.log 2>&1 &    # resumable
#   ./flash_campaign.sh status            # products on EOS against the plan
#
# Recalled files go back to tape-only in under a day (measured 2026-08-13 for the
# waveform pull), so a batch is recalled, submitted as soon as it is online, and
# the next batch is requested while this one reads.  Each job is minutes, so a
# batch is consumed well inside the staging lifetime.  Every decision is a file
# under $WORK; nothing is inferred from a log.
set -eo pipefail

cd "$(dirname "$0")"
WORK=${X17_FLASH_WORK:-$PWD/campaign}
XRD=${X17_CTA_XRD:-root://eosctapublicdisk.cern.ch/}
DESTDIR=${X17_FLASH_DESTDIR:-/eos/experiment/ntof/data/x17/flash_tail}
BATCH=${X17_FLASH_BATCH:-40}        # runs per tape batch (~80 files, ~0.3 TB)
POLL=${X17_FLASH_POLL:-600}
MAX_WAIT=${X17_FLASH_MAX_WAIT:-86400}
USAGE='usage: flash_campaign.sh plan | disk | tape | status'
cmd=${1:?$USAGE}
mkdir -p "$WORK" logs

lcg() { source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1 || true; }

tape_paths() {   # runs file -> tape paths (no endpoint) of their tape rows
    awk -F, 'NR==FNR {want[$1]=1; next} FNR>1 && ($1 in want) && $3=="tape" {print $4}' "$1" plan.csv |
        sed "s|^$XRD||"
}

case "$cmd" in
plan)
    lcg
    python -m ntof_processing.flash_pull.plan inputs/run_inventory.csv --out plan.csv
    # a run goes to the disk list only if EVERY planned file of it is on disk
    awk -F, 'FNR>1 {n[$1]++; if ($3=="disk") d[$1]++} END {for (r in n) print r, (d[r]==n[r]) ? "disk" : "tape"}' plan.csv |
        sort -n > "$WORK/runs_source.txt"
    awk '$2=="disk" {print $1}' "$WORK/runs_source.txt" > "$WORK/runs_disk.txt"
    awk '$2=="tape" {print $1}' "$WORK/runs_source.txt" > "$WORK/runs_tape.txt"
    rm -f "$WORK"/batch_*.txt
    split -l "$BATCH" -d -a 3 --additional-suffix=.txt "$WORK/runs_tape.txt" "$WORK/batch_"
    echo "$(wc -l < "$WORK/runs_disk.txt") runs on disk, $(wc -l < "$WORK/runs_tape.txt") need tape" \
         "in $(ls "$WORK"/batch_*.txt | wc -l) batches"
    ;;
disk)
    condor_submit flash.sub -a "runs=$WORK/runs_disk.txt" | tee "$WORK/disk.submitted"
    ;;
tape)
    batches=("$WORK"/batch_*.txt)
    request() {
        [ -f "$1.requested" ] && return
        tape_paths "$1" | xargs -r -n 100 xrdfs "$XRD" prepare -s > /dev/null
        date -u > "$1.requested"; echo "[$(date -u +%T)] requested $(basename "$1")"
    }
    for i in "${!batches[@]}"; do
        b=${batches[$i]}
        [ -f "$b.submitted" ] && continue
        request "$b"
        t0=$(date +%s)
        while :; do
            want=$(tape_paths "$b" | wc -l)
            on=$(tape_paths "$b" | xargs -r -n 100 xrdfs "$XRD" query prepare 0 2>/dev/null |
                 tr -d ' \n' | grep -o '"online":true' | wc -l)
            echo "[$(date -u +%T)] $(basename "$b"): $on / $want online"
            [ "$on" -ge "$want" ] && break
            if [ $(( $(date +%s) - t0 )) -gt "$MAX_WAIT" ]; then
                echo "  giving up on $(basename "$b") after $MAX_WAIT s"; date -u > "$b.timeout"; continue 2
            fi
            sleep "$POLL"
        done
        condor_submit flash.sub -a "runs=$b" > "$b.submitted"
        cat "$b.submitted"
        nxt=${batches[$((i + 1))]:-}
        [ -n "$nxt" ] && request "$nxt"
    done
    echo "all tape batches submitted"
    ;;
status)
    planned=$(awk -F, 'FNR>1' plan.csv | wc -l)
    done_=$(find "$DESTDIR" -name 'flash_*_provenance.json' 2>/dev/null | wc -l)
    bytes=$(find "$DESTDIR" -name 'flash_*.root' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s/1e9}')
    echo "products: $done_ / $planned planned files, $bytes GB on $DESTDIR"
    condor_q -totals 2>/dev/null | tail -2
    ;;
*) echo "$USAGE"; exit 2 ;;
esac
