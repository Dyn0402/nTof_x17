#!/bin/bash
# Drive the flash pull: disk runs at once, tape runs in recall-then-submit batches.
#
#   ./flash_campaign.sh plan              # plan.csv + runs_disk.txt + tape batches
#   ./flash_campaign.sh disk              # submit every run whose files are all on disk
#   nohup ./flash_campaign.sh tape > tape.log 2>&1 &    # resumable (tape_submitted.txt)
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
    # Per RUN, not per batch: a run is submitted the moment all its tape files are
    # online, and the rest are recalled and polled.  A batch gate let three
    # offline files hold back 160 runs whose files were already online (CTA's disk
    # buffer still held most of the campaign on 2026-10-20).  Recall is
    # requested only for what is offline, at most BATCH runs' worth at a time, so
    # nothing sits staged for long before its job reads it.
    touch "$WORK/tape_submitted.txt"
    t0=$(date +%s)
    while :; do
        todo=$(comm -23 <(sort "$WORK/runs_tape.txt") <(sort "$WORK/tape_submitted.txt"))
        [ -z "$todo" ] && { echo "all tape runs submitted"; break; }
        ready=() waiting=()
        for r in $todo; do
            echo "$r" > "$WORK/one.txt"
            want=$(tape_paths "$WORK/one.txt" | wc -l)
            on=$(tape_paths "$WORK/one.txt" | xargs -r -n 100 xrdfs "$XRD" query prepare 0 2>/dev/null |
                 tr -d ' \n' | grep -o '"online":true' | wc -l || true)
            # `|| true`: with none online grep exits 1, pipefail carries it and
            # set -e killed the driver silently on its first offline run
            if [ "$on" -ge "$want" ]; then ready+=("$r"); else waiting+=("$r"); fi
        done
        if [ "${#ready[@]}" -gt 0 ]; then
            printf '%s\n' "${ready[@]}" > "$WORK/ready_$(date +%s).txt"
            condor_submit flash.sub -a "runs=$(ls -t "$WORK"/ready_*.txt | head -1)" | tail -1
            printf '%s\n' "${ready[@]}" >> "$WORK/tape_submitted.txt"
        fi
        echo "[$(date -u +%T)] submitted ${#ready[@]}, waiting on ${#waiting[@]} runs"
        [ "${#waiting[@]}" -eq 0 ] && continue
        printf '%s\n' "${waiting[@]:0:$BATCH}" > "$WORK/recall.txt"
        tape_paths "$WORK/recall.txt" | xargs -r -n 100 xrdfs "$XRD" prepare -s > /dev/null
        if [ $(( $(date +%s) - t0 )) -gt "$MAX_WAIT" ]; then
            echo "giving up after $MAX_WAIT s; still waiting: ${waiting[*]}"; exit 1
        fi
        sleep "$POLL"
    done
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
