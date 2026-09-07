#!/bin/bash
# Drive the whole waveform pull as a pipeline of recall-then-read batches.
#
#   ./campaign.sh plan   <runs.txt>          # classify runs, write the batches
#   ./campaign.sh run    [batch_dir]         # execute; resumable, run under nohup
#   ./campaign.sh status [batch_dir]         # where it got to
#
# WHY THIS EXISTS, rather than "recall everything, then submit everything":
#
#   A CTA recall does not stay staged.  Measured 2026-08-13: a recall verified
#   at 3187/3187 files online was back to `online:false` on 26 of 27 runs inside
#   a day, holding no pin (`requested:false`), and two jobs that reached the
#   head of the queue after it lapsed died on `[3005] no disk replica exists`.
#   Recalled data is a perishable good.  It must be CONSUMED, not banked.
#
#   So the unit of work is a batch small enough to recall and read inside one
#   staging lifetime, and batch N+1 is requested while batch N is being read --
#   the tape system does the slow thing in parallel with the farm doing the slow
#   thing, and nothing sits staged waiting.
#
# Measured inputs to the batch sizing (2026-08-13 fleet, 41 jobs):
#   read     median 82 min per run, max 137 min, ~2.7 GB per segment out
#   staging  132-199 MB/s per job from CTA, ~350 GB per run
#   recall   9.5 TB / 27 runs completed inside one night
#
# Every decision this script makes is written to $WORK as a file.  Nothing about
# the campaign's state is inferred from the tail of a log.
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${X17_WF_CAMPAIGN:-$HOME/x17wf/campaign}
SUB=${X17_WF_SUBMIT:-$HOME/x17wf}
XRD=${X17_CTA_XRD:-root://eosctapublicdisk.cern.ch/}
CTA=${X17_CTA_BASE:-/eos/ctapublicdisk/archive/ntof/2026/EAR2/X17_measurement}
DISK=${X17_NTOF_RAW:-/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement}
SLIM=${X17_SLIM_BASE:-/eos/experiment/ntof/data/x17/wf_pull/slim_input}
SCHEDD=${X17_SCHEDD:-}

BATCH=${X17_WF_BATCH:-8}          # runs per tape batch
POLL=${X17_WF_POLL:-600}          # s between staging polls
MAX_STAGE_WAIT=${X17_WF_MAX_STAGE_WAIT:-43200}   # 12 h, then give up on a batch

USAGE='usage: campaign.sh plan <runs.txt> | run [batch_dir] | status [batch_dir]'
cmd=${1:?$USAGE}

cq() { if [ -n "$SCHEDD" ]; then condor_q -name "$SCHEDD" "$@"; else condor_q "$@"; fi; }

# ---------------------------------------------------------------------- plan
# A run is worth submitting only if it has a slim to build windows from, and the
# raw source decides how it is scheduled: disk runs can go at any time, tape
# runs must be paced against the staging lifetime.
plan() {
    local LIST=${1:?$USAGE}
    mkdir -p "$WORK"
    : > "$WORK/disk.txt"; : > "$WORK/tape.txt"; : > "$WORK/noslim.txt"
    : > "$WORK/missing.txt"
    while read -r R; do
        [ -z "$R" ] && continue
        case "$R" in \#*) continue ;; esac
        local ns nd nc
        ns=$(find "$SLIM" -name "ntof_hits_*_${R}.root" 2>/dev/null | wc -l)
        if [ "$ns" -eq 0 ]; then echo "$R" >> "$WORK/noslim.txt"; continue; fi
        nd=$(ls "$DISK/$R/stream1" 2>/dev/null | grep -c '_s1\.raw' || true)
        nc=$(xrdfs "$XRD" ls "$CTA/$R/stream1" 2>/dev/null |
             grep -c '_s1\.raw\.finished$' || true)
        if   [ "$nd" -gt 0 ] && [ "$nd" -ge "$nc" ]; then echo "$R" >> "$WORK/disk.txt"
        elif [ "$nc" -gt 0 ]; then                        echo "$R" >> "$WORK/tape.txt"
        else echo "$R" >> "$WORK/missing.txt"; fi
        printf '%s slim=%s disk=%s tape=%s\n' "$R" "$ns" "$nd" "$nc" >> "$WORK/inventory.txt"
    done < "$LIST"

    rm -f "$WORK"/batch_*.txt
    split -l "$BATCH" -d -a 3 --additional-suffix=.txt "$WORK/tape.txt" "$WORK/batch_"
    printf 'plan written to %s\n' "$WORK"
    printf '  %5s runs on disk      -> batch_disk (submit any time)\n' "$(wc -l < "$WORK/disk.txt")"
    printf '  %5s runs on tape      -> %s batches of %s\n' \
        "$(wc -l < "$WORK/tape.txt")" "$(ls "$WORK"/batch_*.txt 2>/dev/null | wc -l)" "$BATCH"
    printf '  %5s runs with NO SLIM  (skipped -- nothing to build a window from)\n' \
        "$(wc -l < "$WORK/noslim.txt")"
    printf '  %5s runs with no raw anywhere\n' "$(wc -l < "$WORK/missing.txt")"
    cp "$WORK/disk.txt" "$WORK/batch_disk.txt"
}

# --------------------------------------------------------------------- stage
# Returns 0 once every file of every run in the batch is online, 1 on timeout.
wait_staged() {
    local B=$1 t0=$SECONDS
    while :; do
        local ready total
        ready=$("$HERE/recall.sh" ready "$B" 2>/dev/null | wc -l)
        total=$(grep -cve '^\s*$' "$B")
        printf '[%s] %s: %s/%s runs fully staged\n' \
            "$(date -u +%H:%M:%S)" "$(basename "$B")" "$ready" "$total"
        [ "$ready" -eq "$total" ] && return 0
        if [ $((SECONDS - t0)) -gt "$MAX_STAGE_WAIT" ]; then
            echo "TIMEOUT staging $(basename "$B") after $MAX_STAGE_WAIT s"
            "$HERE/recall.sh" ready "$B" > "$B.ready" 2>/dev/null || true
            return 1
        fi
        sleep "$POLL"
    done
}

submit() {   # submit one list; echo the cluster id
    local B=$1 out
    ( cd "$SUB" && out=$(condor_submit pull.sub -a "runs=$B" 2>&1) &&
      echo "$out" | sed -n 's/.*submitted to cluster \([0-9]*\).*/\1/p' )
}

wait_drain() {   # block until the cluster leaves the queue
    local CL=$1
    while :; do
        local n
        n=$(cq "$CL" -af ClusterId 2>/dev/null | wc -l)
        # An empty condor_q is NOT proof of completion -- it is also what a
        # dead or wrong schedd returns. Only a query that DEMONSTRABLY reached a
        # schedd is allowed to end the wait.
        if ! cq -totals >/dev/null 2>&1; then
            echo "  (schedd unreachable, not treating as drained)"; sleep "$POLL"; continue
        fi
        [ "$n" -eq 0 ] && return 0
        printf '[%s] cluster %s: %s jobs left\n' "$(date -u +%H:%M:%S)" "$CL" "$n"
        sleep "$POLL"
    done
}

# ----------------------------------------------------------------------- run
run() {
    [ -d "$WORK" ] || { echo "no plan at $WORK -- run 'campaign.sh plan' first"; exit 1; }
    local batches=("$WORK"/batch_*.txt) done_f="$WORK/completed.txt"
    touch "$done_f"

    for i in "${!batches[@]}"; do
        local B=${batches[$i]} name; name=$(basename "$B")
        grep -qx "$name" "$done_f" && { echo "skip $name (done)"; continue; }
        echo "=== $name  $(date -u)"

        if [ "$name" != batch_disk.txt ]; then
            # Request THIS batch if nothing has been requested for it yet, then
            # wait. The next batch is requested below, while this one reads.
            [ -f "$B.requested" ] || {
                "$HERE/recall.sh" request "$B" && touch "$B.requested"; }
            wait_staged "$B" || { echo "$name STAGING FAILED" >> "$WORK/failed.txt"; continue; }
        fi

        local CL; CL=$(submit "$B")
        [ -n "$CL" ] || { echo "$name SUBMIT FAILED" >> "$WORK/failed.txt"; continue; }
        echo "$name cluster $CL" >> "$WORK/submitted.txt"
        echo "  submitted cluster $CL"

        # Fire the NEXT batch's recall now, so the tape system works while the
        # farm works. This is the whole point of the pipeline: a batch should
        # come online at about the moment the farm is free to read it, never
        # hours before.
        local N=${batches[$((i + 1))]:-}
        if [ -n "$N" ] && [ "$(basename "$N")" != batch_disk.txt ] && [ ! -f "$N.requested" ]; then
            echo "  pre-requesting $(basename "$N")"
            "$HERE/recall.sh" request "$N" && touch "$N.requested" || true
        fi

        wait_drain "$CL"
        echo "$name" >> "$done_f"
        python -m ntof_processing.waveform_pull.fleet_report \
            --out "$WORK/report_$name.txt" || true
        echo "=== $name drained $(date -u)"
    done
    echo "campaign complete; $(wc -l < "$done_f") batches done"
}

status() {
    [ -d "$WORK" ] || { echo "no campaign at $WORK"; exit 1; }
    echo "campaign work dir: $WORK"
    for f in disk tape noslim missing; do
        [ -f "$WORK/$f.txt" ] && printf '  %-8s %s runs\n' "$f" "$(wc -l < "$WORK/$f.txt")"
    done
    echo "  batches: $(ls "$WORK"/batch_*.txt 2>/dev/null | wc -l), completed: $(wc -l < "$WORK/completed.txt" 2>/dev/null || echo 0)"
    [ -s "$WORK/failed.txt" ] && { echo "  FAILURES:"; sed 's/^/    /' "$WORK/failed.txt"; }
    [ -s "$WORK/submitted.txt" ] && { echo "  clusters:"; sed 's/^/    /' "$WORK/submitted.txt"; }
    cq -totals 2>/dev/null | tail -2 || echo "  (schedd unreachable)"
}

case "$cmd" in
    plan)   plan "${2:?$USAGE}" ;;
    run)    WORK=${2:-$WORK} run ;;
    status) WORK=${2:-$WORK} status ;;
    *)      echo "$USAGE"; exit 1 ;;
esac
