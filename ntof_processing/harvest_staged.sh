#!/bin/bash
# harvest_staged.sh -- verify staged runs and move them to the ntof disk.
#
#   ssh -K lxplus
#   krenew -K 60 -t -- ./harvest_staged.sh 224689 224690 ...
#
# Use when processing finished but the move did not -- e.g. the campaign driver
# died with its AFS token. Safe to re-run: a run already on the ntof disk is
# skipped, and nothing is deleted unless every file's size matches at the
# destination.
set -uo pipefail

STAGE=${X17_STAGE:-/eos/user/d/dneff/x17/reproc/prod_v12}
FINAL=${X17_FINAL:-/eos/experiment/ntof/data/x17/reproc/prod_v12}
RAW=/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement
LOG=${X17_LOG:-/afs/cern.ch/work/d/dneff/x17_reproc/harvest.log}

# Append per call so each line is flushed. A single long-lived redirection onto
# AFS loses everything if the process dies before close -- that is exactly how
# the 2026-08-10 campaign log ended up 0 bytes despite having been readable.
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ $# -ge 1 ] || { echo "usage: $0 <run> [run ...]"; exit 2; }
mkdir -p "$FINAL"

ok=0; skip=0; bad=0
for run in "$@"; do
    S=$STAGE/$run/completed/$run
    D=$FINAL/$run/completed/$run

    if [ -d "$D" ] && [ "$(ls "$D" 2>/dev/null | grep -c '\.root$')" -gt 0 ]; then
        log "$run: already on the ntof disk, skip"; skip=$((skip+1)); continue
    fi
    [ -d "$S" ] || { log "$run: nothing staged, skip"; skip=$((skip+1)); continue; }

    n_raw=$(ls "$RAW/$run/stream1" 2>/dev/null | wc -l)
    want=$(( (n_raw + 3) / 4 ))
    got=$(ls "$S" | grep -c "^run${run}_[0-9]*\.root$")
    miss=0
    for ((i=1; i<=want; i++)); do
        [ -f "$(printf "%s/run%s_%04d.root" "$S" "$run" "$i")" ] || miss=$((miss+1))
    done
    if [ "$want" -eq 0 ] || [ "$got" -ne "$want" ] || [ "$miss" -ne 0 ] || [ ! -f "$S/history_$run.root" ]; then
        log "$run: INCOMPLETE ($got/$want partials, $miss gaps) -- not moving"
        bad=$((bad+1)); continue
    fi

    log "$run: $got/$want partials contiguous, copying"
    mkdir -p "$FINAL/$run/completed"
    if ! cp -r "$S" "$FINAL/$run/completed/"; then
        log "$run: COPY FAILED -- staged copy kept"; bad=$((bad+1)); continue
    fi

    mism=0
    for f in "$S"/*.root; do
        b=$(basename "$f")
        [ "$(stat -c %s "$f")" = "$(stat -c %s "$D/$b" 2>/dev/null)" ] || { mism=$((mism+1)); log "  MISMATCH $b"; }
    done
    if [ "$mism" -ne 0 ]; then
        log "$run: $mism size mismatch(es) -- staged copy KEPT"; bad=$((bad+1)); continue
    fi

    rm -rf "${STAGE:?}/$run"
    log "$run: VERIFIED -> $D ($got partials), staging freed"
    ok=$((ok+1))
done
log "harvest done: $ok moved, $skip skipped, $bad need attention"
