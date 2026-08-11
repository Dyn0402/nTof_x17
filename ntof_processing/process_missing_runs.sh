#!/bin/bash
# process_missing_runs.sh -- process n_TOF runs ourselves, landing them on the
# ntof disk, in ONE pass over any number of runs.
#
#   ssh -K lxplus                          # MANDATORY: AFS token + condor auth
#   nohup setsid krenew -K 60 -t -- ./process_missing_runs.sh 224695 ... &
#
# RUN IT UNDER `krenew`. The AFS token inherited from the login session expires
# (~24 h, and less if the session is older than it looks), and when it does the
# driver loses /afs and dies mid-campaign with its condor submissions
# half-done -- which is what happened on 2026-08-10. `krenew -K 60 -t` renews
# the ticket and re-runs aklog every hour for as long as the ticket is
# renewable (`klist` shows "renew until"), which covers any realistic campaign.
#
# Rolling pipeline: keep at most MAX_INFLIGHT runs staged at once, and move each
# run to the ntof disk the moment its partials verify. Staging therefore never
# exceeds ~MAX_INFLIGHT x 35 GB regardless of how many runs you pass, so the
# 2 TB user quota stops being a reason to split the campaign into batches.
#
# Why a staging hop at all. `ProcessFileList.sh` is a compiled binary whose -o
# validation accepts exactly two prefixes, `/eos/user/` and `/eos/project-`;
# `/eos/experiment/...` is rejected with "Output path ... is not supported at
# the moment!" and every job exits 255. The DESTINATION has no such limit --
# /eos/experiment/ntof/data/x17/... is writable and its quota is unenforced.
#
# The merge node ALWAYS fails (condor's 1024 MB transfer cap + a 3 MB disk
# request against ~58 GB of use). Expected and harmless: we read the partials.
# See SELF_PROCESSING_RUNBOOK.md.
set -uo pipefail

W=/afs/cern.ch/work/d/dneff/x17_reproc
UI=$W/userinputs/v12_liqpileup/UserInput.h
STAGE=/eos/user/d/dneff/x17/reproc/prod_v12
FINAL=${X17_FINAL:-/eos/experiment/ntof/data/x17/reproc/prod_v12}
RAW=/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement
RP=/eos/experiment/ntof/repositories/processingscripts/RunProcessing.sh

MAX_INFLIGHT=${MAX_INFLIGHT:-6}      # ~210 GB staged at 35 GB/run
POLL=${POLL:-180}
STALL_S=${STALL_S:-5400}             # no new partial for this long -> give up on it

[ $# -ge 1 ] || { echo "usage: $0 <run> [run ...]"; exit 2; }
[ -f "$UI" ] || { echo "UserInput not staged: $UI"; exit 2; }
mkdir -p "$FINAL"

# Append per call so every line is flushed to AFS immediately. A single
# long-lived `> log` redirection onto AFS buffers in the local cache and loses
# EVERYTHING if the process dies before closing the file -- that is exactly how
# the first run of this campaign left a 0-byte log that had been readable while
# it ran. Do not "simplify" this back to plain echo.
LOG=${X17_LOG:-/afs/cern.ch/work/d/dneff/x17_reproc/campaign.log}
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Liveness via a PID FILE, not `pgrep -f process_missing_runs.sh`. Any monitoring
# command that merely MENTIONS the script name matches its own pgrep pattern, so
# the pattern check reports a dead driver as alive -- which is exactly what
# masked this driver's death for two hours on 2026-08-11. Check with:
#   kill -0 "$(cat /afs/cern.ch/work/d/dneff/x17_reproc/campaign.pid)" 2>/dev/null
PIDFILE=${X17_PIDFILE:-/afs/cern.ch/work/d/dneff/x17_reproc/campaign.pid}
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
want_parts() { local n; n=$(ls "$RAW/$1/stream1" 2>/dev/null | wc -l); echo $(( (n + 3) / 4 )); }

# complete = every index 1..N present, plus the history file
is_complete() {
    local run=$1 want=$2 d=$STAGE/$run/completed/$run i
    [ -f "$d/history_$run.root" ] || return 1
    [ "$want" -gt 0 ] || return 1
    for ((i=1; i<=want; i++)); do
        [ -f "$(printf "%s/run%s_%04d.root" "$d" "$run" "$i")" ] || return 1
    done
    return 0
}

submit() {
    local run=$1 A=$W/aux_prod_$1
    rm -rf "$A"; mkdir -p "$A"; cd "$A" || return 1
    mkdir -p "$STAGE/$run"
    "$RP" -y 2026 -a EAR2 -c X17_measurement -r "$run" -p "$UI" -o "$STAGE/$run" \
        > "$A/runproc.log" 2>&1
    grep -q "All jobs submitted" "$A/runproc.log"
}

# verify -> copy -> verify sizes -> delete source. Never deletes unverified.
harvest() {
    local run=$1 S=$STAGE/$run/completed/$run bad=0 f b
    mkdir -p "$FINAL/$run/completed"
    cp -r "$S" "$FINAL/$run/completed/" || { log "$run: COPY FAILED, source kept"; return 1; }
    for f in "$S"/*.root; do
        b=$(basename "$f")
        [ "$(stat -c %s "$f")" = "$(stat -c %s "$FINAL/$run/completed/$run/$b" 2>/dev/null)" ] || bad=$((bad+1))
    done
    [ "$bad" -eq 0 ] || { log "$run: $bad size mismatch(es), source kept"; return 1; }
    rm -rf "${STAGE:?}/$run"
    return 0
}

# ---------------------------------------------------------------- the pipeline
declare -a QUEUE=() ; declare -A WANT=() LASTN=() LASTT=()
for run in "$@"; do
    if [ -d "$FINAL/$run/completed/$run" ]; then log "$run: already on the ntof disk, skip"; continue; fi
    w=$(want_parts "$run")
    [ "$w" -gt 0 ] || { log "$run: no raw staged, skip"; continue; }
    QUEUE+=("$run"); WANT[$run]=$w
done
log "${#QUEUE[@]} run(s) to process, max $MAX_INFLIGHT in flight, dest $FINAL"

declare -a INFLIGHT=()
done_n=0 fail_n=0
while [ ${#QUEUE[@]} -gt 0 ] || [ ${#INFLIGHT[@]} -gt 0 ]; do
    # top up
    while [ ${#INFLIGHT[@]} -lt "$MAX_INFLIGHT" ] && [ ${#QUEUE[@]} -gt 0 ]; do
        run=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
        if submit "$run"; then
            INFLIGHT+=("$run"); LASTN[$run]=0; LASTT[$run]=$SECONDS
            log "$run: submitted, ${WANT[$run]} jobs expected"
        else
            log "$run: SUBMIT FAILED"; fail_n=$((fail_n+1))
        fi
    done

    sleep "$POLL"

    declare -a STILL=()
    for run in "${INFLIGHT[@]}"; do
        n=$(ls "$STAGE/$run/completed/$run" 2>/dev/null | grep -c "^run${run}_[0-9]*\.root$")
        if [ "$n" -ne "${LASTN[$run]}" ]; then LASTN[$run]=$n; LASTT[$run]=$SECONDS; fi
        if is_complete "$run" "${WANT[$run]}"; then
            log "$run: ${WANT[$run]}/${WANT[$run]} partials, moving"
            if harvest "$run"; then
                done_n=$((done_n+1)); log "$run: DONE -> $FINAL/$run/completed/$run"
            else
                fail_n=$((fail_n+1))
            fi
        elif [ $(( SECONDS - LASTT[$run] )) -gt "$STALL_S" ]; then
            log "$run: STALLED at $n/${WANT[$run]} partials, leaving staged for inspection"
            fail_n=$((fail_n+1))
        else
            STILL+=("$run")
        fi
    done
    INFLIGHT=("${STILL[@]+"${STILL[@]}"}")
    log "progress: done $done_n, failed $fail_n, in flight ${#INFLIGHT[@]}, queued ${#QUEUE[@]}"
done
log "campaign complete: $done_n moved, $fail_n need attention"
