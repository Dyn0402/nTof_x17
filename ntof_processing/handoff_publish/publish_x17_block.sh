#!/bin/bash
# publish_x17_block.sh -- copy the X17 runs we processed into official/completed/.
#
# For the n_TOF processing account (ntofpro). Run it from lxplus; it only needs
# the two EOS paths below and no AFS.
#
#   ./publish_x17_block.sh --dry-run     # show exactly what it would do
#   ./publish_x17_block.sh --go          # do it
#   ./publish_x17_block.sh --verify      # re-check what is already published
#
# Optionally give run numbers to restrict it:
#   ./publish_x17_block.sh --go 224688 224690
#
# WHAT THIS IS
#   n_TOF's 2026 X17 EAR2 pass stopped at run 224687 (last output 08-07 19:56).
#   The X17 group processed the remaining block themselves with the SAME
#   configuration -- `UserInput_2026_EAR2_X17_v4.h`; every parameter line and all
#   26 pulse-shape templates are byte-identical, and on runs that exist in both
#   processings the output is identical hit for hit, on all 22 per-hit columns.
#   This script copies that output into the official area so it is where everyone
#   expects to find it.
#
# WHAT IT WILL NOT DO
#   * It never overwrites. A destination directory that already holds partials is
#     left completely alone -- if you have since processed a run yourselves,
#     yours wins and the script says so.
#   * It never deletes anything, anywhere.
#   * It refuses to copy a run whose source is incomplete.
#   * It is resumable: a file already at the destination with the right size is
#     skipped, so re-running after an interruption costs nothing.
#
# CONTACT  Dylan Neff <dneff@cern.ch>, X17 / DREAM group.
set -uo pipefail

SRC=${X17_SRC:-/eos/experiment/ntof/data/x17/reproc/prod_v12}
DST=${X17_DST:-/eos/experiment/ntof/processing/official/completed}
RAW=${X17_RAW:-/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement}
LOG=${X17_LOG:-./publish_x17_block.log}
FILES_PER_JOB=4          # the split RunProcessing.sh used for this block

# The block, minus 224709 -- that one finished later and has its own script,
# publish_224709.sh. 224576 is deliberately NOT here: we hold it only in an
# older variant (v11_pssfit_width, which differs from v4 on the four LIQ rows),
# and you are reprocessing it yourselves anyway.
RUNS_DEFAULT=(224688 224689 224690 224691 224692 224693 224694 224695 224696
              224697 224698 224699 224700 224701 224702 224703 224704 224705
              224706 224707 224708 224710 224711 224712 224713 224714 224715
              224716 224717 224718)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

MODE=""
declare -a RUNS=()
for a in "$@"; do
    case "$a" in
        --dry-run|-n) MODE=dry ;;
        --go)         MODE=go ;;
        --verify)     MODE=verify ;;
        -h|--help)    sed -n '2,45p' "$0"; exit 0 ;;
        [0-9]*)       RUNS+=("$a") ;;
        *)            echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
[ -n "$MODE" ] || { echo "give one of --dry-run, --go, --verify (see -h)" >&2; exit 2; }
[ ${#RUNS[@]} -gt 0 ] || RUNS=("${RUNS_DEFAULT[@]}")

# ---------------------------------------------------------------- source check
# Complete means: history present, partials contiguous 1..N, and N equal to
# ceil(raw stream1 files / 4). Where the raw has aged off disk the count test is
# skipped and contiguity is all that is checked.
src_ok() {
    local run=$1 d=$SRC/$run/completed/$run
    [ -d "$d" ] || { echo "no source directory"; return 1; }
    [ -f "$d/history_$run.root" ] || { echo "history_$run.root missing"; return 1; }
    local got i
    got=$(ls "$d" 2>/dev/null | grep -c "^run${run}_[0-9]*\.root$")
    [ "$got" -gt 0 ] || { echo "no partials"; return 1; }
    for ((i = 1; i <= got; i++)); do
        [ -f "$(printf '%s/run%s_%04d.root' "$d" "$run" "$i")" ] || {
            echo "partial $(printf '%04d' "$i") missing (have $got)"; return 1; }
    done
    local n_raw want
    n_raw=$(ls "$RAW/$run/stream1" 2>/dev/null | wc -l)
    if [ "$n_raw" -gt 0 ]; then
        want=$(( (n_raw + FILES_PER_JOB - 1) / FILES_PER_JOB ))
        [ "$got" -eq "$want" ] || { echo "$got partials, expected $want from $n_raw raw"; return 1; }
    fi
    echo "$got"
    return 0
}

# ---------------------------------------------------------------- the work
ok=0; skipped=0; failed=0; copied_files=0; copied_bytes=0
for run in "${RUNS[@]}"; do
    S=$SRC/$run/completed/$run
    D=$DST/$run

    detail=$(src_ok "$run")
    if [ $? -ne 0 ]; then
        log "$run: SOURCE NOT USABLE -- $detail; skipping"
        failed=$((failed + 1)); continue
    fi
    n_parts=$detail

    if [ "$MODE" != verify ] && [ -d "$D" ] && \
       [ "$(ls "$D" 2>/dev/null | grep -c "^run${run}_[0-9]*\.root$")" -gt 0 ]; then
        log "$run: destination already has partials -- leaving it alone"
        skipped=$((skipped + 1)); continue
    fi

    if [ "$MODE" = dry ]; then
        sz=$(ls -l "$S" | awk '{n += $5} END {printf "%.1f", n/1073741824}')
        log "$run: WOULD COPY $n_parts partials + history, ${sz} GB -> $D"
        ok=$((ok + 1)); continue
    fi

    [ "$MODE" = go ] && mkdir -p "$D"
    bad=0
    for f in "$S"/history_"$run".root "$S"/run"$run"_[0-9]*.root; do
        b=$(basename "$f")
        s_size=$(stat -c %s "$f" 2>/dev/null) || { bad=$((bad + 1)); continue; }
        d_size=$(stat -c %s "$D/$b" 2>/dev/null || echo -1)
        if [ "$s_size" = "$d_size" ]; then
            continue                     # already there and the right size
        fi
        if [ "$MODE" = verify ]; then
            log "  $run/$b: MISSING or wrong size at destination ($d_size vs $s_size)"
            bad=$((bad + 1)); continue
        fi
        if ! cp "$f" "$D/$b"; then
            log "  $run/$b: COPY FAILED"; bad=$((bad + 1)); continue
        fi
        d_size=$(stat -c %s "$D/$b" 2>/dev/null || echo -1)
        if [ "$s_size" != "$d_size" ]; then
            log "  $run/$b: SIZE MISMATCH after copy ($d_size vs $s_size)"
            bad=$((bad + 1)); continue
        fi
        copied_files=$((copied_files + 1))
        copied_bytes=$((copied_bytes + s_size))
    done

    if [ "$bad" -eq 0 ]; then
        log "$run: OK -- $n_parts partials + history verified at $D"
        ok=$((ok + 1))
    else
        log "$run: $bad FILE(S) BAD -- re-run this script to retry just those"
        failed=$((failed + 1))
    fi
done

log "----"
case "$MODE" in
    dry)    log "dry run: $ok run(s) would be copied, $failed unusable at source" ;;
    verify) log "verify: $ok run(s) complete at destination, $failed with problems" ;;
    go)     log "done: $ok run(s) OK, $skipped left alone, $failed with problems; \
copied $copied_files file(s), $((copied_bytes / 1073741824)) GB" ;;
esac
[ "$failed" -eq 0 ]
