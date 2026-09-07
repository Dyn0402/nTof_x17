#!/bin/bash
# Recall raw stream1 from CTA tape for the runs whose EOS disk copy has expired.
#
#   ./recall.sh request runs_to_recall.txt    # fire the staging requests
#   ./recall.sh check   runs_to_recall.txt    # how many files are online yet
#   ./recall.sh ready   runs_to_recall.txt    # print only the runs that are 100 %
#
# The recall is the slow step and the n_TOF wiki quotes up to 72 h (usually
# hours).  Files come back online progressively, so `check` is worth polling and
# the pull can start on whatever is ready.
#
# TWO TRAPS, both already paid for once by the 224526 recovery:
#
#   * StageRuns.sh finds any EOS remnant, asks "stage anyway?" and DEFAULTS TO
#     NO.  Answer yes or it silently skips the run -- that preference is exactly
#     what produced a 13 %-coverage product for 224526.
#   * `xrdfs query prepare` takes many paths in ONE call and answers instantly;
#     one call per file takes minutes, and this gets polled for hours.
set -u

XRD=${X17_CTA_XRD:-root://eosctapublicdisk.cern.ch/}
CTA=${X17_CTA_BASE:-/eos/ctapublicdisk/archive/ntof/2026/EAR2/X17_measurement}
PS=${X17_PROC_SCRIPTS:-/eos/experiment/ntof/repositories/processingscripts}
YEAR=${X17_YEAR:-2026}
AREA=${X17_AREA:-EAR2}
CAMPAIGN=${X17_CAMPAIGN:-X17_measurement}
WORK=${X17_RECALL_WORK:-$HOME/x17wf/recall}

# The usage text must not contain a closing brace: bash ends ${1:?word} at the
# FIRST unquoted `}`, so "{request|check|ready}" truncates the expansion and the
# remainder is parsed as shell syntax -- `<file of run numbers>` then becomes an
# input redirect and the script dies without firing anything.
USAGE='usage: recall.sh request|check|ready <file of run numbers>'
cmd=${1:?$USAGE}
LIST=${2:?$USAGE}
mkdir -p "$WORK"

list_cta() {   # run -> file indices on tape
    xrdfs "$XRD" ls "$CTA/$1/stream1" 2>/dev/null |
        sed -n "s|.*/run$1_\([0-9]*\)_s1\.raw\.finished$|\1|p" | sort -n
}

paths_for() {  # run -> full tape paths, one per line
    local R=$1
    list_cta "$R" | sed "s|^|$CTA/$R/stream1/run${R}_|; s|$|_s1.raw.finished|"
}

case "$cmd" in

request)
    while read -r R; do
        [ -z "$R" ] && continue
        n=$(list_cta "$R" | wc -l)
        if [ "$n" -eq 0 ]; then
            echo "$R: NOT ON TAPE -- nothing to recall, and nothing to pull"
            continue
        fi
        echo "$R: $n files on tape, requesting recall"
        printf 'y\n' | "$PS/StageRuns.sh" -y "$YEAR" -a "$AREA" \
            -c "$CAMPAIGN" -r "$R" -l "$R" || echo "  REQUEST FAILED: $R"
    done < "$LIST"
    echo "requested; poll with '$0 check $LIST'"
    ;;

check|ready)
    tot_on=0; tot_all=0
    while read -r R; do
        [ -z "$R" ] && continue
        paths_for "$R" > "$WORK/$R.list"
        all=$(wc -l < "$WORK/$R.list")
        [ "$all" -eq 0 ] && { [ "$cmd" = check ] && echo "$R: not on tape"; continue; }
        on=$(xargs -a "$WORK/$R.list" -n 50 xrdfs "$XRD" query prepare 0 2>/dev/null |
             tr -d ' \n' | grep -o '"online":true' | wc -l)
        tot_on=$((tot_on + on)); tot_all=$((tot_all + all))
        if [ "$cmd" = ready ]; then
            [ "$on" -eq "$all" ] && echo "$R"
        else
            printf '%s: %d / %d online%s\n' "$R" "$on" "$all" \
                   "$([ "$on" -eq "$all" ] && echo '  READY')"
        fi
    done < "$LIST"
    [ "$cmd" = check ] && echo "TOTAL: $tot_on / $tot_all files online"
    ;;

*)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
