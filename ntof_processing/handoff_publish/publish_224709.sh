#!/bin/bash
# publish_224709.sh -- publish run 224709 once it has finished on our side.
#
# 224709 is the one run of the X17 block 224688-224718 that was not ready when
# the rest were handed over: it is the largest run of the block (344 raw files,
# 86 jobs) and its last job was evicted and retried. Everything else is already
# in official/completed/ via publish_x17_block.sh.
#
#   ./publish_224709.sh              # check whether it is ready, publish if so
#   ./publish_224709.sh --wait       # poll every 10 min until ready, then publish
#   ./publish_224709.sh --dry-run    # say what it would do, copy nothing
#
# It refuses to publish a partial set, so running it early is harmless -- it
# just tells you the run is not ready yet.
#
# CONTACT  Dylan Neff <dneff@cern.ch>, X17 / DREAM group.
set -uo pipefail

RUN=224709
HERE=$(cd "$(dirname "$0")" && pwd)
MAIN=$HERE/publish_x17_block.sh
SRC=${X17_SRC:-/eos/experiment/ntof/data/x17/reproc/prod_v12}
RAW=${X17_RAW:-/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement}
POLL=600

[ -x "$MAIN" ] || { echo "cannot find $MAIN (keep the two scripts together)" >&2; exit 2; }

WAIT=0; ACT=--go
for a in "$@"; do
    case "$a" in
        --wait)       WAIT=1 ;;
        --dry-run|-n) ACT=--dry-run ;;
        -h|--help)    sed -n '2,17p' "$0"; exit 0 ;;
        *)            echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

want() {   # how many partials the run should have
    local n; n=$(ls "$RAW/$RUN/stream1" 2>/dev/null | wc -l)
    echo $(( (n + 3) / 4 ))
}
have() {
    ls "$SRC/$RUN/completed/$RUN" 2>/dev/null | grep -c "^run${RUN}_[0-9]*\.root$"
}

W=$(want)
while :; do
    H=$(have)
    if [ "$W" -gt 0 ] && [ "$H" -eq "$W" ]; then
        echo "$RUN: $H/$W partials present -- publishing"
        exec "$MAIN" "$ACT" "$RUN"
    fi
    if [ "$W" -eq 0 ]; then
        echo "$RUN: cannot see the raw under $RAW/$RUN/stream1;" \
             "publish by hand with: $MAIN $ACT $RUN"
        exit 1
    fi
    echo "$RUN: not ready yet -- $H of $W partials on the X17 disk"
    [ "$WAIT" -eq 1 ] || exit 1
    sleep "$POLL"
done
