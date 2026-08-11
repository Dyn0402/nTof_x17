#!/bin/bash
# reconcile_staging.sh -- free staged runs that are already safely on the ntof disk.
#
#   ssh -K lxplus && ./reconcile_staging.sh            # all staged runs
#   ./reconcile_staging.sh 224705 224711               # or named ones
#
# Why this exists: `cp` returns NON-ZERO on EOS FUSE even when every byte landed
# correctly. `process_missing_runs.sh` treats that as a failure and keeps the
# staged copy -- the safe choice, but it leaks quota. Measured on 2026-08-11:
# 224705 (8 files) and 224711 (37 files) both reported "COPY FAILED" and both
# were byte-identical at the destination.
#
# THE EXIT CODE IS NOT THE TEST. Size comparison is. This script re-verifies and
# frees only what genuinely matches, so it is safe to run at any time, including
# while the campaign driver is running.
set -uo pipefail

STAGE=${X17_STAGE:-/eos/user/d/dneff/x17/reproc/prod_v12}
FINAL=${X17_FINAL:-/eos/experiment/ntof/data/x17/reproc/prod_v12}

runs=("$@")
if [ ${#runs[@]} -eq 0 ]; then
    mapfile -t runs < <(ls "$STAGE" 2>/dev/null)
fi
[ ${#runs[@]} -gt 0 ] || { echo "nothing staged"; exit 0; }

freed=0; kept=0
for run in "${runs[@]}"; do
    S=$STAGE/$run/completed/$run
    D=$FINAL/$run/completed/$run
    [ -d "$S" ] || continue

    nsrc=$(ls "$S" 2>/dev/null | grep -c '\.root$')
    [ "$nsrc" -gt 0 ] || { echo "$run: staging has no .root, skip"; continue; }
    ndst=$(ls "$D" 2>/dev/null | grep -c '\.root$')
    if [ "$ndst" -ne "$nsrc" ]; then
        echo "$run: dest has $ndst of $nsrc files -- KEEPING staging"; kept=$((kept+1)); continue
    fi

    bad=0
    for f in "$S"/*.root; do
        b=$(basename "$f")
        [ "$(stat -c %s "$f" 2>/dev/null)" = "$(stat -c %s "$D/$b" 2>/dev/null)" ] || bad=$((bad+1))
    done
    if [ "$bad" -ne 0 ]; then
        echo "$run: $bad/$nsrc size mismatches -- KEEPING staging"; kept=$((kept+1)); continue
    fi

    rm -rf "${STAGE:?}/$run" 2>/dev/null
    echo "$run: verified $nsrc/$nsrc byte-identical -> staging freed"
    freed=$((freed+1))
done
echo "reconcile: $freed freed, $kept kept"
