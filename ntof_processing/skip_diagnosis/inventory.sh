#!/bin/bash
# inventory.sh -- what n_TOF output exists for every X17 campaign run.
#
# Emits CSV to stdout:
#   run,raw_files,raw_MB,parts,parts_MB,merged_bytes,state
#
# state is one of
#   MERGED         a non-empty run<run>.root in official/done/   (usable, best)
#   PARTIALS_ONLY  no usable merged file, but completed/<run>/ has partials
#                  -- SAME processing, only the merge is absent
#   MERGE_EMPTY    a ZERO-BYTE run<run>.root exists; partials are the truth
#   RAW_ONLY       raw stream1 staged, nothing processed
#   NOTHING        no raw staged and nothing processed (tape recall needed)
#
# Sizes are summed from `ls -l` rather than `du`, which is much cheaper on EOS
# FUSE over ~450 directories.
set -uo pipefail

DAQ=/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement
DONE=/eos/experiment/ntof/processing/official/done
COMPLETED=/eos/experiment/ntof/processing/official/completed

sum_bytes() {  # $1 = dir, $2 = grep pattern; prints MB
    ls -l "$1" 2>/dev/null | awk -v pat="$2" \
        '$NF ~ pat {n+=$5} END {printf "%d", n/1048576}'
}

echo "run,raw_files,raw_MB,parts,parts_MB,merged_bytes,state"
for d in "$DAQ"/*/; do
    r=$(basename "$d")
    [[ "$r" =~ ^[0-9]+$ ]] || continue

    s1="$d/stream1"
    if [ -d "$s1" ]; then
        raw_n=$(ls "$s1" 2>/dev/null | wc -l)
        raw_mb=$(sum_bytes "$s1" '.')
    else
        raw_n=0; raw_mb=0
    fi

    cd_="$COMPLETED/$r"
    if [ -d "$cd_" ]; then
        parts=$(ls "$cd_" 2>/dev/null | grep -c "^run${r}_[0-9]*\.root$")
        parts_mb=$(sum_bytes "$cd_" "^run${r}_[0-9]*\.root$")
    else
        parts=0; parts_mb=0
    fi

    m="$DONE/run$r.root"
    if [ -f "$m" ]; then merged=$(stat -c %s "$m" 2>/dev/null || echo 0); else merged=-1; fi

    if   [ "$merged" -gt 0 ];  then state=MERGED
    elif [ "$merged" -eq 0 ];  then state=$([ "$parts" -gt 0 ] && echo MERGE_EMPTY || echo NOTHING)
    elif [ "$parts" -gt 0 ];   then state=PARTIALS_ONLY
    elif [ "$raw_n" -gt 0 ];   then state=RAW_ONLY
    else                            state=NOTHING
    fi

    echo "$r,$raw_n,$raw_mb,$parts,$parts_mb,$merged,$state"
done
