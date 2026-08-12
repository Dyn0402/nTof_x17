#!/bin/bash
# Recover run 224526 from CTA tape and reprocess it at v12.
#
# WHY THIS EXISTS
#   224526's raw partly expired from the EOS disk staging area (which holds data
#   for only two weeks). n_TOF reprocessed the run on 2026-08-07 from the 22
#   files that were left of ~167, and the official product covers 440 of 3313
#   beam bunches -- 13.3 %. The tape copy is complete: 167 files, indices 0..166,
#   no gaps, 313.8 GB.
#
#   The trap that produced the short product is that RunProcessing.sh builds its
#   file list from whatever is on EOS. This script never touches the EOS
#   remnant: it lists from CTA and calls ProcessFileList.sh with `-c 1`.
#
# USAGE (on lxplus, from a path inside /afs/ -- the tooling requires that)
#   ./recover_224526.sh stage      # request the tape recall (313.8 GB)
#   ./recover_224526.sh check      # how many files are online yet
#   ./recover_224526.sh filelists  # build the 42 per-job .files lists
#   ./recover_224526.sh process    # submit the 42 processing jobs
#   ./recover_224526.sh verify     # partial count, contiguity, bunch coverage
#
# The recall is the slow step: the n_TOF wiki quotes up to 72 h, usually hours.
set -u

RUN=224526
YEAR=2026
AREA=EAR2
CAMPAIGN=X17_measurement

XRD=root://eosctapublicdisk.cern.ch/
CTA=/eos/ctapublicdisk/archive/ntof/$YEAR/$AREA/$CAMPAIGN/$RUN
IDX=$CTA/stream0/run$RUN.idx.finished
PS=/eos/experiment/ntof/repositories/processingscripts

W=/afs/cern.ch/work/d/dneff/x17_reproc
UI=$W/userinputs/v12_liqpileup/UserInput.h
WORK=$W/recover_$RUN
OUT=${X17_RECOVER_OUT:-/eos/experiment/ntof/data/x17/reproc/prod_v12_recover/$RUN}

FILES_PER_JOB=4

list_cta() {
    xrdfs $XRD ls $CTA/stream1 2>/dev/null |
        sed -n "s|.*/run${RUN}_\([0-9]*\)_s1\.raw\.finished$|\1|p" | sort -n
}

case "${1:-}" in

stage)
    n=$(list_cta | wc -l)
    echo "$RUN: $n files on tape, requesting recall"
    # StageRuns.sh finds the 22-file EOS remnant, asks "stage anyway?" and
    # DEFAULTS TO NO -- the same EOS-preference that produced the short product
    # in the first place. Answer yes, or it silently skips the run.
    printf 'y\n' | $PS/StageRuns.sh -y $YEAR -a $AREA -c $CAMPAIGN -r $RUN -l $RUN
    echo "recall requested; poll with '$0 check'"
    ;;

check)
    mkdir -p "$WORK"
    list_cta | sed "s|^|$CTA/stream1/run${RUN}_|; s|$|_s1.raw.finished|" > "$WORK/all.list"
    tot=$(wc -l < "$WORK/all.list")
    # query prepare takes many paths in one call and answers instantly; one
    # call per file takes minutes and this gets polled for hours.
    on=$(xargs -a "$WORK/all.list" -n 50 xrdfs $XRD query prepare 0 2>/dev/null |
         tr -d ' \n' | grep -o '"online":true' | wc -l)
    echo "$RUN: $on / $tot files online"
    [ "$on" -eq "$tot" ] && echo "ready -- run '$0 filelists'"
    ;;

filelists)
    [ -f "$UI" ] || { echo "UserInput missing: $UI"; exit 2; }
    mkdir -p "$WORK/lists"
    rm -f "$WORK/lists"/*.files
    idx=($(list_cta))
    n=${#idx[@]}
    [ "$n" -eq 167 ] || echo "WARNING: expected 167 raw files, found $n"
    job=0
    for ((i = 0; i < n; i += FILES_PER_JOB)); do
        job=$((job + 1))
        f=$(printf "%s/lists/run%s_%04d.files" "$WORK" "$RUN" "$job")
        for ((k = i; k < i + FILES_PER_JOB && k < n; k++)); do
            echo "$IDX $CTA/stream1/run${RUN}_${idx[$k]}_s1.raw.finished"
        done > "$f"
    done
    echo "$RUN: $n raw files -> $job job lists in $WORK/lists"
    echo "expected partials: $job"
    ;;

process)
    ls "$WORK"/lists/*.files >/dev/null 2>&1 || { echo "run '$0 filelists' first"; exit 2; }
    mkdir -p "$OUT"
    for f in "$WORK"/lists/*.files; do
        echo "submitting $(basename "$f")"
        $PS/ProcessFileList.sh -c 1 -f "$f" -o "$OUT" -p "$UI" -r $RUN || \
            echo "  SUBMIT FAILED: $f"
    done
    echo "submitted; watch with condor_q, then '$0 verify'"
    ;;

verify)
    want=$(ls "$WORK"/lists/*.files 2>/dev/null | wc -l)
    d=$OUT/completed/$RUN
    got=$(ls "$d" 2>/dev/null | grep -c "^run${RUN}_.*\.root$")
    echo "$RUN: $got / $want partials in $d"
    ls "$d" 2>/dev/null | grep -o "_[0-9]\{4\}\.root$" | tr -cd '0-9\n' |
        sed 's/^0*//' | sort -n | awk -v w="$want" '
        {a[NR]=$1} END {
            ok=(NR==w); for(i=1;i<=NR;i++) if(a[i]!=i) ok=0
            print (ok ? "contiguous 1.." NR : "NOT contiguous (" NR " files)")
        }'
    echo "--- bunch coverage (target: beam bunches with hits ~= 3313) ---"
    python3 - "$d" $RUN <<'PY'
import glob, sys
try:
    import uproot, numpy as np
except ImportError:
    sys.exit("source the LCG view first")
d, run = sys.argv[1], sys.argv[2]
ps = sorted(glob.glob(f'{d}/run{run}_*.root'))
if not ps:
    sys.exit('no partials yet')
ix = uproot.open(ps[0])['index']
bn = ix['BunchNumber'].array(library='np')
beam = ix['PulseIntensity'].array(library='np') > 1e12
hits = set()
for p in ps:
    try:
        hits |= set(uproot.open(p)['WALA']['BunchNumber'].array(library='np').tolist())
    except Exception as e:
        print('  unreadable', p, e)
hb = np.array([b in hits for b in bn])
print(f'  {len(bn)} bunches, {beam.sum()} beam, {(beam & hb).sum()} with hits '
      f'-> {100 * (beam & hb).sum() / max(beam.sum(), 1):.1f} % '
      f'(official product: 13.3 %)')
PY
    ;;

*)
    sed -n '2,30p' "$0"
    exit 1
    ;;
esac
