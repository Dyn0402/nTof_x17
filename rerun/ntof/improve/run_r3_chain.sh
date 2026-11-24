#!/usr/bin/env bash
# run_r3_chain.sh -- the n_TOF analysis view with the spline angle map (r3), from an UNMAPPED v(t) view.
#
# The same chain serves two cases.
#
#   Now (stage 3 r2, lines already built, the committed map):
#       FIT=0 bash rerun/ntof/improve/run_r3_chain.sh
#     applies rerun/ntof/stage3/angle_map/angle_map_r3.json to stage3_r2_vt and builds stage3_r3_vtmap/ and
#     stage3_r3_final/ under $WORK_DATA/ntof_pass1.
#
#   After a second reconstruction pass (new unmapped view, lines rebuilt, map refitted):
#       SRC_VT=$P/stage3_r4_vt TRAIN=$P/stage3_r4_train REBUILD_TRAIN=1 REBUILD_LINES=1 \
#       LINES_DIR=$P/ana_r4/track_quality ANA=$P/ana_r4/spline_map MAP_OUT=$P/ana_r4/spline_map/angle_map_r4.json \
#       VTMAP=$P/stage3_r4_vtmap FINAL=$P/stage3_r4_final FIT=1 bash rerun/ntof/improve/run_r3_chain.sh
#     The refitted map is written to $MAP_OUT, never over a committed map file: the chain refuses a MAP_OUT inside
#     rerun/ntof/stage3/angle_map/ and refuses an existing MAP_OUT (OVERWRITE_MAP=1 to replace a scratch map).
#     Commit the new map under its own name after reading the held-out table ($ANA/heldout.csv).
#
# Steps (each logged to $LOG):
#   1. training copy TRAIN = SRC_VT + censored-window cut (reused when it exists and REBUILD_TRAIN=0)
#   2. cosmic lines (REBUILD_LINES=1): line_residuals build (A-C) and bd_map (B-D) on TRAIN, into LINES_DIR
#   3. pointing sample: the late capsule tracks of TRAIN (spline_map_fit.py points) -> $ANA/late_tracks_train.parquet
#   4. FIT=1: spline_map_fit.py fit -> MAP_OUT, and heldout -> $ANA/heldout.csv; the map applied is then MAP_OUT
#   5. map view: map_view.py --src SRC_VT --out VTMAP --map MAP_APPLY
#   6. FINAL = VTMAP + censored-window cut + quality references and flags (censor_join, quality_join refs + apply)
#   7. beam check on FINAL: late sample (mapped) and spline_map_fit.py beam -> $ANA/beam_final.csv
#
# Variables (defaults in brackets):
#   WORK_DATA [/media/ucla/work]   P [$WORK_DATA/ntof_pass1]
#   SRC_VT [$P/stage3_r2_vt]  TRAIN [$P/stage3_r2_train]  REBUILD_TRAIN [0]  N_TABLES [543]
#   LINES_DIR [$P/ana_r2/track_quality]  REBUILD_LINES [0]
#   ANA [$P/ana_r3/spline_map]  FIT [1]  MAP_OUT [$ANA/angle_map_r3.json]  OVERWRITE_MAP [0]
#   MAP_APPLY [MAP_OUT if FIT=1, else rerun/ntof/stage3/angle_map/angle_map_r3.json]
#   VTMAP [$P/stage3_r3_vtmap]  FINAL [$P/stage3_r3_final]  JOBS [6] (view build)  FIT_JOBS [3] (readers)
#   STOP_AFTER [ ] (a step number: stop after it)  BEAM_CHECK [1]
set -euo pipefail
cd "$(dirname "$0")/../../.."
REPO=$(pwd)
WORK_DATA=${WORK_DATA:-/media/ucla/work}
P=${P:-$WORK_DATA/ntof_pass1}
SRC_VT=${SRC_VT:-$P/stage3_r2_vt}
TRAIN=${TRAIN:-$P/stage3_r2_train}
REBUILD_TRAIN=${REBUILD_TRAIN:-0}
N_TABLES=${N_TABLES:-543}
LINES_DIR=${LINES_DIR:-$P/ana_r2/track_quality}
REBUILD_LINES=${REBUILD_LINES:-0}
ANA=${ANA:-$P/ana_r3/spline_map}
FIT=${FIT:-1}
MAP_OUT=${MAP_OUT:-$ANA/angle_map_r3.json}
OVERWRITE_MAP=${OVERWRITE_MAP:-0}
COMMITTED_MAP=$REPO/rerun/ntof/stage3/angle_map/angle_map_r3.json
if [ "$FIT" = 1 ]; then MAP_APPLY=${MAP_APPLY:-$MAP_OUT}; else MAP_APPLY=${MAP_APPLY:-$COMMITTED_MAP}; fi
VTMAP=${VTMAP:-$P/stage3_r3_vtmap}
FINAL=${FINAL:-$P/stage3_r3_final}
JOBS=${JOBS:-6}
FIT_JOBS=${FIT_JOBS:-3}
STOP_AFTER=${STOP_AFTER:-}
BEAM_CHECK=${BEAM_CHECK:-1}
PY="nice -n 10 $REPO/.venv/bin/python"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
mkdir -p "$ANA"
LOG=${LOG:-$ANA/r3_chain.log}
say() { echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }
stop_after() { if [ "$STOP_AFTER" = "$1" ]; then say "STOP_AFTER=$1: stop"; exit 0; fi; }
ntab() { ls "$1"/tracks_run_*.parquet 2>/dev/null | wc -l; }

say "run_r3_chain: SRC_VT=$SRC_VT TRAIN=$TRAIN LINES_DIR=$LINES_DIR ANA=$ANA FIT=$FIT MAP_APPLY=$MAP_APPLY VTMAP=$VTMAP FINAL=$FINAL JOBS=$JOBS"
say "git $(git rev-parse --short HEAD); free on $(df -h --output=target,avail "$P" | tail -1)"
n=$(ntab "$SRC_VT"); [ "$n" = "$N_TABLES" ] || { say "$SRC_VT has $n of $N_TABLES tables -- stop"; exit 1; }
tabs=("$SRC_VT"/tracks_run_*.parquet); first=${tabs[0]}
if $REPO/.venv/bin/python -c "import pyarrow.parquet as pq, sys
sys.exit(0 if any(c.endswith('_nomap') for c in pq.read_schema(sys.argv[1]).names) else 1)" "$first"; then
  say "$SRC_VT is already mapped (has *_nomap columns): give the UNMAPPED v(t) view -- stop"; exit 1
fi
if [ "$FIT" = 1 ]; then
  case "$(realpath -m "$MAP_OUT")" in
    "$REPO"/rerun/ntof/stage3/angle_map/*) say "MAP_OUT=$MAP_OUT is a committed map location: write the refit elsewhere -- stop"; exit 1;;
  esac
  if [ -e "$MAP_OUT" ] && [ "$OVERWRITE_MAP" != 1 ]; then say "MAP_OUT=$MAP_OUT exists (OVERWRITE_MAP=1 to replace it) -- stop"; exit 1; fi
fi
for d in "$VTMAP" "$FINAL"; do
  case "$(realpath -m "$d")" in "$(realpath -m "$SRC_VT")"|"$(realpath -m "$TRAIN")") say "$d is an input -- stop"; exit 1;; esac
done

say "1. training copy (unmapped + censored-window cut): $TRAIN"
TRAIN_BUILT=0
if [ "$REBUILD_TRAIN" = 1 ] || [ "$(ntab "$TRAIN")" != "$N_TABLES" ]; then
  TRAIN_BUILT=1
  rm -rf "$TRAIN"; cp -r "$SRC_VT" "$TRAIN"
  $PY rerun/ntof/stage3/censor_join.py --view "$TRAIN" --jobs "$JOBS" >> "$LOG" 2>&1
else
  say "   reusing $TRAIN ($(ntab "$TRAIN") tables)"
fi
stop_after 1

say "2. cosmic lines: $LINES_DIR"
export X17_TQ_OUT=$LINES_DIR
mkdir -p "$LINES_DIR"
if [ "$REBUILD_LINES" = 1 ]; then
  X17_PASS1_VIEW=$TRAIN $PY -m rerun.ntof.improve.line_residuals build >> "$LOG" 2>&1
  X17_PASS1_VIEW=$TRAIN $PY -m rerun.ntof.improve.bd_map --view "$TRAIN" >> "$LOG" 2>&1
else
  say "   reusing lines.parquet and bd_lines.parquet"
fi
for f in lines.parquet bd_lines.parquet; do [ -s "$LINES_DIR/$f" ] || { say "$LINES_DIR/$f missing -- stop"; exit 1; }; done
stop_after 2

LATE_TRAIN=$ANA/late_tracks_train.parquet
say "3. pointing sample (late capsule tracks, unmapped): $LATE_TRAIN"
if [ "$TRAIN_BUILT" = 1 ] || [ ! -s "$LATE_TRAIN" ]; then
  $PY rerun/ntof/improve/spline_map_fit.py points --view "$TRAIN" --out "$LATE_TRAIN" --jobs "$FIT_JOBS" >> "$LOG" 2>&1
else
  say "   reusing $LATE_TRAIN"
fi
stop_after 3

if [ "$FIT" = 1 ]; then
  say "4. fit the spline map -> $MAP_OUT; held-out table -> $ANA/heldout.csv"
  $PY rerun/ntof/improve/spline_map_fit.py fit --lines "$LINES_DIR/lines.parquet" --bd-lines "$LINES_DIR/bd_lines.parquet" \
      --points "$LATE_TRAIN" --out "$MAP_OUT" --status "refit by run_r3_chain.sh from $SRC_VT (not committed)" >> "$LOG" 2>&1
  $PY rerun/ntof/improve/spline_map_fit.py heldout --lines "$LINES_DIR/lines.parquet" --bd-lines "$LINES_DIR/bd_lines.parquet" \
      --points "$LATE_TRAIN" --out "$ANA/heldout.csv" >> "$LOG" 2>&1
else
  say "4. no refit (FIT=0)"
fi
[ -s "$MAP_APPLY" ] || { say "map $MAP_APPLY missing -- stop"; exit 1; }
say "   map applied: $MAP_APPLY md5 $(md5sum "$MAP_APPLY" | cut -c1-32)"
stop_after 4

say "5. map view -> $VTMAP"
rm -rf "$VTMAP"
$PY rerun/ntof/stage3/map_view.py --src "$SRC_VT" --out "$VTMAP" --map "$MAP_APPLY" --jobs "$JOBS" >> "$LOG" 2>&1
[ "$(ntab "$VTMAP")" = "$N_TABLES" ] || { say "$VTMAP incomplete -- stop"; exit 1; }
stop_after 5

say "6. censored-window cut + quality flags -> $FINAL"
rm -rf "$FINAL"; cp -r "$VTMAP" "$FINAL"
$PY rerun/ntof/stage3/censor_join.py --view "$FINAL" --jobs "$JOBS" >> "$LOG" 2>&1
$PY rerun/ntof/stage3/quality_join.py refs --view "$FINAL" >> "$LOG" 2>&1
$PY rerun/ntof/stage3/quality_join.py apply --view "$FINAL" --jobs "$JOBS" >> "$LOG" 2>&1
stop_after 6

if [ "$BEAM_CHECK" = 1 ]; then
  say "7. beam check on $FINAL -> $ANA/beam_final.csv"
  $PY rerun/ntof/improve/spline_map_fit.py points --view "$FINAL" --out "$ANA/late_tracks_final.parquet" --jobs "$FIT_JOBS" >> "$LOG" 2>&1
  $PY rerun/ntof/improve/spline_map_fit.py beam --late "$ANA/late_tracks_final.parquet" --out "$ANA/beam_final.csv" >> "$LOG" 2>&1
fi
say "done: $FINAL (free on $(df -h --output=target,avail "$P" | tail -1))"
