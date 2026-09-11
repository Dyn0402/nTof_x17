#!/usr/bin/env bash
# Full waveform pass (NO allowlist) on run_145/stat090_0002 -- the one sub-run
# of run_145 that has never been reconstructed without a selection.  Closing it
# makes run_145 complete and gives the funnel a third independent sub-run.
#
# ~2.5 h on 16 cores.  Arms run sequentially so peak RSS stays near
# jobs x 1 GB rather than 4 x that; within an arm the tags are parallel.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# The tree comes from paths.py, so a chain and the Python it calls cannot
# disagree about where it is; $X17_ROOT / $X17_SEPT26_OUT still move both.
BASE=$($PY -m sept26_prelim_analysis.paths --path out) || exit 1

RUN=run_145
SUB=stat090_0002
OUT=$BASE/fullpass/$RUN/$SUB
JOBS=${JOBS:-14}
LOG=$OUT/pass.log

mkdir -p "$OUT"
echo "=== full pass $RUN/$SUB, jobs=$JOBS, started $(date -Is) ===" | tee -a "$LOG"

for ARM in A B C D; do
  DEST=$OUT/mx17_$ARM
  mkdir -p "$DEST"
  if [ -s "$DEST/events_prelim.parquet" ]; then
    echo "[$(date -Is)] arm $ARM already done, skipping" | tee -a "$LOG"
    continue
  fi
  echo "[$(date -Is)] arm $ARM starting" | tee -a "$LOG"
  # Reuse the sub-run 0000 bundle: same run, same conditions, same gas and
  # sample grid.  Passing it explicitly beats letting make_bundle re-derive
  # one, which would give this sub-run a bundle nothing else shares.
  $PY -W ignore -m ntof_tracking.wft_beam reco \
      --det "$ARM" --run "$RUN" --subrun "$SUB" --jobs "$JOBS" \
      --bundle "$BASE/stage2/reco_run145_stat090_0000/mx17_$ARM/calib_bundle_prelim" \
      --out "$DEST/events_prelim.parquet" >>"$LOG" 2>&1
  rc=$?
  echo "[$(date -Is)] arm $ARM finished rc=$rc" | tee -a "$LOG"
  [ $rc -ne 0 ] && echo "ARM $ARM FAILED -- see $LOG" | tee -a "$LOG"
done

echo "=== done $(date -Is) ===" | tee -a "$LOG"
for ARM in A B C D; do
  f=$OUT/mx17_$ARM/events_prelim.parquet
  [ -s "$f" ] && echo "  $ARM  $(du -h "$f" | cut -f1)" || echo "  $ARM  MISSING"
done
