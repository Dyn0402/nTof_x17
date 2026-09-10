#!/usr/bin/env bash
# fullpass_chain_2026-09-10.sh -- take the condor FULL pass through to tracks.
#
#   1. guard the existing k_arm JSONs   (they are the allowlist/calib-pass
#      measurements and `k_arm` has NO --out: it always writes
#      kcal/k_arm_<run>.json and would overwrite them silently)
#   2. k_arm PER RUN on the full pass   -- the decision of 2026-09-10 is that
#      run_145's k is not transferable, so every run measures its own
#   3. campaign_tracks --fullpass       -- the track database, per-run k
#   4. tracking_qa + figures            -- the distributions, on the new table
#
# The merge (`merge_campaign --fullpass <reco>`) must already have run; step 2
# reads `events_prelim.parquet` and stage 2 does not write it.
#
# k_arm failing to certify a run is a RESULT, not an error: that run's tracks
# carry raw angles and `angle_calibrated` false. So the loop does not stop on
# it, and the count of runs that certified is printed at the end.
#
#   bash sept26_prelim_analysis/fullpass_chain_2026-09-10.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=/media/dylan/data/x17/sept26_prelim
RECO=$OUT/reco_fullpass
ARCHIVE=$OUT/kcal/pre_fullpass_2026-09-10

log () { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 1. guard
mkdir -p "$ARCHIVE"
n=0
for f in "$OUT"/kcal/k_arm_run_*.json; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  [ -e "$ARCHIVE/$b" ] || { cp -p "$f" "$ARCHIVE/$b"; n=$((n+1)); }
done
log "archived $n existing k_arm JSON(s) to $ARCHIVE"

# ---------------------------------------------------------------- 2. k_arm
log "=== k_arm per run on the FULL pass"
certified=0; attempted=0
for d in "$RECO"/run_*; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  subs=$(ls "$d" 2>/dev/null | grep '^stat090_' | paste -sd, -)
  [ -n "$subs" ] || { log "  $run: no sub-runs, skipped"; continue; }
  attempted=$((attempted+1))
  log "--- $run ($(echo "$subs" | tr ',' '\n' | wc -l) sub-runs)"
  # --merged is REQUIRED: the default is a hardcoded <out>/fullpass/run_145.
  $PY -W ignore -m sept26_prelim_analysis.k_arm --run "$run" \
      --subruns "$subs" --merged "$d" 2>&1 | tail -4
  if $PY - "$OUT/kcal/k_arm_$run.json" <<'EOF'
import json, sys
try:
    sys.exit(0 if json.load(open(sys.argv[1]))['apply'] else 1)
except Exception:
    sys.exit(1)
EOF
  then certified=$((certified+1)); else log "  !! $run did not certify any arm"; fi
done
log "k_arm: $certified of $attempted run(s) certified at least one arm"

# ---------------------------------------------------------------- 3. tracks
# NO --k-from. Per-run k is the whole point; a run that did not certify
# contributes raw angles and is stamped so downstream can see it.
log "=== campaign_tracks on the FULL pass"
$PY -W ignore -m sept26_prelim_analysis.campaign_tracks \
    --fullpass "$RECO" --out "$OUT/stage3_fullpass" --jobs "${JOBS:-8}" \
  || { log "!! campaign_tracks FAILED"; exit 1; }

# ---------------------------------------------------------------- 4. QA
log "=== tracking QA on the new table"
$PY -W ignore -m sept26_prelim_analysis.tracking_qa \
    --src "$OUT/stage3_fullpass/tracks_campaign.parquet" \
    --out "$OUT/tracking_qa_fullpass" || exit 1
$PY -W ignore -m sept26_prelim_analysis.make_tracking_qa_figures \
    --qa-dir "$OUT/tracking_qa_fullpass" || exit 1

log "DONE"
