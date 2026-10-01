#!/bin/bash
# Validation of the "fixed" two-track chain against the contract, at thresholds
# matched to production's false-split rate on real singles (TWO_TRACK_FIT_LOG,
# 2026-09-30). Resumable: a step whose output already exists is skipped, so
# stopping between steps and re-running loses nothing. A step killed midway
# restarts from its beginning (no checkpoints inside a step).
#
#   sept26_prelim_analysis/two_track_validate.sh [JOBS] > validate.log 2>&1
#
# Per chamber, in order:
#   bench     intra_bench build, fixed chain, --overlay replace   -> fixed_<arm>_replace/
#   split-ab  real triggers, one file tag, fixed chain            -> split_ab_fixed_<arm>/
#   split-ab  the same with current production, for comparison    -> split_ab_current_<arm>/
# Matched thresholds: A F=1200, C F=2400 (corroborated at 0.4 F).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
J=${1:-12}
OUT=$($PY -c "import sys; sys.path.insert(0,'.'); from sept26_prelim_analysis.intra_bench import out_dir; print(out_dir())" 2>/dev/null)
FIX="--worker-opt TWO_TRACK_SCALE=two --worker-opt TWO_TRACK_SEARCH=grid --worker-opt TWO_TRACK_RESID_Z=-inf --worker-opt TWO_TRACK_MAX_TRY=99 --worker-opt TWO_TRACK_SELECTED_ONLY=false"
BASE="--pairing --local-mm 16 --local-mode rescue --two-track --two-track-t0 tied --two-track-resid-z 8 --overlay replace"

step() {   # step <done-marker> <label> <command...>
  local marker=$1 label=$2; shift 2
  if [ -e "$OUT/$marker" ]; then echo "=== skip $label (have $marker)"; return; fi
  echo "=== $label $(date)"
  "$@" 2>&1 | grep -v Warn | tail -4
}

for spec in "A 1200 480" "C 2400 960"; do
  set -- $spec
  ARM=$1; F=$2; FC=$3
  step fixed_${ARM}_replace/build.meta.json "bench fixed $ARM" \
    $PY -m sept26_prelim_analysis.intra_bench build --arms $ARM --jobs $J \
        --variant fixed_${ARM}_replace $BASE --two-track-f $F --two-track-f-corrob $FC $FIX
  step split_ab_fixed_${ARM}/summary.csv "split-ab fixed $ARM" \
    $PY -m sept26_prelim_analysis.intra_bench split-ab --arms $ARM --jobs $J --tags 1 --pairing \
        --variant fixed_${ARM} --worker-opt TWO_TRACK_F=$F --worker-opt TWO_TRACK_F_CORROB=$FC $FIX
  step split_ab_current_${ARM}/summary.csv "split-ab current $ARM" \
    $PY -m sept26_prelim_analysis.intra_bench split-ab --arms $ARM --jobs $J --tags 1 --pairing \
        --variant current_${ARM} --worker-opt TWO_TRACK_F=300 --worker-opt TWO_TRACK_F_CORROB=120
done
echo "=== done $(date)"
