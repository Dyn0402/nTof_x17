#!/usr/bin/env bash
# rollout_lp.sh — put a detector on the RC-ladder (share_lp) production config.
#
# The approved per-detector procedure of HANDOFF_2026-07-30.md, as a script:
#
#   0. back up the current wft outputs (they are the old-kernel generation)
#   1. reco with the lp bundle, production env
#   2. w0/kw production retrofit (pass 2 -- the corridor values are biased)
#   3. re-reco so the retrofitted constants are in the table
#   4. analysis chain 01-04 + digest
#
#   mx_june_wft/rollout_lp.sh <run_key> [<run_key> ...] [--jobs N] [--bundle NAME]
#
# The calibration itself is NOT done here: det4/6/7's lp bundles were fitted on
# condor (mx_june_wft/condor/) and are already installed as calib_bundle_lp.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
JOBS=6
BUNDLE_NAME=calib_bundle_lp
KEYS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs)   JOBS=$2; shift 2;;
    --bundle) BUNDLE_NAME=$2; shift 2;;
    *)        KEYS+=("$1"); shift;;
  esac
done
[ ${#KEYS[@]} -gt 0 ] || { echo "usage: rollout_lp.sh <run_key> ..."; exit 2; }

export WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1 WFT_CHI2DOF_BAD=250
LOG=/tmp/wft_logs; mkdir -p "$LOG"
STAMP=$(date +%Y%m%d)
say () { echo "[$(date +%H:%M:%S)] $*"; }

for KEY in "${KEYS[@]}"; do
  OUT=$($PY -c "
import sys; sys.path.insert(0,'mx_june_cosmic_qa')
from qa_config import get_config, setup_paths; setup_paths()
print(get_config('$KEY').OUT_BASE)") || { say "$KEY: no config"; continue; }
  W="$OUT/wft"
  B="$W/$BUNDLE_NAME"
  if [ ! -f "$B/bundle.json" ]; then say "$KEY: no $BUNDLE_NAME, skipped"; continue; fi

  say "===== $KEY  ($BUNDLE_NAME, $JOBS jobs) ====="
  PREV="$W/prev_${STAMP}_prelp"
  if [ ! -d "$PREV" ]; then
    mkdir -p "$PREV"
    for f in events.parquet events.meta.json alignment efficiency angles maps; do
      [ -e "$W/$f" ] && cp -a "$W/$f" "$PREV/"
    done
    say "  backed up the old-kernel generation -> $(basename "$PREV")"
  else
    say "  backup $(basename "$PREV") exists, not overwriting"
  fi

  say "  [1/4] reco (pass 1)"
  $PY -m wft.cli reco "$KEY" --jobs "$JOBS" --matched-only --bundle "$B" \
      > "$LOG/${KEY}_lp_reco1.log" 2>&1 || { say "  RECO FAILED, see $LOG/${KEY}_lp_reco1.log"; continue; }

  say "  [2/4] w0/kw production retrofit"
  $PY mx_june_wft/bench/set_w0.py "$KEY" --bundle "$B" --write \
      > "$LOG/${KEY}_lp_w0.log" 2>&1 || say "  w0 retrofit FAILED (see log) -- continuing with pass-1 constants"
  grep -E "^[xy]:" "$LOG/${KEY}_lp_w0.log" || true

  say "  [3/4] reco (pass 2, retrofitted constants)"
  $PY -m wft.cli reco "$KEY" --jobs "$JOBS" --matched-only --bundle "$B" \
      > "$LOG/${KEY}_lp_reco2.log" 2>&1 || { say "  RECO2 FAILED"; continue; }

  say "  [4/4] analysis chain"
  {
    $PY mx_june_wft/01_alignment.py "$KEY"
    $PY mx_june_wft/02_efficiency.py "$KEY" --max-dropped -1
    $PY mx_june_wft/02_efficiency.py "$KEY"
    $PY mx_june_wft/02_efficiency.py "$KEY" --source hits || echo "hits comparison unavailable"
    $PY mx_june_wft/03_angles.py "$KEY"
    $PY mx_june_wft/04_maps.py "$KEY" || echo "maps failed (non-fatal)"
    $PY mx_june_wft/digest.py "$KEY"
  } > "$LOG/${KEY}_lp_chain.log" 2>&1
  say "  chain exit=$?"
  grep -E "within 5|core sigma|^x:|^y:" "$LOG/${KEY}_lp_chain.log" | tail -10
done
say "rollout complete: ${KEYS[*]}"
