#!/usr/bin/env bash
# gap_consistency.sh — one dataset, from scratch to a drift-gap map.
#
# Per-dataset pipeline (the approved fleet-rollout recipe of HANDOFF_2026-07-30
# plus the gap stage), used here to ask whether each chamber reproduces the SAME
# charge-visible gap map in independent runs / slots / subruns:
#
#   1. RC-ladder (lp) calibration for THIS run condition   (wft.calibrate --share-lp)
#   2. reconstruction with the production env               (wft.cli reco)
#   3. alignment                                           (01_alignment.py)
#   4. benchmark cache (windows + M3 truth + active box)   (bench/build_cache.py)
#   5. w0/kw production retrofit (pass 2 -> mm scale)      (bench/set_w0.py --write)
#   6. gap study: stacked endpoint + topography            (bench/gap_study.py)
#
#   gap_consistency.sh <run_key> <v_pin> [--limit N] [--lowgain] [--backup]
#                      [--jobs N] [--chain] [--skip-calib] [--from STAGE]
#
# --limit N     cap reco/cache at the first N M3-matched events (big runs)
# --lowgain     template cuts for the low-gain chambers (det4/6/7)
# --backup      move an existing old-kernel reco/analysis aside first
# --chain       also run 02/03/04 + digest (only meaningful without --limit)
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
KEY=${1:?run key}; VPIN=${2:?v pin [um/ns]}; shift 2

LIMIT=""; LOWGAIN=0; BACKUP=0; JOBS=8; CHAIN=0; SKIP_CALIB=0; FROM=calib
while [ $# -gt 0 ]; do
  case "$1" in
    --limit) LIMIT=$2; shift 2;;
    --jobs) JOBS=$2; shift 2;;
    --lowgain) LOWGAIN=1; shift;;
    --backup) BACKUP=1; shift;;
    --chain) CHAIN=1; shift;;
    --skip-calib) SKIP_CALIB=1; shift;;
    --from) FROM=$2; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

LOG=/tmp/wft_logs; mkdir -p "$LOG"
export WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1 WFT_CHI2DOF_BAD=250

OUT=$($PY - "$KEY" <<'EOF'
import sys, os
sys.path[:0] = ['.', 'mx_june_cosmic_qa', 'cosmic_bench_analysis']
from qa_config import get_config
print(os.path.join(get_config(sys.argv[1]).OUT_BASE, 'wft'))
EOF
)
B="$OUT/calib_bundle_lp"
say() { echo "[$(date +%H:%M:%S)] $KEY: $*"; }
die() { say "FAILED at $1 (see $LOG/gap_${KEY}_$1.log)"; exit 1; }
stage_idx() {  # stage ordering, so --from can resume a partial run
  case "$1" in calib) echo 1;; reco) echo 2;; align) echo 3;;
               cache) echo 4;; w0) echo 5;; gap) echo 6;;
               *) echo "bad stage $1" >&2; exit 2;; esac
}
FROM_IDX=$(stage_idx "$FROM")
stage_at() { [ "$(stage_idx "$1")" -ge "$FROM_IDX" ]; }
say "wft dir $OUT"

if [ "$BACKUP" = 1 ] && [ -f "$OUT/events.parquet" ] && [ ! -d "$OUT/prev_20260730_oldkernel" ]; then
  say "backing up the old-kernel generation"
  mkdir -p "$OUT/prev_20260730_oldkernel"
  for f in events.parquet events.meta.json alignment angles efficiency maps; do
    [ -e "$OUT/$f" ] && cp -r "$OUT/$f" "$OUT/prev_20260730_oldkernel/"
  done
fi

if [ "$SKIP_CALIB" = 0 ] && stage_at calib; then
  say "1/6 lp calibration (v pinned $VPIN)"
  ARGS=(--jobs "$JOBS" --share-lp --fix-v "$VPIN" --out "$B")
  [ "$LOWGAIN" = 1 ] && ARGS+=(--tmpl-tan-min 0.10 --tmpl-min-amp 250)
  $PY -m wft.calibrate "$KEY" "${ARGS[@]}" > "$LOG/gap_${KEY}_calib.log" 2>&1 || die calib
fi

if stage_at reco; then
  say "2/6 reconstruction"
  ARGS=(--jobs "$JOBS" --matched-only --bundle "$B")
  [ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
  $PY -m wft.cli reco "$KEY" "${ARGS[@]}" > "$LOG/gap_${KEY}_reco.log" 2>&1 || die reco
fi

if stage_at align; then
  say "3/6 alignment"
  $PY mx_june_wft/01_alignment.py "$KEY" > "$LOG/gap_${KEY}_align.log" 2>&1 || die align
fi

if stage_at cache; then
  say "4/6 bench cache"
  ARGS=(); [ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
  $PY mx_june_wft/bench/build_cache.py "$KEY" "${ARGS[@]}" \
      > "$LOG/gap_${KEY}_cache.log" 2>&1 || die cache
fi

if stage_at w0; then
  say "5/6 w0/kw retrofit"
  $PY mx_june_wft/bench/set_w0.py "$KEY" --bundle "$B" --write \
      > "$LOG/gap_${KEY}_w0.log" 2>&1 || die w0
  grep -h "kw =" "$LOG/gap_${KEY}_w0.log" || true
fi

if stage_at gap; then
  say "6/6 gap study"
  $PY mx_june_wft/bench/gap_study.py "$KEY" --bundle "$B" --jobs "$JOBS" \
      --limit ${LIMIT:-40000} > "$LOG/gap_${KEY}_gap.log" 2>&1 || die gap
  grep -h "sharp:\|== \|n_cont" "$LOG/gap_${KEY}_gap.log" || true
fi

if [ "$CHAIN" = 1 ]; then
  say "analysis chain 02/03/04 + digest"
  $PY mx_june_wft/02_efficiency.py "$KEY" --max-dropped -1 > "$LOG/gap_${KEY}_eff.log" 2>&1 || say "eff failed"
  $PY mx_june_wft/03_angles.py "$KEY" > "$LOG/gap_${KEY}_ang.log" 2>&1 || say "angles failed"
  $PY mx_june_wft/04_maps.py "$KEY" > "$LOG/gap_${KEY}_maps.log" 2>&1 || say "maps failed"
  $PY mx_june_wft/digest.py "$KEY" > "$LOG/gap_${KEY}_digest.log" 2>&1 || say "digest failed"
fi

say "COMPLETE"
