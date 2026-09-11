#!/usr/bin/env bash
# Unattended work for the afternoon of 2026-09-09, in dependency order.
# Launched with setsid+nohup so it survives the session ending.
#
#   1. tight coincidence + figures on the now-calibrated campaign tracks (mins)
#   2. run_86 LOCAL FULL PASS, 4 sub-runs x 4 arms (hours)  <-- the ask
#   3. k_arm on that full pass
#   4. the comparison the full pass exists to make
#
# WHY run_86: it is 8.9 days before run_145 and on the SAME side of the 27 Jul
# access, so it is the farthest-in-time run at an unchanged detector condition.
# The question is whether the -6.5 % offset between full-pass and
# calibration-pass k seen at run_145 is a METHOD effect or a TIME effect.
# run_86's calibration-pass k is already measured (A 1.184, C 1.350); this
# produces its full-pass k for the same run.
#
# Resumable: every step skips if its output is already there.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# The tree comes from paths.py, so a chain and the Python it calls cannot
# disagree about where it is; $X17_ROOT / $X17_SEPT26_OUT still move both.
OUT=$($PY -m sept26_prelim_analysis.paths --path out) || exit 1
R86=$OUT/fullpass_r86/run_86
JOBS=${JOBS:-14}
log () { echo "[$(date -Is)] $*"; }

log "=== STEP 1: tight coincidence on the calibrated campaign tracks"
$PY -W ignore -m sept26_prelim_analysis.tight_coincidence --all-runs --campaign \
  && $PY -W ignore -m sept26_prelim_analysis.make_tight_figures --run campaign \
  || log "!! tight coincidence failed -- continuing to the full pass, which is the priority"

log "=== STEP 2: run_86 local full pass (no allowlist -- every seeded trigger)"
for SUB in stat090_0000 stat090_0002 stat090_0004 stat090_0006; do
  for ARM in A B C D; do
    DEST=$R86/$SUB/mx17_$ARM
    mkdir -p "$DEST"
    if [ -s "$DEST/events_prelim.parquet" ]; then
      log "  $SUB/$ARM already done, skipping"; continue
    fi
    log "  $SUB/$ARM starting"
    # Arms sequential, tags parallel inside: peak RSS stays near jobs x 1 GB
    # rather than 4x that (same reasoning as run_fullpass_0002.sh).
    $PY -W ignore -m ntof_tracking.wft_beam reco \
        --det "$ARM" --run run_86 --subrun "$SUB" --jobs "$JOBS" \
        --out "$DEST/events_prelim.parquet" >>"$R86/pass.log" 2>&1
    log "  $SUB/$ARM finished rc=$? $(du -h "$DEST/events_prelim.parquet" 2>/dev/null|cut -f1)"
  done
done

log "=== STEP 3: k_arm on the run_86 FULL pass"
SUBS=$(ls "$R86" 2>/dev/null | grep '^stat090_' | paste -sd, -)
log "  sub-runs: $SUBS"
$PY -W ignore -m sept26_prelim_analysis.k_arm --run run_86 --subruns "$SUBS" \
    --merged "$R86" --out "$OUT/kcal_r86" 2>&1 | tail -25

log "=== STEP 4: the comparison"
OUT="$OUT" $PY -W ignore - <<'PYEOF'
import json, os
from pathlib import Path
out = Path(os.environ['OUT'])      # resolved once, by paths.py, above
rows = []
for label, p in (('run_86 FULL pass',  out/'kcal_r86'/'k_arm_run_86.json'),
                 ('run_86 calib pass', out/'kcal'/'k_arm_run_86.json'),
                 ('run_145 FULL pass', out/'kcal'/'k_arm_run_145.json')):
    if p.exists():
        d = json.load(open(p))
        rows.append((label, d.get('apply') or {},
                     {a: v.get('verdict') for a, v in (d.get('arms') or {}).items()}))
print('\n=== k by arm, full pass vs calibration pass ===')
for label, ap, ver in rows:
    print(f'{label:20s} ' + '  '.join(f'{a}={ap.get(a, float("nan")):.4f}' for a in 'ACD'
                                      if a in ap) or f'{label:20s} (nothing certified)')
    print(f'{"":20s} verdicts: {ver}')
print("""
READING IT:
  If run_86's FULL-pass k sits ~6 % ABOVE its calibration-pass k (1.184 A,
  1.350 C), the offset is a METHOD effect -- the allowlist's SINGLE prescale
  biases the estimator -- and one k campaign-wide is the right call.
  If run_86's full-pass k instead lands NEAR its calibration-pass k, then
  run_145 is the outlier and the offset is run-specific, which would undermine
  using run_145's k everywhere.""")
PYEOF
log "=== ALL DONE"
