#!/usr/bin/env bash
# Everything that has to happen after the first fleet pass:
#   1. det4, which crashed on its mixed 32/37-sample windows (now fixed)
#   2. det3 and det2 re-run on the candidate-cluster seeder, so the whole fleet
#      sits on one seeder
#   3. rebuild det2/det6/det7's HITS caches with the significance floor -- they
#      predate the 2026-07-25 fix, so `--source hits` scores them unfairly
#   4. the fleet digest
# Waits for the running fleet to go idle first.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=/tmp/wft_logs; mkdir -p "$LOG"
say () { echo "[$(date +%H:%M:%S)] $*"; }

while pgrep -f "run_fleet.sh" > /dev/null; do sleep 60; done
say "fleet idle"

# 1. det4
say "===== det4 (mixed-window fix) ====="
bash mx_june_wft/run_chain.sh g_det4 --jobs 12 \
     --legacy /home/dylan/x17/cosmic_bench/Analysis/mx17_det4_day_6-24-26/long_run/mx17_4/waveform_first \
     --hyper-file hyper_det4.json > "$LOG/det4_retry.log" 2>&1
say "det4 exit=$?"
grep -E "within 5|core sigma|^x:|^y:" "$LOG/det4_retry.log" | tail -8

# 2. det3 / det2 on the candidate seeder
for KEY in sat_det3 o22_long_det2; do
  say "===== $KEY (candidate seeder) ====="
  $PY -m wft.cli reco "$KEY" --jobs 12 --matched-only > "$LOG/${KEY}_cand_reco.log" 2>&1
  $PY mx_june_wft/01_alignment.py "$KEY"                    > "$LOG/${KEY}_cand.log" 2>&1
  $PY mx_june_wft/02_efficiency.py "$KEY" --max-dropped -1 >> "$LOG/${KEY}_cand.log" 2>&1
  $PY mx_june_wft/02_efficiency.py "$KEY"                  >> "$LOG/${KEY}_cand.log" 2>&1
  $PY mx_june_wft/03_angles.py "$KEY"                      >> "$LOG/${KEY}_cand.log" 2>&1
  grep -E "within 5|core sigma|^x:|^y:" "$LOG/${KEY}_cand.log" | tail -8
done

# 3. floor-corrected hits caches for the three detectors that never got the fix
for KEY in o22_long_det2 g_det6_long g_det7_long; do
  say "===== $KEY: hits cache rebuild with the significance floor ====="
  # This regenerates the OLD chain's alignment and cache in place. They are
  # derived products and the recipe is the documented-correct one, but back them
  # up first so the pre-refit state stays recoverable.
  OUT=$($PY -c "
import sys; sys.path.insert(0,'mx_june_cosmic_qa')
from qa_config import get_config, setup_paths; setup_paths()
print(get_config('$KEY').OUT_BASE)")
  STAMP=$(date +%Y%m%d_%H%M%S)
  for D in "$OUT/cache" "$OUT/alignment_tpc_veto50"; do
    if [ -d "$D" ]; then
      cp -a "$D" "${D}_prefloor_$STAMP" && say "  backed up $(basename "$D") -> $(basename "$D")_prefloor_$STAMP"
    fi
  done
  $PY mx_june_cosmic_qa/03_alignment_and_tpc.py "$KEY" --veto=50 --refit \
      > "$LOG/${KEY}_hits_refit.log" 2>&1 || say "  refit failed, see $LOG/${KEY}_hits_refit.log"
  $PY mx_june_wft/02_efficiency.py "$KEY" --source hits >> "$LOG/${KEY}_hits_refit.log" 2>&1 || true
  grep -E "within 5|core sigma" "$LOG/${KEY}_hits_refit.log" | tail -2
done

say "===== fleet digest ====="
$PY mx_june_wft/digest.py sat_det3 o22_long_det2 g_det4 g_det6_long g_det7_long \
    --out mx_june_wft/FLEET_DIGEST.md 2>&1 | tail -30
say "all done"
