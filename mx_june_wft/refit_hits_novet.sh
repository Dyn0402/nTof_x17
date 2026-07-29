#!/usr/bin/env bash
# finish_fleet.sh rebuilt det2/det6/det7's hits caches with `03 --veto=50
# --refit`, which writes event_results_veto50.pkl. But 09_efficiency_breakdown
# -- and therefore `02_efficiency.py --source hits`, which reproduces it -- reads
# the UN-vetoed event_results.pkl. So the rescore came back identical (det2
# 80.86% before and after) because the file it reads was never rebuilt.
#
# This runs the missing pass: `03 --no-veto --refit`, which is what puts the
# significance floor into event_results.pkl, then rescores.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=/tmp/wft_logs; mkdir -p "$LOG"
say () { echo "[$(date +%H:%M:%S)] $*"; }

while pgrep -f "finish_fleet.sh" > /dev/null || pgrep -f "redo_det7.sh" > /dev/null; do
  sleep 60
done
say "queue idle, rebuilding the un-vetoed hits caches"

for KEY in o22_long_det2 g_det6_long g_det7_long; do
  say "===== $KEY: un-vetoed cache with the significance floor ====="
  $PY mx_june_cosmic_qa/03_alignment_and_tpc.py "$KEY" --no-veto --refit \
      > "$LOG/${KEY}_hits_novet.log" 2>&1 || say "  refit failed, see $LOG/${KEY}_hits_novet.log"
  $PY mx_june_wft/02_efficiency.py "$KEY" --source hits >> "$LOG/${KEY}_hits_novet.log" 2>&1 || true
  grep -E "within 5|core sigma|median" "$LOG/${KEY}_hits_novet.log" | tail -3
done

say "===== final digest ====="
$PY mx_june_wft/digest.py sat_det3 o22_long_det2 g_det4 g_det6_long g_det7_long \
    --out mx_june_wft/FLEET_DIGEST.md 2>&1 | tail -32
say "all passes complete"
