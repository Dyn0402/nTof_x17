#!/usr/bin/env bash
# Re-run the detectors that were reconstructed before the candidate-cluster
# selector landed (det3, det2), so the whole fleet is on one seeder.
# Waits for any running reco to finish first.
set -u
cd "$(dirname "$0")/.."
LOG=/tmp/wft_logs; mkdir -p "$LOG"

while pgrep -f "wft.cli reco" > /dev/null || pgrep -f "run_fleet.sh" > /dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] fleet idle, re-running det3 and det2 with candidate selection"

for KEY in sat_det3 o22_long_det2; do
  echo "[$(date +%H:%M:%S)] ===== $KEY (candidates) ====="
  bash mx_june_wft/run_chain.sh "$KEY" --jobs 12 --skip-reco >/dev/null 2>&1 || true
  .venv/bin/python -m wft.cli reco "$KEY" --jobs 12 --matched-only \
      > "$LOG/${KEY}_cand_reco.log" 2>&1
  .venv/bin/python mx_june_wft/01_alignment.py "$KEY"  > "$LOG/${KEY}_cand.log" 2>&1
  .venv/bin/python mx_june_wft/02_efficiency.py "$KEY" --max-dropped -1 >> "$LOG/${KEY}_cand.log" 2>&1
  .venv/bin/python mx_june_wft/02_efficiency.py "$KEY"                  >> "$LOG/${KEY}_cand.log" 2>&1
  .venv/bin/python mx_june_wft/03_angles.py "$KEY"     >> "$LOG/${KEY}_cand.log" 2>&1
  grep -E "within 5|core sigma|^x:|^y:" "$LOG/${KEY}_cand.log" | tail -8
done
# The hits caches for det2/det6/det7 predate the 2026-07-25 significance floor
# (no .meta.json sidecar), so --source hits scores them unfairly. Rebuild them
# before any position comparison is quoted. Also closes the July-25 open item.
for KEY in o22_long_det2 g_det6_long g_det7_long; do
  echo "[$(date +%H:%M:%S)] ===== $KEY: rebuilding the hits cache with the significance floor ====="
  .venv/bin/python mx_june_cosmic_qa/03_alignment_and_tpc.py "$KEY" --veto=50 --refit \
      > "$LOG/${KEY}_hits_refit.log" 2>&1 || echo "  (failed, see $LOG/${KEY}_hits_refit.log)"
  .venv/bin/python mx_june_wft/02_efficiency.py "$KEY" --source hits \
      >> "$LOG/${KEY}_hits_refit.log" 2>&1 || true
  grep -E "within 5|core sigma" "$LOG/${KEY}_hits_refit.log" | tail -2
done

echo "[$(date +%H:%M:%S)] candidate re-runs done"
.venv/bin/python mx_june_wft/digest.py sat_det3 o22_long_det2 g_det4 g_det6_long g_det7_long \
    --out mx_june_wft/FLEET_DIGEST.md 2>&1 | tail -25
