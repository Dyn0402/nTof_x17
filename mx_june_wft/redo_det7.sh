#!/usr/bin/env bash
# det7's first calibration was degenerate: c1 = 0.004 (no charge sharing, which
# is impossible on resistive strips -- the June summary measures c1 ~ 0.25 for
# det7's X plane), kY = 6.6, sigma_p0 = 0.52 mm, and v = 36.68 um/ns at a field
# where the drift scan measures 26.4. The fit traded the sharing kernel against
# v, a degeneracy documented in WAVEFORM_FIRST_THREADING.md 17.2.
#
# Re-fit with the two guards added afterwards: a physical floor on c1, and v
# pinned to the drift-scan value for this field (det6's free fit landed 1.3%
# from it, which is the evidence that pinning is safe).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=/tmp/wft_logs; mkdir -p "$LOG"
say () { echo "[$(date +%H:%M:%S)] $*"; }

while pgrep -f "finish_fleet.sh" > /dev/null || pgrep -f "run_fleet.sh" > /dev/null; do
  sleep 60
done
say "queue idle, recalibrating det7"

# keep the degenerate bundle as evidence
B=/home/dylan/x17/cosmic_bench/Analysis/mx17_det6_det7_overnight_6-26-26/long_run/mx17_7/wft
[ -d "$B/calib_bundle" ] && cp -a "$B/calib_bundle" "$B/calib_bundle_degenerate_$(date +%Y%m%d_%H%M%S)"

$PY -m wft.calibrate g_det7_long --jobs 12 --fix-v 26.4 \
    --seed-bundle /home/dylan/x17/cosmic_bench/Analysis/mx17_det6_det7_overnight_6-26-26/long_run/mx17_6/wft/calib_bundle \
    > "$LOG/det7_recalib.log" 2>&1
say "recalibration exit=$?"
grep -E "^\[calib\] mx17|template" "$LOG/det7_recalib.log" | tail -4

$PY -m wft.cli reco g_det7_long --jobs 12 --matched-only > "$LOG/det7_reco2.log" 2>&1
$PY mx_june_wft/01_alignment.py g_det7_long                    > "$LOG/det7_v2.log" 2>&1
$PY mx_june_wft/02_efficiency.py g_det7_long --max-dropped -1 >> "$LOG/det7_v2.log" 2>&1
$PY mx_june_wft/02_efficiency.py g_det7_long --source hits    >> "$LOG/det7_v2.log" 2>&1
$PY mx_june_wft/03_angles.py g_det7_long                      >> "$LOG/det7_v2.log" 2>&1
grep -E "z scan|within 5|core sigma|^x: n=|^y: n=" "$LOG/det7_v2.log" | tail -10
say "det7 redo done"

$PY mx_june_wft/digest.py sat_det3 o22_long_det2 g_det4 g_det6_long g_det7_long \
    --out mx_june_wft/FLEET_DIGEST.md 2>&1 | tail -30
say "final digest written"
