#!/usr/bin/env bash
# run_hv_scans.sh
#
# Pull every resist-HV-scan point and run the HV-scan efficiency+resolution
# analysis for the viable June scans, then compile the combined PDF.
#
#   6-22 (det2_det3): det2+det3 resist stepped together, drift 1000 V  -> g_det2, g_det3
#   6-26 (det6_det7): det6/det7 stepped at different V, drift 700 V     -> g_det6_long, g_det7_long
#   6-23 EXCLUDED: degraded M3 reference (alignment railed z=569/411) -> no reliable
#                  track reference for efficiency.
#
# Alignment is SEEDED from each run's long_run subrun (already analysed); each HV
# point gets a translation-only re-alignment. resist combined_hits are tiny
# (~15 MB / ~2 MB each). Continue-on-error; per-step timeout. Logs under
# Analysis/_grand_logs/.
set -uo pipefail
cd "$(dirname "$0")"

PY=../venv/bin/python
HOST=rays_daplxa
STEP_TIMEOUT=3600
LOG_DIR=/home/dylan/x17/cosmic_bench/Analysis/_grand_logs
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/hv_scans_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $MAIN_LOG"
exec >>"$MAIN_LOG" 2>&1

echo "=== HV scans started $(date) ==="

run_step() { local desc="$1"; shift; echo ">>> [$(date +%H:%M:%S)] $desc"
    local rc=0; timeout "$STEP_TIMEOUT" "$@" || rc=$?
    [ "$rc" -eq 0 ] && echo "    OK   : $desc" || echo "    WARN : $desc (exit $rc)"; return 0; }

pull_scan() {   # pull_scan <run> <area>
    local run="$1" area="$2"
    echo "### pull resist points: $run -> $area ###"
    local subs
    subs=$(ssh -o BatchMode=yes "$HOST" "ls -d /mnt/cosmic_data/MX17/Run/$run/resist_* 2>/dev/null | xargs -n1 basename")
    for sub in $subs; do
        run_step "PULL $run/$sub" ./pull_run.sh "$run" "$sub" "$area" "$HOST"
    done
}

# ---- 6-22: det2 + det3 ----
pull_scan mx17_det2_det3_overnight_6-22-26 det2_det3
run_step "HV scan g_det2" $PY 10_hv_scan_efficiency.py g_det2
run_step "HV scan g_det3" $PY 10_hv_scan_efficiency.py g_det3

# ---- 6-26: det6 + det7 ----
pull_scan mx17_det6_det7_overnight_6-26-26 det6_det7
run_step "HV scan g_det6" $PY 10_hv_scan_efficiency.py g_det6_long
run_step "HV scan g_det7" $PY 10_hv_scan_efficiency.py g_det7_long

# ---- combined PDF ----
run_step "build HV PDF" $PY build_hv_scan_pdf.py g_det2 g_det3 g_det6_long g_det7_long

echo "=== HV scans finished $(date) ==="
echo "=== PDF: /home/dylan/x17/cosmic_bench/Analysis/june_hv_scans.pdf ==="
