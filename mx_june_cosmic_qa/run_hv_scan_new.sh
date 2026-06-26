#!/usr/bin/env bash
# run_hv_scan_new.sh
#
# Analyse the dedicated 6-26 det6+det7 resist-HV scan (mx17_det6_det7_hv_scan_6-26-26;
# det6 & det7 stepped together 400->500 V, drift 700 V). This run has NO long_run, so
# alignment is seeded from the 6-26 OVERNIGHT long_run (same detectors). Waits for the
# last points to finish decoding, then runs 10, rebuilds the combined HV PDF (overlaying
# the new low-V scan with the overnight higher-V scan), and copies both summary PDFs to
# rays.
set -uo pipefail
cd "$(dirname "$0")"

PY=../venv/bin/python
HOST=rays_daplxa
RUN=mx17_det6_det7_hv_scan_6-26-26
AREA=det6_det7
ANALYSIS=/home/dylan/x17/cosmic_bench/Analysis
SEED6=$ANALYSIS/mx17_det6_det7_overnight_6-26-26/long_run/mx17_6/alignment_tpc_veto50/alignment.json
SEED7=$ANALYSIS/mx17_det6_det7_overnight_6-26-26/long_run/mx17_7/alignment_tpc_veto50/alignment.json
LOG_DIR=$ANALYSIS/_grand_logs
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/hv_scan_new_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $MAIN_LOG"
exec >>"$MAIN_LOG" 2>&1
echo "=== new det6/det7 HV scan started $(date) ==="

run_step(){ local desc="$1"; shift; echo ">>> [$(date +%H:%M:%S)] $desc"
    local rc=0; timeout 3600 "$@" || rc=$?
    [ "$rc" -eq 0 ] && echo "    OK   : $desc" || echo "    WARN : $desc (exit $rc)"; return 0; }

# ---- wait for all points to decode, pulling each as it lands (deadline 45 min) ----
mapfile -t POINTS < <(ssh -o BatchMode=yes "$HOST" "ls -d /mnt/cosmic_data/MX17/Run/$RUN/resist_* | xargs -n1 basename")
echo "scan has ${#POINTS[@]} resist points"
deadline=$(( $(date +%s) + 45*60 ))
while :; do
    pending=0
    for sub in "${POINTS[@]}"; do
        local_ch="/home/dylan/x17/cosmic_bench/$AREA/$RUN/$sub/combined_hits_root"
        if [ -d "$local_ch" ] && ls "$local_ch"/*.root >/dev/null 2>&1; then continue; fi
        nch=$(ssh -o BatchMode=yes "$HOST" "ls /mnt/cosmic_data/MX17/Run/$RUN/$sub/combined_hits_root/*.root 2>/dev/null | wc -l" 2>/dev/null || echo 0)
        nm3=$(ssh -o BatchMode=yes "$HOST" "ls /mnt/cosmic_data/MX17/Run/$RUN/$sub/m3_tracking_root/*.root 2>/dev/null | wc -l" 2>/dev/null || echo 0)
        if [ "${nch:-0}" -ge 1 ] && [ "${nm3:-0}" -ge 1 ]; then
            run_step "PULL $sub" ./pull_run.sh "$RUN" "$sub" "$AREA" "$HOST"
        else
            pending=$((pending+1))
        fi
    done
    [ "$pending" -eq 0 ] && { echo "all points decoded+pulled"; break; }
    [ "$(date +%s)" -ge "$deadline" ] && { echo "deadline reached, $pending point(s) still pending — proceeding with what we have"; break; }
    echo "[$(date +%H:%M:%S)] $pending point(s) pending decode; wait 120s"
    sleep 120
done

# ---- analyse (seed from the overnight long_run alignment) ----
run_step "HV new det6" $PY 10_hv_scan_efficiency.py g_det6_hv --seed="$SEED6"
run_step "HV new det7" $PY 10_hv_scan_efficiency.py g_det7_hv --seed="$SEED7"

# ---- rebuild combined PDF (new low-V scan + overnight higher-V scan overlaid) ----
run_step "build HV PDF" $PY build_hv_scan_pdf.py \
    g_det2 g_det3 g_det6_hv g_det6_long g_det7_hv g_det7_long

# ---- copy both summary PDFs to rays ----
run_step "scp PDFs to rays" bash -c \
    "scp '$ANALYSIS/june_detectors_overview.pdf' '$ANALYSIS/june_hv_scans.pdf' $HOST:/mnt/cosmic_data/MX17/Analysis/"

echo "=== finished $(date) ==="
echo "=== PDF: $ANALYSIS/june_hv_scans.pdf ==="