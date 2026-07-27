#!/usr/bin/env bash
# rerun_det3_full.sh
#
# Full detailed chain for det3 ONLY, on the significance-floor reco
# (DET3_RECO_FIX_2026-07-25.md). Same steps and same flags as
# rerun_june_analysis.sh's process_key(), plus the det3-specific fleet/scan
# steps, so results are directly comparable to RERUN_RESULTS_20260725_011307.md.
#
#   ./rerun_det3_full.sh [--dry] [key]      (default key: sat_det3)
#
# NOT included, deliberately: the fleet-level steps (30, 46, 46b, 46c, 47, 47b)
# and build_final_pdf. Those aggregate across detectors, and only det3/det4 are
# on the fixed reco right now, so they would silently mix hit generations.
# 03 is also skipped by default (--with-03 to force): it was already run with
# exactly these flags and its cache carries a matching .meta.json sidecar.
set -uo pipefail
cd "$(dirname "$0")"

PY=../.venv/bin/python
STEP_TIMEOUT=2700
DRY=0; WITH03=0
KEY=""
for a in "$@"; do
    case "$a" in
        --dry)     DRY=1 ;;
        --with-03) WITH03=1 ;;
        -*)        echo "unknown flag $a" >&2; exit 2 ;;
        *)         KEY="$a" ;;
    esac
done
KEY=${KEY:-sat_det3}

LOG_DIR=/home/dylan/x17/cosmic_bench/Analysis/_grand_logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/det3_full_${STAMP}.log"
echo "Logging to $MAIN_LOG"
if [ "$DRY" -eq 0 ]; then exec >>"$MAIN_LOG" 2>&1; fi

N_OK=0; N_WARN=0; WARNS=()

run_step() {
    local desc="$1"; shift
    echo ">>> [$(date +%H:%M:%S)] $desc"
    if [ "$DRY" -eq 1 ]; then echo "    (dry) $*"; return 0; fi
    local rc=0
    timeout "$STEP_TIMEOUT" "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then echo "    OK   : $desc"; N_OK=$((N_OK+1))
    else echo "    WARN : $desc  (exit $rc)"; N_WARN=$((N_WARN+1)); WARNS+=("$desc (exit $rc)"); fi
    return 0
}

cache_dir_of() {
    "$PY" - "$1" <<'PYEOF'
import sys, os, qa_config
print(os.path.join(qa_config.get_config(sys.argv[1]).OUT_BASE, 'cache'))
PYEOF
}
outbase_of() {
    "$PY" - "$1" <<'PYEOF'
import sys, qa_config
print(qa_config.get_config(sys.argv[1]).OUT_BASE)
PYEOF
}
align_seed_of() {
    "$PY" - "$1" <<'PYEOF'
import sys, os, qa_config
print(os.path.join(qa_config.get_config(sys.argv[1]).OUT_BASE, 'alignment_tpc_veto50', 'alignment.json'))
PYEOF
}

# 26 stdout -> cache/cshare.json  (verbatim from rerun_june_analysis.sh)
write_cshare() {
    [ "$DRY" -eq 1 ] && return 0
    local k="$1" cd; cd=$(cache_dir_of "$k") || return 0
    "$PY" - "$k" "$cd" "$MAIN_LOG" <<'PYEOF' || echo "    cshare: none parsed (26 found no leads?)"
import sys, os, re, json
key, cd, log = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(log, errors='ignore').read()
blk = txt.rsplit('== measured sharing', 1)[-1][:2000] if '== measured sharing' in txt else ''
cs = {int(m[0]): [float(m[1]), float(m[2])]
      for m in re.findall(r'FEU\s*(\d+):\s*c1\s*=\s*([-\d.]+)\s+c2\s*=\s*([-\d.]+)', blk)}
if cs:
    os.makedirs(cd, exist_ok=True)
    json.dump(cs, open(os.path.join(cd,'cshare.json'),'w'))
    print(f'    cshare.json <- {cs}')
else:
    sys.exit(1)
PYEOF
}

echo "======================================================================"
echo "=== det3 full detailed chain — key=$KEY — started $(date) ==="
echo "=== reco: adaptive per-plane significance floor (cm.SIG_REL_FLOOR) ==="
echo "======================================================================"

# Waveform-derived caches are hit-generation dependent; 31/33/36 --rebuild would
# regenerate them anyway, but drop them explicitly so nothing stale can be read.
BASE=$(outbase_of "$KEY")
echo "OUT_BASE=$BASE"
if [ "$DRY" -eq 0 ]; then
    find "$BASE" \( -name microtpc_segments.csv -o -name headon_features.csv \) 2>/dev/null | xargs -r rm -f
fi

run_step "01 raw            $KEY" $PY 01_raw_detector_qa.py "$KEY"
run_step "02 m3-reference   $KEY" $PY 02_m3_reference_qa.py "$KEY"
run_step "04 deep-qa        $KEY" $PY 04_detector_deep_qa.py "$KEY"
if [ "$WITH03" -eq 1 ]; then
    run_step "03 align --refit  $KEY" $PY 03_alignment_and_tpc.py "$KEY" --refit --full
    run_step "03 --refit noveto $KEY" $PY 03_alignment_and_tpc.py "$KEY" --refit --no-veto
else
    echo ">>> [$(date +%H:%M:%S)] 03 SKIPPED (already run with these flags; cache sidecar matches)"
fi
run_step "08 efficiency     $KEY" $PY 08_efficiency_maps.py "$KEY"
run_step "09 breakdown      $KEY" $PY 09_efficiency_breakdown.py "$KEY"
run_step "12 sliding-eff    $KEY" $PY 12_efficiency_map_sliding.py "$KEY" --kernel=25 --grid=120

# --- waveform micro-TPC / unsharing chain ---
run_step "26 unsharing      $KEY" $PY 26_unsharing_analysis.py "$KEY" --veto=50 --refit
write_cshare "$KEY"
run_step "27 kernel         $KEY" $PY 27_unsharing_refinement.py "$KEY" --veto=50
run_step "28 calibration    $KEY" $PY 28_angle_calibration.py "$KEY" --veto=50
run_step "31 microtpc       $KEY" $PY 31_microtpc_metrics.py "$KEY" --veto=50 --rebuild
run_step "33 headon         $KEY" $PY 33_headon_tracks.py "$KEY" --veto=50 --rebuild
run_step "34 hybrid self    $KEY" $PY 34_hybrid_tracking.py "$KEY" --veto=50 --dump-events --save-model
run_step "36 position       $KEY" $PY 36_position_estimators.py "$KEY" --veto=50 --rebuild
run_step "42 time-res       $KEY" $PY 42_time_resolution.py "$KEY" --veto=50
run_step "38 charge-balance $KEY" $PY 38_xy_charge_balance.py "$KEY" --veto=50
run_step "38b cb-figs       $KEY" $PY 38b_charge_balance_report_figs.py "$KEY" --veto=50
run_step "39 spark-deadtime $KEY" $PY 39_spark_deadtime.py "$KEY" --veto=50 --rebuild-amp
run_step "40 spark-waveform $KEY" $PY 40_spark_waveforms.py "$KEY" --veto=50 --rebuild

# --- det3-specific drift/gas/geometry scans (all take an explicit key) ---
echo "###################### det3 SCANS ######################"
run_step "14 drift-velocity --refit" $PY 14_drift_velocity_scan.py "$KEY" --refit
run_step "21 geometry vdrift"        $PY 21_geometry_vdrift_scan.py "$KEY"
run_step "23 core geometry vdrift"   $PY 23_core_geometry_vdrift.py "$KEY"
run_step "15 vdrift vs magboltz"     $PY 15_drift_velocity_vs_magboltz.py "$KEY"
run_step "17 gap attachment"         $PY 17_gap_attachment_test.py "$KEY"
run_step "18 attachment vs magboltz" $PY 18_attachment_vs_magboltz.py "$KEY"
run_step "19 amplitude attachment"   $PY 19_amplitude_attachment_plot.py "$KEY"
run_step "43 window truncation"      $PY 43_drift_window_truncation.py "$KEY" --veto=50
run_step "44 final vdrift plot"      $PY 44_final_vdrift_plot.py "$KEY"
run_step "45 slope-ref vdrift"       $PY 45_slope_reference_vdrift_scan.py "$KEY"
run_step "10 hv scan"                $PY 10_hv_scan_efficiency.py "$KEY" --seed="$(align_seed_of "$KEY")"

MODEL=$(find "$BASE" -name hybrid_model.json 2>/dev/null | head -1 || true)
echo "hybrid model -> ${MODEL:-<none saved>}"

echo "======================================================================"
echo "=== det3 full chain DONE $(date):  $N_OK OK / $N_WARN WARN ==="
for w in "${WARNS[@]:-}"; do [ -n "$w" ] && echo "    WARN: $w"; done
echo "=== log: $MAIN_LOG ==="
echo "======================================================================"
