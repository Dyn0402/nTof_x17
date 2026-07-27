#!/usr/bin/env bash
# rerun_june_analysis.sh
#
# Full blind rerun of the June cosmic analysis on the a1cce79 (matched-filter)
# hits. Plan: RERUN_PLAN_2026-07-24.md. Launch in the background and walk away;
# continue-on-error, per-step timeout, one timestamped log. NO DAQ pull (data is
# local and freshly reprocessed). NO commits, NO lxplus writes.
#
#   nohup ./rerun_june_analysis.sh > /tmp/.../rerun_launch.log 2>&1 &
#
# Dry preview of the step list without executing:
#   ./rerun_june_analysis.sh --dry
set -uo pipefail
cd "$(dirname "$0")"

PY=../.venv/bin/python                       # NB: .venv, not the stale ../venv in docstrings
STEP_TIMEOUT=2700                            # 45 min ceiling per step
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

LOG_DIR=/home/dylan/x17/cosmic_bench/Analysis/_grand_logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/rerun_${STAMP}.log"
DIGEST="$(pwd)/RERUN_RESULTS_${STAMP}.md"
echo "Logging to $MAIN_LOG"
echo "Digest    $DIGEST"
if [ "$DRY" -eq 0 ]; then exec >>"$MAIN_LOG" 2>&1; fi

echo "======================================================================"
echo "=== Full June rerun on a1cce79 hits — started $(date) ==="
echo "=== analyzer generation on disk: matched-filter (significance branch) ==="
echo "======================================================================"

# Primary golden runs (full waveform chain). det3 reference FIRST so its saved
# hybrid model exists for the other detectors' transfer cross-check.
PRIMARY=(sat_det3 g_det3_wknd o22_long_det2 g_det4 g_det6_long g_det7_long)
DET3_MODEL=""     # filled after sat_det3's 34 --save-model

run_step() {                                # run_step "desc" cmd args...
    local desc="$1"; shift
    echo ">>> [$(date +%H:%M:%S)] $desc"
    if [ "$DRY" -eq 1 ]; then echo "    (dry) $*"; return 0; fi
    local rc=0
    timeout "$STEP_TIMEOUT" "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then echo "    OK   : $desc"
    else echo "    WARN : $desc  (exit $rc)"; fi
    return 0
}

# Resolve the Analysis cache dir for a key (run/sub/det) from qa_config, so we
# can wipe stale hit-derived caches before rebuilding.
cache_dir_of() {                            # -> <OUT_BASE>/cache for a key
    "$PY" - "$1" <<'PYEOF'
import sys, os, qa_config
c = qa_config.get_config(sys.argv[1])
print(os.path.join(c.OUT_BASE, 'cache'))
PYEOF
}
outbase_of() {                              # -> <OUT_BASE> for a key
    "$PY" - "$1" <<'PYEOF'
import sys, qa_config
print(qa_config.get_config(sys.argv[1]).OUT_BASE)
PYEOF
}
# 10_hv_scan_efficiency.py's --seed wants a PATH to an alignment.json, not a key.
# Resolve the long_run alignment of the seed key (freshly refit by 03 above).
align_seed_of() {                           # -> <OUT_BASE>/alignment_tpc_veto50/alignment.json
    "$PY" - "$1" <<'PYEOF'
import sys, os, qa_config
c = qa_config.get_config(sys.argv[1])
print(os.path.join(c.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json'))
PYEOF
}

wipe_caches() {                             # wipe_caches <key>
    local cd; cd=$(cache_dir_of "$1" 2>/dev/null) || return 0
    [ -z "$cd" ] && return 0
    echo "    wipe: $cd/event_results*.pkl + segment/feature CSVs"
    [ "$DRY" -eq 1 ] && return 0
    rm -f "$cd"/event_results*.pkl 2>/dev/null || true
    # waveform-derived caches live under alignment_tpc_veto*/ (segments, headon)
    local base; base=$(dirname "$cd")
    find "$base" -name microtpc_segments.csv -o -name headon_features.csv 2>/dev/null | xargs -r rm -f
}

process_key() {                             # full per-detector chain
    local k="$1"
    echo "###################### PRIMARY $k ######################"
    wipe_caches "$k"
    run_step "01 raw            $k" $PY 01_raw_detector_qa.py "$k"
    run_step "02 m3-reference   $k" $PY 02_m3_reference_qa.py "$k"
    run_step "04 deep-qa        $k" $PY 04_detector_deep_qa.py "$k"
    run_step "03 align --refit  $k" $PY 03_alignment_and_tpc.py "$k" --refit --full
    run_step "03 --refit noveto $k" $PY 03_alignment_and_tpc.py "$k" --refit --no-veto
    run_step "08 efficiency     $k" $PY 08_efficiency_maps.py "$k"
    run_step "09 breakdown      $k" $PY 09_efficiency_breakdown.py "$k"
    run_step "12 sliding-eff    $k" $PY 12_efficiency_map_sliding.py "$k" --kernel=25 --grid=120
    # --- waveform micro-TPC / unsharing chain ---
    run_step "26 unsharing      $k" $PY 26_unsharing_analysis.py "$k" --veto=50 --refit
    write_cshare "$k"                       # 26 stdout -> cache/cshare.json
    run_step "27 kernel         $k" $PY 27_unsharing_refinement.py "$k" --veto=50
    run_step "28 calibration    $k" $PY 28_angle_calibration.py "$k" --veto=50
    run_step "31 microtpc       $k" $PY 31_microtpc_metrics.py "$k" --veto=50 --rebuild
    run_step "33 headon         $k" $PY 33_headon_tracks.py "$k" --veto=50 --rebuild
    run_step "34 hybrid self    $k" $PY 34_hybrid_tracking.py "$k" --veto=50 --dump-events --save-model
    if [ -n "$DET3_MODEL" ] && [ "$k" != "sat_det3" ]; then
        run_step "34 hybrid xfer $k" $PY 34_hybrid_tracking.py "$k" --veto=50 --model="$DET3_MODEL"
    fi
    run_step "36 position       $k" $PY 36_position_estimators.py "$k" --veto=50 --rebuild
    run_step "42 time-res       $k" $PY 42_time_resolution.py "$k" --veto=50
    run_step "38 charge-balance $k" $PY 38_xy_charge_balance.py "$k" --veto=50
    run_step "38b cb-figs       $k" $PY 38b_charge_balance_report_figs.py "$k" --veto=50
    run_step "39 spark-deadtime $k" $PY 39_spark_deadtime.py "$k" --veto=50 --rebuild-amp
    run_step "40 spark-waveform $k" $PY 40_spark_waveforms.py "$k" --veto=50 --rebuild
    # remember det3's saved hybrid model for the transfer cross-check
    if [ "$k" = "sat_det3" ]; then
        DET3_MODEL=$(find "$(outbase_of "$k")" -name hybrid_model.json 2>/dev/null | head -1 || true)
        echo "    det3 hybrid model -> ${DET3_MODEL:-<none saved>}"
    fi
    echo "###################### DONE    $k ######################"
}

# Parse 26's "== measured sharing ==" block from the tail of the main log and
# emit cache/cshare.json for this key. 26 prints lines like  "FEU 6: c1=.. c2=..".
write_cshare() {
    [ "$DRY" -eq 1 ] && return 0
    local k="$1" cd; cd=$(cache_dir_of "$k") || return 0
    "$PY" - "$k" "$cd" "$MAIN_LOG" <<'PYEOF' || echo "    cshare: none parsed (26 found no leads?)"
import sys, os, re, json
key, cd, log = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(log, errors='ignore').read()
# take the LAST measured-sharing block (this key's 26 run is the most recent)
blk = txt.rsplit('== measured sharing', 1)[-1][:2000] if '== measured sharing' in txt else ''
# 26 prints: "  FEU 6: c1 = 0.247  c2 = 0.057  neighbour dt = ..."
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

# ---------------------------------------------------------------- primary
for k in "${PRIMARY[@]}"; do process_key "$k"; done

# ---------------------------------------------------------------- fleet / scans
echo "###################### FLEET / SCANS ######################"
run_step "30 fleet gas"                 $PY 30_fleet_gas_survey.py
run_step "14 drift-velocity --refit"    $PY 14_drift_velocity_scan.py sat_det3 --refit
run_step "21 geometry vdrift"           $PY 21_geometry_vdrift_scan.py sat_det3
run_step "23 core geometry vdrift"      $PY 23_core_geometry_vdrift.py sat_det3
run_step "15 vdrift vs magboltz"        $PY 15_drift_velocity_vs_magboltz.py sat_det3
run_step "17 gap attachment"            $PY 17_gap_attachment_test.py sat_det3
run_step "18 attachment vs magboltz"    $PY 18_attachment_vs_magboltz.py sat_det3
run_step "19 amplitude attachment"      $PY 19_amplitude_attachment_plot.py sat_det3
# 44 needs the key explicitly: config_from_argv() otherwise falls back to
# DEFAULT_RUN (the 6-16 ArIso det1 run) and looks for a CSV that does not exist.
run_step "44 final vdrift plot"         $PY 44_final_vdrift_plot.py sat_det3
run_step "45 slope-ref vdrift"          $PY 45_slope_reference_vdrift_scan.py sat_det3

# HV/drift-scan efficiency curves. 10 discovers the subruns of the key's RUN; its
# --seed is a PATH to an alignment.json (default guess is <RUN>/long_run/... which
# does not exist for these runs -- it would silently fall back to the hardcoded
# SEED_DEFAULT), so every call gets an explicit, freshly-refit seed.
run_step "10 hv det6 scan"  $PY 10_hv_scan_efficiency.py g_det6_hv --seed="$(align_seed_of g_det6_long)"
run_step "10 hv det7 scan"  $PY 10_hv_scan_efficiency.py g_det7_hv --seed="$(align_seed_of g_det7_long)"
run_step "10 det3 6-27 scan" $PY 10_hv_scan_efficiency.py sat_det3 --seed="$(align_seed_of sat_det3)"
run_step "10 hv det2 6-22"  $PY 10_hv_scan_efficiency.py o22_long_det2 --seed="$(align_seed_of o22_long_det2)"
run_step "10 hv det3 6-22"  $PY 10_hv_scan_efficiency.py o22_long_det3 --seed="$(align_seed_of o22_long_det3)"
# 6-23: the alignment lives under the run's `long_run` subrun, which IS 10's
# default seed path -- passing the key's own (longer_run) path would miss.
run_step "10 hv det3 6-23"  $PY 10_hv_scan_efficiency.py o23_long_det3
run_step "10 hv det4 6-23"  $PY 10_hv_scan_efficiency.py o23_long_det4

# vdrift reference-metric scan + window-truncation / Y-slow-rise re-confirm
# (plan §3; these take no key argument -- they scan the fleet themselves)
run_step "46 vdrift ref scan"    $PY 46_vdrift_ref_metric_scan.py
run_step "46b bias anatomy"      $PY 46b_bias_anatomy.py
run_step "46c gas scale/angles"  $PY 46c_gas_scale_and_anglebins.py
run_step "43 window truncation"  $PY 43_drift_window_truncation.py sat_det3 --veto=50
run_step "47 truncation survey"  $PY 47_window_truncation_survey.py
run_step "47b pulse shape"       $PY 47b_pulse_shape_and_leadtrunc.py

# per-detector overview PDF
run_step "build_final_pdf" $PY build_final_pdf.py "${PRIMARY[@]}"

echo "======================================================================"
echo "=== Full June rerun finished $(date) ==="
echo "=== log: $MAIN_LOG ==="
echo "======================================================================"

# digest is assembled by a companion python (reads the fresh outputs)
[ "$DRY" -eq 0 ] && $PY rerun_digest.py "$MAIN_LOG" "$DIGEST" "${PRIMARY[@]}" || true
