#!/usr/bin/env bash
# run_full_june_qa.sh
#
# Grand June cosmic-bench QA in a SINGLE long-running process (launch in the
# background and walk away). Pulls the chosen best-stats subrun per detector from
# the DAQ PC, runs the full QA pipeline, builds the sliding-window within-5mm
# efficiency map, and compiles a one-page-per-detector PDF.
#
#   det2 + det3 : 6-22 overnight long_run  (shared pull, area det2_det3)
#   det4        : 6-24 daytime  long_run   (area det4_day)
#   det6 + det7 : 6-26 overnight short_run (area det6_det7) -- POLLED: not decoded
#                 at launch; we wait for combined_hits_root/m3_tracking_root on rays.
#
# Continue-on-error per step (NOT set -e); every step has a timeout so a single
# hang cannot stall the night. All output goes to a timestamped log under
# .../Analysis/_grand_logs/; the path is echoed once at launch.
set -uo pipefail
cd "$(dirname "$0")"

PY=../venv/bin/python
HOST=rays_daplxa
STEP_TIMEOUT=3600                 # max seconds per pipeline step
POLL_INTERVAL=1200               # 20 min between decode polls
POLL_HOURS=7                     # give up on det6/det7 after this long

LOG_DIR=/home/dylan/x17/cosmic_bench/Analysis/_grand_logs
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/grand_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $MAIN_LOG"      # -> tool-captured stdout, so the path is known
exec >>"$MAIN_LOG" 2>&1          # everything below -> log file

echo "=================================================================="
echo "=== Grand June QA started $(date) ==="
echo "=================================================================="

run_step() {                      # run_step "desc" cmd args...
    local desc="$1"; shift
    echo ">>> [$(date +%H:%M:%S)] $desc"
    local rc=0
    timeout "$STEP_TIMEOUT" "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then echo "    OK   : $desc"
    else echo "    WARN : $desc  (exit $rc)"; fi
    return 0
}

pull_one() {                      # pull_one run sub area
    run_step "PULL $1/$2 -> $3" ./pull_run.sh "$1" "$2" "$3" "$HOST"
}

process_key() {                   # process_key <qa_config key>
    local k="$1"
    echo "###################### PROCESS $k ######################"
    run_step "01 raw          $k" $PY 01_raw_detector_qa.py "$k"
    run_step "02 m3-reference $k" $PY 02_m3_reference_qa.py "$k"
    run_step "04 deep-qa      $k" $PY 04_detector_deep_qa.py "$k"
    run_step "03 align --full $k" $PY 03_alignment_and_tpc.py "$k" --full
    run_step "03 --no-veto    $k" $PY 03_alignment_and_tpc.py "$k" --no-veto
    run_step "08 efficiency   $k" $PY 08_efficiency_maps.py "$k"
    run_step "09 breakdown    $k" $PY 09_efficiency_breakdown.py "$k"
    run_step "amp-vs-strip    $k" $PY plot_amplitude_vs_strip.py "$k"
    run_step "12 sliding-eff  $k" $PY 12_efficiency_map_sliding.py "$k" --kernel=25 --grid=120
    echo "###################### DONE    $k ######################"
}

remote_ready() {                  # remote_ready run sub  -> 0 if decoded
    local nch nm3
    nch=$(ssh -o BatchMode=yes "$HOST" "ls /mnt/cosmic_data/MX17/Run/$1/$2/combined_hits_root/*.root 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    nm3=$(ssh -o BatchMode=yes "$HOST" "ls /mnt/cosmic_data/MX17/Run/$1/$2/m3_tracking_root/*.root 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    echo "[$(date +%H:%M:%S)] decode poll $1/$2: combined_hits=$nch m3=$nm3"
    [ "${nch:-0}" -ge 1 ] && [ "${nm3:-0}" -ge 1 ]
}

# ---------------------------------------------------------------- det2 & det3
pull_one mx17_det2_det3_overnight_6-22-26 long_run det2_det3
process_key g_det2
process_key g_det3

# ---------------------------------------------------------------- det4
pull_one mx17_det4_day_6-24-26 long_run det4_day
process_key g_det4

# interim PDF for the three detectors we can do immediately
run_step "PDF interim (2,3,4)" $PY build_final_pdf.py g_det2 g_det3 g_det4

# ---------------------------------------------------------------- det6 & det7
# Process EACH subrun of the 6-26 run as it finishes decoding. We poll all known
# cosmic subruns; when one becomes ready (combined_hits + m3) and hasn't been done
# yet, we pull it and run BOTH detectors, then refresh the PDF so partial results
# are visible by morning. det6 = FEU 3/4 key suffix, det7 = FEU 6/8.
RUN67=mx17_det6_det7_overnight_6-26-26
# subrun -> "det6key det7key"  (most-stats subruns first)
declare -A SUB_KEYS=(
    [long_run]="g_det6_long g_det7_long"
    [longer_run]="g_det6_longer g_det7_longer"
    [short_run]="g_det6 g_det7"
)
declare -A DONE_SUB=()
echo "### Polling for det6/det7 subrun decodes (up to ${POLL_HOURS} h) ###"
deadline=$(( $(date +%s) + POLL_HOURS*3600 ))
n67=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    progressed=0
    for sub in long_run longer_run short_run; do
        [ -n "${DONE_SUB[$sub]:-}" ] && continue
        if remote_ready "$RUN67" "$sub"; then
            echo "    $sub decoded; settle 120 s, re-check..."
            sleep 120
            remote_ready "$RUN67" "$sub" || continue
            pull_one "$RUN67" "$sub" det6_det7
            for key in ${SUB_KEYS[$sub]}; do process_key "$key"; done
            DONE_SUB[$sub]=1
            n67=$((n67+1))
            progressed=1
            run_step "PDF refresh after $sub" $PY build_final_pdf.py
        fi
    done
    # stop early once the highest-stats subrun (long_run) is in hand
    [ -n "${DONE_SUB[long_run]:-}" ] && { echo "    long_run done — stop polling."; break; }
    [ "$progressed" -eq 0 ] && sleep "$POLL_INTERVAL"
done
[ "$n67" -eq 0 ] && echo "### det6/det7 NO subrun decoded within ${POLL_HOURS} h — left as TODO ###"

# ---------------------------------------------------------------- final PDF
run_step "PDF final (all available)" $PY build_final_pdf.py

echo "=================================================================="
echo "=== Grand June QA finished $(date) ==="
echo "=== det6/det7 subruns processed this run: $n67  (${!DONE_SUB[*]}) ==="
echo "=== PDF: /home/dylan/x17/cosmic_bench/Analysis/june_grand_qa.pdf ==="
echo "=================================================================="
