#!/usr/bin/env bash
# det_a_scint_chain_2026-09-10.sh
#
# Detector A, every track, against the scintillators behind it, end to end.
#
#   1. det_a_scint        project every gated arm-A track onto the SiPM wall,
#                         the plastic bars and the liquid cell; match against
#                         what fired; measure the position tolerance in situ;
#                         measure the angle scale from the surveyed boundaries
#   2. figures + report
#
# Prerequisites, none of which this script builds:
#   <out>/stage3_fullpass      the track table -- arm A MUST be angle-calibrated
#                              (run_126, run_154 and run_156 are not, and the
#                              module refuses them by name rather than
#                              reporting their rate as zero)
#   <out>/reco_fullpass        the condor FULL pass, for the sub-run list
#   <out>/slim/*.parquet       the exported n_TOF slim
#   <runs>/<run>/run_config.json   the DAQ survey; every layer position and the
#                              plastic detn map are read from it
#
# Runtime is dominated by step 1, which reads the slim for 31 runs.  The
# per-track tables it writes under <out>/det_a_scint/tracks/ are the reusable
# product; the aggregates are rebuilt from them on every pass.
#
#   bash sept26_prelim_analysis/det_a_scint_chain_2026-09-10.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=${X17_SEPT26_OUT:-/media/dylan/data/x17/sept26_prelim}
LOG=$OUT/det_a_scint_run.log
JOBS=${JOBS:-6}

echo "=== detector A -> scintillators, $(date -Is) ===" | tee "$LOG"

echo "--- 1/2  project, match, tolerance, angle scale ---" | tee -a "$LOG"
$PY -m sept26_prelim_analysis.det_a_scint --jobs "$JOBS" 2>&1 | tee -a "$LOG"

echo "--- 2/2  figures and report ---" | tee -a "$LOG"
$PY -m sept26_prelim_analysis.make_det_a_scint_figures 2>&1 | tee -a "$LOG"
$PY -m sept26_prelim_analysis.make_det_a_scint_report 2>&1 | tee -a "$LOG"

echo "=== done, $(date -Is) ===" | tee -a "$LOG"
echo "report: $OUT/det_a_scint/report.html" | tee -a "$LOG"
