#!/usr/bin/env bash
# det_a_chain_2026-09-10.sh
#
# Detector A, intra-chamber pairs, end to end.
#
#   1. det_a_intra        pairs + event-mixed null, the 2D efficiency map per
#                         run, the in-situ two-track resolution, the acceptance
#                         toy that carries both, and the folded comparison
#   2. figures + report
#
# Prerequisites, none of which this script builds:
#   <out>/stage3_fullpass      the track table (arm A must be angle-calibrated)
#   <out>/reco_fullpass        the condor FULL pass, for the efficiency map
#   <out>/stage1/census_*.csv  trigger totals, for the accidental control p0
#   <out>/slim/*.parquet       the exported n_TOF slim
#   <out>/imaging_campaign/    per-run source position
#
# Runtime is dominated by the second pass, which throws N pairs per run once
# the two-track resolution has been measured on the pooled sample.
#
#   bash sept26_prelim_analysis/det_a_chain_2026-09-10.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=${X17_SEPT26_OUT:-/media/dylan/data/x17/sept26_prelim}
LOG=$OUT/det_a_chain_2026-09-10.log
JOBS=${JOBS:-6}
N=${N:-3000000}

echo "=== detector A intra chain, $(date -Is) ===" | tee "$LOG"

echo "--- 1/2  pairs, map, resolution, acceptance, fold ---" | tee -a "$LOG"
$PY -m sept26_prelim_analysis.det_a_intra --jobs "$JOBS" --n "$N" 2>&1 \
    | tee -a "$LOG"

echo "--- 2/2  figures and report ---" | tee -a "$LOG"
$PY -m sept26_prelim_analysis.make_det_a_figures 2>&1 | tee -a "$LOG"
$PY -m sept26_prelim_analysis.make_det_a_report 2>&1 | tee -a "$LOG"

echo "=== done, $(date -Is) ===" | tee -a "$LOG"
echo "report: $OUT/det_a_intra/report.html" | tee -a "$LOG"
