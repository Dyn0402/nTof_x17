#!/usr/bin/env bash
# acceptance_fold_chain_2026-09-10.sh
#
# Per-run acceptance and the aluminium capsule fold, end to end.
#
#   1. campaign_efficiency  -- the scintillator-tagged efficiency, all 36 runs,
#                              with the u map on single-track events and the
#                              response-versus-incidence table
#   2. campaign_acceptance  -- A(theta) per run, three efficiency variants,
#                              two vertex models (gas volume and capsule wall)
#   3. campaign_fold        -- the capsule and gas continua folded through it,
#                              against the measured campaign spectra
#   4. figures + report
#
# Prerequisites, none of which this script builds:
#   <out>/reco_fullpass        the condor FULL pass  (NOT <out>/fullpass, which
#                              is the allowlist pass despite its name)
#   <out>/stage1/census_*.csv  trigger totals, for the accidental control p0
#   <out>/slim/*.parquet       the exported n_TOF slim, all 293 sub-runs
#   <out>/imaging_campaign/    per-run source position
#   <out>/angle_campaign/      the measured spectra and the pair weights
#
# Nothing here writes into <out>/efficiency or <out>/angle: the published
# single-run products stay put so the comparison against them survives.
#
#   bash sept26_prelim_analysis/acceptance_fold_chain_2026-09-10.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
# The tree comes from paths.py, so a chain and the Python it calls cannot
# disagree about where it is; $X17_ROOT / $X17_SEPT26_OUT still move both.
OUT=$($PY -m sept26_prelim_analysis.paths --path out) || exit 1
LOG=$OUT/acceptance_fold_chain_2026-09-10.log
JOBS=${JOBS:-6}
N=${N:-8000000}

echo "=== acceptance + fold chain, $(date -Is) ===" | tee "$LOG"

echo "--- 1/4  per-run efficiency ---" | tee -a "$LOG"
$PY -m sept26_prelim_analysis.campaign_efficiency --jobs "$JOBS" 2>&1 \
    | tee -a "$LOG"

echo "--- 2/4  per-run acceptance ---" | tee -a "$LOG"
$PY -m sept26_prelim_analysis.campaign_acceptance --jobs "$JOBS" --n "$N" \
    2>&1 | tee -a "$LOG"

echo "--- 3/4  the fold ---" | tee -a "$LOG"
$PY -m sept26_prelim_analysis.campaign_fold --variant incidence 2>&1 \
    | tee -a "$LOG"

echo "--- 4/4  figures and report ---" | tee -a "$LOG"
$PY -m sept26_prelim_analysis.make_fold_figures 2>&1 | tee -a "$LOG"
$PY -m sept26_prelim_analysis.make_fold_report 2>&1 | tee -a "$LOG"

echo "=== done, $(date -Is) ===" | tee -a "$LOG"
echo "report: $OUT/fold_campaign/report.html" | tee -a "$LOG"
