#!/usr/bin/env bash
# campaign_qa_chain_2026-09-10.sh -- per-run capsule imaging, then the
# campaign opening angle, on the condor FULL pass.
#
# Run this AFTER `fullpass_chain_2026-09-10.sh` (which builds the per-run k and
# the stage-3 track database).  Nothing here reconstructs anything.
#
#   1. campaign_imaging      the SCALE-FREE crossing, once per run.  This is
#                            the calibration QA: k cancels out of it, so it
#                            checks the alignment WITHOUT touching the scale.
#   2. tight_coincidence     the coincident pair sample, campaign-wide, on the
#                            full pass -- NOT --campaign, which reads the old
#                            allowlist tree.
#   3. campaign_angle        the opening-angle spectra per topology, against
#                            ipc_channels.thermal_spectrum() folded through the
#                            acceptance.
#   4. figures + reports     one report.html per analysis directory.
#
# Step 2 OVERWRITES <out>/tight_coincidence/*_campaign.*, so the previous
# pass's copies are archived first.  The same trap deleted run_145's published
# k calibration on 2026-09-09.
#
#   bash sept26_prelim_analysis/campaign_qa_chain_2026-09-10.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=/media/dylan/data/x17/sept26_prelim
JOBS=${JOBS:-8}

log () { echo; echo "[$(date +%H:%M:%S)] === $*"; }
step () { local what="$1"; shift; log "$what"; "$@" || { echo "!! $what FAILED"; exit 1; }; }

# ------------------------------------------------------- 1. imaging, per run
step "capsule imaging, per run" \
  $PY -W ignore -u -m sept26_prelim_analysis.campaign_imaging --jobs "$JOBS"

# ----------------------------------------- 2. the coincident pairs, campaign
ARCHIVE=$OUT/tight_coincidence/pre_$(date +%Y-%m-%d)
mkdir -p "$ARCHIVE"
for f in "$OUT"/tight_coincidence/*campaign*; do
  [ -f "$f" ] || continue
  [ -e "$ARCHIVE/$(basename "$f")" ] || cp "$f" "$ARCHIVE/" 2>/dev/null
done
log "archived the previous campaign tight-coincidence products to $ARCHIVE"

# NO --campaign: that flag points at <out>/stage3_campaign, the ALLOWLIST
# pass.  The default is <out>/stage3_fullpass, which is the full pass.
step "tight coincidence, campaign" \
  $PY -W ignore -u -m sept26_prelim_analysis.tight_coincidence --all-runs

# ------------------------------------------------------ 3. the opening angle
step "opening angle, campaign" \
  $PY -W ignore -u -m sept26_prelim_analysis.campaign_angle --jobs "$JOBS"

# ------------------------------------------------------ 4. figures, reports
step "imaging figures" \
  $PY -W ignore -m sept26_prelim_analysis.make_campaign_imaging_figures
step "imaging report" \
  $PY -W ignore -m sept26_prelim_analysis.make_campaign_imaging_report
step "angle figures" \
  $PY -W ignore -m sept26_prelim_analysis.make_campaign_angle_figures
step "angle report" \
  $PY -W ignore -m sept26_prelim_analysis.make_campaign_angle_report

log "DONE"
echo "  $OUT/imaging_campaign/report.html"
echo "  $OUT/angle_campaign/report.html"
