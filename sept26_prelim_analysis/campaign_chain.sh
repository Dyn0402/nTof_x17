#!/usr/bin/env bash
# Everything downstream of the stage-2 condor pass, in dependency order.
#
#   fetch_stage2     EOS tarballs -> <out>/fullpass/<run>/<subrun>/mx17_<arm>/
#   merge_campaign   per-tag events_*.parquet -> events_prelim.parquet
#   k_arm            the in-situ angle scale, PER RUN
#   campaign_tracks  stage 3 per sub-run, then one concatenated table
#   tight_coincidence the two-arm scintillator cut on the campaign sample
#
# ORDER IS NOT COSMETIC. `k_arm` must run BEFORE `campaign_tracks`, because
# `build_tracks` stamps `angle_calibrated` at build time and
# `source_imaging._track_table` drops every track where it is False -- so a run
# whose k is computed afterwards contributes NOTHING to the pair analysis, and
# does so silently. `merge_campaign` must run before `k_arm` for the same kind
# of reason: k_arm reads `events_prelim.parquet`, which stage 2 does not write.
#
# Idempotent: re-run it as more of the campaign lands. Steps that are already
# done are cheap no-ops.
#
#   bash campaign_chain.sh              # everything present
#   SKIP_FETCH=1 bash campaign_chain.sh # already pulled
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=/media/dylan/data/x17/sept26_prelim

step () { echo; echo "=== $1  $(date -Is)"; shift; if ! "$@"; then echo "!! FAILED: $*"; exit 1; fi; }

if [ "${SKIP_FETCH:-0}" != "1" ]; then
  step "fetch stage2" bash sept26_prelim_analysis/condor/fetch_stage2.sh
  # The calibration pass is a SEPARATE product tree. Its allowlist takes SINGLE
  # at 0.25 against the main pass's 0.05, so it is a superset with the same
  # salt -- richer in exactly the pointing-coincident single tracks k_arm
  # measures, which the main pass prescales away. It is fetched apart so its
  # different event selection never overwrites the main reco under the same
  # run/subrun/arm/tag name.
  step "fetch calib" env \
      REMOTE=lxplus:/eos/user/d/dneff/x17/sept26_calib \
      LOCALPKG=/home/dylan/x17/sept26_calib \
      OUT="$OUT/fullpass_calib" \
      bash sept26_prelim_analysis/condor/fetch_stage2.sh
fi
step "merge" $PY -W ignore -m sept26_prelim_analysis.merge_campaign
step "merge calib" $PY -W ignore -m sept26_prelim_analysis.merge_campaign \
     --fullpass "$OUT/fullpass_calib"

# k_arm, per run, over the sub-runs that actually arrived. A run with too few
# coincident tracks fails to certify -- that is a real outcome, not an error,
# so a non-zero exit here is reported and the chain continues.
echo; echo "=== k_arm per run  $(date -Is)"
# Calibrate from the CALIB tree where it exists, the main tree otherwise. The
# main pass prescales SINGLE to 0.05 and cannot reach k_arm's 200-track-per-
# (arm, sub-run) floor -- measured 2026-09-09, 4-168 tracks. Runs with no calib
# products still get a k_arm attempt, which will honestly fail to certify
# rather than quietly inventing one.
for d in "$OUT"/fullpass_calib/run_* "$OUT"/fullpass/run_*; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  [ -f "$OUT/kcal/k_arm_$run.json" ] && continue     # already calibrated
  subs=$(ls "$d" 2>/dev/null | grep '^stat090_' | paste -sd, -)
  [ -n "$subs" ] || continue
  echo "--- $run  ($(echo "$subs" | tr ',' '\n' | wc -l) sub-run(s))"
  # --merged is REQUIRED: k_arm's default is a hardcoded <out>/fullpass/run_145,
  # so without it every run is calibrated from run_145's tables and stamped with
  # the wrong run's name. Found 2026-09-09 when run_156 "measured" 179 tracks
  # that were actually run_145's.
  $PY -W ignore -m sept26_prelim_analysis.k_arm --run "$run" --subruns "$subs" \
      --merged "$d" \
      2>&1 | tail -6 || echo "  !! k_arm did not certify $run -- its tracks will carry raw angles only"
done

step "tracks" $PY -W ignore -m sept26_prelim_analysis.campaign_tracks --jobs "${JOBS:-8}"
# Pooled over every run that has stage-3 tracks. run_79/run_81 are excluded by
# default -- they are the pre-27-Jul-access condition and carry ~2x the INTER
# fraction (OVERNIGHT_2026-09-08.md). Set INCLUDE_PRE_ACCESS=1 to override,
# knowing what that mixes.
step "tight coincidence" $PY -W ignore -m sept26_prelim_analysis.tight_coincidence \
     --all-runs --campaign ${INCLUDE_PRE_ACCESS:+--include-pre-access}
step "tight figures" $PY -W ignore -m sept26_prelim_analysis.make_tight_figures \
     --run campaign

echo; echo "=== done $(date -Is)"
du -sh "$OUT/stage3_campaign" "$OUT/slim" 2>/dev/null
