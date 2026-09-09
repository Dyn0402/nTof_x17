#!/usr/bin/env bash
# Re-run everything downstream of the waveform pass, in dependency order, and
# publish. Idempotent: safe to run twice, and safe to run from cron.
#
#   merge_fullpass  the flat CERN pass -> the nested per-sub-run layout
#   noisy_channels  D's hot/dead/noisy strips, from raw hits (HANDOFF_D_NOISY_CHANNELS.md)
#   hot_seed_strata per-trigger hot content -> the production hot-channel cut
#   k_arm           the in-situ angle scale, per chamber (needs the pass)
#   gas_chain       v along the A->B->C->D line, and the H2O it implies
#   funnel          trigger -> track -> n_TOF confirmation (needs stage-1 census)
#   pairs           the controlled two-chamber rate (needs k_arm)
#   make_figures    the five figures (needs funnel + k_arm)
#   make_report     report.html + body.html (needs all of the above)
#   scintillators   S1: the wall read at both ends -> position along the bar
#   source_imaging  S2: the pointing crossing -> where the capsule is
#   pair_physics    S4a: X17 and IPC birth spectra, validated against Geant4
#   acceptance      S4b: the geometric toy, thrown flat in opening angle
#   opening_angle   S4c: the measured spectrum against the folded expectation
#   ipc_born        S4d: the Born multipole IPC continuum, self-validated
#   ipc_channels    S4d: which multipoles the >1 ms window actually makes
#   k_robustness    is k a property of the chamber, or of the sample?
#   normal_incidence what a head-on track costs the opening angle
#   chamber_b       B on its own chain, as the hit detector it is
#   rsync           -> lxplus:/eos/user/d/dneff/www/x17/<page>/
#
# noisy_channels and hot_seed_strata moved to the FRONT on 2026-09-08: the
# hot-channel cut they define is applied by k_arm.coincident_tracks, so it now
# feeds the angle scale and everything downstream of it. Running them last, as
# they were, would silently calibrate on the previous run's strata.
#
# Each step must succeed before the next runs: a report built on a half-updated
# k_arm would be worse than no report.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN=${RUN:-run_145}
SUBRUNS=${SUBRUNS:-stat090_0000,stat090_0001,stat090_0002}
OUT=/media/dylan/data/x17/sept26_prelim/funnel

step () {  # step <name> <command...>
  echo; echo "=== $1  $(date -Is)"
  shift
  if ! "$@"; then echo "!! FAILED: $*"; exit 1; fi
}

step "merge_fullpass" $PY -W ignore -m sept26_prelim_analysis.merge_fullpass --run "$RUN"
step "noisy channels" $PY -W ignore -m sept26_prelim_analysis.noisy_channels --run "$RUN" --subruns "$SUBRUNS"
step "hot strata"   $PY -W ignore -m sept26_prelim_analysis.hot_seed_strata --run "$RUN" --arm all
step "k_arm"        $PY -W ignore -m sept26_prelim_analysis.k_arm --run "$RUN" --subruns "$SUBRUNS"
step "gas_chain"   $PY -W ignore -m sept26_prelim_analysis.gas_chain --run "$RUN" --subruns "$SUBRUNS"
step "funnel"       $PY -W ignore -m sept26_prelim_analysis.funnel --run "$RUN" --subruns "$SUBRUNS"
step "pairs"        $PY -W ignore -m sept26_prelim_analysis.pairs --run "$RUN" --subruns "$SUBRUNS"
step "figures"      $PY -W ignore -m sept26_prelim_analysis.make_figures --run "$RUN"
step "report"       $PY -W ignore -m sept26_prelim_analysis.make_funnel_report --run "$RUN"

# --- S1-S4: the alignment and spectrum phase (PLAN.md sec 10) --------------- #
# Order is a real dependency chain: the acceptance toy throws from the position
# source_imaging measured, and opening_angle folds that acceptance.
step "scintillators" $PY -W ignore -m sept26_prelim_analysis.scintillators --run "$RUN" --subruns "$SUBRUNS"
step "scint figures" $PY -W ignore -m sept26_prelim_analysis.make_scint_figures --run "$RUN"
step "scint report"  $PY -W ignore -m sept26_prelim_analysis.make_scint_report --run "$RUN"
step "imaging"      $PY -W ignore -m sept26_prelim_analysis.source_imaging --run "$RUN" --subruns "$SUBRUNS"
step "imaging figs" $PY -W ignore -m sept26_prelim_analysis.make_imaging_figures --run "$RUN" --subruns "$SUBRUNS"
step "imaging rpt"  $PY -W ignore -m sept26_prelim_analysis.make_imaging_report --run "$RUN"
step "pair physics" $PY -W ignore -m sept26_prelim_analysis.pair_physics
step "acceptance"   $PY -W ignore -m sept26_prelim_analysis.acceptance --run "$RUN" --subruns "$SUBRUNS"
step "opening angle" $PY -W ignore -m sept26_prelim_analysis.opening_angle --run "$RUN" --subruns "$SUBRUNS"
step "angle figures" $PY -W ignore -m sept26_prelim_analysis.make_angle_figures --run "$RUN"
step "angle report"  $PY -W ignore -m sept26_prelim_analysis.make_angle_report --run "$RUN"

# --- S4d: what the IPC continuum should look like -------------------------- #
# No run dependence at all -- these are nuclear physics, not data -- so they
# sit after the angle page they are the companion to, and cost seconds.
step "ipc born"     $PY -W ignore -m sept26_prelim_analysis.ipc_born --al --write
step "ipc channels" $PY -W ignore -m sept26_prelim_analysis.ipc_channels --write
step "ipc figures"  $PY -W ignore -m sept26_prelim_analysis.make_ipc_figures
step "ipc report"   $PY -W ignore -m sept26_prelim_analysis.make_ipc_report

# --- detector studies: no page of their own, products only ----------------- #
step "k robustness"  $PY -W ignore -m sept26_prelim_analysis.k_robustness --run "$RUN" --subruns "$SUBRUNS"
step "normal incid"  $PY -W ignore -m sept26_prelim_analysis.normal_incidence --run "$RUN" --subruns "$SUBRUNS"
step "chamber B"     $PY -W ignore -m sept26_prelim_analysis.chamber_b --run "$RUN" --subruns "$SUBRUNS"

echo; echo "=== publish  $(date -Is)"
cp "$OUT/report.html" "$OUT/index.html"
BASE=/media/dylan/data/x17/sept26_prelim
for pair in "funnel:reco-funnel" "scint:scintillators" \
            "imaging:source-imaging" "angle:opening-angle" \
            "ipc:ipc-continuum"; do
  dir=${pair%%:*}; slug=${pair##*:}
  [ -f "$BASE/$dir/report.html" ] || continue
  cp "$BASE/$dir/report.html" "$BASE/$dir/index.html"
  rsync -a --delete -e "ssh -o BatchMode=yes -o ConnectTimeout=25" \
        "$BASE/$dir/index.html" "$BASE/$dir/figures" \
        "lxplus:/eos/user/d/dneff/www/x17/$slug/" || {
    echo "!! rsync failed for $slug -- the local products are still good"
    exit 1; }
  code=$(curl -sS -o /dev/null -w '%{http_code}' \
         "https://dylan-neff.web.cern.ch/x17/$slug/" || echo 000)
  echo "live: $slug HTTP $code"
done
echo "=== done $(date -Is)"
