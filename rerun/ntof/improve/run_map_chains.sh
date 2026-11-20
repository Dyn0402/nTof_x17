#!/usr/bin/env bash
# run_map_chains.sh -- move the n_TOF analysis view onto the in-situ angle map + censored-window cut (Dylan 2026-11-20)
# and rerun the downstream on it, keeping the previous products for the comparison.
#
#   bash rerun/ntof/improve/run_map_chains.sh            (waits for map_view.py to finish first)
#
# Layout after it (all under $WORK_DATA/ntof_pass1, regenerable):
#   stage3_vt_nomap/  the v(t) view as of 11-20 (was stage3_vt)        ana_nomap/  its downstream (was ana)
#   stage3_vtmap/     + angle map (map_view.py)                       ana_map/    its downstream
#   stage3_vt/        + angle map + censored-window cut: THE VIEW     ana/        its downstream: THE PASS-1 ROOT
set -uo pipefail
cd "$(dirname "$0")/../../.."
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
PY=.venv/bin/python
LOG=$P/map_chains.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

while pgrep -f "stage3/map_view.py" > /dev/null; do sleep 60; done
n=$(ls $P/stage3_vtmap/tracks_run_*.parquet 2>/dev/null | wc -l)
m=$(ls $P/stage3_vt/tracks_run_*.parquet | wc -l)
if [ "$n" != "$m" ] || grep -q Traceback $P/stage3_vtmap_build.log; then say "map view incomplete ($n of $m) -- stop"; exit 1; fi
say "map view complete: $n tables"

# 1. the previous view and products become the no-map baseline
if [ ! -e $P/stage3_vt_nomap ]; then mv $P/stage3_vt $P/stage3_vt_nomap; fi
if [ ! -e $P/ana_nomap ]; then mv $P/ana $P/ana_nomap; ln -sfn ../stage3_vt_nomap $P/ana_nomap/stage3_pass1; fi

# 2. the analysis view: map + censored-window cut
if [ ! -e $P/stage3_vt ]; then
  say "copy stage3_vtmap -> stage3_vt"; cp -r $P/stage3_vtmap $P/stage3_vt.part && mv $P/stage3_vt.part $P/stage3_vt
fi
say "censor_join on stage3_vt"
$PY rerun/ntof/stage3/censor_join.py --view $P/stage3_vt --jobs 14 > $P/stage3_vt_censor.log 2>&1 || { say "censor_join FAILED"; exit 1; }
mkdir -p $P/ana
[ -e $P/ana/track_quality ] || mv $P/ana_nomap/track_quality $P/ana/track_quality
# the reco-tree imaging does not read the view: the same product for every chain
for A in ana_map ana; do mkdir -p $P/$A; [ -e $P/$A/imaging_campaign ] || cp -r $P/ana_nomap/imaging_campaign $P/$A/; done

STEPS="resolution focus focus_split pose_data pose_data_z pose_data_depth pose_shear
  pose_pose2_xb0 pose_pose2z_xb0 pose_pose2d_xb0 pose_pose2_xb-5 pose_pose2z_xb-5 pose_pose2d_xb-5
  pose_pose2_xb-10 pose_pose2z_xb-10 pose_pose2d_xb-10 capsule_tof capsule_tilt
  sc_A sc_late_A sc_C sc_late_C sc_events sc_phys sc_cmp scint_x scint_ana scintillators chamber_b tight calo_beam
  pv_vertex_lab pv_diagnostics pv_vertex_image pv_z_image pv_vertex3d pv_y_image pv_intra_vertex"
R=rerun/ntof/stage3/resolution

# 3-4. map only, and map + censored-window cut (the analysis view), in parallel (16 cores; each chain ~6 h alone)
say "chains: map, and map + cut (the view), in parallel"
( X17_PASS1_VIEW=stage3_vtmap X17_PASS1_ANA=$P/ana_map LABEL=pass1map RESOLUTION_OUT=$R/resolution_pass1map.csv \
    X17_RESOLUTION=$R/resolution_pass1map.csv bash rerun/ntof/PASS1_DOWNSTREAM.sh $STEPS > $P/map_chain_map.log 2>&1 \
    && say "chain map done" || say "chain map FAILED" ) &
( X17_PASS1_VIEW=stage3_vt X17_PASS1_ANA=$P/ana LABEL=pass1 RESOLUTION_OUT=$R/resolution_pass1.csv \
    X17_RESOLUTION=$R/resolution_pass1.csv bash rerun/ntof/PASS1_DOWNSTREAM.sh $STEPS > $P/map_chain_view.log 2>&1 \
    && say "chain view done" || say "chain view FAILED" ) &
wait
git checkout -- ntof_athens_26/pair_vertex_imaging/figures 2>/dev/null

# 5. the baseline's same-chamber comparison with today's rate-fit window (FIT_MS), so all three use one method
say "baseline: sc_phys, sc_cmp with FIT_MS"
X17_PASS1_VIEW=stage3_vt_nomap X17_PASS1_ANA=$P/ana_nomap X17_RESOLUTION=$R/resolution_pass1nomap.csv \
  bash rerun/ntof/PASS1_DOWNSTREAM.sh sc_phys sc_cmp >> $LOG 2>&1 || say "baseline sc FAILED"

say "compare"
$PY -m rerun.ntof.improve.compare_chains >> $LOG 2>&1
say "done"
