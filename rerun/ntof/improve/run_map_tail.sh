#!/usr/bin/env bash
# run_map_tail.sh -- finish both map chains from scint_ana on (2026-11-21: PASS1_DOWNSTREAM.sh ran ntof_scint_stack.ana
# on its default arms ABCD; pass 1 gives B no angles, so B's wall v fit had no tracks and both chains stopped there.
# The 11-20 baseline had been run by hand with --arms ACD.)  The two chains run one after the other (memory).
set -uo pipefail
cd "$(dirname "$0")/../../.."
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
LOG=$P/map_chains.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
R=rerun/ntof/stage3/resolution
TAIL=${TAIL:-"scint_ana scintillators chamber_b tight calo_beam
  pv_vertex_lab pv_diagnostics pv_vertex_image pv_z_image pv_vertex3d pv_y_image pv_intra_vertex"}
say "chain view (tail from scint_ana)"
X17_PASS1_VIEW=stage3_vt X17_PASS1_ANA=$P/ana LABEL=pass1 RESOLUTION_OUT=$R/resolution_pass1.csv \
  X17_RESOLUTION=$R/resolution_pass1.csv bash rerun/ntof/PASS1_DOWNSTREAM.sh $TAIL \
  > $P/map_chain_view_tail.log 2>&1 && say "chain view (tail) done" || say "chain view (tail) FAILED"
say "chain map (tail from scint_ana)"
X17_PASS1_VIEW=stage3_vtmap X17_PASS1_ANA=$P/ana_map LABEL=pass1map RESOLUTION_OUT=$R/resolution_pass1map.csv \
  X17_RESOLUTION=$R/resolution_pass1map.csv bash rerun/ntof/PASS1_DOWNSTREAM.sh $TAIL \
  > $P/map_chain_map_tail.log 2>&1 && say "chain map (tail) done" || say "chain map (tail) FAILED"
git checkout -- ntof_athens_26/pair_vertex_imaging/figures 2>/dev/null
.venv/bin/python -m rerun.ntof.improve.compare_chains >> $LOG 2>&1
say "tail all done"
