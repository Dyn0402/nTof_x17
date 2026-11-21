#!/usr/bin/env bash
# run_map_rest.sh -- finish the map-only chain after run_map_chains.sh (2026-11-21: the two chains in parallel ran the
# machine out of memory in det_a_intra, ~45 GB for two copies; the map-only chain was stopped at sc_A and finishes here,
# alone, once the view chain and the comparison are done).
set -uo pipefail
cd "$(dirname "$0")/../../.."
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
LOG=$P/map_chains.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
while pgrep -f "improve/run_map_chains.sh" > /dev/null; do sleep 60; done
R=rerun/ntof/stage3/resolution
say "chain: map (rest, alone)"
X17_PASS1_VIEW=stage3_vtmap X17_PASS1_ANA=$P/ana_map LABEL=pass1map RESOLUTION_OUT=$R/resolution_pass1map.csv \
  X17_RESOLUTION=$R/resolution_pass1map.csv bash rerun/ntof/PASS1_DOWNSTREAM.sh \
  sc_A sc_late_A sc_C sc_late_C sc_events sc_phys sc_cmp scint_x scint_ana scintillators chamber_b tight calo_beam \
  pv_vertex_lab pv_diagnostics pv_vertex_image pv_z_image pv_vertex3d pv_y_image pv_intra_vertex \
  > $P/map_chain_map_rest.log 2>&1 && say "chain map (rest) done" || say "chain map (rest) FAILED"
git checkout -- ntof_athens_26/pair_vertex_imaging/figures 2>/dev/null
.venv/bin/python -m rerun.ntof.improve.compare_chains >> $LOG 2>&1
say "all done"
