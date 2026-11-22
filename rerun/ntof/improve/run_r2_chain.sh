#!/usr/bin/env bash
# run_r2_chain.sh -- the analysis view of the rebuilt stage 3 (tracking audit fixes, 2026-11-22), from stage3_r2/.
#
#   bash rerun/ntof/improve/run_r2_chain.sh            (after ntof_stage3.py --out $P/stage3_r2 has finished)
#
# Under $WORK_DATA/ntof_pass1 (all regenerable from stage3_r2):
#   stage3_r2_vt/      v(t) view, unmapped (vt_view.py)                     -> map_view's input
#   stage3_r2_train/   the same + censored-window cut: the cosmic lines the map is fitted on
#   ana_r2/            line_residuals, angle_map, bd_map outputs (track_quality out dir)
#   stage3_r2_vtmap/   + the refitted map (map_view.py --map angle_map_r2.json)
#   stage3_r2_final/   + censored-window cut + quality flags: THE NEW ANALYSIS VIEW (renamed stage3_vt at the switch)
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
PY=.venv/bin/python
LOG=$P/r2_chain.log
MAP=rerun/ntof/stage3/angle_map/angle_map_r2.json
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
n=$(ls $P/stage3_r2/tracks_run_*.parquet | wc -l)
[ "$n" = 543 ] || { say "stage3_r2 has $n of 543 tables -- stop"; exit 1; }

say "1. v(t) view"
$PY rerun/ntof/stage3/vt_view.py --src $P/stage3_r2 --out $P/stage3_r2_vt --jobs 12 >> $LOG 2>&1
say "2. training copy + censored-window cut"
rm -rf $P/stage3_r2_train; cp -r $P/stage3_r2_vt $P/stage3_r2_train
$PY rerun/ntof/stage3/censor_join.py --view $P/stage3_r2_train --jobs 12 >> $LOG 2>&1
say "3. cosmic lines and the maps (registry beam-off, unmapped slopes)"
export X17_TQ_OUT=$P/ana_r2/track_quality
mkdir -p $X17_TQ_OUT
X17_PASS1_VIEW=stage3_r2_train $PY -m rerun.ntof.improve.line_residuals build >> $LOG 2>&1
X17_PASS1_VIEW=stage3_r2_train $PY -m rerun.ntof.improve.angle_map fit >> $LOG 2>&1
X17_PASS1_VIEW=stage3_r2_train $PY -m rerun.ntof.improve.bd_map --view $P/stage3_r2_train >> $LOG 2>&1
$PY -m rerun.ntof.improve.merge_maps --ac $X17_TQ_OUT/angle_map.json --d $X17_TQ_OUT/bd_map.json --out $MAP >> $LOG 2>&1
say "4. map view"
$PY rerun/ntof/stage3/map_view.py --src $P/stage3_r2_vt --out $P/stage3_r2_vtmap --map $MAP --jobs 12 >> $LOG 2>&1
say "5. censored-window cut + quality flags"
rm -rf $P/stage3_r2_final; cp -r $P/stage3_r2_vtmap $P/stage3_r2_final
$PY rerun/ntof/stage3/censor_join.py --view $P/stage3_r2_final --jobs 12 >> $LOG 2>&1
$PY rerun/ntof/stage3/quality_join.py refs --view $P/stage3_r2_final >> $LOG 2>&1
$PY rerun/ntof/stage3/quality_join.py apply --view $P/stage3_r2_final --jobs 12 >> $LOG 2>&1
say "done: $P/stage3_r2_final"
