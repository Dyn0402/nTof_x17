#!/usr/bin/env bash
# chain_partial.sh <cfg-name> <eos out/<cfg> dir> <local dir> <subruns> -- the phase-4 stage3_chain.sh, adapted:
#   * stage 3 through ntof_stage3_partial.py (keeps only the tags/arms the test output has; skips chamber B)
#   * output under <local dir> (not $WORK_DATA/ntof_pass1/phase4), --jobs 3 with nice, --vt passed explicitly
#   * the fetch tars only what is on EOS (no rm of an existing stage2/ unless FETCH=1)
set -euo pipefail
CFG=$1; EOS=$2; D=$3; SUBS=$4
REPO=/home/dylan/PycharmProjects/nTof_x17
HERE=$(cd "$(dirname "$0")" && pwd)
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
PY="nice -n 10 $REPO/.venv/bin/python"
MAP=$REPO/rerun/ntof/stage3/angle_map/angle_map_r2.json
cd $REPO
mkdir -p $D; LOG=$D/chain.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
if [ "${FETCH:-1}" = 1 ]; then
  say "0. fetch $EOS"; rm -rf $D/stage2; mkdir -p $D/stage2
  ssh -o BatchMode=yes lxplus "tar -C $EOS -cf - ." | tar -C $D/stage2 -xf -
fi
say "   $(find $D/stage2 -name '*.candidates.parquet' | wc -l) candidates tables"
say "1. stage 3 (partial)"; rm -rf $D/stage3
$PY $HERE/ntof_stage3_partial.py --stage2 $D/stage2 --side $D/stage2 --tests $D/stage2 --rerun-dbeam $D/stage2 \
    --subruns $SUBS --stage1-dir $P/context_r2/stage1 --trigtime-dir $P/context_r2/trigtime \
    --vt $P/vt/vt_smooth.parquet --out $D/stage3 --jobs 3 >> $LOG 2>&1
say "2. v(t) view"; rm -rf $D/vt
$PY rerun/ntof/stage3/vt_view.py --src $D/stage3 --out $D/vt --jobs 3 >> $LOG 2>&1
say "3. map + censor + quality"; rm -rf $D/final
$PY rerun/ntof/stage3/map_view.py --src $D/vt --out $D/final --map $MAP --jobs 3 >> $LOG 2>&1
$PY rerun/ntof/stage3/censor_join.py --view $D/final --stage2 $D/stage2 --side $D/stage2 --tests $D/stage2 \
    --rerun-dbeam $D/stage2 --jobs 3 >> $LOG 2>&1
cp $P/stage3_r2_final/quality_refs.json $D/final/
$PY rerun/ntof/stage3/quality_join.py apply --view $D/final --jobs 3 >> $LOG 2>&1
say "done: $D/final ($(ls $D/final/tracks_run_*.parquet | wc -l) tables)"
