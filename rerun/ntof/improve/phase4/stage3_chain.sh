#!/usr/bin/env bash
# stage3_chain.sh <cfg> -- the analysis view of one phase-4 test configuration, built exactly as stage3_r2_final
# (../run_r2_chain.sh) but with the r2 angle map and quality references held FIXED (a five-sub-run test cannot refit
# them, and holding them fixed keeps the comparison about the reconstruction alone).
#
#   bash rerun/ntof/improve/phase4/stage3_chain.sh ref
#
# Under $WORK_DATA/ntof_pass1/phase4/<cfg>/ (all regenerable from EOS tests/phase4_<code>/out/<cfg>):
#   stage2/  the reconstruction outputs, fetched over xrootd
#   stage3/  ntof_stage3.py (context from context_r2, as stage3_r2)
#   vt/      v(t) view;  final/  + map (angle_map_r2.json) + censored-window cut + quality flags (stage3_r2_final's refs)
# X17_QUALITY_REF_LATE_MS (roadmap 4.6) is passed through to ntof_stage3 if set.
set -euo pipefail
CFG=$1
CODE=${PHASE4_CODE:-e86028b5}
cd "$(dirname "$0")/../../../.."
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
D=$P/phase4/$CFG
PY=.venv/bin/python
EOS=/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/phase4_$CODE/out/$CFG
SUBS=run_145/stat090_0000,run_145/stat090_0001,run_145/stat090_0002,run_104/stat090_0003,run_139/stat090_0003
MAP=rerun/ntof/stage3/angle_map/angle_map_r2.json
mkdir -p $D
LOG=$D/chain.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "0. fetch $EOS"
rm -rf $D/stage2; mkdir -p $D/stage2
# interactive lxplus reads /eos directly (the desktop has no xrootd binaries); the shared ssh connection, no 2FA
ssh lxplus "tar -C $EOS -cf - ." | tar -C $D/stage2 -xf -
n=$(find $D/stage2 -name '*.candidates.parquet' | wc -l); say "   $n candidates tables"
say "1. stage 3"
rm -rf $D/stage3
$PY rerun/ntof/stage3/ntof_stage3.py --stage2 $D/stage2 --side $D/stage2 --tests $D/stage2 --rerun-dbeam $D/stage2 \
    --subruns $SUBS --stage1-dir $P/context_r2/stage1 --trigtime-dir $P/context_r2/trigtime --out $D/stage3 \
    --jobs 5 >> $LOG 2>&1
say "2. v(t) view"
$PY rerun/ntof/stage3/vt_view.py --src $D/stage3 --out $D/vt --jobs 5 >> $LOG 2>&1
say "3. map view + censored-window cut + quality flags"
rm -rf $D/final
$PY rerun/ntof/stage3/map_view.py --src $D/vt --out $D/final --map $MAP --jobs 5 >> $LOG 2>&1
$PY rerun/ntof/stage3/censor_join.py --view $D/final --stage2 $D/stage2 --side $D/stage2 --tests $D/stage2 \
    --rerun-dbeam $D/stage2 --jobs 5 >> $LOG 2>&1
cp $P/stage3_r2_final/quality_refs.json $D/final/
$PY rerun/ntof/stage3/quality_join.py apply --view $D/final --jobs 5 >> $LOG 2>&1
say "done: $D/final ($(ls $D/final/tracks_run_*.parquet | wc -l) tables)"
