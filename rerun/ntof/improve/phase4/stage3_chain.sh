#!/usr/bin/env bash
# stage3_chain.sh <cfg> -- the analysis view of one phase-4 test configuration, built exactly as stage3_r2_final
# (../run_r2_chain.sh) but with the r2 angle map and quality references held FIXED (a five-sub-run test cannot refit
# them, and holding them fixed keeps the comparison about the reconstruction alone).
#
#   bash rerun/ntof/improve/phase4/stage3_chain.sh ref
#   PHASE4_OUT=reftest PHASE4_DIR=/some/dir bash rerun/ntof/improve/phase4/stage3_chain.sh ref
#
# Under $PHASE4_DIR (default $WORK_DATA/ntof_pass1/phase4/<cfg>; all regenerable from EOS tests/phase4_<code>/out/<cfg>):
#   stage2/  the reconstruction outputs, fetched as a tar stream over ssh (PHASE4_FETCH=0 keeps an existing copy)
#   stage3/  ntof_stage3.py --allow-missing (context from context_r2, as stage3_r2)
#   vt/      v(t) view;  final/  + map (angle_map_r2.json) + censored-window cut + quality flags (stage3_r2_final's refs)
# Then compare with:  python rerun/ntof/improve/phase4/compare.py --cfg <dir>/final --ref <ref dir>/final ...
#
# The test never runs chamber B, and a sizing cluster covers only some acquisitions, so stage 3 runs with
# --allow-missing: (tag, arm) pairs without a candidates table are skipped and listed in the sidecar ("partial").
# Before 2026-11-24 this script called stage 3 without it and failed on every sub-run (re-audit, T1/T2).
# Limits: x/y_quality_ok take their reference median per sub-run over the tags present, so they are not comparable
# with pass 1 on a partial sub-run (they are not part of `gated`); stage 3 reads the pass-1 bundles for the bow patch
# and the pins, which is right for the osc bundles only because they differ in the hot lists alone.
# X17_QUALITY_REF_LATE_MS (roadmap 4.6) is passed through to ntof_stage3 if set: build it into its own PHASE4_DIR.
set -euo pipefail
CFG=$1
CODE=${PHASE4_CODE:-e86028b5}
cd "$(dirname "$0")/../../../.."
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
D=${PHASE4_DIR:-$P/phase4/$CFG}
J=${PHASE4_JOBS:-5}
PY="nice -n 10 .venv/bin/python"
EOS=/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/phase4_$CODE/out/${PHASE4_OUT:-$CFG}
SUBS=${PHASE4_SUBS:-run_145/stat090_0000,run_145/stat090_0001,run_145/stat090_0002,run_104/stat090_0003,run_139/stat090_0003}
MAP=rerun/ntof/stage3/angle_map/angle_map_r2.json
mkdir -p $D
LOG=$D/chain.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
if [ "${PHASE4_FETCH:-1}" = 1 ]; then
    say "0. fetch $EOS"
    rm -rf $D/stage2; mkdir -p $D/stage2
    # interactive lxplus reads /eos directly (the desktop has no xrootd binaries); the shared ssh connection, no 2FA
    ssh -o BatchMode=yes lxplus "tar -C $EOS -cf - ." | tar -C $D/stage2 -xf -
fi
n=$(find $D/stage2 -name '*.candidates.parquet' | wc -l); say "   $n candidates tables"
[ "$n" -gt 0 ] || { say "no candidates tables under $D/stage2 -- stop"; exit 1; }
say "1. stage 3 (--allow-missing)"
rm -rf $D/stage3
$PY rerun/ntof/stage3/ntof_stage3.py --stage2 $D/stage2 --side $D/stage2 --tests $D/stage2 --rerun-dbeam $D/stage2 \
    --subruns $SUBS --stage1-dir $P/context_r2/stage1 --trigtime-dir $P/context_r2/trigtime --out $D/stage3 \
    --allow-missing --jobs $J >> $LOG 2>&1
say "2. v(t) view"
rm -rf $D/vt
$PY rerun/ntof/stage3/vt_view.py --src $D/stage3 --out $D/vt --jobs $J >> $LOG 2>&1
say "3. map view + censored-window cut + quality flags"
rm -rf $D/final
$PY rerun/ntof/stage3/map_view.py --src $D/vt --out $D/final --map $MAP --jobs $J >> $LOG 2>&1
$PY rerun/ntof/stage3/censor_join.py --view $D/final --stage2 $D/stage2 --side $D/stage2 --tests $D/stage2 \
    --rerun-dbeam $D/stage2 --jobs $J >> $LOG 2>&1
cp $P/stage3_r2_final/quality_refs.json $D/final/
$PY rerun/ntof/stage3/quality_join.py apply --view $D/final --jobs $J >> $LOG 2>&1
say "done: $D/final ($(ls $D/final/tracks_run_*.parquet | wc -l) tables)"
