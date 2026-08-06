#!/usr/bin/env bash
# run_calib.sh — condor worker payload for one RC-ladder calibration.
#
#   run_calib.sh <calib_cache.pkl> <label> <detector> <run_key> <fix_v> <lowgain 0|1>
#
# Input is the 2 MB corridor cache; no bench data is touched. Writes
# bundle_<label>.tar.gz for condor to bring back.
set -eu

CACHE=$1; LABEL=$2; DET=$3; KEY=$4; FIXV=$5; LOWGAIN=${6:-0}

echo "[$(date -u +%H:%M:%S)] $(hostname) calibrating $LABEL (v pinned $FIXV)"

LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
if [ -f "$LCG" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$LCG"
  set -u
fi
python -c "import numpy, scipy; print('numpy', numpy.__version__)"

tar xzf payload.tar.gz

ARGS=(--cache "$CACHE" --out "bundle_$LABEL" --detector "$DET"
      --run-key "$KEY" --share-lp --jobs "${OMP_NUM_THREADS:-4}")
[ "$FIXV" != "none" ] && ARGS+=(--fix-v "$FIXV")
[ "$LOWGAIN" = "1" ] && ARGS+=(--tmpl-tan-min 0.10 --tmpl-min-amp 250)

time python payload/mx_june_wft/bench/calib_hyper.py "${ARGS[@]}"

tar czf "bundle_${LABEL}.tar.gz" "bundle_$LABEL"
echo "[$(date -u +%H:%M:%S)] done"
