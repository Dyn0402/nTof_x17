#!/usr/bin/env bash
# run_gap_fit.sh — the condor worker payload for one gap-fit shard.
#
#   run_gap_fit.sh <cache.pkl> <bundle.tgz> <label> <shard_i> <n_shards> [v_override] [k_bins]
#
# Runs in the job scratch directory with payload.tar.gz, the bench cache and the
# bundle tarball already transferred in. Writes result_<label>_<i>.tar.gz, which
# condor transfers back.
set -eu

CACHE=$1; BUNDLE_TGZ=$2; LABEL=$3; SHARD=$4; NSHARD=$5; VOVR=${6:-}; KBINS=${7:-}

echo "[$(date -u +%H:%M:%S)] $(hostname) starting $LABEL shard $SHARD/$NSHARD"

LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
if [ -f "$LCG" ]; then
  set +u                      # the LCG setup references unset vars (COMPILER)
  # shellcheck disable=SC1090
  source "$LCG"
  set -u
else
  echo "WARNING: $LCG not found, falling back to the system python" >&2
fi
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

tar xzf payload.tar.gz
tar xzf "$BUNDLE_TGZ"                 # -> bundle/
BUNDLE=$(dirname "$(find . -name bundle.json | head -1)")

export WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1 WFT_CHI2DOF_BAD=250

ARGS=(--cache "$CACHE" --bundle "$BUNDLE" --out out --label "$LABEL"
      --shard "$SHARD/$NSHARD" --jobs 1)
# 'none' is the placeholder the job list uses for an unset optional column:
# condor splits the line on whitespace, so an empty field cannot travel.
[ -n "$VOVR" ] && [ "$VOVR" != none ] && ARGS+=(--v-override "$VOVR")
[ -n "$KBINS" ] && [ "$KBINS" != none ] && ARGS+=(--k-bins "$KBINS")

time python payload/mx_june_wft/bench/gap_fit.py "${ARGS[@]}"

tar czf "result_${LABEL}_${SHARD}.tar.gz" out
echo "[$(date -u +%H:%M:%S)] done"
