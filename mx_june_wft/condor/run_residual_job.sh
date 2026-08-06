#!/usr/bin/env bash
# run_residual_job.sh — condor worker for one shard of the residual audit.
#
#   run_residual_job.sh <cache.pkl> <bundle.tgz> <shard_i> <n_shards> <tag>
#
# Stacks (data - model) in the fit's own (strip, sample) frame; see
# bench/residual_audit.py. Writes result_resid_<tag>.tar.gz.
set -eu

CACHE=$1; BUNDLE_TGZ=$2; SHARD=$3; NSHARD=$4; TAG=$5

echo "[$(date -u +%H:%M:%S)] $(hostname) residual shard $SHARD/$NSHARD"

LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
if [ -f "$LCG" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$LCG"
  set -u
fi

tar xzf payload.tar.gz
tar xzf "$BUNDLE_TGZ"
BUNDLE=$(dirname "$(find . -name bundle.json | head -1)")

export WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1 WFT_CHI2DOF_BAD=250

mkdir -p out
time python payload/mx_june_wft/bench/residual_audit.py \
    --cache "$CACHE" --bundle "$BUNDLE" --out out \
    --shard "$SHARD/$NSHARD" --events 1200

tar czf "result_resid_${TAG}.tar.gz" out
echo "[$(date -u +%H:%M:%S)] done"
