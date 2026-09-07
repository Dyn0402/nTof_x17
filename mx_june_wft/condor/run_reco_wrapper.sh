#!/bin/bash
# Condor executable for one campaign reco job.
#   run_reco_wrapper.sh <row> <tag> [extra run_reco_job.py args...]
# <tag> "prod" | "t0p" | "offcond" | ... ("-" = no tag). Output:
#   reco_r<row>_<tag>.tar.gz  containing out/<key>[__tag]/
set -e
ROW=$1; TAG=$2; shift 2 || true

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

tar xzf code.tar.gz          # -> code/
tar xzf bundles.tar.gz       # -> bundles/

ARGS=()
[ "$TAG" != "-" ] && [ "$TAG" != "prod" ] && ARGS+=(--out-tag "$TAG")
python3 run_reco_job.py "$ROW" --jobs "${RECO_JOBS:-8}" "${ARGS[@]}" "$@"

tar czf "reco_r${ROW}_${TAG}.tar.gz" out
echo "[wrapper] wrote reco_r${ROW}_${TAG}.tar.gz"
