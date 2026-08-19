#!/bin/bash
# Condor executable for one beam reco job.
#   run_beam_wrapper.sh <arm> <tag> [extra run_beam_job.py args...]
# Output: beam_<arm>_<tag>.tar.gz containing out/mx17_<arm>/
set -e
ARM=$1; TAG=$2; shift 2 || true

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

tar xzf code.tar.gz          # -> code/
tar xzf bundles.tar.gz       # -> bundles/

python3 run_beam_job.py "$ARM" "$TAG" --jobs "${RECO_JOBS:-8}" "$@"

tar czf "beam_${ARM}_${TAG}.tar.gz" out
echo "[wrapper] wrote beam_${ARM}_${TAG}.tar.gz"
