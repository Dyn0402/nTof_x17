#!/bin/bash
# Condor wrapper for the 2026-07-28 two-source (Y-88 + Cs-137) plastic calibration
# read pass. arg1 = run number (e.g. 224588).
#
# One job per run: xrdcp the official file to node-local scratch, run
# 33_srccal_spectra.py on it, tar the cache + calib output back. The campaign
# files are only ~0.3-0.6 GB (source runs, 6 min each), so this is minutes, not
# the ~89 min of the beam-run read pass.
#
# Output is a SINGLE tarball per job (srccal_out_<run>.tgz) rather than the
# cache/ + calib/ directories: nine jobs write into the same submit dir, and a
# tarball per job makes that unambiguous. Unpack on lxplus with
#   for t in srccal_out_*.tgz; do tar xzf "$t"; done
set -eo pipefail
RUN=$1
echo "START $(date '+%F %T') host $(hostname) run $RUN scratch=$_CONDOR_SCRATCH_DIR"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
python3 -c "import numpy,uproot;print('numpy',numpy.__version__,'uproot',uproot.__version__)"

LOCAL=$_CONDOR_SCRATCH_DIR/run${RUN}.root
SRC=root://eosexperiment.cern.ch//eos/experiment/ntof/processing/official/done/run${RUN}.root
echo "xrdcp start $(date '+%T')  $SRC"
t0=$SECONDS
xrdcp -f -s "$SRC" "$LOCAL"
echo "xrdcp done  $((SECONDS-t0)) s  size $(du -h "$LOCAL" | cut -f1)"

mkdir -p cache calib
/usr/bin/time -f "TIMING 33_srccal wall=%es maxrss=%MkB" \
    python3 -u 33_srccal_spectra.py "$LOCAL"

tar czf "srccal_out_${RUN}.tgz" \
    cache/33_srccal_run${RUN}.npz calib/adc_to_mv_run${RUN}.json
echo "DONE $(date '+%T')"
ls -la "srccal_out_${RUN}.tgz"
