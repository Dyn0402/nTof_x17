#!/bin/bash
# Condor wrapper for the FIT stage of the two-source plastic calibration:
# 34_srccal_edges.py (bootstrapped edge fits) + 35_srccal_calib.py (energy scale
# and controls). Normally these run on the laptop from the ~1 MB caches, but the
# bootstrap is a few thousand curve_fits and this offloads it when the local
# machine is busy.
#
# Input: the nine cache/33_srccal_run2245XX.npz files (transferred in).
# Output: srccalfit_out.tgz = calib/srccal_*.json + figures/33_srccal/*.png +
#         SRCCAL_RESULTS_2026-07-28.md
set -eo pipefail
echo "START $(date '+%F %T') host $(hostname)"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
python3 -c "import numpy,scipy,matplotlib;print('numpy',numpy.__version__,
'scipy',scipy.__version__,'matplotlib',matplotlib.__version__)"

mkdir -p cache calib figures
mv 33_srccal_run*.npz cache/ 2>/dev/null || true
# the 07-17 calibration, if it was shipped along, enables the transport table
mv y88_energy_calib.json calib/ 2>/dev/null || true
ls -la cache

/usr/bin/time -f "TIMING 34 wall=%es maxrss=%MkB" python3 -u 34_srccal_edges.py
/usr/bin/time -f "TIMING 35 wall=%es maxrss=%MkB" python3 -u 35_srccal_calib.py

tar czf srccalfit_out.tgz calib/srccal_*.json figures/33_srccal \
    SRCCAL_RESULTS_2026-07-28.md
echo "DONE $(date '+%T')"
ls -la srccalfit_out.tgz
