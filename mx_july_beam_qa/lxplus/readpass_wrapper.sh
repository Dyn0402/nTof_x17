#!/bin/bash
# Condor read-pass wrapper for the July-beam QA. arg1 = run number (e.g. 224461).
#
# Runs the 6 pure-read scripts (01,02,03,06,07,09) on lxplus next to the EOS data,
# so the 13-18 GB raw file never has to be pulled local. Emits ~2.4 MB of caches +
# calib JSON, which condor transfers back (see readpass.sub). Plot locally afterwards
# with the 0Xb / 08 / 10 / 11 scripts, passing the run stem (e.g. `python 07b_... run224461`).
#
# Smoke-tested 2026-07-16 on run224461 (18.4 GB): ~89 min wall, ~15.9 GB peak RSS.
set -eo pipefail
RUN=$1
echo "START $(date '+%F %T') host $(hostname) scratch=$_CONDOR_SCRATCH_DIR"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
python3 -c "import numpy,uproot;print('numpy',numpy.__version__,'uproot',uproot.__version__)"

# Copy the official file to node-local scratch (fast NVMe) via xrootd.
LOCAL=$_CONDOR_SCRATCH_DIR/run${RUN}.root
SRC=root://eosexperiment.cern.ch//eos/experiment/ntof/processing/official/done/run${RUN}.root
echo "xrdcp start $(date '+%T')  $SRC"
t0=$SECONDS
xrdcp -f -s "$SRC" "$LOCAL"
echo "xrdcp done  $((SECONDS-t0)) s  size $(du -h "$LOCAL" | cut -f1)"

mkdir -p cache calib
rc_all=0
# Order matters: 01 first (bunch-selection cache), then 02; 07 needs 03's JSON,
# 09 needs 06's cache.
for s in 01_signal_qa 02_coincidence_scan 03_time_offsets 06_wall_geometry_test 07_mip_amplitude 09_late_inclusive; do
  echo "===== $s  start $(date '+%T') ====="
  ts=$SECONDS
  if /usr/bin/time -f "TIMING $s wall=%es maxrss=%MkB" python3 -u "$s.py" "$LOCAL"; then
    echo "----- $s OK  $((SECONDS-ts)) s -----"
  else
    echo "##### $s FAILED rc=$? #####"; rc_all=1
  fi
done

# ADC->mV factors from this run's DAQsettings, so mV plotting is fully run-portable
# (adc_mv writes calib/adc_to_mv_<stem>.json; plotters read it without opening the raw file).
python3 -c "from adc_mv import mv_factors; mv_factors('$LOCAL')" || echo "WARN: adc_to_mv gen failed"

echo "ALL DONE $(date '+%T')  outputs:"
ls -la cache calib
exit $rc_all
