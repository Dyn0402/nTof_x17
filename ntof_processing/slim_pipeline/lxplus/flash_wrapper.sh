#!/bin/bash
# Condor wrapper for flash_reference_sweep.py. arg1 = DREAM run (e.g. run_124).
#
# Nothing is staged: the sweep reads two branches of each DREAM sub-run's
# combined hits off EOS FUSE, and the `bunches` tree of any published slim
# product. No n_TOF source, no scratch, no xrdcp -- a sub-run is ~3 s of work
# on local disk and the read is the cost.
set -eo pipefail
RUN=$1
[ -z "$RUN" ] && { echo "usage: $0 <dream_run> [extra args]"; exit 2; }
EXTRA=("${@:2}")

echo "START $(date '+%F %T')  host $(hostname)  $RUN  extra: ${EXTRA[*]}"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
export X17_BEAM_JULY=/eos/experiment/ntof/data/x17/july_beam
mkdir -p sweep

for f in ntof_processing/slim_pipeline/flash_reference_sweep.py \
         ntof_july_analysis/pulse_match.py common/beam_july_paths.py; do
  [ -f "$f" ] || { echo "MISSING INPUT: $f"; ls -la; exit 3; }
done
ls ntof_processing/slim_pipeline/cache_burst_census/${RUN}_*.json >/dev/null 2>&1 \
  || { echo "no burst censuses for $RUN in the sandbox"; exit 3; }

/usr/bin/time -f "TIMING sweep wall=%es maxrss=%MkB" \
  python3 -u ntof_processing/slim_pipeline/flash_reference_sweep.py \
    "$RUN" --out sweep "${EXTRA[@]}"
rc=$?
echo "ALL DONE $(date '+%T')"
find sweep -type f -printf '%10s  %p\n'
exit $rc
