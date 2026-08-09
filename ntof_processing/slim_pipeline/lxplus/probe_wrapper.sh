#!/bin/bash
# Condor wrapper for pss_tail_probe.py. arg1 = n_TOF run number.
# Same staging as slim_wrapper.sh; only the payload differs.
set -eo pipefail
RUN=$1
[ -z "$RUN" ] && { echo "usage: $0 <ntof_run>"; exit 2; }

echo "START $(date '+%F %T')  host $(hostname)  run $RUN"
df -h "$_CONDOR_SCRATCH_DIR" | tail -1
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

export X17_BEAM_JULY=/eos/experiment/ntof/data/x17/july_beam
export X17_SLIM_CACHE=$_CONDOR_SCRATCH_DIR/ntof_cache
mkdir -p "$X17_SLIM_CACHE" probe

for f in ntof_processing/slim_study/pss_tail_probe.py \
         ntof_dream_merge/ntof_io.py; do
  [ -f "$f" ] || { echo "MISSING INPUT: $f"; ls -la; exit 3; }
done

SRCDIR=$_CONDOR_SCRATCH_DIR/ntof
mkdir -p "$SRCDIR"
SRC=root://eosexperiment.cern.ch//eos/experiment/ntof/processing/official/done/run${RUN}.root
echo "xrdcp start $(date '+%T')"
t0=$SECONDS
xrdcp -f -s "$SRC" "$SRCDIR/run${RUN}.root"
echo "xrdcp done $((SECONDS-t0)) s"

/usr/bin/time -f "TIMING probe wall=%es maxrss=%MkB" \
  python3 -u ntof_processing/slim_study/pss_tail_probe.py \
    --ntof "$RUN" --bunches 150 --source "$SRCDIR" \
    --out probe/slim --json probe/pss_tail_probe.json
rc=$?
echo "ALL DONE $(date '+%T')"
find probe -type f -printf '%10s  %p\n' | sort -k2
exit $rc
