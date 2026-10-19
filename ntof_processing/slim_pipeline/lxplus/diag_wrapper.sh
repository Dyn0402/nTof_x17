#!/bin/bash
# Condor wrapper for segment_diagnose.py.
#   arguments = <ntof_run> <dream_run> <dream_subrun>
set -eo pipefail
RUN=$1; DRUN=$2; DSUB=$3
[ -z "$DSUB" ] && { echo "usage: $0 <ntof_run> <dream_run> <dream_subrun>"; exit 2; }

echo "START $(date '+%F %T')  host $(hostname)  $DRUN/$DSUB x $RUN"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
export X17_BEAM_JULY=/eos/experiment/ntof/data/x17/july_beam
export X17_SLIM_CACHE=$_CONDOR_SCRATCH_DIR/ntof_cache
mkdir -p "$X17_SLIM_CACHE" diag

SRCDIR=$_CONDOR_SCRATCH_DIR/ntof
mkdir -p "$SRCDIR"
xrdcp -f -s "root://eosexperiment.cern.ch//eos/experiment/ntof/processing/official/done/run${RUN}.root" \
      "$SRCDIR/run${RUN}.root"
echo "xrdcp done"

python3 -u ntof_processing/slim_pipeline/segment_diagnose.py \
    "$DRUN" "$DSUB" "$RUN" --ntof-source "$SRCDIR" --span 5 \
    2>&1 | tee "diag/${DRUN}_${DSUB}_${RUN}.txt"
echo "ALL DONE $(date '+%T')"
