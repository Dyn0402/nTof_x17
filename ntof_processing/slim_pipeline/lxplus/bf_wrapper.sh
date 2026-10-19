#!/bin/bash
# Condor wrapper for burst_bruteforce.py: stage one n_TOF run to node scratch,
# then brute-force the requested DREAM sub-run's bursts against it.
#   arguments = <ntof_run> <dream_run> <dream_subrun> [burst_bruteforce.py args]
set -eo pipefail
RUN=$1; DRUN=$2; DSUB=$3
[ -z "$DSUB" ] && { echo "usage: $0 <ntof_run> <dream_run> <dream_subrun> [args]"; exit 2; }
EXTRA=("${@:4}")

echo "START $(date '+%F %T')  host $(hostname)  $DRUN/$DSUB x $RUN  extra: ${EXTRA[*]}"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
export X17_BEAM_JULY=/eos/experiment/ntof/data/x17/july_beam
export X17_SLIM_CACHE=$_CONDOR_SCRATCH_DIR/ntof_cache
mkdir -p "$X17_SLIM_CACHE" bf

SRCDIR=$_CONDOR_SCRATCH_DIR/ntof
mkdir -p "$SRCDIR"
MGM=root://eosexperiment.cern.ch
DONE=/eos/experiment/ntof/processing/official/done/run${RUN}.root
COMPLETED=/eos/experiment/ntof/processing/official/completed/${RUN}
MERGED_BYTES=$(xrdfs "$MGM" stat "$DONE" 2>/dev/null | awk '/^Size:/ {print $2}') || true
: "${MERGED_BYTES:=0}"
t0=$SECONDS
if [ "$MERGED_BYTES" -gt 0 ]; then
  xrdcp -f -s "$MGM/$DONE" "$SRCDIR/run${RUN}.root"
else
  echo "no usable merged file; taking partials from $COMPLETED"
  xrdcp -f -s -r "$MGM/$COMPLETED" "$SRCDIR/" || { echo "PARTIAL COPY FAILED"; exit 4; }
  if [ -d "$SRCDIR/$RUN" ]; then mv "$SRCDIR/$RUN"/* "$SRCDIR/"; rmdir "$SRCDIR/$RUN"; fi
fi
echo "staged in $((SECONDS-t0)) s: $(du -sh "$SRCDIR" | cut -f1)"

TAG=${DRUN}_${DSUB}_${RUN}
python3 -u ntof_processing/slim_pipeline/burst_bruteforce.py \
    "$DRUN" "$DSUB" "$RUN" --ntof-source "$SRCDIR" --out "bf/$TAG.json" "${EXTRA[@]}" \
    2>&1 | tee "bf/$TAG.txt"
echo "ALL DONE $(date '+%T')"
