#!/bin/bash
# Condor wrapper for the n_TOF -> DREAM slim. arg1 = n_TOF run number.
#
# One job = one n_TOF run = every DREAM sub-run that overlaps it. The 30 GB
# source is copied to node-local scratch once via xrootd, the bunch index is
# built once there, and each sub-run is a ~6 min segment on top.
#
# Reads DREAM (combined_hits, n1081b_config, beam CSVs) straight off EOS FUSE --
# only a few branches of ~1 GB per sub-run, so it is not worth staging.
#
# Writes into ./out, which condor transfers back (see slim.sub). Nothing is
# written to EOS from the worker: that needs a token the job does not have, and
# the outputs are ~35 MB per sub-run. Push them to EOS afterwards from lxplus
# with publish_to_eos.sh.
set -eo pipefail
RUN=$1
[ -z "$RUN" ] && { echo "usage: $0 <ntof_run>"; exit 2; }

echo "START $(date '+%F %T')  host $(hostname)  run $RUN"
echo "scratch $_CONDOR_SCRATCH_DIR"
df -h "$_CONDOR_SCRATCH_DIR" | tail -1

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
python3 -c "import numpy,uproot,pandas;print('numpy',numpy.__version__,
 'uproot',uproot.__version__,'pandas',pandas.__version__)"

# DREAM tree and the n_TOF cache both come from the environment so nothing in
# the package has to know it is on a worker.
export X17_BEAM_JULY=/eos/experiment/ntof/data/x17/july_beam
export X17_SLIM_CACHE=$_CONDOR_SCRATCH_DIR/ntof_cache
mkdir -p "$X17_SLIM_CACHE" out

# Preflight BEFORE the 30 GB copy. The first run of this job wasted two 135 s
# xrdcps discovering that transfer_input_files had flattened the package tree.
for f in ntof_processing/slim_pipeline/slim_run.py \
         ntof_dream_merge/ntof_io.py ntof_july_analysis/pulse_match.py \
         common/beam_july_paths.py \
         mx_july_beam_qa/calib/adc_to_mv_run224524.json; do
  [ -f "$f" ] || { echo "MISSING INPUT: $f"; echo "sandbox contains:"; ls -la; exit 3; }
done
python3 -c "import sys; sys.path.insert(0,'.'); import ntof_processing.slim_pipeline.slim_run" \
  || { echo "package imports are broken in the sandbox"; exit 3; }
echo "preflight OK"

# The n_TOF file -> node-local NVMe. The whole run is touched (the bunch index
# reads every tree's BunchNumber, then two passes read entry ranges), so a
# staged copy beats FUSE random access by a wide margin.
#
# Two layouts to handle. Most runs are a single merged run<run>.root in done/.
# For 28 runs the 5-7 August pass finished the RECONSTRUCTION but not the merge,
# so done/ has nothing (or a ZERO-BYTE file -- run224405 and run224667 are both
# 0 bytes) while completed/<run>/ holds the full, contiguous, v12 partial set.
# Those partials are the same processing, so prefer a merged file only when it
# is actually non-empty. See ../../skip_diagnosis/README.md.
SRCDIR=$_CONDOR_SCRATCH_DIR/ntof
mkdir -p "$SRCDIR"
MGM=root://eosexperiment.cern.ch
DONE=/eos/experiment/ntof/processing/official/done/run${RUN}.root
COMPLETED=/eos/experiment/ntof/processing/official/completed/${RUN}

MERGED_BYTES=$(xrdfs "$MGM" stat "$DONE" 2>/dev/null | awk '/^Size:/ {print $2}')
: "${MERGED_BYTES:=0}"

t0=$SECONDS
if [ "$MERGED_BYTES" -gt 0 ]; then
  echo "xrdcp start $(date '+%T')  merged $DONE ($MERGED_BYTES bytes)"
  xrdcp -f -s "$MGM/$DONE" "$SRCDIR/run${RUN}.root"
else
  echo "no usable merged file (size=$MERGED_BYTES); taking partials from $COMPLETED"
  xrdcp -f -s -r "$MGM/$COMPLETED" "$SRCDIR/" \
    || { echo "PARTIAL COPY FAILED for run $RUN"; exit 4; }
  # -r reproduces the remote directory name; flatten it into $SRCDIR
  if [ -d "$SRCDIR/$RUN" ]; then mv "$SRCDIR/$RUN"/* "$SRCDIR/"; rmdir "$SRCDIR/$RUN"; fi
  n=$(ls "$SRCDIR"/run${RUN}_[0-9]*.root 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "NO PARTIALS COPIED for run $RUN"; exit 4; }
  echo "copied $n partials"
fi
echo "xrdcp done  $((SECONDS-t0)) s  $(du -sh "$SRCDIR" | cut -f1)"

/usr/bin/time -f "TIMING slim wall=%es maxrss=%MkB" \
  python3 -u ntof_processing/slim_pipeline/slim_run.py "$RUN" \
    --out out --ntof-source "$SRCDIR"
rc=$?

echo "ALL DONE $(date '+%T')  outputs:"
find out -type f -printf '%10s  %p\n' | sort -k2
exit $rc
