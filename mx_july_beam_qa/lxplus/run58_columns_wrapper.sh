#!/bin/bash
# Condor wrapper: drift-column statistics for one run_58 sub-run.
# arg1 = sub-run directory name, e.g. sngPS_dr700_r560_004
#
# Copies that sub-run's combined_hits_root (~130 MB) to node-local scratch via
# xrootd, runs run58_columns.py on it, and leaves a ~100 kB parquet in out/ for
# condor to transfer back. Nothing large ever leaves the node.
set -eo pipefail
SUB=$1
RUN=${RUN:-run_58}
EOSDIR=root://eospublic.cern.ch//eos/experiment/ntof/data/x17/july_beam/runs/${RUN}/${SUB}/combined_hits_root

echo "START $(date '+%F %T') host $(hostname) subrun=$SUB scratch=$_CONDOR_SCRATCH_DIR"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
python3 -c "import numpy,pandas,uproot;print('numpy',numpy.__version__,'pandas',pandas.__version__,'uproot',uproot.__version__)"

LOCAL=$_CONDOR_SCRATCH_DIR/hits
mkdir -p "$LOCAL" out
t0=$SECONDS
# xrdcp the directory recursively; -s silent, -f overwrite
xrdcp -f -s -r "$EOSDIR" "$LOCAL" || { echo "##### xrdcp FAILED"; exit 3; }
echo "xrdcp done $((SECONDS-t0)) s  $(du -sh "$LOCAL" | cut -f1)"
# xrdcp -r recreates the trailing directory name under $LOCAL
HITS=$(find "$LOCAL" -name '*combined_hits.root' -printf '%h\n' | head -1)
[ -n "$HITS" ] || { echo "##### no hits files found under $LOCAL"; exit 4; }

/usr/bin/time -f "TIMING wall=%es maxrss=%MkB" \
  python3 -u run58_columns.py "$HITS" "$SUB" --stripmap run58_stripmap.npz --out out
echo "ALL DONE $(date '+%T')"
ls -la out
