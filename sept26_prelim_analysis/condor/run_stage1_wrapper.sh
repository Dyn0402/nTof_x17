#!/bin/bash
# Condor executable for ONE stage-1 sub-run census.
#   run_stage1_wrapper.sh <run> <subrun>
# Output: stage1_<run>_<subrun>.tar.gz containing out/stage1/
#
# WHY SUB-RUN GRANULARITY AND NOT PER-TAG. A sub-run's ~13 file tags would be
# ~6 min each and schedule faster, but candidate_filter measures the hot-channel
# mask ONCE on the first tag and reuses it for the rest -- the mask is a
# property of the run condition, not of a file (CLAUDE.md). Per-tag jobs would
# each measure their own and make the tags non-comparable inside one sub-run.
#
# WHY xrdcp AND NOT THE FUSE MOUNT. /eos/experiment is mounted on the workers,
# but a fuse mount held open for the 1-1.5 h this job runs is a good way to
# lose the job to a stale handle. Copying 1.2 GB up front costs ~2 min and the
# job is then independent of EOS for its whole life.
set -e
RUN=$1; SUB=$2

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

tar xzf code.tar.gz          # -> code/

EOSDIR=/eos/experiment/ntof/data/x17/july_beam/runs
# Trailing slash is REQUIRED: the listed paths are absolute, so
# "$HOST$f" must produce root://host//eos/... -- with a single
# slash xrootd calls it a relative path and refuses to open it.
HOST=root://eosexperiment.cern.ch/
# Layout must be <base>/runs/<run>/<sub>/...: sept26 resolves through
# $X17_RUNS but ntof_tracking.reco.io resolves through
# common/beam_july_paths.py, which wants the PARENT of runs/.
# Both are set below and must agree.
mkdir -p "stage/runs/$RUN/$SUB/combined_hits_root" "stage/runs/$RUN/$SUB/ntof_hits" out

echo "[wrapper] staging $RUN/$SUB from EOS"
xrdcp -f -s "$HOST$EOSDIR/$RUN/run_config.json" "stage/runs/$RUN/run_config.json"
n=0
for f in $(xrdfs eosexperiment.cern.ch ls "$EOSDIR/$RUN/$SUB/combined_hits_root" | grep '\.root$'); do
  xrdcp -f -s "$HOST$f" "stage/runs/$RUN/$SUB/combined_hits_root/"; n=$((n+1))
done
echo "[wrapper] staged $n combined_hits tag file(s)"
[ "$n" -gt 0 ] || { echo "[wrapper] FATAL: no combined_hits for $RUN/$SUB"; exit 2; }
# The slim: without it the IMPLIED class can never fire, and candidate_filter
# would carry on with empty n_TOF columns and write a census that looks fine.
# Missing slim is a real condition for some sub-runs, so warn, do not abort --
# but say it loudly enough that the merge can flag those sub-runs.
for f in $(xrdfs eosexperiment.cern.ch ls "$EOSDIR/$RUN/$SUB/ntof_hits" 2>/dev/null || true); do
  xrdcp -f -s "$HOST$f" "stage/runs/$RUN/$SUB/ntof_hits/" || true
done
ls "stage/runs/$RUN/$SUB/ntof_hits/"*.root >/dev/null 2>&1 \
  || echo "[wrapper] WARNING: no n_TOF slim for $RUN/$SUB -- IMPLIED is not measured here"

export X17_BEAM_JULY="$PWD/stage"
export X17_RUNS="$PWD/stage/runs"
export X17_SEPT26_OUT="$PWD/out"

cd code
python3 -W ignore -m sept26_prelim_analysis.candidate_filter \
        --run "$RUN" --subrun "$SUB" --all-tags --out ../out/stage1
cd ..

tar czf "stage1_${RUN}_${SUB}.tar.gz" out
echo "[wrapper] wrote stage1_${RUN}_${SUB}.tar.gz"
