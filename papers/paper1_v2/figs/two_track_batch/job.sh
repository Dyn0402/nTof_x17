#!/bin/bash
# job.sh <code dir> <shard> <nshard> -- one shard of the chamber A two-track overlay (make_two_track_separation.py run)
# <code dir>: a `git archive` of the commit carrying papers/paper1_v2/figs/make_two_track_separation.py, on AFS
# (~/x17rerun/code_<commit>). Python runs through lockpy (staged lock venv), never from EOS. Inputs are copied into the
# scratch dir with xrdcp (2.1 GB), outputs (a few MB) go to EOS with eosput, and the scratch dir is emptied before exit.
CODE=$1; SHARD=$2; NSHARD=$3
set -u
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/tt_$$}/tt; mkdir -p $W
MGM=root://eospublic.cern.ch/
B=/eos/experiment/ntof/data/x17/cosmic_bench
REL=mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V
OUT=$B/x17rerun/paper1_v2/two_track
get() {   # get <eos path> <local path>: xrdcp with retries
  mkdir -p "$(dirname "$2")"
  for k in 1 2 3 4 5; do xrdcp -s -f "$MGM$1" "$2" && return 0; echo "xrdcp attempt $k failed: $1" >&2; sleep $((10*k)); done
  return 1
}
fail() { echo "job.sh: $*" >&2; rm -rf $W; exit 1; }

RAW=$W/raw/$REL; TAB=$W/tables
for v in 07 08; do for t in 000 001 002 003; do
  f=MX17_long_run_resist_490V_drift_1000V_datrun_260627_22H49_${t}_${v}.root
  get $B/june_tests/$REL/decoded_root_fa80c06/$f $RAW/decoded_root_fa80c06/$f || fail "input $f"
done; done
for t in 000 001 002 003; do
  f=MX17_long_run_resist_490V_drift_1000V_datrun_260627_22H49_${t}_feu-combined_hits.root
  get $B/june_tests/$REL/combined_hits_root_rr1_cm0/$f $RAW/combined_hits_root_rr1_cm0/$f || fail "input $f"
done
get $B/june_tests/mx17_det3_saturday_scan_6-27-26/run_config.json $W/raw/mx17_det3_saturday_scan_6-27-26/run_config.json || fail run_config
for f in events.parquet events.candidates.parquet calib_bundle/bundle.json calib_bundle/arrays.npz calib_bundle/w0kw.json; do
  get $B/Analysis_pass1_twotrack/$REL/mx17_3/wft/$f $TAB/$f || fail "table $f"
done
get $B/x17rerun/paper1_eff/budget/$REL/mx17_3/rays_det3.parquet $W/rays_det3.parquet || fail rays
# a cut ROOT file reads as a short one, with exit 0 (rerun/condor/README.md, 11-08): check every input
python3 $CODE/rerun/condor/root_complete $RAW/decoded_root_fa80c06/*.root $RAW/combined_hits_root_rr1_cm0/*.root \
  || fail "incomplete ROOT input"

export TT_RAW=$RAW TT_TABLES=$TAB TT_RAYS=$W/rays_det3.parquet TT_WORK=$W/work \
       TT_RUNCFG=$W/raw/mx17_det3_saturday_scan_6-27-26/run_config.json
/afs/cern.ch/user/d/dneff/bin/lockpy $CODE/papers/paper1_v2/figs/make_two_track_separation.py run \
    --per-bin 300 --n-val 200 --shard $SHARD --nshard $NSHARD --jobs ${REQUEST_CPUS:-4}
rc=$?
if [ $rc -eq 0 ]; then
  for f in $W/work/run_$(printf %03d $SHARD)_*; do
    $CODE/rerun/condor/eosput $f $OUT/$(basename $f) || rc=3
  done
fi
# condor copies whatever is left in the scratch dir back to AFS: leave nothing behind
rm -rf $W
exit $rc
