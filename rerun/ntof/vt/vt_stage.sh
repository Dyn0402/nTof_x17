#!/bin/bash
# vt_stage.sh <code_dir> <run> <subrun> <arms> <tags,comma,separated>
# Condor entry point for P7a (NTOF_PASS_PLAN §3): the column duration T of one sub-run, per (acquisition, arm), on the
# pass's clean tracks (vt_acq.py).  All EOS I/O over xrootd through ~/bin/eosio (checksummed, retried), never the
# worker's /eos FUSE mount; the lock venv through ~/bin/lockpy.  Per tag and arm: stage the arm's two FEU files of
# decoded_root_fa80c06 (as decoded_root) and the pass's events parquet, fit, stage the .npz out, delete the inputs.
# Exit 2 = an input could not be staged, 3 = an output did not verify, 4 = vt_acq failed.
set -u
CODE=$1; RUN=$2; SUB=$3; ARMS=$4; TAGS=$5
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
export WFT_REPO=$CODE MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
EOSR=/eos/experiment/ntof/data/x17/july_beam/runs
A1=/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1
OUTD=$A1/vt/run_$RUN/$SUB
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/vt_$$}/w; SD=$W/runs/run_$RUN/$SUB
mkdir -p $SD/decoded_root $W/pass $W/tmpl $W/out
EOSIO=/afs/cern.ch/user/d/dneff/bin/eosio
XRD=${EOSIO_HOST:-eospublic.cern.ch}
xls() {
  local k out
  for k in 1 2 3 4 5; do out=$(xrdfs $XRD ls "$1" 2>/dev/null) && { echo "$out"; return 0; }; sleep $((15*k)); done
  echo "listing failed: $1" >&2; return 1
}
$EOSIO in $EOSR/run_$RUN/run_config.json $W/runs/run_$RUN/ || exit 2
DEC=$(xls $EOSR/run_$RUN/$SUB/decoded_root_fa80c06 | sort) || exit 2
rc=0
for ARM in $(echo $ARMS | sed 's/./& /g'); do
  case $ARM in A) FX=03; FY=04;; B) FX=05; FY=06;; C) FX=07; FY=08;; D) FX=01; FY=02;; esac
  [ -f $W/tmpl/$ARM.npz ] || $EOSIO in $A1/bundles/${ARM}_E2_beam/arrays.npz $W/tmpl/ || exit 2
  [ -f $W/tmpl/$ARM.npz ] || mv $W/tmpl/arrays.npz $W/tmpl/$ARM.npz
  for TAG in $(echo $TAGS | tr ',' ' '); do
    NAME=run_${RUN}_${SUB}_${TAG}_$ARM
    for F in $FX $FY; do
      f=$(echo "$DEC" | while read -r e; do case $(basename "$e") in *_datrun_${TAG}_$F.root) echo "$e"; break;; esac; done)
      [ -n "$f" ] || { echo "no decoded file for FEU $F tag $TAG"; exit 2; }
      $EOSIO in $f $SD/decoded_root/ || exit 2
    done
    $EOSIO in $A1/stage2/run_$RUN/$SUB/$ARM/$NAME.parquet $W/pass/ || exit 2
    /afs/cern.ch/user/d/dneff/bin/lockpy $CODE/rerun/ntof/vt/vt_acq.py $RUN $SUB $TAG $ARM --base $W/runs \
      --pass $W/pass/$NAME.parquet --tmpl $W/tmpl/$ARM.npz --out $W/out/vt_$NAME.npz --jobs ${NTOF_JOBS:-4} || { rc=4; }
    if [ -f $W/out/vt_$NAME.npz ]; then
      $EOSIO out $W/out/vt_$NAME.npz $OUTD/vt_$NAME.npz || rc=3
    fi
    rm -f $SD/decoded_root/* $W/pass/* $W/out/*
  done
done
rm -rf $W
exit $rc
