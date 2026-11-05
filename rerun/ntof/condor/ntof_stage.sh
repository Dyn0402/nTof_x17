#!/bin/bash
# ntof_stage.sh <code_dir> <run> <subrun> <tag> <arm> <bundle dir> <out dir> [extra ntof_reco.py args]
# Condor entry point for one n_TOF (acquisition, arm) of pass 1 (NTOF_PASS_PLAN P6).
#  - stages from EOS only what the arm reads: run_config.json, the arm's two FEU files of this acquisition from
#    decoded_root_fa80c06 (staged as decoded_root, rerun/inputs.py rule) and its rr1_cm0 combined hits (as
#    combined_hits_root);
#  - runs rerun/ntof/ntof_reco.py from the frozen code archive with the lock venv, staged into scratch by ~/bin/lockpy;
#  - copies <out>/<run>_<subrun>_<tag>_<arm>.{parquet,candidates.parquet,splits.parquet,meta.json} to EOS and
#    leaves nothing large in the scratch dir (condor copies it back to AFS otherwise).
set -u
CODE=$1; RUN=$2; SUB=$3; TAG=$4; ARM=$5; BUNDLE=$6; OUT=$7; shift 7
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') WFT_REPO=$CODE MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
EOSR=/eos/experiment/ntof/data/x17/july_beam/runs
case $ARM in A) FX=03; FY=04;; B) FX=05; FY=06;; C) FX=07; FY=08;; D) FX=01; FY=02;; esac
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/ntof_$$}/w; SD=$W/runs/run_$RUN/$SUB
mkdir -p $SD/decoded_root $SD/combined_hits_root $W/bundle $W/out
cpr() { for k in 1 2 3 4 5; do cp "$@" && return 0; sleep $((15*k)); done; echo "copy failed: $*"; return 1; }
cpr $EOSR/run_$RUN/run_config.json $W/runs/run_$RUN/ || exit 2
for F in $FX $FY; do
  f=$(ls $EOSR/run_$RUN/$SUB/decoded_root_fa80c06/*_datrun_${TAG}_$F.root 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "no decoded file for FEU $F tag $TAG"; exit 2; }
  cpr $f $SD/decoded_root/ || exit 2
done
h=$(ls $EOSR/run_$RUN/$SUB/combined_hits_root_rr1_cm0/*_datrun_${TAG}_feu-combined_hits.root 2>/dev/null | head -1)
[ -n "$h" ] || { echo "no rr1_cm0 hits for tag $TAG"; exit 2; }
cpr $h $SD/combined_hits_root/ || exit 2
cpr -r $BUNDLE/. $W/bundle/ || exit 2
NAME=run_${RUN}_${SUB}_${TAG}_$ARM
/afs/cern.ch/user/d/dneff/bin/lockpy $CODE/rerun/ntof/ntof_reco.py \
  --run $RUN --subrun $SUB --arm $ARM --tag $TAG --bundle $W/bundle --out $W/out/$NAME.parquet --base $W/runs \
  --jobs ${NTOF_JOBS:-4} "$@"
rc=$?
if [ $rc -eq 0 ]; then
  mkdir -p $OUT
  for f in $W/out/$NAME*; do cpr $f $OUT/$(basename $f).part && mv $OUT/$(basename $f).part $OUT/$(basename $f) || rc=3; done
fi
rm -rf $W
exit $rc
