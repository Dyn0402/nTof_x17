#!/bin/bash
# job.sh <run> <subrun> <tag>  -- P1 decoder spot check for one n_TOF acquisition (NTOF_PASS_PLAN_2026-11-02 §3 P1).
# For each of the 8 FEUs: decode the .fdf fresh with the pinned decoder (fa80c06, as the bench/SPS pass 1), compare
# branch by branch with the EOS decoded_root (rerun/spotcheck/compare_decoded.py), and audit it against the
# independent numpy decoder (rerun/decoder_audit/audit_decode.py: complete/incomplete events, missing eventIDs).
# Nothing is written next to the data; results -> $OUT/<run>_<subrun>_<tag>_f<FEU>.{cmp,audit}.json
set -u
RUN=$1; SUB=$2; TAG=$3
D=/eos/experiment/ntof/data/x17/july_beam/runs/run_$RUN/$SUB
OUT=/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/checks/decode_spot
for k in 1 2 3; do mkdir -p $OUT 2>/dev/null; [ -d $OUT ] && break; sleep 5; done
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
DEC=$HOME/x17rerun/mm_strip_reconstruction/build/decoder/decode
PY=/afs/cern.ch/user/d/dneff/bin/lockpy
TOOLS=$HOME/x17rerun/ntof/decode_spot
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/ds_${RUN}_${TAG}; mkdir -p $W; cd $W
# AFS times out under many parallel jobs (11-02, 'decode: Connection timed out'): copy the tools to scratch first
for k in 1 2 3 4 5; do cp $DEC ./decode && cp $TOOLS/compare_decoded.py $TOOLS/audit_decode.py . && break; sleep $((20*k)); done
[ -x ./decode ] || { echo "cannot fetch decoder"; exit 1; }
DEC=./decode; TOOLS=.
for F in 01 02 03 04 05 06 07 08; do
  B=${RUN}_${SUB}_${TAG}_f$F
  [ -s $OUT/$B.cmp.json ] && [ -s $OUT/$B.audit.json ] && ! grep -q '"error"' $OUT/$B.cmp.json && continue
  base=Mx17_${SUB}_datrun_${TAG}_$F
  fdf=$D/raw_daq_data/$base.fdf; old=$D/decoded_root/$base.root
  [ -s $fdf ] || { echo "{\"error\":\"no fdf\"}" > $OUT/$B.cmp.json; continue; }
  cp $fdf in.fdf || continue
  rm -f new.root
  $DEC in.fdf new.root > dec.log 2>&1 || { echo "{\"error\":\"decode failed\"}" > $OUT/$B.cmp.json; cp dec.log $OUT/$B.log; rm -f in.fdf new.root; continue; }
  env -u PYTHONPATH -u PYTHONHOME $PY $TOOLS/compare_decoded.py new.root $old c.json || echo "{\"error\":\"compare failed\"}" > c.json
  env -u PYTHONPATH -u PYTHONHOME $PY $TOOLS/audit_decode.py in.fdf new.root a.json || echo "{\"error\":\"audit failed\"}" > a.json
  cp c.json $OUT/$B.cmp.json; cp a.json $OUT/$B.audit.json; cp dec.log $OUT/$B.log
  rm -f in.fdf new.root c.json a.json a.npz
done
cd /; rm -rf $W
