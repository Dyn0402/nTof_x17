#!/bin/bash
# job.sh <run> <subrun> <tag> -- full n_TOF re-decode (Dylan 2026-11-02): the 8 FEU .fdf of one acquisition decoded with
# the pinned decoder (fa80c06, same build as bench/SPS pass 1) into <subrun>/decoded_root_fa80c06/, next to the old
# decoded_root (untouched).  Per file: written as .part, size-checked, renamed; a record (events, bytes, decoder log
# tail) -> $REC/<run>_<subrun>_<tag>_f<FEU>.json.  Idempotent: a file with a record and the same size is skipped.
set -u
RUN=$1; SUB=$2; TAG=$3
D=/eos/experiment/ntof/data/x17/july_beam/runs/run_$RUN/$SUB
DOUT=$D/decoded_root_fa80c06
REC=/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/checks/redecode
for k in 1 2 3; do mkdir -p $DOUT $REC 2>/dev/null; [ -d $DOUT ] && [ -d $REC ] && break; sleep 10; done
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/rd_${RUN}_${TAG}; mkdir -p $W; cd $W
for k in 1 2 3 4 5; do cp $HOME/x17rerun/mm_strip_reconstruction/build/decoder/decode ./decode && break; sleep $((20*k)); done
[ -x ./decode ] || { echo "cannot fetch decoder"; exit 1; }
fail=0
for F in 01 02 03 04 05 06 07 08; do
  base=Mx17_${SUB}_datrun_${TAG}_$F
  B=${RUN}_${SUB}_${TAG}_f$F
  if [ -s $REC/$B.json ] && [ -s $DOUT/$base.root ]; then continue; fi
  fdf=$D/raw_daq_data/$base.fdf
  [ -s $fdf ] || { echo "{\"error\":\"no fdf\"}" > $REC/$B.json; continue; }
  ok=0
  for k in 1 2 3; do cp $fdf in.fdf && ok=1 && break; sleep 15; done
  [ $ok = 1 ] || { echo "copy failed $base"; fail=1; continue; }
  if ! ./decode in.fdf out.root > dec.log 2>&1; then echo "decode failed $base"; tail -5 dec.log; fail=1; rm -f in.fdf out.root; continue; fi
  S=$(stat -c %s out.root)
  ok=0
  for k in 1 2 3; do cp out.root $DOUT/$base.root.part && [ $(stat -c %s $DOUT/$base.root.part) -eq $S ] && mv $DOUT/$base.root.part $DOUT/$base.root && ok=1 && break; sleep 15; done
  [ $ok = 1 ] || { echo "push failed $base"; fail=1; rm -f in.fdf out.root; continue; }
  EV=$(grep -m1 "Events analysed" dec.log | sed 's/.*: *//')
  printf '{"run":%s,"subrun":"%s","tag":"%s","feu":"%s","events":"%s","bytes":%s,"fdf_bytes":%s,"decoder":"fa80c06","log_tail":%s}\n' \
    $RUN $SUB $TAG $F "$EV" $S $(stat -c %s in.fdf) "$(tail -12 dec.log | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" > $REC/$B.json
  rm -f in.fdf out.root dec.log
done
cd /; rm -rf $W
exit $fail
