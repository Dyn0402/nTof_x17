#!/bin/bash
# job.sh <fdf on EOS> -- W9 re-decode of one June bench .fdf with the pinned decoder (fa80c06) + independent audit.
#   decoded -> <subrun>/decoded_root_fa80c06/<base>.root (next to the old decoded_root, which is left untouched)
#   audit   -> /eos/user/d/dneff/x17rerun/decode_audit/<run>/<subrun>/<base>.{json,npz,log}
set -u
FDF=$1
SUBDIR=$(dirname $(dirname $FDF)); SUB=$(basename $SUBDIR); RUN=$(basename $(dirname $SUBDIR))
FB=$(basename $FDF .fdf)
DOUT=$SUBDIR/decoded_root_fa80c06
AOUT=/eos/user/d/dneff/x17rerun/decode_audit/$RUN/$SUB
mkdir -p $DOUT $AOUT
if [ -s $AOUT/$FB.json ] && [ -s $DOUT/$FB.root ]; then echo "done already"; exit 0; fi
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
DEC=$HOME/x17rerun/mm_strip_reconstruction/build/decoder/decode
PY=/eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/w_$FB; mkdir -p $W; cd $W
cp $FDF in.fdf
$DEC in.fdf $FB.root > dec.log 2>&1 || { echo "decode failed"; tail -20 dec.log; cp dec.log $AOUT/$FB.log; rm -rf $W; exit 1; }
env -u PYTHONPATH -u PYTHONHOME $PY $HOME/x17rerun/decoder_audit/audit_decode.py in.fdf $FB.root a.json a.npz \
  || { echo "audit failed"; cp dec.log $AOUT/$FB.log; rm -rf $W; exit 1; }
cp $FB.root $DOUT/$FB.root.part && mv $DOUT/$FB.root.part $DOUT/$FB.root
[ $(stat -c %s $DOUT/$FB.root) -eq $(stat -c %s $FB.root) ] || { echo "copy size mismatch"; rm -rf $W; exit 1; }
cp dec.log $AOUT/$FB.log; cp a.npz $AOUT/$FB.npz; cp a.json $AOUT/$FB.json
cd /; rm -rf $W
