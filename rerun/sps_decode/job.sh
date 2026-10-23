#!/bin/bash
# job.sh <fdf on EOS> -- W9: decode one SPS det4 (FEU 03) .fdf with the pinned decoder (fa80c06) into
# <subrun>/decoded_root_fa80c06/; RAW files are also audited against the independent decoder
# (rerun/decoder_audit/audit_decode.py); zero-suppressed files are decoded only (the ZS path of the decoder is
# unchanged since the 8-03 byte-identical regression).  Audit -> /eos/user/d/dneff/x17rerun/decode_audit_sps/...
FDF=$1
SUBDIR=$(dirname $(dirname $FDF)); SUB=$(basename $SUBDIR); RUN=$(basename $(dirname $SUBDIR))
FB=$(basename $FDF .fdf)
DOUT=$SUBDIR/decoded_root_fa80c06
AOUT=/eos/user/d/dneff/x17rerun/decode_audit_sps/$RUN/$SUB
for k in 1 2 3; do mkdir -p $DOUT $AOUT 2>/dev/null; [ -d $DOUT ] && [ -d $AOUT ] && break; sleep 5; done
if [ -s $AOUT/$FB.done ] && [ -s $DOUT/$FB.root ]; then echo "done already"; exit 0; fi
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
DEC=$HOME/x17rerun/mm_strip_reconstruction/build/decoder/decode
PY=/eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/w_$FB; mkdir -p $W; cd $W
cp $FDF in.fdf || exit 1
$DEC in.fdf $FB.root > dec.log 2>&1 || { echo "decode failed"; tail -20 dec.log; cp dec.log $AOUT/$FB.log; rm -rf $W; exit 1; }
MODE=$(grep -m1 "mode  " dec.log | sed 's/.*: //')
if echo "$MODE" | grep -q "^RAW"; then
  env -u PYTHONPATH -u PYTHONHOME $PY $HOME/x17rerun/decoder_audit/audit_decode.py in.fdf $FB.root a.json a.npz \
    || { echo "audit failed"; cp dec.log $AOUT/$FB.log; rm -rf $W; exit 1; }
  cp a.json $AOUT/$FB.json; cp a.npz $AOUT/$FB.npz
fi
cp $FB.root $DOUT/$FB.root.part && mv $DOUT/$FB.root.part $DOUT/$FB.root
[ $(stat -c %s $DOUT/$FB.root) -eq $(stat -c %s $FB.root) ] || { echo "copy size mismatch"; rm -rf $W; exit 1; }
cp dec.log $AOUT/$FB.log; echo "$MODE" > $AOUT/$FB.done
[ -s $AOUT/$FB.done ] || { echo "output copy failed"; rm -rf $W; exit 1; }
cd /; rm -rf $W
