#!/bin/bash
# job.sh <eos_run_dir> <subrun> <feus dot-sep> <tag>   -- decode the first acquisition of each FEU, compare with EOS
set -u
RUN=$1; SUB=$2; FEUS=$3; TAG=$4
OUT=/eos/user/d/dneff/x17rerun/spotcheck/decoded
mkdir -p $OUT
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
DEC=$HOME/x17rerun/mm_strip_reconstruction/build/decoder/decode
PY=/eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/w; mkdir -p $W; cd $W
for F in ${FEUS//./ }; do
  F2=$(printf %02d $F)
  fdf=$(ls $RUN/$SUB/raw_daq_data/*_000_${F2}.fdf 2>/dev/null | grep -v pedestal | head -1)
  if [ -z "$fdf" ]; then echo "{\"error\":\"no fdf for FEU $F2\"}" > $OUT/${TAG}_f${F2}.json; continue; fi
  base=$(basename $fdf .fdf)
  old=$RUN/$SUB/decoded_root/$base.root
  if [ ! -f "$old" ]; then echo "{\"error\":\"no EOS decoded_root $base\"}" > $OUT/${TAG}_f${F2}.json; continue; fi
  cp $fdf in.fdf
  $DEC in.fdf new.root > dec.log 2>&1 || { echo "{\"error\":\"decode failed\"}" > $OUT/${TAG}_f${F2}.json; tail -5 dec.log; rm -f in.fdf new.root; continue; }
  env -u PYTHONPATH -u PYTHONHOME $PY $HOME/x17rerun/spotcheck/compare_decoded.py new.root $old $OUT/${TAG}_f${F2}.json
  rm -f in.fdf new.root
done
rm -rf $W
