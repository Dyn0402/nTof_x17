#!/bin/bash
# job.sh <subrun dir> <fnum> <feu=decoded=pedfdf=flags;...> -- analyze + combine one acquisition, both CM generations.
set -u
SD=$1; FNUM=$2; SPEC=$3
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1; set -u
SOFT=$HOME/x17rerun/mmsr_36eb894/build
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/rh_$(basename $SD)_$FNUM; mkdir -p $W; cd $W
for M in 0 5; do
  G=rr1_cm$M; HO=$SD/hits_root_$G; CO=$SD/combined_hits_root_$G; mkdir -p $HO $CO
  : > list_$M.txt
  for item in ${SPEC//;/ }; do
    IFS='=' read -r FEU DEC PED FL <<< "$item"
    FL=${FL//_/ }
    [ -s ped_$FEU.root ] || { cp $PED ped_$FEU.fdf && $SOFT/decoder/decode ped_$FEU.fdf ped_$FEU.root > ped_$FEU.log 2>&1 && rm -f ped_$FEU.fdf; } \
      || { echo "pedestal decode failed $PED"; exit 1; }
    out=$(basename $DEC .root)_hits.root
    $SOFT/waveform_analysis/analyze_waveforms $DEC $out ped_$FEU.root $FL --cns-mask $M > an_${M}_$FEU.log 2>&1 \
      || { echo "analyze failed $DEC m$M"; tail -5 an_${M}_$FEU.log; exit 1; }
    echo "$W/$out $((10#$FEU))" >> list_$M.txt
  done
  first=$(head -1 list_$M.txt | cut -d' ' -f1); comb=$(basename $first | sed -E 's/(_[0-9]{3}_)[0-9]{2}_hits/\1feu-combined_hits/')
  $SOFT/feu_hit_combiner/combine_feus_hits list_$M.txt $comb > comb_$M.log 2>&1 || { echo "combine failed m$M"; tail -5 comb_$M.log; exit 1; }
  for f in $(cut -d' ' -f1 list_$M.txt); do cp $f $HO/ && rm -f $f; done
  cp $comb $CO/$comb.part && mv $CO/$comb.part $CO/$comb && rm -f $comb
done
cd /; rm -rf $W
