#!/bin/bash
# var_job.sh <variant> <subrun> <cosbase> <pedbase> <NNN> -- one M3 tracking variant on one file (rerun M3 audit).
#   base: the patched build with no switch (must equal production); out5: M3_OUTLIER_CHI2=5; cm0/cm1: M3_CM_MASK=0/1;
#   out5cm1: both.  Output /eos/user/d/dneff/x17rerun/m3_audit/<variant>/<run>__<subrun>__<NNN>.root
V=$1; SUB=$2; COS=$3; PED=$4; NUM=$5
case $V in
  base) ;;
  out5) export M3_OUTLIER_CHI2=5 ;;
  cm0) export M3_CM_MASK=0 ;;
  cm1) export M3_CM_MASK=1 ;;
  out5cm1) export M3_OUTLIER_CHI2=5 M3_CM_MASK=1 ;;
esac
RUN=$(basename $(dirname $SUB)); S=$(basename $SUB)
O=/eos/user/d/dneff/x17rerun/m3_audit/$V; mkdir -p $O 2>/dev/null; sleep 1; mkdir -p $O
D=/afs/cern.ch/work/d/dneff/m3_var
$D/run_m3.sh $D $D/config_ref_june2026.json $SUB $(basename $COS) $(basename $PED) $NUM $O/${RUN}__${S}__${NUM}.root
[ -s $O/${RUN}__${S}__${NUM}.root ] || { echo "no output"; exit 1; }
