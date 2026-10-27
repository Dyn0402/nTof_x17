#!/bin/bash
# fa_row.sh <code_dir> <mode> <key> <bundle> [limit] -- FA condor job for one bench row (stages via rerun/stage_job.py).
#   mode colsum : model-free column duration of every trigger            -> <EOSA>/<row>/fa/colsum.npz
#   mode reco   : wft reco (matched, first <limit> events) + waveform alignment + per-event table
#                                                                         -> <EOSA>/<row>/wft/{events.parquet,alignment/}, fa/table.parquet
#   mode full   : reco then colsum
#   mode arm:<name> : wft reco of every matched event with <bundle> into wft/arm_<name>/events.parquet (study arm)
# Outputs go to the FA Analysis tree on EOS (never Analysis_rr1).  Leaves nothing large in the scratch dir.
CODE=$1; MODE=$2; KEY=$3; BUNDLE=$4; LIMIT=${5:-0}
EOSA=/eos/user/d/dneff/x17rerun/followups/fa/Analysis
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') MPLBACKEND=Agg
export WFT_RND_DIR=/eos/user/d/dneff/x17rerun/regression/wft_model_sat_det3
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/fa_$$}; mkdir -p $W
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/stage_job.py --work $W --eos-analysis $EOSA"
FA=rerun/model_audit/followups/fa_gas_v_tilt
LIM=""; [ "$LIMIT" != "0" ] && LIM="--limit $LIMIT"
rc=0
run_reco() {
  $PY $KEY --parts run_config,m3,decoded,hits --module wft.cli -- reco {key} --bundle $BUNDLE --matched-only $LIM --jobs 8 || return 1
  rm -rf $W/bench
  $PY $KEY --parts run_config,m3 --script mx_june_wft/01_alignment.py || return 1
  $PY $KEY --parts run_config,m3 --script $FA/rowtable.py -- --table {out}/wft/events.parquet \
      --alignment {out}/wft/alignment/alignment.json --out {out}/fa/table.parquet || return 1
}
run_colsum() {
  $PY $KEY --parts run_config,m3,decoded --script $FA/colsum.py -- --bundle $BUNDLE --out {out}/fa/colsum.npz --jobs 8 || return 1
}
case $MODE in
  colsum) run_colsum || rc=1 ;;
  reco)   run_reco || rc=1 ;;
  full)   run_reco || rc=1; rm -rf $W/bench $W/Analysis; run_colsum || rc=1 ;;
  arm:*)  ARM=${MODE#arm:}
          $PY $KEY --parts run_config,m3,decoded,hits --module wft.cli -- reco {key} --bundle $BUNDLE --matched-only $LIM \
              --jobs 8 --out {out}/wft/arm_$ARM/events.parquet || rc=1 ;;
  *) echo "unknown mode $MODE"; rc=2 ;;
esac
rm -rf $W/bench $W/Analysis
exit $rc
