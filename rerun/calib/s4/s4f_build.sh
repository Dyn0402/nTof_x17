#!/bin/bash
# s4f_build.sh <code_dir> <key> <row> <gen> -- S4 follow-up: s4f_build.py (the ref-only head-on cache through the pass
# alignment, on the <gen> decode, plus the hits-chain selections of the prod and rr1 chains) for one golden row.
# Reads Analysis_pass1 (alignment, rr1 hits cache) and the uploaded prod hits cache; writes only under
# /eos/user/d/dneff/x17rerun/s4_followup/Analysis_<gen>/<row>/s4f/
CODE=$1; KEY=$2; ROW=$3; GEN=$4
F=/eos/user/d/dneff/x17rerun/s4_followup
EOSA=$F/Analysis_$GEN
P1=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/$ROW
PROD=$F/prod_inputs/$ROW
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export RERUN_CODE_COMMIT=$(basename $CODE | sed 's/^code_//') MPLBACKEND=Agg
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/s4f_$$}; mkdir -p $W
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/stage_job.py --work $W --eos-analysis $EOSA --gen $GEN"
$PY $KEY --parts run_config,m3,decoded --script rerun/calib/s4/s4f_build.py -- \
    --alignment $P1/wft/alignment/alignment.json --out {out}/s4f \
    --prod-results $PROD/event_results_veto50.pkl --prod-alignment $PROD/alignment.json \
    --rr1-results $P1/cache/event_results_veto50.pkl --rr1-alignment $P1/alignment_tpc_veto50/alignment.json
rc=$?
rm -rf $W/bench $W/Analysis
exit $rc
