#!/bin/bash
# vtc_row.sh <key> <row> -- v(t) cross-check (CROSSCHECK_2026-10-31): the two M3-scale-free timing estimators of one
# pass-1 row, with the deployed tools that made them on the golden rows (S4, 10-30):
#   colsum   FA's model-free column duration, every trigger (code_b5a29af, the row's own pass bundle)
#   s4       the head-on cache + S4 stack (s4_stacks.py, code_75287c0, the row's own pass alignment and bundle)
# <row> = <run>/<subrun>/<det> under Analysis_pass1, read only.
# -> /eos/user/d/dneff/x17rerun/vt_check/Analysis/<row>/{fa/colsum.npz, s4/{headon,incl}.pkl, s4/s4_stacks.json}
KEY=$1; ROW=$2
EOSA=/eos/user/d/dneff/x17rerun/vt_check/Analysis
P1=/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/$ROW/wft
C1=$HOME/x17rerun/code_b5a29af
C2=/eos/user/d/dneff/x17rerun/s4_inputs/code_75287c0
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export MPLBACKEND=Agg
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/vtc_$$}; mkdir -p $W
PYX="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
RERUN_CODE_COMMIT=b5a29af $PYX $C1/rerun/stage_job.py --work $W --eos-analysis $EOSA $KEY --parts run_config,m3,decoded \
    --script rerun/model_audit/followups/fa_gas_v_tilt/colsum.py -- --bundle $P1/calib_bundle --out {out}/fa/colsum.npz --jobs 8
rc1=$?
rm -rf $W/bench $W/Analysis
RERUN_CODE_COMMIT=75287c0 $PYX $C2/rerun/stage_job.py --work $W --eos-analysis $EOSA $KEY --parts run_config,m3,decoded \
    --script rerun/calib/s4/s4_stacks.py -- --alignment $P1/alignment/alignment.json --bundle $P1/calib_bundle \
    --out {out}/s4 --workers 8
rc2=$?
rm -rf $W/bench $W/Analysis
echo "colsum rc $rc1, s4 rc $rc2"
exit $(( rc1 || rc2 ))
