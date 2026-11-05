#!/bin/bash
# ana.sh <code_dir> <script> <out name> <script args...> -- one hits-DAQ analysis job (tiers.py / h2c_run.py / resim.py)
# from the frozen code archive with the lock venv.  The script writes $W/<out name>; it is copied to EOS
# (cosmic_bench/x17rerun/hits_daq/results/) and removed from the scratch dir.  '{W}' in the args becomes the scratch dir.
CODE=$1; SCRIPT=$2; NAME=$3; shift 3
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/hdana_$$}; mkdir -p $W
ARGS=(); for x in "$@"; do ARGS+=("${x//\{W\}/$W}"); done
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/papers/paper2/hits_daq/$SCRIPT "${ARGS[@]}" --out $W/$NAME
rc=$?
E=root://eosexperiment.cern.ch//eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/hits_daq/results
if [ $rc -eq 0 ]; then xrdcp -f $W/$NAME $E/$NAME || rc=1; fi
rm -f $W/$NAME $W/*.parquet $W/*.json $W/*.pkl
exit $rc
