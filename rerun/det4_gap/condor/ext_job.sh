#!/bin/bash
# ext_job.sh <key> -- rerun/det4_gap/sps_extent.py on the final-rule SPS builder (code_48244778), output on n_TOF EOS
KEY=$1
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$HOME/x17rerun/code_48244778 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $HOME/x17rerun/det4_gap/sps_extent.py $KEY \
    --registry $HOME/x17rerun/det4_gap/sps_registry.csv --out /eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/det4_gap/sps
