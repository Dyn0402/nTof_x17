#!/bin/bash
# fetch.sh [dest] -- pull the v(t) results (small JSONs, < 10 MB) of the vt_all DAG and the n_TOF test from EOS
# (/eos/user/d/dneff/x17rerun/vt_all) through `ssh lxplus`.
set -euo pipefail
D=${1:-${VT_ALL_DATA:-${WORK_DATA:-$HOME/x17}/vt_all}}
mkdir -p "$D"; cd "$D"
ssh lxplus 'cd /eos/user/d/dneff/x17rerun/vt_all && tar cf - bench/*.json sps/*/vt_sps.json ntof/vt_ntof_campaign.json 2>/dev/null' | tar xf -
echo "pulled into $D"
