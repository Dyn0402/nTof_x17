#!/bin/bash
# collect_all.sh -- merge the condor chunks of headon_sigma.py / zs_treatment.py (after rsync from EOS ws_sps/out)
cd "$(dirname "$0")/../../.."
PY=../../../.venv/bin/python; [ -x $PY ] || PY=.venv/bin/python
for a in "raw700 0 1" "raw450 0 1" "raw275 0 1" "raw700 4 1" "raw700 5 1" "raw450 4 1" "raw275 4 1" "raw700 0 0"; do
  set -- $a
  $PY rerun/model_audit/ws_sps/headon_sigma.py collect --plateau $1 --zs $2 --censor $3 | tail -2
done
for a in "raw700 4" "raw700 5" "raw275 4"; do
  set -- $a
  $PY rerun/model_audit/ws_sps/zs_treatment.py collect --plateau $1 --zs $2
done
