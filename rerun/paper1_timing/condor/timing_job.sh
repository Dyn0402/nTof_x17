#!/bin/bash
# timing_job.sh <code dir> <first line> <count> -- run timing_row.py on lines first..first+count-1 of jobs.txt.
CODE=$1; FIRST=$2; COUNT=$3
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
OUT=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_timing/rows
mkdir -p "$OUT"
rc=0
sed -n "${FIRST},$((FIRST + COUNT - 1))p" "$CODE/rerun/paper1_timing/jobs.txt" | while read -r colsum bundle stem meta; do
  [ -s "$stem.json" ] && grep -q '"estimators"\|"skip"' "$stem.json" && { echo "done: $stem"; continue; }
  for try in 1 2 3; do
    $PY "$CODE/rerun/paper1_timing/timing_row.py" "$colsum" --bundle "$bundle" --out "$stem" --meta $meta && break
    echo "retry $try: $stem"; sleep 30
  done
done
exit $rc
