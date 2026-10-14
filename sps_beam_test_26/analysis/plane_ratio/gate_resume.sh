#!/usr/bin/env bash
# gate.sh -- full-reconstruction gate for one per-view kernel arm on one golden
# key: the R06_GATE procedure (21_r06_gate.sh) with the arm's tag, nothing
# overwritten.  Everything lands next to the production products as *_<tag>.
#
#   1. reco with the arm bundle            -> events_<tag>.parquet (matched only)
#   2. set_w0 --write, apply_w0 --write    -> w0/kw measured from THAT reco
#   3. 01 alignment, 03 angles (+ per-event dump), 02 efficiency (headline)
#
#   gate.sh <run_key> <wft dir> <tag> <bundle dir> [jobs]
set -euo pipefail
KEY=$1; W=$2; TAG=$3; B=$4; J=${5:-7}
REPO=/home/dylan/PycharmProjects/nTof_x17_paper
PY=/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python
cd "$REPO"
echo "[$(date +%T)] $KEY $TAG reco"
[ -f "$W/events_$TAG.parquet" ] && echo "reco exists, skipped" || "$PY" -m wft.cli reco "$KEY" --bundle "$B" --out "$W/events_$TAG.parquet" --matched-only --jobs "$J"
echo "[$(date +%T)] w0/kw"
"$PY" mx_june_wft/bench/set_w0.py   "$KEY" --bundle "$B" --events "events_$TAG.parquet" --write
"$PY" mx_june_wft/bench/apply_w0.py "$KEY" --bundle "$B" --events "events_$TAG.parquet" --write
echo "[$(date +%T)] alignment / angles / efficiency"
"$PY" mx_june_wft/01_alignment.py "$KEY" --table "$W/events_$TAG.parquet" --out "$W/alignment_$TAG"
"$PY" mx_june_wft/03_angles.py    "$KEY" --table "$W/events_$TAG.parquet" \
      --alignment "$W/alignment_$TAG/alignment.json" --out "$W/angles_$TAG"
"$PY" mx_june_wft/02_efficiency.py "$KEY" --table "$W/events_$TAG.parquet" \
      --alignment "$W/alignment_$TAG/alignment.json" --max-dropped -1 --out "$W/efficiency_$TAG"
echo "[$(date +%T)] $KEY $TAG done"
