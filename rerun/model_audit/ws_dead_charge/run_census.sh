#!/bin/bash
# The W3 channel census (rerun/seedfit/channel_census.py) on every other June row of each golden chamber, for the
# cross-run dead-list comparison (F12).  4 at a time.  Output: census/<det>_<key>.json + .log
cd "$(dirname "$0")"
PY=../../../../../.venv/bin/python
[ -x "$PY" ] || PY=/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python
mkdir -p census
run() {  # det key feux feuy
    "$PY" ../../seedfit/channel_census.py "$2" --feu-x "$3" --feu-y "$4" --out "census/$1_$2.json" > "census/$1_$2.log" 2>&1
    echo "done $1 $2 rc=$?"
}
run det2 ovn_det2 7 8 &
run det2 g_det2 6 8 &
run det3 g_det3 3 4 &
run det3 o23_long_det3 3 4 &
wait
run det3 g_det3_wknd 7 8 &
run det4 o23_long_det4 6 8 &
run det6 g_det6_longer 3 4 &
run det7 g_det7_longer 6 8 &
wait
