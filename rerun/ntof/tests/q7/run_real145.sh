#!/bin/bash
# Q7: real run_145 triggers, every configuration (q7_real.py), 4 acquisitions x 1000 seeded events per arm
cd /home/dylan/PycharmProjects/nTof_x17
L=/media/ucla/work/ntof_pass1/q7/logs; mkdir -p $L
T="260805_14H06_000 260805_14H06_001 260805_14H06_002 260805_14H06_003"
for a in ${ARMS:-A C}; do
  .venv/bin/python rerun/ntof/tests/q7/q7_real.py --run 145 --subrun stat090_0000 --arm $a --tags $T --limit ${LIMIT:-1000} --stem r145_$a --jobs 5 > $L/r145_$a.log 2>&1
done
