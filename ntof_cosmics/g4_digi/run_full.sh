#!/usr/bin/env bash
# the 100-file reruns of HANDOFF_TRACKING §13 (A then C, ~1 h each locally)
cd /home/dylan/PycharmProjects/nTof_x17
export PYTHONPATH=.
.venv/bin/python ntof_cosmics/g4_digi/run_digi.py g4 --arm A --bundle is2_A --steps /media/dylan/data/x17/ntof_cosmics/g4_digi/steps_nose/*.parquet --g4-arm 2 --n 100000 --jobs 15 --seed 11 --label g4_A_is2_full > /media/dylan/data/x17/ntof_cosmics/g4_digi/g4_A_is2_full.log 2>&1
.venv/bin/python ntof_cosmics/g4_digi/run_digi.py g4 --arm C --bundle is2_C --steps /media/dylan/data/x17/ntof_cosmics/g4_digi/steps_nose/*.parquet --g4-arm 3 --n 100000 --jobs 15 --seed 12 --label g4_C_is2_full > /media/dylan/data/x17/ntof_cosmics/g4_digi/g4_C_is2_full.log 2>&1
