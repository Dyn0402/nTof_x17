#!/usr/bin/env bash
# Digitise one EAR2-gun Geant4 run (capsule_pose's track library) locally, A then C.
# Needs, pulled from full_sim/angle_scale/: steps_<tag>/ -> g4_digi/steps_<tag>/ and
# neutrons_<tag>/ -> g4_angle/neutrons_<tag>/ (the wall reduction selects the events).
# Writes g4_digi/g4_{A,C}_<tag>/all.parquet, which capsule_pose.library reads (POSE2_LIB=<tag>).
#   bash ntof_cosmics/g4_digi/run_ear2.sh ear2_L19p5
set -euo pipefail
cd /home/dylan/PycharmProjects/nTof_x17
export PYTHONPATH=.
T=$1
D=/media/dylan/data/x17/ntof_cosmics/g4_digi
# the event selection reads the wall reduction of THIS run; the default is the
# July neutrons_nose, whose (file, eventID) keys match almost nothing here
export G4DIGI_REDUCED=/media/dylan/data/x17/ntof_cosmics/g4_angle/neutrons_$T
ls "$G4DIGI_REDUCED"/r*.parquet >/dev/null
for arm in A C; do
  g=$([ $arm = A ] && echo 2 || echo 3)
  s=$([ $arm = A ] && echo 11 || echo 12)
  .venv/bin/python ntof_cosmics/g4_digi/run_digi.py g4 --arm $arm --bundle is2_$arm --steps "$D"/steps_$T/*.parquet \
      --g4-arm $g --n 100000 --jobs 15 --seed $s --label g4_${arm}_$T > "$D"/g4_${arm}_$T.log 2>&1
  mkdir -p "$D"/g4_${arm}_$T
  mv "$D"/g4_${arm}_$T.parquet "$D"/g4_${arm}_$T/all.parquet
  grep -E "selection|rows" "$D"/g4_${arm}_$T.log
done
