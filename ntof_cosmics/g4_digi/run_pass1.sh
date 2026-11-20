#!/usr/bin/env bash
# Geant4 step libraries through the n_TOF PASS-1 reconstruction (NTOF_PASS_PLAN P5 test 8), A then C.
# Frozen pass-1 E2 beam bundles, RC-kernel signal generator (digitise.plane_waveform_rc), t0 = the bundle's pin for the
# overlay trigger's clock phase + a pass-1 residual, charge scale and x/y split set to pass-1 run_145 (median x q_sum,
# median y/x).  Output: $WORK_DATA/ntof_pass1/g4_digi/g4_<arm>_<tag>_pass1.parquet
#   bash ntof_cosmics/g4_digi/run_pass1.sh [nose|ear2_L19p5|...]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
T=${1:-nose}
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
D=/media/dylan/data/x17/ntof_cosmics/g4_digi
export G4DIGI_OUT=$P/g4_digi
mkdir -p "$G4DIGI_OUT"
if [ "$T" != nose ]; then export G4DIGI_REDUCED=/media/dylan/data/x17/ntof_cosmics/g4_angle/neutrons_$T; fi
for arm in A C; do
  g=$([ $arm = A ] && echo 2 || echo 3); s=$([ $arm = A ] && echo 11 || echo 12)
  q=$([ $arm = A ] && echo 1626 || echo 1661); yx=$([ $arm = A ] && echo 1.041 || echo 1.016)
  .venv/bin/python ntof_cosmics/g4_digi/run_digi.py g4 --arm $arm --bundle $P/bundles/pf/${arm}_E2_beam \
      --steps "$D"/steps_$T/*.parquet --g4-arm $g --n 100000 --jobs 14 --seed $s --label g4_${arm}_${T}_pass1 \
      --t0-tracks $P/stage3_vt/tracks_run_145_stat090_0000.parquet --q-x-median $q --yx $yx \
      > "$G4DIGI_OUT"/g4_${arm}_${T}_pass1.log 2>&1
  grep -E "selection|events|rows" "$G4DIGI_OUT"/g4_${arm}_${T}_pass1.log
done
