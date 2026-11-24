#!/bin/bash
# nose then ear2, three workers
cd /media/ucla/work/claude-jobs/eb8200b3/W3_g4_scale
for r in "0 34" "34 67" "67 100"; do set -- $r
  ( nice -n 10 /tmp/w3v/bin/python extract_origin.py neutrons_thermal_trig_2cm_nose nose $1 $2
    nice -n 10 /tmp/w3v/bin/python extract_origin.py neutrons_thermal_ear2_L19p5 ear2_L19p5 $1 $2 ) > ext_$1.log 2>&1 &
done
wait
