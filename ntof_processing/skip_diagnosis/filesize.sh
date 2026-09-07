#!/bin/bash
# For every X17-campaign run whose stream1 is still staged, report:
#   run  n_files  total_GB  mean_GB_per_file  processed(yes/no)
D=/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement
for d in $D/2243*/ $D/2244*/ $D/2245*/ $D/2246*/ $D/2247*/; do
  r=$(basename $d)
  s=$d/stream1
  [ -d "$s" ] || continue
  n=$(ls $s 2>/dev/null | wc -l)
  [ "$n" -ge 20 ] || continue
  mb=$(du -sm $s 2>/dev/null | cut -f1)
  if [ -f /eos/experiment/ntof/processing/official/done/run$r.root ]; then p=done; else p=SKIPPED; fi
  echo "$r $n $mb $p"
done
