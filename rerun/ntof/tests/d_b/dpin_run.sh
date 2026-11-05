#!/bin/bash
# D t0 pin checks (plan §6 11-05): run_149 cosmic line/wall candidates x {d2 free, d2pin cosmic latency};
# run_145 beam x {d2 free, d2pinb beam latency, d2pin cosmic latency}.  Bundles in d_b/bundles/.
cd /home/dylan/PycharmProjects/nTof_x17
B=/media/ucla/work/ntof_pass1; O=$B/d_b/reco_pin; BU=$B/d_b/bundles; EV=$B/d_b/reco/events; mkdir -p $O
r() { .venv/bin/python rerun/ntof/ntof_reco.py "$@" --base $B/runs --jobs ${JOBS:-7} 2>&1 | grep "^\[ntof\]"; }
if [ "$1" = cosmic ]; then
for i in $(seq -w 0 11); do s=cosbounce_cos_00$i; for lab in d2 d2pin; do
  out=$O/r149_${s}_D_$lab.parquet; [ -s $out ] && continue
  if [ $i = 00 ]; then r --run 149 --subrun $s --arm D --bundle $BU/${lab}_D --out $out --events $EV/$s.txt --tag 260806_14H51_001 260806_14H51_002
  else r --run 149 --subrun $s --arm D --bundle $BU/${lab}_D --out $out --events $EV/$s.txt; fi
  echo "$(date +%H:%M) $s $lab"
done; done
else
for lab in d2pinb d2 d2pin; do out=$O/r145_s0_D_$lab.parquet; [ -s $out ] && continue
  r --run 145 --subrun stat090_0000 --arm D --bundle $BU/${lab}_D --out $out --tag 260805_14H06_000 --limit-per-tag 1500
  echo "$(date +%H:%M) beam $lab"
done
fi
echo DONE $1
