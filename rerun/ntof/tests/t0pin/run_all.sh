#!/bin/bash
# t0 pinned vs free on every staged run_149 sub-run, A-C candidate triggers only (ac_candidates.py). Plan §6 "Next".
cd /home/dylan/PycharmProjects/nTof_x17
B=/media/ucla/work/ntof_pass1; O=$B/t0pin; J=${JOBS:-6}
for i in $(seq -w 0 11); do for lab in v1k v1kpin; do for a in A C; do
  out=$O/r149s${i}_${a}_$lab.parquet; [ -s $out ] && continue
  .venv/bin/python rerun/ntof/ntof_reco.py --run 149 --subrun cosbounce_cos_00$i --arm $a --bundle $B/bundles/${lab}_$a \
    --out $out --base $B/runs --events $O/cand_00$i.txt --jobs $J 2>&1 | grep "^\[ntof\]" | sed "s/^/s$i $lab $a /"
done; done; done
echo DONE
