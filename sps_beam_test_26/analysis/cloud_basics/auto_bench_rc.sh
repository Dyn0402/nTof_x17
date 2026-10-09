#!/bin/bash
# auto_bench_rc.sh -- queue the RC bench of every chamber whose three refits have landed
cd ~/cloud_basics_condor/rc; mkdir -p bench log; touch bench_rc_submitted.txt; : > jobs_bench_rc.txt
for det in det2 det3 det4 det6 det7; do
  grep -qx "$det" bench_rc_submitted.txt && continue
  [ -f out/arm_${det}_rcf3.json ] && [ -f out/arm_${det}_rcf4x.json ] && [ -f out/arm_${det}_rcfD.json ] || continue
  case $det in det3|det4) tr=calib_cache;; *) tr=big_cache;; esac
  echo "$det $tr" >> jobs_bench_rc.txt; echo "$det" >> bench_rc_submitted.txt
done
n=$(wc -l < jobs_bench_rc.txt); [ "$n" -gt 0 ] && condor_submit condor_bench_rc.sub | tail -1; echo "queued $n"
