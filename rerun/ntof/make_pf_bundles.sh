#!/bin/bash
# Pre-freeze pass bundles (plan §6 11-05): arm x mask era (E1/E2) x t0 class (cos = pinned on the run_149 cosmic latency,
# beam = pinned on the beam latency, free = t0 free, for the latency-stability check).  D: its cosmic latency on both
# classes (11-05 D check).  B: positions only, t0 free in every class.  Values: plan §6 11-04/11-05.
cd /home/dylan/PycharmProjects/nTof_x17
W=${1:-/media/ucla/work/ntof_pass1}; L=$W/bundles; O=$W/bundles/pf; mkdir -p $O
declare -A ARGS=(
  [A]="--v 38.6 --kw 0.976,0.965 --dp 0.0245"
  [C]="--v 28.1 --kw 1.009,1.000 --dp 0.0136"
  [D]="--v 31.0 --kw 0.854,0.872 --dp 0.0157"
  [B]="--v 42.6" )
declare -A LAT=( [A_cos]=$L/latency_A_r149.json [A_beam]=$L/latency_A_r145beam.json
                 [C_cos]=$L/latency_C_r149.json [C_beam]=$L/latency_C_r145beam.json
                 [D_cos]=$L/latency_D_r149.json [D_beam]=$L/latency_D_r149.json )
for arm in A B C D; do for era in E1_pre_access E2_post_access; do for lc in cos beam free; do
  out=$O/${arm}_${era%%_*}_$lc; [ -f $out/bundle.json ] && continue
  pin=""; l=${LAT[${arm}_$lc]:-}; [ -n "$l" ] && pin="--latency $l --t0-sigma 15"
  .venv/bin/python rerun/ntof/ntof_bundle.py $arm --src $W/bench_bundles/$arm ${ARGS[$arm]} --mask-era $era --tan-max 0.6 \
     $pin --out $out > $out.log 2>&1 || echo "FAIL $out"
done; done; done
ls $O | grep -v log
