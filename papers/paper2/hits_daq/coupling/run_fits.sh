#!/bin/bash
# the joint kernel fits on det3 Run 1 (see fit_kernel.py); $1 = output dir, $2 = bundle
PY=../../../../.venv/bin/python; F=$1; B=$2; W=$(ls $WORK_DATA/paper2_hits/windows/sat_det3_p*.pkl)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fit() { n=$1; v=$2; free=$3; set=$4; par=${5:--1}; $PY fit_kernel.py $W --view $v --free $free --set "$set" --parity $par --bundle $B --n 2000 --maxfev 320 --jobs 12 --out $F/${n}_$v.json 2>&1 | tail -1; }
for v in x y; do
  fit base $v sigma,Dp,rc ""            # no coupling: the wrapper
  fit free $v sigma,Dp,rc,k1,k2 ""      # everything free
  fit s033 $v Dp,rc,k1,k2 "sigma=0.033" # Garfield-scale avalanche footprint
  fit s070 $v Dp,rc,k1,k2 "sigma=0.07"  # induction floor
  fit s033_nok2 $v Dp,rc,k1 "sigma=0.033"
  fit s033_p0 $v Dp,rc,k1,k2 "sigma=0.033" 0
  fit s033_p1 $v Dp,rc,k1,k2 "sigma=0.033" 1
done
