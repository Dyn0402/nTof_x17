#!/bin/bash
# run_arms.sh <stacks dir> [arm ...] -- the baseline gas fit and its systematic arms (results/gasfit[_<arm>].json).
# With arm names, only those arms (no baseline).
set -e
S=$1; shift; cd "$(dirname "$0")"; PY=${PY:-python3}
declare -A ARM=([zs05]="--zs-scale 0.5" [zs15]="--zs-scale 1.5" [zswin]="--zs-win-hi 2900" [kp]="--k 0.1" [km]="--k -0.1"
                [sat]="--sat" [tmpl]="--bench-template" [h1]="--hx 1 --hy 4" [h4]="--hx 4 --hy 12" [raweta]="--raw-eta")
if [ $# -eq 0 ]; then
    $PY gasfit.py --stacks "$S" --out results/gasfit.json > results/gasfit.log 2>&1
    set -- zs05 zs15 zswin kp km sat h1 h4 raweta tmpl
fi
for tag in "$@"; do
    $PY gasfit.py --stacks "$S" --out results/gasfit_$tag.json --gaps 26.2 27.0 27.9 28.8 ${ARM[$tag]} > results/gasfit_$tag.log 2>&1
done
