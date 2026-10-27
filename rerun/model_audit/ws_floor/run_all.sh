#!/bin/bash
# floor_toy production: det3 breakdown (production estimator), the likelihood-weighted estimator, the other chambers.
set -e
cd "$(dirname "$0")"
PY=/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python
$PY floor_toy.py run --det det3 --config full --n 200 --est poisson
for c in no_noise no_gainfl no_delta no_landau no_diff only_diff only_ion; do
  $PY floor_toy.py run --det det3 --config $c --n 150
done
for d in det2 det4 det6 det7; do
  $PY floor_toy.py run --det $d --config full --n 150
  $PY floor_toy.py run --det $d --config full --n 150 --est poisson
done
for d in det2 det3 det4 det6 det7; do $PY floor_toy.py crb --det $d; done
