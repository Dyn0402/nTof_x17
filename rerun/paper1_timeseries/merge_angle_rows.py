#!/usr/bin/env python3
"""merge_angle_rows.py NEW_DIR -- put the angle rows of a new timeseries.py run into the committed results (step 5, 2026-11-24).

timeseries.py run on the local copy of its inputs (claude-jobs/1ccd2342) does not reproduce every committed number
(interior efficiency 95.70 -> 95.72 %, n 25671 -> 25657, the velocity columns): the inputs of the committed run
(claude-jobs/twotrack/timeseries_tt) are gone.  The angle rows (q == 'ang': s68, its error, n, median per 30-minute bin) and
the angle entries of 'integrated' and 'trends' reproduce exactly on the pass-1 angle (0.9870 / 0.8986), so only those are taken
from the new run (ANGRES_ANGLES=map1: the five-time-block cross-validated map angle, same events, joined by event id) and
written into results/bins.csv and results/timeseries.json.  Everything else stays as committed.

    ANGRES_ANGLES=map1 python timeseries.py --eff-events <rerun/effbudget/single/results of the main checkout>
    cp results/{bins.csv,timeseries.json} NEW_DIR; git checkout results/; python merge_angle_rows.py NEW_DIR
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
new = sys.argv[1]
RES = os.path.join(HERE, 'results')
b0 = pd.read_csv(os.path.join(RES, 'bins.csv'))
b1 = pd.read_csv(os.path.join(new, 'bins.csv'))
k = b0.q == 'ang'
assert (b0[k][['run', 't']].values == b1[b1.q == 'ang'][['run', 't']].values).all(), 'angle bins differ'
cols = ['s68_x', 's68_x_err', 'n_x', 'med_x', 's68_y', 's68_y_err', 'n_y', 'med_y']
b0.loc[k, cols] = b1.loc[b1.q == 'ang', cols].values
b0.to_csv(os.path.join(RES, 'bins.csv'), index=False)
j0 = json.load(open(os.path.join(RES, 'timeseries.json')))
j1 = json.load(open(os.path.join(new, 'timeseries.json')))
for c in ('s68_x', 's68_y'):
    j0['integrated'][c] = j1['integrated'][c]
    j0['trends'][c] = j1['trends'][c]
json.dump(j0, open(os.path.join(RES, 'timeseries.json'), 'w'), indent=1)
print('merged', int(k.sum()), 'angle bins;', {c: j0['integrated'][c] for c in ('s68_x', 's68_y')})
