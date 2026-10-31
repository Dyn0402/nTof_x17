#!/usr/bin/env python3
"""pg_specs.py <out dir> -- the arm grid of the time-resolved bench timing stacks (rerun/paper1_gas/README.md) for
rerun/calib/s4/s4f_fit.py.

Per row of condor/stack_jobs.in it reads (on lxplus, through FUSE) <STACKS>/<row>/s4/headon.pkl (s4_stacks.py on the
pass alignment, rr1 decode; condor/pg_stack.sh) and the row's pass bundle <PASS1>/<row>/wft/calib_bundle.  The estimator is the S4 re-check
recipe (rerun/calib/S4_PHYSICS_INPUTS_2026-10-30.md, last section): pass template; X mask = the bundle's dead list
united with strip_gain_audit's flags of the chamber's golden row; M3 reference >= 30 mm inside the active area; every
unsaturated amplitude (< 3500 ADC).  Arms per row:

    <key>|in30      the recipe (the quoted number)
    <key>|all       the same without the interior cut (more events on short rows; the edge zone biases T, S4 re-check)
    <key>|in30|t<i> the recipe on the i-th third of the row by event id (event id rises with time within a row,
                    one acquisition each), rows with >= 3 x XW_MIN_EV interior head-on events

Writes <out dir>/spec_T<ns>.json (one per starting drift time), sel.json (the event-id thirds) and rows.json.
Then, per spec: XW_MIN_EV=50 s4f_fit.py --spec spec_T<ns>.json --out fit_T<ns>.json --nboot 20 (condor/pg_fit.sh).
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
CB = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')
sys.path[:0] = [CB, REPO]
from strip_gain_audit import bad_set  # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402

D = os.path.abspath(sys.argv[1])
X17 = '/eos/experiment/ntof/data/x17/cosmic_bench'
STACKS = os.environ.get('PG_STACKS', f'{X17}/x17rerun/paper1_gas/Analysis')
PASS1 = os.environ.get('PG_PASS1', f'{X17}/Analysis_pass1')
AUDIT = json.load(open(os.path.join(CB, 'results', 'strip_gain_audit.json')))
AUDIT_KEY = {'mx17_2': 'o22_long_det2', 'mx17_3': 'sat_det3', 'mx17_4': 'g_det4', 'mx17_6': 'g_det6_long',
             'mx17_7': 'g_det7_long'}
V0 = {'mx17_2': 39.3, 'mx17_3': 35.8, 'mx17_4': 33.0, 'mx17_6': 24.17, 'mx17_7': 36.8}   # starting T only
EDGE, TAN, NSPLIT, MIN_EV = 30.0, 0.03, 3, int(os.environ.get('XW_MIN_EV', 50))
V0_SCAN = {100: 2.5, 300: 12.0, 500: 18.0, 700: 25.0, 900: 33.0, 1100: 38.0}   # det3 drift scan starting v


def main():
    jobs = [l.split() for l in open(os.path.join(HERE, 'condor', 'stack_jobs.in')) if l.strip()]
    arms, sel, rows = {}, {}, {}
    for key, row, _arow, brow in jobs:
        det = row.split('/')[-1]
        cache = os.path.join(STACKS, row, 's4', 'headon.pkl')
        if not os.path.exists(cache):
            print('missing', key); continue
        b = os.path.join(PASS1, brow, 'wft', 'calib_bundle')
        dead = set(json.load(open(os.path.join(b, 'bundle.json')))['dead']['x'])
        mask = sorted(dead | bad_set(AUDIT[AUDIT_KEY[det]], 'x'))
        box = TRUE_ACTIVE_BY_DET[det]
        base = dict(cache=cache, sel=None, template=os.path.join(b, 'arrays.npz'), sat=3500.0, amp_lo=0.0, mask_x=mask)
        arms[f'{key}|in30'] = dict(base, edge=[box, EDGE, 'in'])
        arms[f'{key}|all'] = dict(base)
        ev = pickle.load(open(cache, 'rb'))
        (x0, x1), (y0, y1) = box['x'], box['y']
        inn = sorted(k for k, e in ev.items() if abs(e['tan_x']) < TAN and e['x']['W'].max() < 3500
                     and min(e['ref_mesh_x'] - x0, x1 - e['ref_mesh_x'], e['ref_mesh_y'] - y0, y1 - e['ref_mesh_y']) >= EDGE)
        rows[key] = dict(row=row, det=det, n_in30=len(inn), n_all=len(ev))
        if len(inn) >= NSPLIT * MIN_EV:
            # thirds by event id of the interior head-on events; every other event of the cache goes with its eid
            # bracket so the arm's own cuts (edge, mask, saturation) see the same population as |in30
            cuts = [inn[int(round(i * len(inn) / NSPLIT))] for i in range(1, NSPLIT)]
            edges = [-np.inf] + cuts + [np.inf]
            for i in range(NSPLIT):
                name = f'{key}|t{i}'
                sel[name] = [int(k) for k in ev if edges[i] <= k < edges[i + 1]]
                arms[f'{key}|in30|t{i}'] = dict(base, edge=[box, EDGE, 'in'], sel=[os.path.join(D, 'sel.json'), name])
            rows[key]['eid_cuts'] = [int(c) for c in cuts]
        if 'saturday_scan' in row:
            rows[key]['T0'] = 28.08 / (V0_SCAN[int(row.split('drift_')[1].split('V')[0])] * 1e-3)
    json.dump(sel, open(os.path.join(D, 'sel.json'), 'w'))
    json.dump(rows, open(os.path.join(D, 'rows.json'), 'w'), indent=1)
    # one T0 per spec: split by chamber (and the drift scan by field) so each fit starts near its own T
    groups = {}
    for name, a in arms.items():
        key = name.split('|')[0]
        T0 = rows[key].get('T0', 30.0 / (V0[rows[key]['det']] * 1e-3))
        groups.setdefault(round(T0, -1), {})[name] = a
    for T0, g in sorted(groups.items()):
        json.dump(dict(T0=T0, arms=g), open(os.path.join(D, f'spec_T{int(T0)}.json'), 'w'), indent=1)
        print(f'spec_T{int(T0)}.json: {len(g)} arms')


if __name__ == '__main__':
    main()
