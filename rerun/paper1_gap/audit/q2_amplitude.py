"""q2_amplitude.py -- outstanding question 2: is the low gap-term amplitude b on A 6-25 (0.22) and C (0.18) caused by
saturation or by censored drift times?  b as r3_lib.fit_b (quoted at beta 0.68), per mounting, on subsets:
  all            -- as R3
  unsat          -- events with no saturated strip in either view (field from all events, b from the subset)
  unsat_field    -- field AND b from unsaturated events
  uncens         -- events whose charge profile does not reach the last basis bin in either view
  per row        -- for multi-row mountings
Errors: bootstrap over events (b only, field fixed), 100 replicates.  -> $WORK/audit/q16/q2_amplitude.json
"""
import json
import numpy as np
import pandas as pd
from r3_lib import (DL, MOUNT, load_mounting, gap_field, attach_field, fit_b)

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
rng = np.random.default_rng(2)
NB = 100


def rails(name):
    """per-event flag (in load_mounting order): profile reaches the basis end in x or y."""
    out = []
    for key in MOUNT[name]['rows']:
        d = DL.load(key, xhalf=MOUNT[name].get('xhalf', False))
        full = pd.read_parquet(DL.path(key, 'bow_gate_table.parquet'))
        mx = np.nanmax(np.r_[full.x_q_uend, full.y_q_uend])
        out.append(((d.x_q_uend >= mx - 1) | (d.y_q_uend >= mx - 1)).values)
    return np.concatenate(out)


def b_on(D, m):
    S = D.sub(m)
    b = fit_b(S)
    bs = []
    for _ in range(NB):
        i = rng.integers(0, S.n, S.n)
        bs.append(fit_b(S.sub(i))['xy'])
    return dict(n=int(S.n), b=b['xy'], err=float(np.std(bs)), bx=b['x'], by=b['y'])


res = {}
for name in ('A625', 'A628', 'B', 'C', 'D', 'E623', 'E624'):
    D, info, bundle = load_mounting(name)
    spl, cf, ok = gap_field(D)
    attach_field(D, bundle, spl, cf)
    sat = (D.satx > 0) | (D.saty > 0)
    rl = rails(name)
    assert len(rl) == D.n
    R = dict(sat_frac=float(sat.mean()), rail_frac=float(rl.mean()))
    R['all'] = b_on(D, np.ones(D.n, bool))
    if (~sat).sum() >= 300:
        R['unsat'] = b_on(D, ~sat)
        D2, _, _ = load_mounting(name)
        s2, c2, _ = gap_field(D2, only=~sat)
        attach_field(D2, bundle, s2, c2)
        R['unsat_field'] = b_on(D2, ~sat)
    if sat.sum() >= 300:
        R['sat_only'] = b_on(D, sat)
    if rl.any() and (~rl).sum() >= 300:
        R['uncens'] = b_on(D, ~rl)
    if len(info) > 1:
        R['rows'] = {}
        for i, inf in enumerate(info):
            m = D.row == i
            if m.sum() >= 300:
                R['rows'][inf['key']] = dict(**b_on(D, m), sat=float(sat[m].mean()), rail=float(rl[m].mean()))
    res[name] = R
    print(name, json.dumps(R, default=float), flush=True)
json.dump(res, open(f'{OUT}/q2_amplitude.json', 'w'), indent=1, default=float)
print('done')
