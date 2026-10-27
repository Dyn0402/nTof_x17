#!/usr/bin/env python3
"""data_split.py -- where in the data does the excess over the toy live?  Measured X/Y angle resolution (s68 vs M3,
the fleet convention) split by the fitted charge, the chi2/dof, the strip count and the fit's t0 error, per chamber.
Stochastic physics improves with charge; a model error that scales with the signal does not.

    data_split.py   -> results/data_split.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')


def s68(a):
    a = np.asarray(a)
    a = a[np.abs(a) <= 5]
    return float(np.percentile(np.abs(a - np.median(a)), 68)) if len(a) > 20 else np.nan


def main():
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        z = np.load(os.path.join(AUDIT, det, 'final_physmask', 'angles', 'per_event.npz'))
        ev = pd.read_parquet(os.path.join(AUDIT, det, 'final_physmask', 'events.parquet')).set_index('event_id')
        b = json.load(open(os.path.join(AUDIT, det, 'calib_bundle', 'bundle.json')))
        for v in ('x', 'y'):
            tr = z[f'{v}_tan_ref']
            tf = (z[f'{v}_w'] * 1e3 - b['w0'][v]) / (b['kw'][v] * float(b['v_drift']))
            e = ev.reindex(z[f'{v}_eid'])
            d = np.degrees(np.arctan(tf)) - np.degrees(np.arctan(tr))
            ok = np.isfinite(d) & (np.abs(np.degrees(np.arctan(tr))) < 12)
            res = {}
            for key, col in (('q', e[f'{v}_q_sum'].to_numpy()),
                             ('chi2dof', (e[f'{v}_chi2'] / e[f'{v}_dof'].clip(lower=1)).to_numpy()),
                             ('nstrip', e[f'{v}_n_strips'].to_numpy().astype(float)),
                             ('pos', e[f'{v}_p0'].to_numpy())):
                m = ok & np.isfinite(col)
                qs = np.nanpercentile(col[m], [0, 20, 40, 60, 80, 100])
                rows = []
                for lo, hi in zip(qs[:-1], qs[1:]):
                    s = m & (col >= lo) & (col <= hi)
                    rows.append(dict(lo=float(lo), hi=float(hi), med=float(np.median(col[s])), n=int(s.sum()),
                                     s68=s68(d[s])))
                res[key] = rows
            out[f'{det}_{v}'] = res
            print(det, v, ' | '.join(f"{k}: " + ' '.join(f"{r['s68']:.2f}" for r in rows) for k, rows in res.items()))
    json.dump(out, open(os.path.join(HERE, 'results', 'data_split.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
