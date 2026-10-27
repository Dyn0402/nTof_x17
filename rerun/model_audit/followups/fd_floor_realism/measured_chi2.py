#!/usr/bin/env python3
"""measured_chi2.py -- the data side of the realism ladder: per chamber and view, the measured angle resolution
(s68 vs M3, same convention as ws_floor/measured.py) and the production fit's chi2/dof, for the M3-matched single
tracks of the finalize tables, in bins of |theta_ref|.  Also the median fitted charge and strip count, which the
toy must match.

    measured_chi2.py   -> results/measured_chi2.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
EDGES = [0, 5, 15, 25, 35]


def s68(a):
    a = np.asarray(a)
    a = a[np.abs(a) <= 5]
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def main():
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        z = np.load(os.path.join(AUDIT, det, 'final_physmask', 'angles', 'per_event.npz'))
        ev = pd.read_parquet(os.path.join(AUDIT, det, 'final_physmask', 'events.parquet')).set_index('event_id')
        b = json.load(open(os.path.join(AUDIT, det, 'calib_bundle', 'bundle.json')))
        for v in ('x', 'y'):
            eid = z[f'{v}_eid']
            tr = z[f'{v}_tan_ref']
            tf = (z[f'{v}_w'] * 1e3 - b['w0'][v]) / (b['kw'][v] * float(b['v_drift']))
            e = ev.reindex(eid)
            c2 = (e[f'{v}_chi2'] / e[f'{v}_dof'].clip(lower=1)).to_numpy()
            q = e[f'{v}_q_sum'].to_numpy()
            ns = e[f'{v}_n_strips'].to_numpy()
            ok = np.isfinite(tr) & np.isfinite(tf) & np.isfinite(c2)
            a_ref = np.degrees(np.arctan(tr))
            d = np.degrees(np.arctan(tf)) - a_ref
            rows = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = ok & (np.abs(a_ref) >= lo) & (np.abs(a_ref) < hi)
                if m.sum() < 30:
                    continue
                rows.append(dict(lo=lo, hi=hi, mid=float(np.median(np.abs(a_ref[m]))), n=int(m.sum()),
                                 s68=s68(d[m]), chi2dof=float(np.median(c2[m])),
                                 chi2dof_p25=float(np.percentile(c2[m], 25)),
                                 chi2dof_p75=float(np.percentile(c2[m], 75)),
                                 q50=float(np.median(q[m])), nstrip=float(np.median(ns[m]))))
            out[f'{det}_{v}'] = dict(n=int(ok.sum()), all_s68=s68(d[ok]), chi2dof=float(np.median(c2[ok])),
                                     q50=float(np.median(q[ok])), angle_abs_pct=[float(x) for x in
                                     np.percentile(np.abs(a_ref[ok]), [25, 50, 75, 90])], bins=rows,
                                     dead=b['dead'][v], t0_prior_sigma=b.get('t0_prior_sigma'),
                                     sigma_p0=b['hyper'].get(f'sigma_p0_{v}'),
                                     sigma_p0_prof=b['hyper'].get(f'sigma_p0_prof_{v}'), Dp=b['hyper']['Dp'],
                                     v=b['v_drift'])
            print(det, v, f"n {ok.sum()} s68 {s68(d[ok]):.2f} chi2/dof {np.median(c2[ok]):.0f} q50 {np.median(q[ok]):.0f}",
                  ' | '.join(f"{r['mid']:.0f}: {r['s68']:.2f} / {r['chi2dof']:.0f}" for r in rows),
                  'dead', len(b['dead'][v]), 'prof', b['hyper'].get(f'sigma_p0_prof_{v}'), b['hyper'].get(f'sigma_p0_{v}'))
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'measured_chi2.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
