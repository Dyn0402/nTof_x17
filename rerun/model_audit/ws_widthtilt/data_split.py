#!/usr/bin/env python3
"""data_split.py — decompose the bench angle response into a symmetric scale S(|t|) and an even (sign-independent)
lateral-speed term E(|t|), and split both by per-event covariates (position in the chamber, charge, column end,
chi2/dof, profile footprint is not in the table).

For a |tan_ref| bin with medians r+ (tan > 0) and r− (tan < 0) of r = (w − w0)/(v·tan_ref):
    S = (r+ + r−)/2          (1 = the angle scale of the bin agrees with the shallow one, < 1 compression)
    E = (r+ − r−)/2 · v · t̄  [µm/ns]   (an extra w that does not change sign with the track: w0 depending on angle)
Usage: data_split.py [--det det3 ...]  → ws_widthtilt/data_split.json, prints tables.
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
ABS = [(0.03, 0.08), (0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]


def load(det):
    b = json.load(open(f'{ROOT}/{det}/calib_bundle/bundle.json'))
    d = np.load(f'{ROOT}/{det}/final_physmask/angles/per_event.npz')
    ev = pd.read_parquet(f'{ROOT}/{det}/final_physmask/events.parquet').set_index('event_id')
    T = {}
    for p in 'xy':
        df = pd.DataFrame(dict(eid=d[f'{p}_eid'], t=d[f'{p}_tan_ref'], w=d[f'{p}_w'] * 1e3))
        e = ev.loc[df.eid]
        for c in ('p0', 'q_sum', 'q_uend', 'chi2', 'dof', 't0', 'n_strips', 'q_u50'):
            df[c] = e[f'{p}_{c}'].to_numpy()
        df['r'] = (df.w - b['w0'][p]) / (b['v_drift'] * df.t)
        df['chi2dof'] = df.chi2 / df.dof.clip(lower=1)
        T[p] = df
    return b, T


def decompose(df, v):
    S, E, N = [], [], []
    for lo, hi in ABS:
        a = np.abs(df.t)
        mp = (df.t >= lo) & (df.t < hi); mn = (df.t <= -lo) & (df.t > -hi)
        if mp.sum() < 8 or mn.sum() < 8:
            S.append(np.nan); E.append(np.nan); N.append((int(mp.sum()), int(mn.sum()))); continue
        rp, rn = np.median(df.r[mp]), np.median(df.r[mn])
        tm = np.median(a[mp | mn])
        S.append(float(0.5 * (rp + rn))); E.append(float(0.5 * (rp - rn) * v * tm)); N.append((int(mp.sum()), int(mn.sum())))
    return S, E, N


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--det', nargs='*', default=['det2', 'det3', 'det4', 'det6', 'det7'])
    a = ap.parse_args()
    out = {'bins': ABS}
    for det in a.det:
        b, T = load(det); v = b['v_drift']
        for p in 'xy':
            df = T[p]
            res = {}
            S, E, N = decompose(df, v); res['all'] = dict(S=S, E=E, N=N)
            # covariate splits: halves by median within the inclined population
            inc = df[np.abs(df.t) > 0.08]
            for c in ('q_sum', 'chi2dof', 'q_uend', 't0'):
                med = inc[c].median()
                for nm, sub in (('lo', df[df[c] <= med]), ('hi', df[df[c] > med])):
                    S_, E_, N_ = decompose(sub, v); res[f'{c}_{nm}'] = dict(S=S_, E=E_, N=N_, cut=float(med))
            # position in the chamber: central third vs outer, in the fit's own p0
            lo_, hi_ = np.nanpercentile(df.p0, [20, 80])
            cen = df[(df.p0 > lo_) & (df.p0 < hi_)]; out_ = df[(df.p0 <= lo_) | (df.p0 >= hi_)]
            for nm, sub in (('central', cen), ('outer', out_)):
                S_, E_, N_ = decompose(sub, v); res[f'pos_{nm}'] = dict(S=S_, E=E_, N=N_)
            out[f'{det}_{p}'] = res
            fmt = lambda xs: ' '.join(f'{x:6.3f}' for x in xs)
            print(f'{det} {p}  S {fmt(res["all"]["S"])}   E {fmt(res["all"]["E"])}')
            for k in res:
                if k == 'all':
                    continue
                print(f'   {k:12s} S {fmt(res[k]["S"])}   E {fmt(res[k]["E"])}')
    os.makedirs(f'{ROOT}/ws_widthtilt', exist_ok=True)
    json.dump(out, open(f'{ROOT}/ws_widthtilt/data_split.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
