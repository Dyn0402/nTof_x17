#!/usr/bin/env python3
"""score.py -- t0 pinned vs free on the run_149 A-C cosmic line, all staged sub-runs (plan §6, Q2).

    score.py [--dir /media/ucla/work/ntof_pass1/t0pin] [--out results.json]

For each config (v1k = free t0, v1kpin = pinned, sigma 15 ns) and sub-run, rerun/ntof/ac_line.py builds the line
events.  The angle scale is re-measured per config on the even sub-runs (fitted/true tan ratio, kwmed window) and
applied to the odd ones (tan / ratio), so each config is scored at its own calibrated scale on held-out events.
Pinned vs free is compared on the held-out events common to both, with a paired bootstrap over events.
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import ac_line

RC = '/media/ucla/work/ntof_pass1/runs/run_149/run_config.json'
CFG = ('v1k', 'v1kpin')


def lines(d, lab):
    out = []
    for fa in sorted(glob.glob(f'{d}/r149s*_A_{lab}.parquet')):
        s = int(re.search(r'r149s(\d+)_A', fa).group(1))
        fc = fa.replace('_A_', '_C_')
        if not os.path.exists(fc):
            continue
        _, M = ac_line.score(fa, fc, RC)
        M = M.copy(); M['sub'] = s
        out.append(M)
    return pd.concat(out, ignore_index=True) if out else None


def ratio(M, col, tt):
    w = (np.abs(M[tt]) > 0.15) & (np.abs(M[tt]) < 0.45) & np.isfinite(M[col])
    r = M.loc[w, col] / M.loc[w, tt]
    return float(np.median(r)), float(1.2533 * 1.4826 * np.median(np.abs(r - np.median(r))) / np.sqrt(w.sum())), int(w.sum())


def resid_angle(M, s, scale):
    """3-D residual angle (deg) between the chamber's fitted local direction (tan/scale) and the true line."""
    tx = M[f'tanx_{s}'] / scale['x']; ty = M[f'tany_{s}'] / scale['y']
    ux, uy = M[f'true_tanx_{s}'], M[f'true_tany_{s}']
    a = np.stack([tx, ty, np.ones_like(tx)], 1); b = np.stack([ux, uy, np.ones_like(ux)], 1)
    a /= np.linalg.norm(a, axis=1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.abs((a * b).sum(1)).clip(0, 1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='/media/ucla/work/ntof_pass1/t0pin')
    ap.add_argument('--out', default=None)
    ap.add_argument('--cfgs', default=','.join(CFG), help='configs; each compared with the first on common events')
    a = ap.parse_args()
    cfgs = a.cfgs.split(',')
    L = {lab: lines(a.dir, lab) for lab in cfgs}
    res = dict(subruns=sorted(L[cfgs[0]]['sub'].unique().tolist()))
    sc = {}
    for lab in cfgs:
        M = L[lab]; tr = M[M['sub'] % 2 == 0]; te = M[M['sub'] % 2 == 1]
        res[lab] = dict(n_line=int(len(M)), n_train=int(len(tr)), n_test=int(len(te)))
        sc[lab] = {}
        for s, arm in (('a', 'A'), ('c', 'C')):
            sc[lab][arm] = {}
            for ax in 'xy':
                r_tr = ratio(tr, f'tan{ax}_{s}', f'true_tan{ax}_{s}')
                r_te = ratio(te, f'tan{ax}_{s}', f'true_tan{ax}_{s}')
                sc[lab][arm][ax] = r_tr[0]
                res[lab][f'{arm}{ax}_ratio_train'] = r_tr
                res[lab][f'{arm}{ax}_ratio_test'] = r_te
                res[lab][f'{arm}{ax}_ratio_test_after'] = r_te[0] / r_tr[0]
    # held-out comparison on common events, each config against the first
    T = {lab: L[lab][L[lab]['sub'] % 2 == 1].set_index(['sub', 'event_id']) for lab in cfgs}
    rng = np.random.default_rng(1)
    base = cfgs[0]
    for lab in cfgs[1:]:
        common = T[base].index.intersection(T[lab].index)
        res[f'n_common_test_{lab}'] = int(len(common))
        for s, arm in (('a', 'A'), ('c', 'C')):
            R = {k: resid_angle(T[k].loc[common], s, sc[k][arm]) for k in (base, lab)}
            D = {k: {ax: np.abs(T[k].loc[common, f'tan{ax}_{s}'] / sc[k][arm][ax] -
                                T[k].loc[common, f'true_tan{ax}_{s}']).to_numpy() for ax in 'xy'} for k in (base, lab)}
            n = len(common)
            d68 = []
            for _ in range(2000):
                i = rng.integers(0, n, n)
                d68.append(np.percentile(R[lab][i], 68) - np.percentile(R[base][i], 68))
            out = dict(res68_base=float(np.percentile(R[base], 68)), res68=float(np.percentile(R[lab], 68)),
                       d68=float(np.percentile(R[lab], 68) - np.percentile(R[base], 68)), d68_err=float(np.std(d68)),
                       frac_better=float((R[lab] < R[base]).mean()))
            for ax in 'xy':
                out[f'tan{ax}_68_base'] = float(np.percentile(D[base][ax], 68))
                out[f'tan{ax}_68'] = float(np.percentile(D[lab][ax], 68))
            res[f'compare_{lab}_vs_{base}_{arm}'] = out
    res['line_sep_med'] = {lab: float(L[lab]['sep_mm'].median()) for lab in cfgs}
    print(json.dumps(res, indent=1, default=float))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
