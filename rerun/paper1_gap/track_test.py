#!/usr/bin/env python3
"""track_test.py <work> -- does the surface model make chamber A's angles better?  Cross-validated: every map is made
on one run and applied to the other; only two constants (w0, v) are re-measured on the run itself, as production does.

Angle conversions, per view, tan = (w - w0(U, W)) / v(U, W), with (U, W) from the RECONSTRUCTED track at mid-gap:
  pass1    the pass-1 table's tan (constant mapping + FB bow block: drift tilt a(U, W) and linear scale 1 + su U + sw W)
  const    w0, v constant (no position dependence)
  cells    the other run's 4x4 cell maps of the ladder (v_rel, w0 - <w0>), bilinear
  model    the other run's joint surface fit (joint.py, all classes): v_rel = (g / <g>)^-beta, w0 = v (dh/ds + dc/ds) / 2
  model_v  model velocity map only (pass-1 tilt map kept): what the gap adds to what pass 1 already has
Metrics on the interior single tracks (M3 chi2 < 1, 4 clusters, |tan_ref| <= 0.6): s68 of theta_fit - theta_ref (deg),
the same for |tan_ref| >= 0.3, the rms over 4x4 cells of the mean angle offset and of the angle-scale error (%).
-> results/track_test.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gapmap as GM  # noqa: E402
import joint as JT  # noqa: E402


def s68(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    q = np.percentile(a, [16, 84]); return float((q[1] - q[0]) / 2)


def bilinear(cells, key, U, W):
    """cells: list of dicts with U, W, key; 4x4 -> values at (U, W), clamped at the outer cell centres."""
    us = np.unique([c['U'] for c in cells]); ws = np.unique([c['W'] for c in cells])
    Z = np.full((len(us), len(ws)), np.nan)
    for c in cells:
        Z[np.searchsorted(us, c['U']), np.searchsorted(ws, c['W'])] = c[key]
    Z = np.where(np.isfinite(Z), Z, np.nanmean(Z))
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((us, ws), Z, bounds_error=False, fill_value=None)
    return f(np.vstack([np.clip(U, us[0], us[-1]), np.clip(W, ws[0], ws[-1])]).T)


def robust_const(t_ref, w, dw0, vrel):
    """w0, v of w = w0 + dw0 + v vrel t_ref (two constants re-measured on the run itself)."""
    y = w - dw0; x = vrel * t_ref
    keep = np.ones(len(x), bool)
    for _ in range(3):
        A = np.vstack([np.ones(keep.sum()), x[keep]]).T
        c = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - c[0] - c[1] * x; rs = 1.4826 * np.median(np.abs(r[keep])); keep = np.abs(r) < 3.5 * rs
    return c


def metrics(t, tr, U, W):
    d = np.degrees(np.arctan(t) - np.arctan(tr))
    big = np.abs(tr) >= 0.3
    ex = np.linspace(-1.7, 1.7, 5); ew = np.linspace(-1.51, 1.51, 5)
    i = np.searchsorted(ex, U) - 1; j = np.searchsorted(ew, W) - 1
    off, scl = [], []
    for a in range(4):
        for b in range(4):
            m = (i == a) & (j == b)
            if m.sum() < 80:
                continue
            off.append(np.median(d[m]))
            mm = m & big
            if mm.sum() > 30:
                scl.append(np.median(t[mm] / tr[mm]) - 1)
    return dict(s68=s68(d), s68_steep=s68(d[big]), cell_offset_rms=float(np.std(off)), cell_scale_rms_pct=100 * float(np.std(scl)),
                mean_offset=float(np.median(d)), n=int(len(d)))


def model_maps(F, run_label, U, W, p):
    P = F['params']
    qh = [P['h_' + s] for s in JT.SURF]; qc = [P['c_' + s] for s in JT.SURF]
    g = P['g0'] + JT.q(qc, U, W) - JT.q(qh, U, W)
    vrel = (g / np.mean(g)) ** (-P['beta'])
    hu, hw = JT.dq(qh, U, W); cu, cw = JT.dq(qc, U, W)
    tilt = (hu + cu) / 200.0 if p == 'x' else (hw + cw) / 200.0
    dw0 = JT.V_DRIFT * tilt
    return dw0 - np.mean(dw0), vrel


def main():
    work = sys.argv[1]
    obs = json.load(open(f'{work}/cellobs.json'))
    floors = json.load(open(f'{HERE}/results/joint.json'))['floors']
    fits = {r: JT.fit({r: obs[r]}, JT.CLASSES, floors) for r in obs}
    other = {'run1': 'run2', 'run2': 'run1'}
    out = {}
    for run, R in GM.RUNS.items():
        d = GM.load_table(work, R['rel'])
        o = other[run]
        res = {}
        for p in 'xy':
            q = 'y' if p == 'x' else 'x'
            # position of the track from the reconstruction itself (not the reference)
            Uf = (d[f'x_p0'].values + 14 * d['x_tan_theta'].values - 200) / 100
            Wf = (d[f'y_p0'].values + 14 * d['y_tan_theta'].values - 200) / 100
            w = d[f'{p}_w'].values * 1e3; tr = d[f'tan_ref_{p}'].values
            sel = np.abs(tr) <= GM.TAN_LADDER
            w, tr, Uf, Wf = w[sel], tr[sel], Uf[sel], Wf[sel]
            t_pass = d[f'{p}_tan_theta'].values[sel]
            arms = {'pass1': t_pass}
            c = robust_const(tr, w, 0.0, 1.0); arms['const'] = (w - c[0]) / c[1]
            oc = obs[o]
            vk = f'v_{p}'; vmean = np.nanmean([x[vk] for x in oc])
            cells = [dict(U=x['U'], W=x['W'], vrel=x[vk] / vmean, w0=x[f'w0_{p}']) for x in oc]
            w0m = np.nanmean([x['w0'] for x in cells])
            for x in cells:
                x['dw0'] = x['w0'] - w0m
            vr = bilinear(cells, 'vrel', Uf, Wf); dw = bilinear(cells, 'dw0', Uf, Wf)
            c = robust_const(tr, w, dw, vr); arms['cells'] = (w - c[0] - dw) / (c[1] * vr)
            dw, vr = model_maps(fits[o], o, Uf, Wf, p)
            c = robust_const(tr, w, dw, vr); arms['model'] = (w - c[0] - dw) / (c[1] * vr)
            # velocity map on top of pass 1: divide out pass 1's own linear scale, apply the model's v map
            b = json.load(open(f'{work}/pass1/{R["rel"]}/calib_bundle/bundle.json'))['bow']
            S = 1 + b[f's_{p}']['su'] * Uf + b[f's_{p}']['sw'] * Wf
            _, vr = model_maps(fits[o], o, Uf, Wf, p)
            t_mv = t_pass * S / vr
            k = np.median(t_mv[np.abs(tr) > 0.2] / tr[np.abs(tr) > 0.2]); arms['model_v'] = t_mv / k
            res[p] = {a: metrics(t, tr, Uf, Wf) for a, t in arms.items()}
            print(run, p, {a: {kk: round(vv, 3) for kk, vv in m.items()} for a, m in res[p].items()}, flush=True)
        out[run] = res
    json.dump(dict(fits={r: dict(params=f['params'], errors=f['errors']) for r, f in fits.items()}, metrics=out),
              open(f'{HERE}/results/track_test.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
