#!/usr/bin/env python3
"""curvature_fit.py — how much genuine angle nonlinearity is left once the angle scale and offset are allowed to vary
with WHERE the track crosses the chamber?

Robust (soft-L1) joint fit per chamber/view, all events with |tan_ref| < 0.45:
  global : w = a + b·t + c·t|t| + e·t²                              (no position dependence)
  posbin : w = a_j + b_j·t + c·t|t| + e·t²   for j = position bin of the track at mid-gap along the SAME axis
  pos2d  : as posbin with j = (same-axis bin) x (other-axis bin), 5 x 3
c is the symmetric curvature (c < 0: steep tracks read shallow), e the sign-independent term.  Quoted as the
fractional scale change from |t| = 0.15 to |t| = 0.36:  dS = c·(0.36 − 0.15)/b, and E(0.36) − E(0.15) in µm/ns.
Errors: bootstrap (30).  Writes ws_widthtilt/curvature_fit.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_split import load, ROOT

RNG = np.random.default_rng(11)


def design(t, j, nb):
    X = np.zeros((len(t), 2 * nb + 2))
    X[np.arange(len(t)), j] = 1.0
    X[np.arange(len(t)), nb + j] = t
    X[:, 2 * nb] = t * np.abs(t); X[:, 2 * nb + 1] = t * t
    return X


def robust(X, y, scale):
    """Huber IRLS (k = scale): ~12 weighted least-squares solves."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(12):
        r = np.abs(X @ beta - y)
        sw = np.sqrt(np.where(r <= scale, 1.0, scale / np.maximum(r, 1e-12)))
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return beta


def fit(t, w, j, nb, scale, nboot=30):
    X = design(t, j, nb)
    keep = X[:, :nb].sum(0) > 0
    beta = robust(X, w, scale)
    b_mean = np.average(beta[nb:2 * nb], weights=np.bincount(j, minlength=nb) + 1e-9)
    c, e = beta[2 * nb], beta[2 * nb + 1]
    boots = []
    for _ in range(nboot):
        i = RNG.integers(0, len(t), len(t))
        bb = robust(design(t[i], j[i], nb), w[i], scale)
        boots.append((bb[2 * nb], bb[2 * nb + 1]))
    boots = np.array(boots)
    dS = c * (0.36 - 0.15) / b_mean; dE = e * (0.36 ** 2 - 0.15 ** 2)
    dS_e = np.std(boots[:, 0]) * (0.36 - 0.15) / b_mean; dE_e = np.std(boots[:, 1]) * (0.36 ** 2 - 0.15 ** 2)
    return dict(b=float(b_mean), c=float(c), e=float(e), dS=float(dS), dS_err=float(dS_e), dE=float(dE), dE_err=float(dE_e))


def qbins(x, n):
    e = np.nanpercentile(x, np.linspace(0, 100, n + 1)); e[-1] += 1e-6
    return np.clip(np.searchsorted(e, x, side='right') - 1, 0, n - 1)


def main():
    out = {}
    for det in ['det2', 'det3', 'det4', 'det6', 'det7']:
        b, T = load(det); v = b['v_drift']
        px = T['x'].set_index('eid'); py = T['y'].set_index('eid')
        for p in 'xy':
            df = T[p].copy()
            oth = (py if p == 'x' else px)
            df['p_mid'] = df.p0 + df.t * 15.0                     # position at mid-gap (15 mm) from the reference angle
            df['p_oth'] = (oth.p0 + oth.t * 15.0).reindex(df.eid).to_numpy()
            df = df[(np.abs(df.t) < 0.45) & np.isfinite(df.w) & np.isfinite(df.p_mid)]
            df = df[np.abs(df.w - v * df.t * b['kw'][p]) < 0.15 * v]   # drop far outliers (> ~8.5 deg)
            t, w = df.t.to_numpy(), df.w.to_numpy(); scale = 0.03 * v
            res = {'n': int(len(df))}
            res['global'] = fit(t, w, np.zeros(len(t), int), 1, scale)
            j = qbins(df.p_mid.to_numpy(), 8); res['posbin'] = fit(t, w, j, 8, scale)
            ok = np.isfinite(df.p_oth.to_numpy())
            j2 = qbins(df.p_mid.to_numpy(), 5) * 3 + qbins(np.where(ok, df.p_oth, np.nanmedian(df.p_oth)), 3)
            res['pos2d'] = fit(t, w, j2, 15, scale)
            out[f'{det}_{p}'] = res
            f = lambda r: f'dS {100 * r["dS"]:+5.1f}±{100 * r["dS_err"]:.1f} %  dE {r["dE"]:+.3f}±{r["dE_err"]:.3f}'
            print(f'{det} {p}  global {f(res["global"])} | posbin {f(res["posbin"])} | pos2d {f(res["pos2d"])}', flush=True)
    json.dump(out, open(f'{ROOT}/ws_widthtilt/curvature_fit.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
