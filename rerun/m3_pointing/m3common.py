"""Shared helpers of the M3 per-track pointing study: loading, the residual sample, robust widths, M3 geometry."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO]
from rows import ROWS, WORK  # noqa: E402

TABLES = os.path.join(WORK, 'tables')
RES_MAX_MM = 15.0
M3R = json.load(open(os.path.join(REPO, 'mx_june_cosmic_qa', 'm3_self_resolution', 'results.json')))
Z_LAYERS = np.array(M3R['z_layers'], float)
SIG_PLANE = {c: np.array([L['sigma_geomean_um'] / 1000.0 for L in M3R['coords'][c]['layers']]) for c in ('X', 'Y')}


def load(tags=None, sample='resolution'):
    """The joined tables of the rows (all rows when tags is None), with the sample flag of res_rows.py:
    interior_good, both views fitted, no spark veto, |residual| < 15 mm in both views ('resolution'), or the
    efficiency denominator (interior_good, every ray: 'efficiency')."""
    out = []
    for r in ROWS:
        if tags is not None and r['tag'] not in tags:
            continue
        D = pd.read_parquet(os.path.join(TABLES, f"{r['tag']}.parquet"))
        D['pair'] = r['pair']
        out.append(D)
    D = pd.concat(out, ignore_index=True)
    D['fit'] = D.x_ok & D.y_ok
    D['res_ok'] = D.fit & ~D.spark_veto & (D.res_x.abs() < RES_MAX_MM) & (D.res_y.abs() < RES_MAX_MM)
    if sample == 'resolution':
        D = D[D.interior_good & D.res_ok].copy()
    elif sample == 'efficiency':
        D = D[D.interior_good].copy()
    return D


def hat(z_used, z_eval):
    """Coefficients g_k(z) of the unweighted straight-line fit: y(z) = sum_k g_k y_k."""
    A = np.vstack([np.ones_like(z_used), z_used]).T
    return np.array([1.0, z_eval]) @ np.linalg.inv(A.T @ A) @ A.T


def leverage():
    A = np.vstack([np.ones_like(Z_LAYERS), Z_LAYERS]).T
    return A @ np.linalg.inv(A.T @ A) @ A.T


def pointing_gauss(z, coord):
    """Gaussian (MS-free) M3 pointing at z for one M3 coordinate [mm]: sqrt(sum g_k^2 sigma_k^2)."""
    g = hat(Z_LAYERS, z)
    return float(np.sqrt(np.sum(g ** 2 * SIG_PLANE[coord] ** 2)))


def view_coords(theta_deg):
    """weights (c^2, s^2) of M3 X and M3 Y in the chamber views x and y."""
    th = np.deg2rad(theta_deg); c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
    return {'x': (c2, s2), 'y': (s2, c2)}


def s68(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 30:
        return np.nan
    return float(np.percentile(np.abs(d - np.median(d)), 68.27))


def core_sigma(d, s=None):
    """Gaussian fitted to the histogram in the fixed window median +- 2 s68 (res_rows.core_sigma)."""
    from scipy.optimize import curve_fit
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 50:
        return np.nan, np.nan
    s = s68(d) if s is None else s
    med = np.median(d); lo, hi = med - 2 * s, med + 2 * s
    k = (d > lo) & (d < hi)
    nb = int(np.clip(np.sqrt(k.sum()), 12, 60))
    h, e = np.histogram(d[k], bins=nb, range=(lo, hi)); c = 0.5 * (e[1:] + e[:-1])
    f = lambda x, a, m, sg: a * np.exp(-0.5 * ((x - m) / sg) ** 2)
    try:
        p, cov = curve_fit(f, c, h, p0=(h.max(), med, s), sigma=np.sqrt(np.maximum(h, 1)), maxfev=5000)
        return float(abs(p[2])), float(np.sqrt(cov[2, 2]))
    except Exception:
        return np.nan, np.nan


def binned(D, col, edges, f):
    """f(sub) per bin of col; returns a DataFrame with the bin centre (mean of col) and the outputs."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = D[(D[col] >= lo) & (D[col] < hi)]
        if len(s) < 50:
            continue
        o = f(s); o.update(lo=lo, hi=hi, x=float(s[col].mean()), n=len(s))
        rows.append(o)
    return pd.DataFrame(rows)
