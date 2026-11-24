#!/usr/bin/env python3
"""T5 task 4: vertical view.  r_v = measured vertical global slope (unmapped) - slope of the line from the CENTRE of the
source line (y_c = -28.5 mm) to the track point.  For a perfect chamber r_v = (y_c - y_s)/L: a box of half-width
43.5/L ~ 0.185 (source -72 .. +15 mm) blurred by scattering.  Two per-cell estimators, on the |tan| < 0.6 sample (the
vertical box needs the room):
  centre  median of r_v within |r_v| < 0.35, tracks with |e_v| <= 0.20 (box + 2 sigma inside |s_v| < 0.6)
  edge    the upper edge of the box (= the source's sharp lower end, y_s = -72 mm): erf-step fit to the r_v histogram
          on [0.02, 0.45], tracks with e_v <= 0.12 (edge + 2.5 sigma inside s_v < 0.6)
Then a degree-2 polynomial in (x, y) is removed from the cell values and from the cosmic y-map prediction
(a_y + s_y t, t = e_v for the centre, e_v + 0.185 for the edge) and the remaining cell-to-cell structures are
correlated.  Also the box width (edges both sides) where available -> 1 + s_y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc

import pm


def poly2(x, y):
    return np.vstack([np.ones_like(x), x, y, x * x, x * y, y * y]).T


def detrend(Z, xc, yc, v):
    X = poly2(xc[v], yc[v])
    b = np.linalg.lstsq(X, Z[v], rcond=None)[0]
    out = np.full(Z.shape, np.nan)
    out[v] = Z[v] - X @ b
    return out


def step(r, B, N, h, s):
    return B + N * 0.5 * erfc((r - h) / (np.sqrt(2) * s))


def lowstep(r, B, N, h, s):
    return B + N * 0.5 * erfc(-(r - h) / (np.sqrt(2) * s))


def edge_fit(r, upper=True):
    eb = np.arange(0.02, 0.4501, 0.01) if upper else np.arange(-0.45, -0.0199, 0.01)
    h, _ = np.histogram(r, eb)
    if h.sum() < 500:
        return np.nan, np.nan
    mid = 0.5 * (eb[1:] + eb[:-1])
    f = step if upper else lowstep
    p0 = [h[-3:].mean() if upper else h[:3].mean(), h.max(), 0.185 if upper else -0.185, 0.05]
    try:
        p, cov = curve_fit(f, mid, h, p0=p0, sigma=np.sqrt(np.maximum(h, 4)), maxfev=5000)
    except Exception:
        return np.nan, np.nan
    if not (0.01 < p[3] < 0.2):
        return np.nan, np.nan
    return float(p[2]), float(np.sqrt(cov[2, 2]))


def run(arm, edges, d):
    g = d[d.arm == arm]
    A = pm.arm_arrays(g, arm)
    r = A['sv'] - A['ev']
    mp = pm.cosmic_maps()[f'{arm}_y']
    a, s = pm.interp(mp, A['x'], A['y'])
    nc = len(edges) - 1
    ix, iy, ok = pm.cell_index(A['x'], A['y'], edges)
    out = {k: np.full((nc, nc), np.nan) for k in ('cen', 'cen_err', 'hi', 'hi_err', 'lo', 'lo_err', 'pred_cen',
                                                    'pred_hi', 'pred_lo', 'ev', 's_cos', 'L')}
    out['n'] = np.zeros((nc, nc), int)
    mc = np.abs(A['ev']) <= 0.20
    mh = A['ev'] <= 0.12
    ml = A['ev'] >= -0.12
    rng = np.random.default_rng(2)
    for i in range(nc):
        for k in range(nc):
            m = ok & (ix == i) & (iy == k)
            out['n'][i, k] = m.sum()
            if m.sum() < 1000:
                continue
            out['ev'][i, k] = A['ev'][m].mean()
            out['s_cos'][i, k] = s[m].mean()
            out['L'][i, k] = A['L'][m].mean()
            q = m & mc
            if q.sum() >= 1000:
                rr = r[q]
                rr = rr[np.abs(rr) < 0.35]
                out['cen'][i, k] = np.median(rr)
                lab = rng.integers(0, 8, len(rr))
                out['cen_err'][i, k] = np.std([np.median(rr[lab == t]) for t in range(8)], ddof=1) / np.sqrt(8)
                out['pred_cen'][i, k] = np.mean(a[q] + s[q] * A['ev'][q])
            for key, qq, up in (('hi', m & mh, True), ('lo', m & ml, False)):
                if qq.sum() >= 2000:
                    hh, he = edge_fit(r[qq], up)
                    out[key][i, k], out[f'{key}_err'][i, k] = hh, he
                    half = 43.5 / A['L'][qq]
                    t_edge = A['ev'][qq] + (half if up else -half)
                    # the edge in r_v moves by a + s t_edge (distortion at the edge track's true angle)
                    out[f'pred_{key}'][i, k] = np.mean(a[qq] + s[qq] * t_edge)
    cen = 0.5 * (edges[1:] + edges[:-1])
    XC, YC = np.meshgrid(cen, cen, indexing='ij')
    rows = []
    for key in ('cen', 'hi', 'lo'):
        v = np.isfinite(out[key]) & np.isfinite(out[f'pred_{key}']) & (out[f'{key}_err'] < 0.01)
        if v.sum() < 8:
            rows.append(dict(arm=arm, cell=int(edges[1] - edges[0]), est=key, n=int(v.sum())))
            continue
        dz = detrend(out[key], XC, YC, v)
        dp = detrend(out[f'pred_{key}'], XC, YC, v)
        out[f'{key}_detr'], out[f'pred_{key}_detr'] = dz, dp
        e = out[f'{key}_err'][v]
        rows.append(dict(arm=arm, cell=int(edges[1] - edges[0]), est=key, n=int(v.sum()),
                         corr_raw=float(np.corrcoef(out[key][v], out[f'pred_{key}'][v])[0, 1]),
                         corr_detr=float(np.corrcoef(dz[v], dp[v])[0, 1]),
                         rms_detr_point=float(np.std(dz[v])), rms_detr_cos=float(np.std(dp[v])),
                         med_err=float(np.median(e)), slope_detr=float(np.polyfit(dp[v], dz[v], 1)[0]),
                         chi2ndf_detr_diff=float(np.sum(((dz - dp)[v] / e) ** 2) / (v.sum() - 6))))
    # box width -> 1 + s_y : width = hi - lo should be (1 + s) * 87 / L  (+ blur-independent: erf midpoints)
    w = out['hi'] - out['lo']
    vw = np.isfinite(w) & (out['hi_err'] < 0.01) & (out['lo_err'] < 0.01)
    out['s_box'] = w / (87.0 / out['L']) - 1
    if vw.sum() >= 4:
        rows.append(dict(arm=arm, cell=int(edges[1] - edges[0]), est='box_s', n=int(vw.sum()),
                         corr_raw=float(np.corrcoef(out['s_box'][vw], out['s_cos'][vw])[0, 1]) if vw.sum() > 2 else np.nan,
                         rms_detr_point=float(np.std(out['s_box'][vw])), rms_detr_cos=float(np.std(out['s_cos'][vw])),
                         med_err=float(np.median(np.hypot(out['hi_err'], out['lo_err'])[vw] / (87.0 / out['L'][vw]))),
                         mean_s_box=float(np.mean(out['s_box'][vw])), mean_s_cos=float(np.mean(out['s_cos'][vw]))))
    return out, rows


def main():
    d = pm.load(ext=True)
    rows = []
    for arm in 'ACD':
        for edges, lab in ((pm.E25, '25'), (pm.E50, '50')):
            o, rr = run(arm, edges, d)
            np.savez(pm.HERE / f'vmap_{arm}_{lab}.npz', edges=edges, **o)
            rows += rr
            for x in rr:
                print(x, flush=True)
    pd.DataFrame(rows).to_csv(pm.HERE / 'vmap_compare.csv', index=False)


if __name__ == '__main__':
    main()
