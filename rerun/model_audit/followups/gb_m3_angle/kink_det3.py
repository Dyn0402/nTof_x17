#!/usr/bin/env python3
"""kink_det3.py -- GB test (a)+(b) on the one run with per-plane M3 positions (sat det3, files 0-3,
mx_june_cosmic_qa/m3_self_resolution/sat_perplane_0to3.root, 47 452 events; det3 at z = 702).

Per event: the two doublet slopes s_top (z 1302/1185) and s_bot (144/24), their difference = the inter-doublet kink
(multiple scattering between the doublets + doublet slope noise; the per-event momentum proxy: a soft muon kinks), and
the 4-point unweighted fit slope (what the reference is).  Joined on evn to det3's FD measured basis (final_physmask
per_event.npz) and, for the chi2 scan, to every fitted event of the table.

  1. resolution (s68, FD convention) of d = theta_det3 - theta_M3 in quintiles of |kink| and of M3 chi2;
  2. the regression of d on the kink (MS at the DUT is e = c * kink_MS, c ~ +-0.5 at z = 702) -> variance explained;
  3. the kink distribution itself: s68 vs the doublet-noise-only expectation (Gaussian, per-plane sigmas of the M3
     self-resolution study) -> the MS kink width with its tails.

    kink_det3.py   -> results/kink_det3.json
"""
import json
import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from common import RES, load_dut, tan_fit_all, fit_map, apply_map, deg, s68, M3_Z

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
PERPLANE = os.path.join(REPO, 'mx_june_cosmic_qa', 'm3_self_resolution', 'sat_perplane_0to3.root')
SIG = {'X': [0.415, 0.410, 0.409, 0.411], 'Y': [0.511, 0.503, 0.445, 0.458]}      # M3_SELF_RESOLUTION.md 4.1


def perplane():
    a = uproot.open(PERPLANE)['T'].arrays(library='ak')
    a = a[ak.num(a['Chi2X']) == 1]
    d = {'evn': ak.to_numpy(a['evn']).astype(np.int64)}
    for c in ('X', 'Y'):
        m = np.c_[[ak.to_numpy(ak.firsts(a[f'm{c}{k}'])) for k in range(4)]].T
        z = np.c_[[ak.to_numpy(ak.firsts(a[f'z{c}{k}'])) for k in range(4)]].T
        d[f'chi2{c}'] = ak.to_numpy(ak.firsts(a[f'Chi2{c}']))
        d[f'n{c}'] = ak.to_numpy(ak.firsts(a[f'NClus{c}']))
        d[f'st{c}'] = (m[:, 0] - m[:, 1]) / (z[:, 0] - z[:, 1])
        d[f'sb{c}'] = (m[:, 2] - m[:, 3]) / (z[:, 2] - z[:, 3])
        zb = z.mean(1, keepdims=True)
        d[f'sf{c}'] = ((z - zb) * (m - m.mean(1, keepdims=True))).sum(1) / ((z - zb) ** 2).sum(1)
        d[f'k{c}'] = d[f'st{c}'] - d[f'sb{c}']
    r = pd.DataFrame(d).drop_duplicates('evn', keep=False).set_index('evn')
    return r


def kink_noise(c):
    """doublet-noise-only sigma of s_top - s_bot [rad] from the per-plane sigmas."""
    s = SIG[c]
    return float(np.sqrt((s[0] ** 2 + s[1] ** 2) / (M3_Z[0] - M3_Z[1]) ** 2 + (s[2] ** 2 + s[3] ** 2) / (M3_Z[2] - M3_Z[3]) ** 2))


def bins_of(x, d, q=(0, 20, 40, 60, 80, 100)):
    ok = np.isfinite(x) & np.isfinite(d)
    e = np.nanpercentile(x[ok], q)
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        m = ok & (x >= lo) & (x <= hi)
        out.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), s68=s68(d[m])))
    return out


def main():
    r = perplane()
    out = {'n_rays': int(len(r))}
    rng = np.random.default_rng(1)
    prod = (r['nX'] == 4) & (r['nY'] == 4) & (r['chi2X'] < 1) & (r['chi2Y'] < 1)
    for c, sel_name in (('X', 'all'), ('Y', 'all'), ('X', 'prod'), ('Y', 'prod')):
        k = r[f'k{c}'][(r[f'n{c}'] == 4) & (prod if sel_name == 'prod' else True)].to_numpy() * 1e3
        noise = kink_noise(c) * 1e3
        sim = rng.normal(0, noise, 200000)
        # MS kink width that, convolved with the noise, reproduces the measured s68 (Gaussian MS)
        tgt = np.percentile(np.abs(k - np.median(k)), 68)
        grid = np.linspace(0, 60, 601)
        s_of = [np.percentile(np.abs(sim + rng.normal(0, g, sim.size)), 68) for g in grid]
        g68 = float(np.interp(tgt, s_of, grid))
        out[f'kink_{c}_{sel_name}'] = dict(n=int(len(k)), s68_mrad=float(tgt), noise_sigma_mrad=noise,
                                noise_s68_mrad=float(np.percentile(np.abs(sim), 68)), ms_sigma_from_s68_mrad=g68,
                                p95_mrad=float(np.percentile(np.abs(k - np.median(k)), 95)),
                                noise_p95_mrad=float(np.percentile(np.abs(sim), 95)),
                                p99_mrad=float(np.percentile(np.abs(k - np.median(k)), 99)),
                                noise_p99_mrad=float(np.percentile(np.abs(sim), 99)))
        print(c, sel_name, out[f'kink_{c}_{sel_name}'])
    for v in ('x', 'y'):
        d, ev, b = load_dut('det3', v)
        m = d.join(r, how='inner')
        cmap, rms = fit_map(m['tan_ref'], m.rename(columns={'sfX': 'tX', 'sfY': 'tY'}))
        # the kink and the chi2 seen in this view: the M3 coordinate the view is made of
        src = 'X' if abs(cmap[0]) > abs(cmap[1]) else 'Y'
        sign = np.sign(cmap[0] if src == 'X' else cmap[1])
        dd = pd.Series(deg(m['tan_fit']) - deg(m['tan_ref']), index=m.index)
        kk = sign * m[f'k{src}'].to_numpy() * 1e3
        res = dict(map=list(map(float, cmap)), map_rms=rms, m3_coord=src, n=int(len(m)), s68_all=s68(dd))
        res['by_abs_kink'] = bins_of(np.abs(kk), dd.to_numpy())
        # regression of d (deg) on the kink (mrad), core |d| < 5
        ok = np.isfinite(dd) & (np.abs(dd) < 5) & np.isfinite(kk) & (np.abs(kk) < 60)
        x, y = kk[ok], dd.to_numpy()[ok] - np.median(dd[ok])
        beta = float(np.cov(x, y)[0, 1] / np.var(x))
        res['beta_deg_per_mrad'] = beta
        res['beta_dimless'] = beta / np.degrees(1e-3)
        res['corr'] = float(np.corrcoef(x, y)[0, 1])
        res['var_explained_deg2'] = float(beta ** 2 * np.var(x))
        res['s68_after_kink_regression'] = s68(dd.to_numpy()[ok] - beta * x)
        # bootstrap on beta
        bs = []
        for _ in range(300):
            i = rng.integers(0, len(x), len(x))
            bs.append(np.cov(x[i], y[i])[0, 1] / np.var(x[i]))
        res['beta_err'] = float(np.std(bs))
        # alternative references: the doublet slopes and the full fit, same events
        for nm, col in (('top', 'st'), ('bot', 'sb'), ('full_refit', 'sf')):
            tref = apply_map(cmap, m[f'{col}X'], m[f'{col}Y'])
            res[f's68_vs_{nm}'] = s68(deg(m['tan_fit']) - deg(tref))
        # chi2 scan on every fitted det3 event with an NClus-4 single ray (not only the production chi2 < 1)
        tf = tan_fit_all(ev, b, v)
        e2 = pd.DataFrame({'tan_fit': tf, 'ok': ev[f'{v}_ok']}).join(r, how='inner')
        e2 = e2[e2['ok'].astype(bool) & (e2['nX'] == 4) & (e2['nY'] == 4)]
        tref = apply_map(cmap, e2['sfX'], e2['sfY'])
        d2 = np.asarray(deg(e2['tan_fit']) - deg(tref))
        a_ref = deg(tref)
        sel = np.abs(a_ref) < 35
        chi = np.maximum(e2['chi2X'], e2['chi2Y']).to_numpy()
        res['chi2_scan'] = []
        for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 20), (20, 1e9)):
            mm = sel & (chi >= lo) & (chi < hi)
            res['chi2_scan'].append(dict(lo=lo, hi=hi, n=int(mm.sum()), s68=s68(d2[mm])))
        own = e2[f'chi2{src}'].to_numpy()
        res['chi2_own_quintiles'] = bins_of(np.where(sel, own, np.nan), d2)
        out[f'det3_{v}'] = res
        print(v, json.dumps({k: res[k] for k in res if k not in ('by_abs_kink', 'chi2_scan', 'chi2_own_quintiles')}))
        print('  |kink| quintiles', [round(b_['s68'], 3) for b_ in res['by_abs_kink']], [round(b_['hi'], 1) for b_ in res['by_abs_kink']])
        print('  chi2 scan', [(b_['lo'], b_['n'], round(b_['s68'], 3)) for b_ in res['chi2_scan']])
    json.dump(out, open(os.path.join(RES, 'kink_det3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
