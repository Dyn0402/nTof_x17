#!/usr/bin/env python3
"""pair_cov.py -- GB test (c), the key measurement: two DUTs in the same stack, same M3 track.

d_i = theta_i - theta_M3 (FD measured basis, per view).  If the reference error e is common to both chambers,
    cov(d_6, d_7) = cov(e_232, e_702)            and     var(d_6 - d_7) = sigma_6^2 + sigma_7^2 (+ MS between them),
so the covariance is the reference's common term, measured without any model of either chamber.  Before taking it,
each chamber's angle-dependent bias (a function of the shared true angle, so it correlates the two residuals without
being a reference error) is removed by a cubic in theta_ref per chamber; a second pass also removes each chamber's
position-dependent mean (40 mm cells, 2-fold cross-validated, the FD tilt-map recipe) to separate a per-event common
term from position-correlated systematics.  Robust estimators throughout (core |d| < 5 deg; s68 of sum and
difference: cov_s68 = (s68(d6+d7)^2 - s68(d6-d7)^2) / 4), bootstrap errors.

Pairs: det6 (z 232) + det7 (z 702), 6-26 overnight long_run (final_physmask tables, the FD basis); a weak second pair
det3 (z 232, 6-22 long_run, s68 ~5 deg: the "unusable" Monday det3) + det2 (z 702), r06 tables of Aug 21, own map.

    pair_cov.py   -> results/pair_cov.json
"""
import json
import os

import numpy as np
import pandas as pd

from common import RES, BENCH, load_dut, load_rays, RAYS, fit_map, apply_map, deg, s68

RNG = np.random.default_rng(3)


def detrend(d, a, deg_=3):
    ok = np.isfinite(d) & (np.abs(d) < 5)
    c = np.polyfit(a[ok], d[ok], deg_)
    return d - np.polyval(c, a)


def detrend_pos(d, x, y, cell=40.0):
    """subtract a 2-fold cross-validated per-cell mean (cells in mm of the M3 track position)."""
    out = d.copy()
    fold = RNG.integers(0, 2, d.size)
    key = np.floor(x / cell).astype(int) * 1000 + np.floor(y / cell).astype(int)
    for f in (0, 1):
        tr = (fold != f) & np.isfinite(d) & (np.abs(d) < 5)
        mu = pd.Series(d[tr]).groupby(key[tr]).median()
        nn = pd.Series(d[tr]).groupby(key[tr]).size()
        mu = mu[nn >= 10]
        te = fold == f
        out[te] = d[te] - pd.Series(key[te]).map(mu).fillna(0.0).to_numpy()
    return out


def stats(a, b, nboot=300):
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 5) & (np.abs(b) < 5)
    a, b = a[ok] - np.median(a[ok]), b[ok] - np.median(b[ok])

    def one(a, b):
        return dict(cov=float(np.cov(a, b)[0, 1]), r=float(np.corrcoef(a, b)[0, 1]),
                    cov_s68=float((s68(a + b) ** 2 - s68(a - b) ** 2) / 4),
                    s68_a=s68(a), s68_b=s68(b), s68_diff=s68(a - b))
    r = one(a, b)
    bs = [one(a[i], b[i]) for i in (RNG.integers(0, a.size, a.size) for _ in range(nboot))]
    for k in ('cov', 'r', 'cov_s68'):
        r[k + '_err'] = float(np.std([x[k] for x in bs]))
    r['n'] = int(a.size)
    # the common term as an angle (sign kept): sqrt|cov| with the sign of cov
    for k in ('cov', 'cov_s68'):
        r[k.replace('cov', 'common_deg')] = float(np.sign(r[k]) * np.sqrt(abs(r[k])))
        r[k.replace('cov', 'common_deg') + '_ul95'] = float(np.sqrt(max(r[k] + 1.645 * r[k + '_err'], 0)))
    return r


def pair_6_7():
    rays = load_rays(RAYS['det6_det7'])
    out = {}
    for v in ('x', 'y'):
        d6, ev6, _ = load_dut('det6', v)
        d7, ev7, _ = load_dut('det7', v)
        j = d6.join(d7, lsuffix='6', rsuffix='7', how='inner').join(rays, how='inner')
        r6 = (deg(j['tan_fit6']) - deg(j['tan_ref6']))
        r7 = (deg(j['tan_fit7']) - deg(j['tan_ref7']))
        a6, a7 = deg(j['tan_ref6']), deg(j['tan_ref7'])
        # M3 track position at mid-stack (467 mm): common coordinates for the position detrend
        zm = 467.0
        tX = (j['X_Down'] - j['X_Up']) / (j['Z_Down'] - j['Z_Up'])
        tY = (j['Y_Down'] - j['Y_Up']) / (j['Z_Down'] - j['Z_Up'])
        px = (j['X_Up'] + tX * (zm - j['Z_Up'])).to_numpy()
        py = (j['Y_Up'] + tY * (zm - j['Z_Up'])).to_numpy()
        res = {'raw': stats(r6, r7)}
        t6, t7 = detrend(r6, a6), detrend(r7, a7)
        res['angle_detrended'] = stats(t6, t7)
        # position detrend per chamber in its own crossing point (each chamber's own tilt map)
        x6 = (j['X_Up'] + tX * (232 - j['Z_Up'])).to_numpy(); y6 = (j['Y_Up'] + tY * (232 - j['Z_Up'])).to_numpy()
        x7 = (j['X_Up'] + tX * (702 - j['Z_Up'])).to_numpy(); y7 = (j['Y_Up'] + tY * (702 - j['Z_Up'])).to_numpy()
        p6, p7 = detrend_pos(t6, x6, y6), detrend_pos(t7, x7, y7)
        res['angle_and_position_detrended'] = stats(p6, p7)
        # covariance split by M3 chi2 (own coordinate) and by |theta|
        src = 'chi2Y' if v == 'x' else 'chi2X'
        ch = j[src].to_numpy()
        q = np.nanpercentile(ch, [0, 33, 67, 100])
        res['by_chi2'] = [dict(lo=float(lo), hi=float(hi), **stats(t6[(ch >= lo) & (ch <= hi)], t7[(ch >= lo) & (ch <= hi)], 100))
                          for lo, hi in zip(q[:-1], q[1:])]
        res['by_angle'] = []
        for lo, hi in ((0, 5), (5, 15), (15, 35)):
            m = (np.abs(a7) >= lo) & (np.abs(a7) < hi)
            res['by_angle'].append(dict(lo=lo, hi=hi, **stats(t6[m], t7[m], 100)))
        # sanity: the two chambers' own tilt maps, cell medians in M3 mid-stack coordinates, correlated?
        key = np.floor(px / 40).astype(int) * 1000 + np.floor(py / 40).astype(int)
        ok = (np.abs(t6) < 5) & (np.abs(t7) < 5)
        g = pd.DataFrame({'k': key[ok], 'a': t6[ok], 'b': t7[ok]}).groupby('k')
        cm = g.median()[g.size() >= 25]
        res['cell_median_corr'] = dict(n_cells=int(len(cm)), r=float(np.corrcoef(cm['a'], cm['b'])[0, 1]),
                                       s_a=float(cm['a'].std()), s_b=float(cm['b'].std()))
        out[v] = res
        for k in ('raw', 'angle_detrended', 'angle_and_position_detrended'):
            s = res[k]
            print(f"det6+det7 {v} {k:30s} n {s['n']} r {s['r']:+.3f}±{s['r_err']:.3f} cov {s['cov']:+.3f}±{s['cov_err']:.3f} "
                  f"cov_s68 {s['cov_s68']:+.3f}±{s['cov_s68_err']:.3f} | s68 6 {s['s68_a']:.2f} 7 {s['s68_b']:.2f} diff {s['s68_diff']:.2f}")
        print('   by chi2', [(round(b['hi'], 2), round(b['cov_s68'], 3), round(b['cov_s68_err'], 3)) for b in res['by_chi2']])
        print('   by angle', [(b['lo'], b['n'], round(b['cov_s68'], 3), round(b['cov_s68_err'], 3)) for b in res['by_angle']])
        print('   cells', res['cell_median_corr'])
    return out


def pair_3_2():
    """6-22 long_run: det3 (mx17_3, z 232) + det2 (mx17_2, z 702), r06 tables; v1 rays (no NClus: chi2 in (1e-4, 1))."""
    base = f'{BENCH}/Analysis/mx17_det2_det3_overnight_6-22-26/long_run'
    rays = load_rays(f'{BENCH}/det2_det3/mx17_det2_det3_overnight_6-22-26/long_run/m3_tracking_root')
    rays = rays[(rays.chi2X > 1e-4) & (rays.chi2Y > 1e-4) & (rays.chi2X < 1) & (rays.chi2Y < 1)]
    tabs = {}
    for det, sub in (('det3', 'mx17_3'), ('det2', 'mx17_2')):
        ev = pd.read_parquet(f'{base}/{sub}/wft/events.parquet')
        ev = ev[~ev.event_id.duplicated(keep=False)].set_index('event_id')
        meta = json.load(open(f'{base}/{sub}/wft/events.meta.json'))
        ac = meta['angle_constants']
        vd = float(meta['bundle']['v_drift']) if isinstance(meta['bundle'], dict) else float(eval(meta['bundle'])['v_drift'])
        tabs[det] = (ev, ac, vd)
    out = {}
    for v in ('x', 'y'):
        cols = {}
        for det, (ev, ac, vd) in tabs.items():
            e = ev[ev[f'{v}_ok'].astype(bool)]
            tf = (e[f'{v}_w'] * 1e3 - ac['w0'][v]) / (ac['kw'][v] * vd)
            cols[det] = tf
        j = pd.DataFrame({'tf3': cols['det3'], 'tf2': cols['det2']}).dropna().join(rays, how='inner')
        res = {'n_join': int(len(j))}
        refs = {}
        for det in ('det3', 'det2'):
            # rotation-only map, sign and angle from the regression of tan_fit on the M3 slopes
            c, _ = fit_map(j[f'tf{det[-1]}'], j)
            ang = np.arctan2(c[1], c[0])
            cu = np.array([np.cos(ang), np.sin(ang), 0.0])
            refs[det] = apply_map(cu, j['tX'], j['tY'])
            res[f'map_{det}'] = [float(c[0]), float(c[1])]
        r3 = deg(j['tf3']) - deg(refs['det3'])
        r2 = deg(j['tf2']) - deg(refs['det2'])
        res['raw'] = stats(r3, r2)
        t3, t2 = detrend(r3, deg(refs['det3'])), detrend(r2, deg(refs['det2']))
        res['angle_detrended'] = stats(t3, t2)
        pos = {}
        for zz in (232.0, 702.0):
            pos[zz] = ((j['X_Up'] + j['tX'] * (zz - j['Z_Up'])).to_numpy(), (j['Y_Up'] + j['tY'] * (zz - j['Z_Up'])).to_numpy())
        res['angle_and_position_detrended'] = stats(detrend_pos(t3, *pos[232.0]), detrend_pos(t2, *pos[702.0]))
        out[v] = res
        s2 = res['angle_and_position_detrended']
        print(f"   + position detrend: r {s2['r']:+.3f}±{s2['r_err']:.3f} cov {s2['cov']:+.3f}±{s2['cov_err']:.3f} cov_s68 {s2['cov_s68']:+.3f}±{s2['cov_s68_err']:.3f}")
        s = res['angle_detrended']
        print(f"det3+det2 {v} n {s['n']} r {s['r']:+.3f}±{s['r_err']:.3f} cov {s['cov']:+.3f}±{s['cov_err']:.3f} cov_s68 {s['cov_s68']:+.3f}"
              f"±{s['cov_s68_err']:.3f} | s68 3 {s['s68_a']:.2f} 2 {s['s68_b']:.2f} diff {s['s68_diff']:.2f}  maps {res['map_det3']} {res['map_det2']}")
    return out


def main():
    out = {'det6_det7': pair_6_7(), 'det3_det2_6-22': pair_3_2()}
    json.dump(out, open(os.path.join(RES, 'pair_cov.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
