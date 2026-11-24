#!/usr/bin/env python3
"""T5 tasks 2, 3, 6 (horizontal view): per-cell pointing residual centre c, the cosmic map's prediction a + s t_p,
comparison and systematics.  Writes hmap_<arm>_<cell>.npz pieces collected later into pointmap.npz, and hmap.csv.

Validity of a cell (all data-driven):
  reach      track |e| <= E_REACH (0.44): the windowed median of the |tan| < 0.5
             sample and the |tan| < 0.6 extended sample agree to < 1e-3 up to |e| = 0.46 (central y), so the acceptance
             truncation does not reach the core there
  normal     track |e| >= deficit half-width of the measured slopes + 0.06 (per arm, from data): near-normal tracks are repelled from tan = 0 by the
             reconstruction (deficit of measured tan near 0, S-shaped residual), a tan-dependent effect, not a map
  hole       no low-density 5 mm strip band (< 0.6 x running median) within HOLE_PAD of the cell in x_local: positions
             next to a dead band are biased, so the expected tan is wrong (C x_local 10-25, D several)
  core       core sigma < 1.3 x the arm's median over cells (y-edge cells hold tracks that passed the vertical gate only
             by scattering or mis-measurement)
  n          >= 1000 tracks
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import pm

E_REACH, E_NORMAL, HOLE_PAD, NMIN = 0.44, 0.08, 10.0, 1000
E_NORMAL_ARM = {}   # filled from the data (normal_halfwidth + 0.06) unless set here


def holes(x, y):
    eb = np.arange(-200, 201, 5.)
    h = np.histogram(x[np.abs(y) < 100], eb)[0]
    ref = median_filter(h, 11, mode='nearest')
    return eb[:-1][h < 0.6 * ref]


def cosmic_pred_err(arm, x, y, e, nboot=30):
    """Bootstrap error of the cosmic map's per-track prediction a + s e (replicas of angle_map.fit_map / bd_map)."""
    from rerun.ntof.improve import angle_map as AM
    if arm in 'AC':
        M = pd.read_parquet(pm.LINES)
        G = AM.clean_lines(M, arm.lower(), 'x')
        edges = AM.EDGES
        nmin = AM.N_MIN
    else:
        from rerun.ntof.improve import bd_map as BD
        G = BD.clean(pd.read_parquet(pm.BDLINES), 'z')
        edges, nmin = BD.EDGES, 60
    old = AM.EDGES, AM.CENTRES, AM.N_MIN
    AM.EDGES, AM.CENTRES, AM.N_MIN = edges, 0.5 * (edges[1:] + edges[:-1]), nmin
    rng = np.random.default_rng(11)
    reps = []
    try:
        for _ in range(nboot):
            mp = AM.fit_map(G.iloc[rng.integers(0, len(G), len(G))])
            a, s = pm.interp(mp, x, y)
            reps.append(a + s * e)
    finally:
        AM.EDGES, AM.CENTRES, AM.N_MIN = old
    return np.array(reps)          # (nboot, ntracks)


def cell_stats(x, y, vals, edges):
    """cell means of several per-track arrays."""
    nc = len(edges) - 1
    ix, iy, ok = pm.cell_index(x, y, edges)
    key = np.where(ok, ix * nc + iy, nc * nc)
    cnt = np.bincount(key, minlength=nc * nc + 1)[:-1]
    out = {}
    for k, v in vals.items():
        if v.ndim == 1:
            s = np.bincount(key, weights=np.nan_to_num(v), minlength=nc * nc + 1)[:-1]
            out[k] = (s / np.where(cnt > 0, cnt, np.nan)).reshape(nc, nc)
        else:
            out[k] = np.stack([(np.bincount(key, weights=np.nan_to_num(vv), minlength=nc * nc + 1)[:-1]
                                / np.where(cnt > 0, cnt, np.nan)).reshape(nc, nc) for vv in v])
    return out


def normal_halfwidth(sh, y):
    """Half-width of the near-normal deficit of MEASURED slopes (0.01 bins below 0.85 x the |s| 0.12-0.25 level,
    contiguous around 0), central y."""
    eb = np.arange(-0.3, 0.3001, 0.01)
    h = np.histogram(sh[np.abs(y) < 100], eb)[0].astype(float)
    mid = 0.5 * (eb[1:] + eb[:-1])
    ref = np.interp(mid, mid[(np.abs(mid) > 0.12) & (np.abs(mid) < 0.25)],
                    h[(np.abs(mid) > 0.12) & (np.abs(mid) < 0.25)])
    low = h < 0.85 * ref
    i0 = int(np.argmin(np.abs(mid)))
    lo = hi = i0
    while lo > 0 and low[lo - 1]:
        lo -= 1
    while hi < len(mid) - 1 and low[hi + 1]:
        hi += 1
    if not low[i0]:
        return 0.0
    return float(max(abs(eb[lo]), abs(eb[hi + 1])))


def track_ok(A, hole_x):
    """track-level reach / near-normal / dead-band masks (see the module docstring)."""
    ae = np.abs(A['e'])
    en = A.get('e_normal', E_NORMAL)
    ok = (ae >= en) & (ae <= E_REACH)
    for h in hole_x:
        ok &= ~((A['x'] > h - HOLE_PAD) & (A['x'] < h + 5 + HOLE_PAD))
    return ok


def validity(m, edges, hole_x, e_lo, e_hi):
    nc = len(edges) - 1
    reach = np.isfinite(m['c'])
    normal = hole = reach
    sig_med = np.nanmedian(m['sig'][reach & (m['n'] >= NMIN)])
    core = m['sig'] < 1.3 * sig_med
    nok = m['n'] >= NMIN
    v = reach & normal & hole & core & nok & np.isfinite(m['c'])
    return v, dict(reach=reach, normal=normal, hole=hole, core=core, nok=nok)


def compare(c, ce, p, pe, e, v):
    """c (pointing) vs p (cosmic) on valid cells: correlation, slope of c on p, chi2/ndf of c - p, and the same after a
    free alpha + gamma e (source x and distance systematics)."""
    c, ce, p, pe, e = c[v], ce[v], p[v], pe[v], e[v]
    n = len(c)
    if n < 4:
        return dict(n=n)
    s2 = ce ** 2 + pe ** 2
    d = c - p
    chi2 = float(np.sum(d ** 2 / s2))
    X = np.vstack([np.ones(n), e]).T
    W = 1 / s2
    beta = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * d))
    d2 = d - X @ beta
    chi2b = float(np.sum(d2 ** 2 / s2))
    # orthogonal-ish: slope of c on p (weighted least squares with combined errors, both centred)
    k = np.polyfit(p, c, 1, w=1 / np.sqrt(s2))
    # correlation after removing each one's own alpha + gamma e
    cc = c - X @ np.linalg.lstsq(X, c, rcond=None)[0]
    pp = p - X @ np.linalg.lstsq(X, p, rcond=None)[0]
    return dict(n=n, corr=float(np.corrcoef(c, p)[0, 1]), corr_detr=float(np.corrcoef(cc, pp)[0, 1]),
                slope=float(k[0]), offset=float(k[1]), chi2=chi2, chi2_ndf=chi2 / n, alpha=float(beta[0]),
                gamma=float(beta[1]), chi2b_ndf=chi2b / (n - 2), rms_c=float(np.std(c)), rms_p=float(np.std(p)),
                rms_diff=float(np.std(d)), rms_diff_b=float(np.std(d2)), med_ce=float(np.median(ce)),
                med_pe=float(np.median(pe)), rms_cc=float(np.std(cc)), rms_pp=float(np.std(pp)))


def run_arm(d, arm, edges, label, sys_kw=None, sel=None, boot=True, ext=None):
    g = d[d.arm == arm]
    if sel is not None:
        g = g[sel(g)]
    A = pm.arm_arrays(g, arm, **(sys_kw or {}))
    mp = pm.cosmic_maps()[f'{arm}_x']
    a, s = pm.interp(mp, A['x'], A['y'])
    pred = a + s * A['e']
    r = A['sh'] - A['e']
    rm = A['shm'] - A['e']
    hx = holes(A['x'], A['y'])
    A['e_normal'] = E_NORMAL_ARM.get(arm) or (normal_halfwidth(A['sh'], A['y']) + 0.06)
    tk = track_ok(A, hx)
    X, Y = np.where(tk, A['x'], np.nan), np.where(tk, A['y'], np.nan)
    m = pm.cell_map(X, Y, r, edges, extra=None, nmin=300)
    mm = pm.cell_map(X, Y, rm, edges, extra=None, nmin=300)
    cs = cell_stats(X, Y, dict(e=A['e'], pred=pred, a=a, s=s), edges)
    # e range of each cell: e at the cell's x edges, from the arm's own position->e relation (linear fit on tracks)
    k = np.polyfit(A['x'], A['e'], 1)
    e_lo = np.add.outer(np.polyval(k, edges[:-1]), np.zeros(len(edges) - 1))
    e_hi = np.add.outer(np.polyval(k, edges[1:]), np.zeros(len(edges) - 1))
    v, flags = validity(m, edges, hx, e_lo, e_hi)
    out = dict(c=m['c'], err=m['err'], n=m['n'], sig=m['sig'], c_map=mm['c'], err_map=mm['err'], sig_map=mm['sig'],
               e=cs['e'], pred=cs['pred'], a_cos=cs['a'], s_cos=cs['s'], valid=v, e_lo=e_lo, e_hi=e_hi,
               **{f'flag_{kk}': vv for kk, vv in flags.items()})
    if boot:
        reps = cosmic_pred_err(arm, A['x'], A['y'], A['e'])
        cr = cell_stats(X, Y, dict(rep=reps), edges)['rep']
        out['pred_err'] = np.nanstd(cr, axis=0, ddof=1)
    if ext is not None:
        ge = ext[ext.arm == arm]
        if sel is not None:
            ge = ge[sel(ge)]
        Ae = pm.arm_arrays(ge, arm, **(sys_kw or {}))
        Ae['e_normal'] = A['e_normal']
        tke = track_ok(Ae, hx)
        me = pm.cell_map(np.where(tke, Ae['x'], np.nan), Ae['y'], Ae['sh'] - Ae['e'], edges, nmin=300)
        out['c_ext06'] = me['c']
        out['err_ext06'] = me['err']
    # pointing core width over the reach region, before / after the cosmic map (|e| <= reach, |y| <= 125, not normal)
    reg = tk & (np.abs(A['y']) <= 125)
    out['e_normal'] = A['e_normal']
    out['holes_x'] = hx
    out['core_before'] = pm.centre(r[reg])[3]
    out['core_after'] = pm.centre(rm[reg])[3]
    out['n_region'] = int(reg.sum())
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'main'
    d = pm.load()
    rows = []
    if which == 'main':
        ext = pm.load(ext=True)
        for arm in 'ACD':
            for edges, lab in ((pm.E25, '25'), (pm.E50, '50')):
                o = run_arm(d, arm, edges, lab, ext=ext)
                np.savez(pm.HERE / f'hmap_{arm}_{lab}.npz', edges=edges, **o)
                cmp = compare(o['c'], o['err'], o['pred'], o['pred_err'], o['e'], o['valid'])
                v = o['valid']
                cmp.update(arm=arm, cell=lab, n_valid=int(v.sum()),
                           ext_maxdiff=float(np.nanmax(np.abs(o['c'] - o['c_ext06'])[v])) if v.any() else np.nan,
                           ext_rmsdiff=float(np.sqrt(np.nanmean(((o['c'] - o['c_ext06'])[v]) ** 2))),
                           core_before=o['core_before'], core_after=o['core_after'],
                           cellrms_before=float(np.std(o['c'][v])), cellrms_after=float(np.std(o['c_map'][v])),
                           n_tracks_valid=int(o['n'][v].sum()))
                # cell-to-cell rms after removing the global alpha + gamma e, before / after map
                X = np.vstack([np.ones(v.sum()), o['e'][v]]).T
                for kk, cc in (('before', o['c'][v]), ('after', o['c_map'][v])):
                    cmp[f'cellrms_detr_{kk}'] = float(np.std(cc - X @ np.linalg.lstsq(X, cc, rcond=None)[0]))
                rows.append(cmp)
                print(pd.Series(cmp).to_string(), flush=True)
        pd.DataFrame(rows).to_csv(pm.HERE / 'hmap_compare.csv', index=False)
    elif which == 'sys':
        # systematics at 50 mm cells: change of c on the valid cells of the nominal map
        variants = {'xs+1': dict(xs=pm.XS + 1), 'xs-1': dict(xs=pm.XS - 1), 'zs+1': dict(zs=pm.ZS + 1),
                    'zs-1': dict(zs=pm.ZS - 1), 'dist+1%': dict(dist_scale=1.01), 'midgap': dict(plane_dz=15.0)}
        rn = d.rn
        half = np.median(rn)
        qcut = {arm: np.nanquantile(d.q_total[d.arm == arm], [1 / 3, 2 / 3]) for arm in 'ACD'}
        sels = {'first_half': lambda g: g.rn <= half, 'second_half': lambda g: g.rn > half}
        for arm in 'ACD':
            nom = dict(np.load(pm.HERE / f'hmap_{arm}_50.npz'))
            v = nom['valid']
            q1, q2 = qcut[arm]
            ss = dict(sels)
            ss['q_low'] = lambda g, q1=q1: g.q_total <= q1
            ss['q_high'] = lambda g, q2=q2: g.q_total > q2
            res = {}
            for lab, kw in variants.items():
                res[lab] = run_arm(d, arm, pm.E50, '50', sys_kw=kw, boot=False)
            for lab, f in ss.items():
                res[lab] = run_arm(d, arm, pm.E50, '50', sel=f, boot=False)
            for lab, o in res.items():
                v = nom['valid'] & np.isfinite(o['c']) & np.isfinite(o['err'])
                dc = (o['c'] - nom['c'])[v]
                de = np.sqrt(o['err'] ** 2 + (0 if lab in variants else 1) * nom['err'] ** 2)[v]
                X = np.vstack([np.ones(v.sum()), nom['e'][v]]).T
                beta = np.linalg.lstsq(X, dc, rcond=None)[0]
                cmpv = compare(o['c'], o['err'], nom['pred'], nom['pred_err'], nom['e'], v)
                rows.append(dict(arm=arm, variant=lab, mean_dc=float(np.mean(dc)), rms_dc=float(np.std(dc)),
                                 dc_const=float(beta[0]), dc_slope_e=float(beta[1]),
                                 rms_dc_detr=float(np.std(dc - X @ beta)), chi2_vs_nom=float(np.sum((dc / de) ** 2) / len(dc))
                                 if lab not in variants else np.nan,
                                 corr_with_cosmic_detr=cmpv.get('corr_detr'), slope_on_cosmic=cmpv.get('slope'),
                                 n_cells=int(v.sum())))
                np.savez(pm.HERE / f'hsys_{arm}_{lab}.npz', **{k: o[k] for k in ('c', 'err', 'n')})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(pm.HERE / 'hmap_sys.csv', index=False)


if __name__ == '__main__':
    main()
