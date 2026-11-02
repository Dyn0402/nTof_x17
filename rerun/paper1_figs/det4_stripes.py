#!/usr/bin/env python3
"""det4_stripes.py --rays R --tel DIR [--out results/det4_stripes.json] -- det4's non-amplifying stripes on pass 1, bench
and SPS, and whether the two agree (Paper I §9).

  bench  the golden row g_det4 (6-24 long_run): per M3 ray, the X view detected (the budget's cat_x_5 == 'detected':
         fitted, |r| <= 5 mm, <= 2 competing strips) against the ray's x at mid-gap, 1 mm bins, y interior; the
         same for Y against y.  No dead-stripe map is used (the budget's 'hw_known' class is just not 'detected').
  SPS    run63 flat700 and run53 (flat mount), telescope-first: every event with one cluster in each of the four uRWELL
         planes, the track at det4's z (Z_DET4), mapped into det4's strip frame by a robust affine fit on the events
         where det4 sees one cluster in both views (centroid estimator, |residual| < 1.5 mm iterated); detected =
         det4 view fitted and its centroid within 3 mm of the prediction.  Profile against det4 x (and y), 1 mm bins.
  agree  live/dead classification per 1 mm bin (efficiency above / below half the view's plateau) over the x range
         both cover; the fraction of bins that agree, for the identity and for the mirror x -> 398.6 - x, and the
         band edges of each.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'sps_audit'))
from analyze import Z_BACK, Z_DET4  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers'))
import paper_style as pst  # noqa: E402  (pst: `ps` is a profile array below)

BIN = 1.0
XMAX = 398.6


def profile(x, ok, lo=0.0, hi=XMAX, step=BIN, nmin=8):
    e = np.arange(lo, hi + step, step); k = np.digitize(x, e) - 1
    c = e[:-1] + step / 2
    n = np.bincount(k[(k >= 0) & (k < len(c))], minlength=len(c)).astype(float)
    s = np.bincount(k[(k >= 0) & (k < len(c))], weights=ok[(k >= 0) & (k < len(c))].astype(float), minlength=len(c))
    p = np.where(n >= nmin, s / np.maximum(n, 1), np.nan)
    return c, p, n


def bands(c, live):
    """[(lo, hi)] of contiguous live runs."""
    out, start = [], None
    for x, l in zip(c, live):
        if l and start is None:
            start = x
        if not l and start is not None:
            out.append((float(start - BIN / 2), float(x - BIN / 2))); start = None
    if start is not None:
        out.append((float(start - BIN / 2), float(c[-1] + BIN / 2)))
    return out


def bench(path):
    R = pd.read_parquet(path)
    R = R[~R.not_measured]
    px, py = R.mesh_x + 15 * R.tan_x, R.mesh_y + 15 * R.tan_y
    iny = R.region_y.isin(['interior', 'edge-other'])
    inx = (px > 30) & (px < XMAX - 30)
    out = {}
    out['x'] = profile(px[iny].values, (R.cat_x_5[iny] == 'detected').values)
    out['y'] = profile(py[inx].values, (R.cat_y_5[inx] == 'detected').values)
    out['xy_area'] = float((R.cat_xy_5[iny & inx] == 'detected').mean())
    return out


def robust_affine(T, E, it=4, cut=1.5):
    ok = np.all(np.isfinite(T), 1) & np.isfinite(E)
    for _ in range(it):
        A = np.c_[T[ok], np.ones(ok.sum())]
        c, *_ = np.linalg.lstsq(A, E[ok], rcond=None)
        r = np.c_[T, np.ones(len(T))] @ c - E
        ok = np.isfinite(r) & (np.abs(r) < max(cut, 3 * 1.4826 * np.nanmedian(np.abs(r[ok]))) if _ == 0 else np.isfinite(r) & (np.abs(r) < cut))
    return c, ok


def sps(tel_dir, prefixes):
    fs = sorted(f for p in prefixes for f in glob.glob(os.path.join(tel_dir, p + '_[0-9][0-9][0-9].parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    tel = np.all([(d[f'{p}_n'] == 1).values for p in ('fx', 'fy', 'bx', 'by')], axis=0)
    d = d[tel].reset_index(drop=True)
    tx = (d.fx_p + (d.bx_p - d.fx_p) * Z_DET4 / Z_BACK).values
    ty = (d.fy_p + (d.by_p - d.fy_p) * Z_DET4 / Z_BACK).values
    T = np.c_[tx, ty]
    ex, ey = d.d4_x_c_pos.values.astype(float), d.d4_y_c_pos.values.astype(float)
    okx = d.d4_x_c_n_strips.fillna(0).values > 0
    oky = d.d4_y_c_n_strips.fillna(0).values > 0
    ex[~okx] = np.nan; ey[~oky] = np.nan
    cx, _ = robust_affine(T, np.where(np.isfinite(ey), ex, np.nan))
    cy, _ = robust_affine(T, np.where(np.isfinite(ex), ey, np.nan))
    Px = np.c_[T, np.ones(len(T))] @ cx; Py = np.c_[T, np.ones(len(T))] @ cy
    detx = np.isfinite(ex) & (np.abs(ex - Px) < 3.0)
    dety = np.isfinite(ey) & (np.abs(ey - Py) < 3.0)
    return dict(x=profile(Px, detx, nmin=20), y=profile(Py, dety, nmin=20), n=int(len(d)), cx=cx.tolist(), cy=cy.tolist(),
                x_range=[float(np.nanpercentile(Px, 1)), float(np.nanpercentile(Px, 99))],
                y_range=[float(np.nanpercentile(Py, 1)), float(np.nanpercentile(Py, 99))])


def live_of(p):
    pl = np.nanpercentile(p, 90)
    return np.where(np.isfinite(p), p > 0.5 * pl, np.nan)


def agree(cb, pb, cs, ps):
    lb, ls = live_of(pb), live_of(ps)
    both = np.isfinite(lb) & np.isfinite(ls)
    a_id = float(np.mean(lb[both] == ls[both])) if both.any() else np.nan
    lbm = live_of(np.interp(XMAX - cb, cb, pb))       # mirror: bench at 398.6 - x
    bothm = np.isfinite(lbm) & np.isfinite(ls)
    a_mi = float(np.mean(lbm[bothm] == ls[bothm])) if bothm.any() else np.nan
    # best shift (identity orientation) within +-5 mm
    best = (0.0, a_id)
    for s in np.arange(-5, 5.01, 0.5):
        lsh = live_of(np.interp(cb + s, cb, pb))
        b = np.isfinite(lsh) & np.isfinite(ls)
        if b.sum() > 20:
            a = float(np.mean(lsh[b] == ls[b]))
            if a > best[1]:
                best = (float(s), a)
    return dict(n_bins=int(both.sum()), agree_identity=a_id, agree_mirror=a_mi, best_shift_mm=best[0], agree_best=best[1])


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('--rays', required=True); ap.add_argument('--tel', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'det4_stripes.json'))
    ap.add_argument('--fig', default=os.path.join(HERE, '..', '..', 'papers', 'paper1', 'figures', 'det4_stripes.pdf'))
    a = ap.parse_args()
    B = bench(a.rays)
    S = {'run63 flat700 (CF4)': sps(a.tel, ['G6302', 'G6303']), 'run53 (Ar/CO2/iso)': sps(a.tel, ['G53'])}
    res = dict(bench=dict(xy_area_interior=B['xy_area']), sps={})
    cb, pb, nb = B['x']
    rng = (cb > 30) & (cb < XMAX - 30) & np.isfinite(pb)
    lb = live_of(pb)
    res['bench']['dead_fraction_x_interior'] = float(1 - np.nanmean(lb[rng]))
    res['bench']['live_bands_x'] = bands(cb[rng], lb[rng] == 1)
    res['bench']['plateau_x'] = float(np.nanpercentile(pb, 90))
    for nm, s in S.items():
        cs, ps, ns = s['x']
        win = (cs >= s['x_range'][0]) & (cs <= s['x_range'][1]) & np.isfinite(ps)
        ps_w = np.where(win, ps, np.nan)
        ag = agree(cb, pb, cs, ps_w)
        ls = live_of(ps_w)
        res['sps'][nm] = dict(n_tracks=s['n'], x_range=s['x_range'], y_range=s['y_range'], plateau_x=float(np.nanpercentile(ps_w, 90)),
                              live_bands_x=bands(cs[win], ls[win] == 1), **ag)
        print(nm, res['sps'][nm])
    print('bench', res['bench'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)

    pst.use()
    gas_col = {'run63 flat700 (CF4)': pst.GAS['CF4'], 'run53 (Ar/CO2/iso)': pst.GAS['CO2']}
    gas_lbl = {'run63 flat700 (CF4)': r'run 63 (CF$_4$ mixture)', 'run53 (Ar/CO2/iso)': r'run 53 (Ar/CO$_2$/iC$_4$H$_{10}$)'}
    prof = [pd.DataFrame(dict(set='bench g_det4', x_mm=cb, eff=pb, n=nb))]
    fig, axs = plt.subplots(2, 1, figsize=(pst.FULL[0], 3.9), sharex=True)
    axs[0].step(cb, 100 * pb, where='mid', color=pst.color('det4'), lw=1.0)
    axs[0].plot([], [], color=pst.color('det4'), lw=3, label='bench, golden row (cosmic, M3 reference)')
    for nm, s in S.items():
        cs, ps, ns = s['x']; win = (cs >= s['x_range'][0]) & (cs <= s['x_range'][1])
        axs[1].step(cs[win], 100 * ps[win], where='mid', lw=1.0, color=gas_col[nm])
        axs[1].plot([], [], color=gas_col[nm], lw=3, label=gas_lbl[nm])
        prof.append(pd.DataFrame(dict(set=nm, x_mm=cs[win], eff=ps[win], n=ns[win])))
    lo, hi = min(r['x_range'][0] for r in res['sps'].values()), max(r['x_range'][1] for r in res['sps'].values())
    for ax in axs:
        ax.axvspan(lo, hi, color=pst.GRID, zorder=0)
        ax.set_ylabel('X-view efficiency [%]'); ax.set_ylim(0, 105); ax.set_xlim(0, XMAX)
    axs[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=2.0, borderaxespad=0.2)
    axs[1].legend(loc='center left', handlelength=2.0, title='SPS H4 (beam, uRWELL telescope)', title_fontsize=pst.BASE_PT * 0.85,
                  alignment='left')
    axs[0].text(0.5 * (lo + hi), 103, 'SPS beam spot', ha='center', va='top', fontsize=7, color=pst.MUTED)
    axs[1].set_xlabel('x in the det4 strip frame [mm] (1 mm bins)')
    fig.tight_layout(h_pad=0.6)
    fig.savefig(a.fig)                                       # the paper copy (pdf)
    pst.save(fig, os.path.join(HERE, 'figures', 'det4_stripes'), data=pd.concat(prof, ignore_index=True))


if __name__ == '__main__':
    main()
