#!/usr/bin/env python3
"""paperfig.py <work> -- the three Paper I (appendix app:gapmap) figures drawn from the audit outputs.

  figures/gap_mountings   r4: spline maps of ln q_u90 of chamber A in its two usable mountings (25 June, 27+28 June).
                          22 June is left out: its drift time is censored by the depth grid of the fit (results/q_RESULT.md).
  figures/gap_driftlines  e : sideways displacement delta against height z above the mesh, five slices, X and Y views
  figures/gap_tiltpred    r1: 25 June tilt, measured / predicted from the mesh shape and the drift-time map / difference

Each writes <name>.{pdf,png,csv} into ./figures/ (next to this script). <work> is the audit work directory
(/media/ucla/work/claude-jobs/e1a7a26e/gap). No analysis is changed: the r4 maps are recomputed with r4's own functions
(the combined 27+28 June map fits both runs' covariate-corrected ln q_u90 in one spline; cached in <work>/audit/paperfig/).
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402

WORK = sys.argv[1] if len(sys.argv) > 1 else '/media/ucla/work/claude-jobs/e1a7a26e/gap'
AUD = f'{WORK}/audit'
OUT = os.path.join(HERE, 'figures')
CACHE = f'{AUD}/paperfig'
os.makedirs(CACHE, exist_ok=True)
CMAP = 'RdBu_r'


# ---------------------------------------------------------------- figure 1
def mountings():
    import r4_lib as L
    import r1_lib as R1
    import r4_06_check_template as T6
    f = f'{CACHE}/mountings.npz'
    if os.path.exists(f):
        z = np.load(f)
        maps = {k: z[k] for k in ('jun22', 'jun25', 'jun27_28', 'jun27', 'jun28')}
    else:
        rel22 = 'mx17_det2_det3_overnight_6-22-26/long_run'
        d22 = L.load_events(rel22); d25 = L.load_events(L.R625)
        d1 = L.load_events(L.GOLD1); d2 = L.load_events(L.GOLD2)
        maps = {'jun22': T6.spline_map(d22)[0], 'jun25': T6.spline_map(d25)[0],
                'jun27': T6.spline_map(d1)[0], 'jun28': T6.spline_map(d2)[0]}
        U, W, Y = [], [], []
        for d in (d1, d2):                      # both golden runs in one fit, each corrected with its own covariates
            y, _ = T6.corrected(d)
            U.append(d.U.values); W.append(d.W.values); Y.append(y - y.mean())
        U, W, Y = map(np.concatenate, (U, W, Y))
        spl = R1.Spl(T6.K)
        co, _ = spl.fit(U, W, Y, T6.LAM, B=spl.basis(U, W))
        maps['jun27_28'] = (spl.basis(T6.UU.ravel(), T6.WW.ravel()) @ co).reshape(T6.UU.shape)
        np.savez(f, **maps)
    gx = 200 + 100 * T6.GRID_U
    gy = 200 + 100 * T6.GRID_W
    P = {k: 100 * (m - m.mean()) for k, m in maps.items()}
    # reproduction check against the exploratory convention: the combined map follows the mean of the two single runs
    c = np.corrcoef(P['jun27_28'].ravel(), 0.5 * (P['jun27'] + P['jun28']).ravel())[0, 1]
    print('combined 27+28 vs mean of the two single-run maps: r = %.3f' % c)
    ps.use()
    fig, axs = ps.fig_for('full', height=3.1, ncols=2, sharey=True)
    titles = ['25 June, drift 500 V', '27 and 28 June, drift 1000 V']
    keys = ['jun25', 'jun27_28']
    lev = np.arange(-10, 10.1, 2.5)
    rows = []
    for ax, k, t in zip(axs, keys, titles):
        m = P[k]
        im = ax.pcolormesh(gx, gy, m.T, cmap=CMAP, vmin=-10, vmax=10, shading='auto', rasterized=True)
        ax.contour(gx, gy, m.T, levels=lev[lev != 0], colors='k', linewidths=0.3, alpha=0.6)
        ax.set_aspect('equal'); ax.grid(False)
        ax.set_title(t, fontsize=ps.BASE_PT * 0.9, loc='center')
        ax.set_xlabel('x [mm]')
        ax.set_xticks([50, 200, 350])
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        rows.append(pd.DataFrame(dict(mounting=k, x_mm=X.ravel(), y_mm=Y.ravel(), rel_drift_time_pct=m.ravel())))
    axs[0].set_ylabel('y [mm]')
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02, ticks=np.arange(-10, 10.1, 5))
    cb.set_label('relative drift time [%]')
    ps.panel_labels(axs, loc='inside upper left', color='k')
    ps.save(fig, f'{OUT}/gap_mountings', data=pd.concat(rows))


# ---------------------------------------------------------------- figure 2
def driftlines():
    S = pickle.load(open(f'{AUD}/e/slices_default.pkl', 'rb'))
    ps.use()
    fig, ax = plt.subplots(2, 2, figsize=(6.1, 4.6), sharex=True, layout='constrained')
    cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
    zz = np.linspace(0, 25, 100)
    rows = []
    for j, (v, axis, lab) in enumerate((('x', 'U', 'x'), ('y', 'W', 'y'))):
        res = S[(v, axis, 'both')]
        for i, r in enumerate(res):
            sel = r['sel']; z = r['z']; dl = r['delta']; a, b, q = r['par']
            pos = 100 * r['coord']                # mm from the chamber centre
            col = cols[i]
            kw = dict(color=col, label=f'{pos:+.0f}')
            ax[0, j].errorbar(z[sel], (dl - a)[sel], yerr=r['ed'][sel], fmt='o', ms=2.8, lw=0.8, **kw)
            ax[0, j].plot(zz, b * zz + q * zz ** 2, '-', color=col, lw=1)
            ax[1, j].errorbar(z[sel], (dl - a - b * z)[sel], yerr=r['ed'][sel], fmt='o', ms=2.8, lw=0.8, color=col)
            ax[1, j].plot(zz, q * zz ** 2, '-', color=col, lw=1)
            for zk, dk, ek in zip(z[sel], dl[sel], r['ed'][sel]):
                rows.append(dict(view=v.upper(), slice_axis=lab, slice_mm_from_centre=pos, z_mm=zk,
                                 delta_minus_const_mm=dk - a, delta_minus_const_minus_linear_mm=dk - a - b * zk,
                                 error_mm=ek, fit_delta_minus_const_mm=b * zk + q * zk ** 2,
                                 fit_curvature_mm=q * zk ** 2))
        ax[0, j].set_title(f'{v.upper()} view, slices along ${lab}$', loc='center', fontsize=ps.BASE_PT)
        h, l = ax[0, j].get_legend_handles_labels()
        ax[1, j].legend(h, l, title=f'${lab}$ from center [mm]', title_fontsize=ps.BASE_PT * 0.8,
                        fontsize=ps.BASE_PT * 0.75, ncol=2, loc='upper left', labelspacing=0.25, handlelength=1.0)
        ax[1, j].set_xlabel('height above the mesh $z$ [mm]')
        ax[1, j].axhline(0, color=ps.MUTED, lw=0.5)
    ax[0, 0].set_ylabel('displacement $\\delta$ [mm]\n(constant removed)')
    ax[1, 0].set_ylabel('$\\delta$ without the linear term [mm]\n(curvature)')
    ps.panel_labels(ax, loc='inside lower left', color='k')
    ps.save(fig, f'{OUT}/gap_driftlines', data=pd.DataFrame(rows))


# ---------------------------------------------------------------- figure 3
def tiltpred():
    R = pickle.load(open(f'{AUD}/r1/resid.pkl', 'rb'))['r625:perview']
    ps.use()
    fig, axs = plt.subplots(2, 3, figsize=(6.1, 3.7), sharex=True, sharey=True, layout='constrained')
    rows = []
    for r, v in enumerate('xy'):
        d = R[v]; M = d['M']
        ext = [(M['Uc'].min() - 0.2125) * 100 + 200, (M['Uc'].max() + 0.2125) * 100 + 200,
               (M['Wc'].min() - 0.19) * 100 + 200, (M['Wc'].max() + 0.19) * 100 + 200]
        for c, (k, t) in enumerate((('A', 'measured'), ('P', 'predicted'), ('res', 'difference'))):
            ax = axs[r, c]
            im = ax.imshow(d[k].T * 1e3, origin='lower', extent=ext, cmap=CMAP, vmin=-25, vmax=25,
                           interpolation='nearest')
            ax.set_aspect('equal'); ax.grid(False)
            ax.set_title(f'{v.upper()}, {t}', loc='center', fontsize=ps.BASE_PT * 0.9)
            if r == 1:
                ax.set_xlabel('x [mm]')
            if c == 0:
                ax.set_ylabel('y [mm]')
            U, W = 100 * np.asarray(M['Uc']) + 200, 100 * np.asarray(M['Wc']) + 200
            rows.append(pd.DataFrame(dict(view=v.upper(), quantity=t, x_mm=U.ravel(), y_mm=W.ravel(),
                                          tilt_mrad=(d[k] * 1e3).ravel())))
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02, ticks=[-25, -12.5, 0, 12.5, 25], format='%g')
    cb.set_label('tilt, constant removed [mrad]')
    ps.panel_labels(axs, loc='inside upper left', color='k', bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0))
    ps.save(fig, f'{OUT}/gap_tiltpred', data=pd.concat(rows))


if __name__ == '__main__':
    for fn in (mountings, driftlines, tiltpred):
        fn()
        print('done', fn.__name__)
