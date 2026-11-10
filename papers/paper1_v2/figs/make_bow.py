#!/usr/bin/env python3
"""make_bow.py -- Paper I v2: the bow of the readout boards, from data (one full-width figure).

  (a) chamber A: mesh height h(x, y) in the 5 x 5 cells of the final-table cross-check (xcheck.py cell_maps, the
      per-cell Huber slope of the position residual against the reference slope), X- and Y-strip maps combined by
      inverse variance after removing each view's own rigid plane (h0 + gu U + gw W of the 2-D fit).
  (b) chamber A: profiles of h through the centre (tracks within BAND mm of the centre line), 9 bins, same per-bin
      regression, rigid plane removed; lines = the 2-D fit's quadratic, 0.5 k U^2, for each view.
  (c) dish depth at 150 mm, five chambers: pooled value with its combined error (bow_2d_fig.depth_summary) and the
      four single terms (X/Y strips, along x and y).
  (d) closure: measured w0 gradient against v h'' (bow_2d_fig.rf_grad), ten views.
Every number of (c) and (d) and the lines of (b) come from fb_bow_maps/results_final/xcheck_<det>.json (the final
tables, final_tables.py run), through the same functions bow_2d_fig.py uses to write summary_paper.json; the script
checks that it reproduces summary_paper.json.  Also: --edgecheck refits the 2-D surfaces without the tracks within
30 mm of the unpassivated x sides (for the text's fringe argument).

Run from the repo root:  .venv/bin/python papers/paper1_v2/figs/make_bow.py [--edgecheck]
Inputs: rerun/model_audit/followups/fb_bow_maps/results_final/*.json, ~/work_data/claude-jobs/bow_final/det*.parquet
Writes: papers/paper1_v2/figs/out/bow.{pdf,png,json}
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FB = os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fb_bow_maps')
RES = os.path.join(FB, 'results_final')
TAB = os.path.join(os.environ.get('WORK_DATA', os.path.expanduser('~/work_data')), 'claude-jobs', 'bow_final')
OUT = os.path.join(HERE, 'out')
sys.path.insert(0, FB)
sys.path.insert(0, os.path.join(REPO, 'rerun', 'paper1_res'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import fbcommon as F                      # noqa: E402
import xcheck as XC                       # noqa: E402
import bow_2d_fig as B2                   # noqa: E402  (import only: summary() would rewrite summary_paper.json)

ORDER = ('det3', 'det2', 'det6', 'det7', 'det4')          # A..E
SLOT = {'det6': 'bottom'}
EDGES5 = XC.EDGES5
EDGES9 = np.linspace(30, 370, 10)
BAND = 50.0                                               # half-width of the central band of (b), mm
RIM = B2.RIM


def J(n):
    return json.load(open(os.path.join(RES, n)))


def table(det):
    b = json.load(open(os.path.join(TAB, f'{det}.bundle.json')))
    return F.load_path(os.path.join(TAB, f'{det}.parquet'), b, det, slot=SLOT.get(det, 'top'), note='final tables', name=det)


def plane(q, U, W):
    return q['h0'] + q['gu'] * U + q['gw'] * W


def cell_map(T, x):
    """5x5 h cells from xcheck_<det>.json, rigid plane of each view removed, views combined."""
    out = {}
    u, w = T.u.to_numpy(), T.w.to_numpy()
    for p in 'xy':
        c = x[f'cells_{p}']
        H, E = np.array(c['h'], float), np.array(c['h_err'], float)
        m = F.view_mask(T, p)
        Ub = np.full((5, 5), np.nan); Wb = Ub.copy()
        for i in range(5):
            for j in range(5):
                k = m & (u >= EDGES5[i]) & (u < EDGES5[i + 1]) & (w >= EDGES5[j]) & (w < EDGES5[j + 1])
                Ub[i, j] = (u[k].mean() - 200) / 100; Wb[i, j] = (w[k].mean() - 200) / 100
        out[p] = dict(h=H - plane(x[f'pos_{p}'], Ub, Wb), err=E, n=np.array(c['n']))
    wx, wy = 1 / out['x']['err'] ** 2, 1 / out['y']['err'] ** 2
    hc = (wx * out['x']['h'] + wy * out['y']['h']) / (wx + wy)
    ec = 1 / np.sqrt(wx + wy)
    ok = np.isfinite(hc)
    return out, np.where(ok, hc, np.nan), np.where(ok, ec, np.nan)


def profiles(T, x, nboot=30):
    """h in 9 bins along each axis, tracks within BAND of the centre line, rigid plane removed."""
    rng = np.random.default_rng(5)
    u, w = T.u.to_numpy(), T.w.to_numpy()
    res = {}
    for p in 'xy':                                # strip view
        m = F.view_mask(T, p)
        t, r = T[f'{p}_t'].to_numpy(), T[f'{p}_r'].to_numpy()
        q = x[f'pos_{p}']
        for ax_, pos, oth in (('x', u, w), ('y', w, u)):
            rows = []
            for lo, hi in zip(EDGES9[:-1], EDGES9[1:]):
                k = m & (pos >= lo) & (pos < hi) & (np.abs(oth - 200) < BAND)
                n = int(k.sum())
                if n < 50 or (t[k] > 0.05).sum() < 10 or (t[k] < -0.05).sum() < 10:
                    rows.append([0.5 * (lo + hi), np.nan, np.nan, n]); continue
                X = np.c_[np.ones(n), t[k]]
                h = F.huber(X, r[k], 0.5)[1]
                hb = [F.huber(X[s], r[k][s], 0.5)[1] for s in (rng.integers(0, n, n) for _ in range(nboot))]
                U = (u[k].mean() - 200) / 100; W = (w[k].mean() - 200) / 100
                rows.append([float(pos[k].mean()), float(h - plane(q, U, W)), float(np.std(hb)), n])
            res[f'{p}_along_{ax_}'] = np.array(rows, float)
    return res


def depths_and_closure():
    S = {}
    SP = J('summary_paper.json')
    for d in ORDER:
        x = J(f'xcheck_{d}.json')
        k, e = B2.terms(x)
        s = B2.depth_summary(k, e)
        s['rf_grad'] = B2.rf_grad(x)
        # must reproduce the section's numbers (summary_paper.json)
        assert abs(s['depth_mm'] - SP[d]['depth_mm']) < 1e-9 and abs(s['total'] - SP[d]['total']) < 1e-9, d
        for p in 'xy':
            assert abs(s['rf_grad'][p]['r_f'] - SP[d]['rf_grad'][p]['r_f']) < 1e-9, (d, p)
        S[d] = s
    return S


def edgecheck(cut=30.0):
    """2-D position and angle routes refitted without tracks whose mid-gap x is within `cut` mm of an x side."""
    out = {}
    for d in ORDER:
        T = table(d)
        x0 = J(f'xcheck_{d}.json')
        keep = ((T.u > cut) & (T.u < 398.6 - cut)).to_numpy()
        Tk = T[keep].copy(); Tk.attrs = T.attrs
        x = dict(pos_x=XC.pos_route(Tk, 'x'), pos_y=XC.pos_route(Tk, 'y'),
                 ang_x=XC.angle_route(Tk, 'x', T.attrs['v']), ang_y=XC.angle_route(Tk, 'y', T.attrs['v']))
        k, e = B2.terms(x)
        s = B2.depth_summary(k, e)
        rf = B2.rf_grad(x)
        s0 = B2.depth_summary(*B2.terms(x0))
        out[d] = dict(cut_mm=cut, frac_removed=float(1 - keep.mean()), depth_mm=s['depth_mm'], total=s['total'],
                      depth_all_mm=s0['depth_mm'], terms_mm=s['terms_mm'], terms_all_mm=s0['terms_mm'],
                      grad_x=rf['x']['measured'], grad_y=rf['y']['measured'],
                      grad_x_all=10 * x0['ang_x']['cu'], grad_y_all=10 * x0['ang_y']['cw'],
                      rf_x=rf['x']['r_f'], rf_y=rf['y']['r_f'])
        print(d, {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in out[d].items() if not isinstance(v, list)}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--edgecheck', action='store_true')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import paper_style as ps
    ps.use()

    xA = J('xcheck_det3.json')
    TA = table('det3')
    views, hc, ec = cell_map(TA, xA)
    PR = profiles(TA, xA)
    S = depths_and_closure()

    fig = plt.figure(figsize=(6.1, 5.0), layout='constrained')
    top, bot = fig.subfigures(2, 1, height_ratios=[1.0, 0.98], hspace=0.03)
    axa, axb1, axb2 = top.subplots(1, 3, gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    axb2.sharey(axb1)
    axc, axd = bot.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.05, 1]))

    # (a) map
    vmin, vmax = -0.5, 2.0
    cmap = plt.get_cmap('viridis')
    im = axa.pcolormesh(EDGES5, EDGES5, hc.T, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    for i in range(5):
        for j in range(5):
            if np.isfinite(hc[i, j]):
                cx, cy = 0.5 * (EDGES5[i] + EDGES5[i + 1]), 0.5 * (EDGES5[j] + EDGES5[j + 1])
                frac = (hc[i, j] - vmin) / (vmax - vmin)
                v_ = 0.0 if abs(hc[i, j]) < 0.05 else hc[i, j]
                axa.text(cx, cy, f'{v_:.1f}'.replace('-', '\u2212'), ha='center', va='center', fontsize=7,
                         color='white' if frac < 0.6 else ps.INK)
    axa.plot([0, 398.6, 398.6, 0, 0], [0, 0, 398.6, 398.6, 0], color=ps.MUTED, lw=0.6)
    axa.set_xlim(-5, 404); axa.set_ylim(-5, 404); axa.set_aspect('equal')
    axa.set_xticks([0, 100, 200, 300, 400]); axa.set_yticks([0, 100, 200, 300, 400])
    axa.grid(False)
    axa.set_xlabel('x at mid-gap [mm]'); axa.set_ylabel('y at mid-gap [mm]')
    cb = top.colorbar(im, ax=axa, location='top', fraction=0.06, pad=0.02, aspect=20)
    cb.set_label('mesh height [mm]', labelpad=3)
    cb.ax.tick_params(labelsize=7.5)

    # (b) profiles
    uu = np.linspace(10, 390, 120); U = (uu - 200) / 100
    c = ps.color('det3')
    for axb, along, kk in ((axb1, 'x', 'kuu'), (axb2, 'y', 'kww')):
        for p, dx in (('x', -3.0), ('y', 3.0)):
            R = PR[f'{p}_along_{along}']
            g = np.isfinite(R[:, 1])
            axb.errorbar(R[g, 0] + dx, R[g, 1], yerr=R[g, 2], ls='none', marker='o', ms=4, color=c, elinewidth=0.8,
                         mfc=c if p == 'x' else 'white', mec=c, zorder=5)
            k = xA[f'pos_{p}'][kk]
            axb.plot(uu, 0.5 * k * U ** 2, color=ps.MODEL, ls=ps.VIEW_LS[p], lw=1.0, zorder=3)
        axb.axhline(0, color=ps.MUTED, lw=0.5)
        axb.set_xlim(0, 400); axb.set_xticks([0, 100, 200, 300, 400])
        axb.set_xlabel(f'{along} at mid-gap [mm]')
    axb1.set_ylabel('mesh height [mm]')
    axb2.tick_params(labelleft=False)
    axb1.set_ylim(-1.2, 4.3)
    axb1.legend(handles=[Line2D([], [], color=c, marker='o', ls='none', label='X strips'),
                         Line2D([], [], color=c, marker='o', mfc='white', ls='none', label='Y strips')],
                loc='upper center', ncol=1, frameon=False, handletextpad=0.2)
    axb2.legend(handles=[Line2D([], [], color=ps.MODEL, ls='-', label='fit, X'),
                         Line2D([], [], color=ps.MODEL, ls='--', label='fit, Y')],
                loc='upper center', ncol=1, frameon=False, handlelength=2.0)

    # (c) depths
    for i, d in enumerate(ORDER):
        s = S[d]; col = ps.color(d)
        t, te = np.array(s['terms_mm']), np.array(s['terms_err'])
        for j, (off, fill) in enumerate(((-0.30, True), (-0.18, True), (0.18, False), (0.30, False))):
            axc.errorbar(i + off, t[j], yerr=te[j], ls='none', marker='o', ms=2.8, elinewidth=0.6, color=col,
                         mfc=col if fill else 'white', mec=col, alpha=0.9, zorder=3)
        axc.errorbar(i, s['depth_mm'], yerr=s['total'], ls='none', marker=ps.DET_MARKER[d], ms=6.5, color=col,
                     elinewidth=1.4, capsize=2.5, zorder=5, mec=ps.INK if d == 'det3' else col, mew=0.6)
    axc.set_xticks(range(5)); axc.set_xticklabels([ps.NAME[d] for d in ORDER])
    axc.set_xlim(-0.55, 4.55); axc.set_ylim(0, 2.8)
    axc.set_xlabel('chamber'); axc.set_ylabel('dish depth at 150 mm [mm]')
    axc.grid(axis='x', visible=False)
    hc_ = [Line2D([], [], color=ps.INK, marker='o', ms=6, ls='none', label='combined'),
           Line2D([], [], color=ps.INK, marker='o', ms=2.8, ls='none', label='X strips'),
           Line2D([], [], color=ps.INK, marker='o', ms=2.8, mfc='white', ls='none', label='Y strips')]
    axc.legend(handles=hc_, loc='lower center', ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.3)

    # (d) closure
    xs = np.linspace(0, 10, 10)
    axd.plot(xs, xs, '--', color=ps.MODEL, lw=1)
    axd.plot(xs, 0.5 * xs, ':', color=ps.MODEL, lw=1)
    axd.text(7.3, 7.9, '$r_f = 1$', ha='right', va='bottom')
    axd.text(8.0, 3.5, '$r_f = 1/2$', ha='center', va='top')
    for d in ORDER:
        for p in 'xy':
            g = S[d]['rf_grad'][p]
            axd.errorbar(g['predicted'], g['measured'], xerr=g['predicted_err'], yerr=g['measured_err'], ls='none',
                         elinewidth=0.7, ms=5.5 if d == 'det3' else 4.5, **ps.det(d, view=p, line=False, label=None))
    hd = [Line2D([], [], ls='none', color=ps.color(d), marker=ps.DET_MARKER[d], label=ps.NAME[d]) for d in ORDER]
    hd += [Line2D([], [], ls='none', marker='o', color=ps.INK, label='X'),
           Line2D([], [], ls='none', marker='o', color=ps.INK, mfc='white', label='Y')]
    axd.legend(handles=hd, loc='upper left', ncol=2, frameon=False, columnspacing=0.6, handletextpad=0.2,
               borderaxespad=0.2)
    axd.set_xlim(0, 10); axd.set_ylim(0, 10)
    axd.set_xlabel(r'predicted gradient $v\,h^{\prime\prime}$ [µm ns$^{-1}$ m$^{-1}$]')
    axd.set_ylabel(r'measured $w_0$ gradient [µm ns$^{-1}$ m$^{-1}$]')

    for ax, l in ((axa, 'a'), (axb1, 'b'), (axc, 'c'), (axd, 'd')):
        ps.panel_label(ax, l, loc='upper left')
    ps.save(fig, os.path.join(OUT, 'bow'))

    J_ = dict(
        source=dict(xcheck='rerun/model_audit/followups/fb_bow_maps/results_final/xcheck_<det>.json',
                    summary='rerun/model_audit/followups/fb_bow_maps/results_final/summary_paper.json (reproduced exactly)',
                    tables='~/work_data/claude-jobs/bow_final/<det>.parquet + .bundle.json'),
        chamber_names=ps.NAME,
        a_map=dict(chamber='A', edges_mm=EDGES5.tolist(), note='h - (h0 + gu U + gw W) per view, inverse-variance '
                   'mean of X and Y strips; [i][j] = x cell i, y cell j', h_combined_mm=np.round(hc, 4).tolist(),
                   err_combined_mm=np.round(ec, 4).tolist(),
                   h_x_mm=np.round(views['x']['h'], 4).tolist(), h_y_mm=np.round(views['y']['h'], 4).tolist(),
                   n_x=views['x']['n'].tolist(), n_y=views['y']['n'].tolist(), min_tracks_per_cell=60),
        b_profiles=dict(band_mm=BAND, bins_mm=EDGES9.tolist(),
                        series={k: dict(pos_mm=v[:, 0].tolist(), h_mm=v[:, 1].tolist(), err_mm=v[:, 2].tolist(),
                                        n=v[:, 3].astype(int).tolist()) for k, v in PR.items()},
                        fit_k={p: dict(kuu=xA[f'pos_{p}']['kuu'], kuu_err=xA[f'pos_{p}']['kuu_err'],
                                       kww=xA[f'pos_{p}']['kww'], kww_err=xA[f'pos_{p}']['kww_err']) for p in 'xy'},
                        model='h = 0.5 k ((pos - 200 mm)/100 mm)^2'),
        c_depth={ps.NAME[d]: dict(depth_mm=S[d]['depth_mm'], total_err_mm=S[d]['total'], stat_mm=S[d]['stat'],
                                  scatter_sem_mm=S[d]['scatter_sem'],
                                  terms_mm=dict(zip(('X along x', 'X along y', 'Y along x', 'Y along y'), S[d]['terms_mm'])),
                                  terms_err_mm=S[d]['terms_err']) for d in ORDER},
        d_closure={f'{ps.NAME[d]} {p.upper()}': S[d]['rf_grad'][p] for d in ORDER for p in 'xy'},
    )
    if a.edgecheck:
        J_['edgecheck_no_x_side_30mm'] = edgecheck()
    elif os.path.exists(os.path.join(OUT, 'bow.json')):
        old = json.load(open(os.path.join(OUT, 'bow.json')))
        if 'edgecheck_no_x_side_30mm' in old:
            J_['edgecheck_no_x_side_30mm'] = old['edgecheck_no_x_side_30mm']
    json.dump(J_, open(os.path.join(OUT, 'bow.json'), 'w'), indent=1, default=float)
    for d in ORDER:
        print(ps.NAME[d], f"{S[d]['depth_mm']:.2f} +- {S[d]['total']:.2f}",
              ' '.join(f"{p}:rf {S[d]['rf_grad'][p]['r_f']:.2f}" for p in 'xy'))
    print('map\n', np.round(hc, 2), '\nerr\n', np.round(ec, 2))


if __name__ == '__main__':
    main()
