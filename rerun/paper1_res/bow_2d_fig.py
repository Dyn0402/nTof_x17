#!/usr/bin/env python3
"""bow_2d_fig.py summary | plot -- the Paper I bow numbers and figures from the SAME estimator as the text:
the two-dimensional event-level fit of fb_bow_maps/xcheck.py, run on the final tables by fb_bow_maps/final_tables.py
(results_final/xcheck_<row>.json, summary_final.json, gate_*.json).

summary  -> results_final/summary_paper.json : per chamber the depth at 150 mm with its uncertainty, the four terms,
         r_f and the gradients with errors, the 19-run stability, the column-duration route, the correction gain.
plot     -> figures/bow_profiles_2d.{pdf,png,csv}, figures/bow_closure_2d.{pdf,png,csv}   (the paper's bow_profiles
         and bow_closure).  The 1-D figures of bow_pass1.py (bow_profiles.pdf, bow_closure.pdf) stay for comparison.
Use --results to point at another results_final (default: fb_bow_maps/results_final).

Uncertainty recipe (depth at 150 mm, d = 0.5 k (1.5)^2, k = curvature in mm per (100 mm)^2).  A chamber has four
curvature terms: kuu and kww of the X-strip fit and of the Y-strip fit.  Their pooled value is the inverse-variance
mean d_bar with the bootstrap error sigma_stat = (sum 1/e_i^2)^(-1/2).  The terms scatter far more than their
bootstrap errors (chi2/3 = 2-8 at the golden rows), so the pooled value carries a second error, the standard error of
the mean of the four terms, sigma_scat = s_4 / 2 (s_4 = sample standard deviation of the four d_i).  The four terms
are not independent draws of one number (they share the telescope, the board, the tilt), so this is a floor, not a
guarantee, and a common bias of the reference (bounded separately in the text) is not in it.
sigma_tot = sqrt(sigma_stat^2 + sigma_scat^2).  The single-term uncertainty is s_4 itself.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RES0 = os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fb_bow_maps', 'results_final')
FIGD = os.path.join(HERE, 'figures')
ORDER = ('det3', 'det2', 'det6', 'det7', 'det4')
NAME = {'det3': 'A', 'det2': 'B', 'det6': 'C', 'det7': 'D', 'det4': 'E'}
RIM = 1.5                  # 150 mm in units of 100 mm
# the 19-run consistency set: (row, chamber)
CH = {'det3': 'det3', 'det3_day625': 'det3', 'det3_d700': 'det3', 'det3_d500': 'det3', 'det3_p2': 'det3',
      'det3_o23': 'det3', 'det2': 'det2', 'det2_o22long': 'det2', 'det2_o17': 'det2', 'det6': 'det6',
      'det6_longer': 'det6', 'det6_hv505': 'det6', 'det6_hv530': 'det6', 'det7': 'det7', 'det7_longer': 'det7',
      'det7_hv480': 'det7', 'det7_hv505': 'det7', 'det4': 'det4', 'det4_o23': 'det4'}
EDGES5 = np.linspace(25, 375, 6)


def X(res, n):
    return json.load(open(os.path.join(res, f'xcheck_{n}.json')))


def terms(x):
    px, py = x['pos_x'], x['pos_y']
    k = np.array([px['kuu'], px['kww'], py['kuu'], py['kww']])
    e = np.array([px['kuu_err'], px['kww_err'], py['kuu_err'], py['kww_err']])
    return k, e


def depth_summary(k, e):
    """Pooled depth at 150 mm and its two errors (see the module docstring)."""
    f = 0.5 * RIM ** 2
    d, de = f * k, f * e
    w = 1 / de ** 2
    dbar = float(np.sum(w * d) / np.sum(w))
    st = float(1 / np.sqrt(np.sum(w)))
    s4 = float(np.std(d, ddof=1))
    sc = s4 / 2
    return dict(depth_mm=dbar, stat=st, scatter_sem=sc, total=float(np.hypot(st, sc)), s4=s4,
                chi2_3=float(np.sum(w * (d - dbar) ** 2) / 3), terms_mm=d.tolist(), terms_err=de.tolist(),
                curvature_per_mm=dbar / (0.5 * RIM ** 2) * 1e-4, curvature_err_per_mm=float(np.hypot(st, sc) / (0.5 * RIM ** 2) * 1e-4))


def rf_grad(x):
    """r_f and the drift-tilt gradient (um/ns per m) of both views, with bootstrap errors; predicted gradient v h''."""
    out = {}
    for p, cu, kk in (('x', 'cu', 'kuu'), ('y', 'cw', 'kww')):
        a, q = x[f'ang_{p}'], x[f'pos_{p}']
        B, k = a['B'], q[kk]
        meas, meas_e = 10 * a[cu], 10 * a[cu + '_err']
        pred = 0.1 * B * k
        pred_e = abs(pred) * np.hypot(a['B_err'] / B, q[kk + '_err'] / k)
        rf = meas / pred
        # r_f = cu * 100 / (B k): B enters once
        rf_e = abs(rf) * np.sqrt((a[cu + '_err'] / a[cu]) ** 2 + (a['B_err'] / B) ** 2 + (q[kk + '_err'] / k) ** 2)
        out[p] = dict(measured=meas, measured_err=meas_e, predicted=pred, predicted_err=pred_e, r_f=rf, r_f_err=rf_e)
    return out


def column_route(x):
    """Quadratic surface through the 5x5 cells of column duration; curvature -> equivalent depth v * dT at 150 mm
    (positive = longer column at the centre = deeper gap, the sign a dish predicts)."""
    cen = 0.5 * (EDGES5[:-1] + EDGES5[1:])
    U, W = np.meshgrid((cen - 200) / 100, (cen - 200) / 100, indexing='ij')
    v = x['v']
    out = {}
    for p in 'xy':
        for k in ('uend', 'u90m50'):
            M = np.array(x['column'][p][k], float); E = np.array(x['column'][p][k + '_err'], float)
            g = np.isfinite(M) & np.isfinite(E) & (E > 0)
            S = np.c_[np.ones(g.sum()), U[g], W[g], .5 * U[g] ** 2, .5 * W[g] ** 2, U[g] * W[g]]
            sw = 1 / E[g]
            A = S * sw[:, None]
            beta, *_ = np.linalg.lstsq(A, M[g] * sw, rcond=None)
            r = (M[g] - S @ beta) / E[g]; dof = g.sum() - 6
            c2 = float((r ** 2).sum() / dof)
            cov = np.linalg.inv(A.T @ A) * max(1.0, c2)
            f = -0.5 * RIM ** 2 * v * 1e-3        # k in ns/(100 mm)^2 -> mm
            out[f'{p}_{k}'] = dict(equiv_depth_u=f * beta[3], equiv_depth_w=f * beta[4],
                                   err_u=abs(f) * np.sqrt(cov[3, 3]), err_w=abs(f) * np.sqrt(cov[4, 4]), chi2_dof=c2)
    return out


def summary(res):
    S = {}
    for n in ORDER:
        x = X(res, n)
        k, e = terms(x)
        d = depth_summary(k, e)
        d['views'] = dict(x=dict(depth=[0.5 * RIM ** 2 * k[0], 0.5 * RIM ** 2 * k[1]]),
                          y=dict(depth=[0.5 * RIM ** 2 * k[2], 0.5 * RIM ** 2 * k[3]]))
        d['rf_grad'] = rf_grad(x)
        d['column'] = column_route(x)
        d['v'] = x['v']
        S[n] = d
    # per-view ranges (term level, over chambers)
    xs = [S[n]['views']['x']['depth'] for n in ORDER]; ys = [S[n]['views']['y']['depth'] for n in ORDER]
    xv = [np.mean(v) for v in xs]; yv = [np.mean(v) for v in ys]
    S['_view_means_mm'] = dict(x=dict(zip(ORDER, xv)), y=dict(zip(ORDER, yv)))
    alld = np.array([S[n]['terms_mm'] for n in ORDER])
    S['_along'] = dict(u_mean=dict(zip(ORDER, [float(np.mean([t[0], t[2]])) for t in alld])),
                       w_mean=dict(zip(ORDER, [float(np.mean([t[1], t[3]])) for t in alld])),
                       w_minus_u=dict(zip(ORDER, [float(np.mean([t[1], t[3]]) - np.mean([t[0], t[2]])) for t in alld])))
    S['_ranges'] = dict(
        depth_total=[min(S[n]['depth_mm'] for n in ORDER), max(S[n]['depth_mm'] for n in ORDER)],
        s4=[min(S[n]['s4'] for n in ORDER), max(S[n]['s4'] for n in ORDER)],
        scatter_sem=[min(S[n]['scatter_sem'] for n in ORDER), max(S[n]['scatter_sem'] for n in ORDER)],
        curvature_per_mm=[min(S[n]['curvature_per_mm'] for n in ORDER), max(S[n]['curvature_per_mm'] for n in ORDER)],
        grad_x=[min(S[n]['rf_grad']['x']['measured'] for n in ORDER), max(S[n]['rf_grad']['x']['measured'] for n in ORDER)],
        grad_y=[min(S[n]['rf_grad']['y']['measured'] for n in ORDER), max(S[n]['rf_grad']['y']['measured'] for n in ORDER)],
        grad_err=[min(S[n]['rf_grad'][p]['measured_err'] for n in ORDER for p in 'xy'),
                  max(S[n]['rf_grad'][p]['measured_err'] for n in ORDER for p in 'xy')],
        rf=[min(S[n]['rf_grad'][p]['r_f'] for n in ORDER for p in 'xy'), max(S[n]['rf_grad'][p]['r_f'] for n in ORDER for p in 'xy')],
        rf_x=[min(S[n]['rf_grad']['x']['r_f'] for n in ORDER), max(S[n]['rf_grad']['x']['r_f'] for n in ORDER)],
        rf_err=[min(S[n]['rf_grad'][p]['r_f_err'] for n in ORDER for p in 'xy'),
                max(S[n]['rf_grad'][p]['r_f_err'] for n in ORDER for p in 'xy')],
        y_over_x=[min(S[n]['rf_grad']['y']['measured'] / S[n]['rf_grad']['x']['measured'] for n in ORDER),
                  max(S[n]['rf_grad']['y']['measured'] / S[n]['rf_grad']['x']['measured'] for n in ORDER)])
    # stability: every row against the error-weighted mean of its chamber (per-run error: the same recipe, as in
    # summary_final.json = bootstrap inflated by the four-term scatter)
    SF = json.load(open(os.path.join(res, 'summary_final.json')))
    runs = {}
    for r, c in CH.items():
        runs[r] = dict(chamber=c, depth=SF[r]['depth_mm'], err=SF[r]['depth_err'])
    st = {}
    zs, zl = [], []
    for c in ORDER:
        rs = [r for r in runs if runs[r]['chamber'] == c]
        d = np.array([runs[r]['depth'] for r in rs]); e = np.array([runs[r]['err'] for r in rs]); w = 1 / e ** 2
        m = np.sum(w * d) / np.sum(w); me = 1 / np.sqrt(np.sum(w))
        for i, r in enumerate(rs):
            z = (d[i] - m) / e[i]
            o = np.arange(len(rs)) != i
            if o.sum():
                mo = np.sum(w[o] * d[o]) / np.sum(w[o]); eo = np.hypot(e[i], 1 / np.sqrt(np.sum(w[o])))
                zl.append((r, (d[i] - mo) / eo))
            zs.append((r, z))
        st[c] = dict(mean=float(m), mean_err=float(me), runs=rs)
    good = [r for r in runs if runs[r]['err'] < 0.25]
    fr = []
    for r in good:
        c = runs[r]['chamber']; fr.append(runs[r]['depth'] / st[c]['mean'] - 1)
    S['_stability'] = dict(chamber_means=st, n_runs=len(runs), max_abs_z=float(max(abs(z) for _, z in zs)),
                           n_within_2sigma=int(sum(abs(z) < 2 for _, z in zs)),
                           n_loo_within_2sigma=int(sum(abs(z) < 2 for _, z in zl)), n_loo=len(zl),
                           n_err_below_0p25=len(good), rms_frac_err_below_0p25=float(np.sqrt(np.mean(np.square(fr)))),
                           run_depths={r: runs[r] for r in runs})
    # correction gain
    G = json.load(open(os.path.join(res, 'gate_bow.json')))
    gain = {a: [1 - G[d][p][a]['s68_deg'] / G[d][p]['const']['s68_deg'] for d in G for p in 'xy'] for a in ('map1d', 'bow2d', 'phys')}
    gy = {a: [1 - G[d]['y'][a]['s68_deg'] / G[d]['y']['const']['s68_deg'] for d in G] for a in gain}
    pos = [1 - G[d][p]['pos_s68_mm']['height'] / G[d][p]['pos_s68_mm']['align'] for d in G for p in 'xy']
    T = json.load(open(os.path.join(res, 'gate_transfer.json')))
    skip = ('det2_o17', 'det3_o23')
    rec = [(k, p, (v[p]['const']['s68_deg'] - v[p]['static_plus_tilt']['s68_deg']) /
            (v[p]['const']['s68_deg'] - v[p]['refit_full']['s68_deg'])) for k, v in T.items() for p in 'xy']
    rec_ok = [r[2] for r in rec if not any(k in r[0] for k in skip)]
    S['_correction'] = dict(
        bow2d_median=float(np.median(gain['bow2d'])), bow2d_range=[min(gain['bow2d']), max(gain['bow2d'])],
        bow2d_y_range=[min(gy['bow2d']), max(gy['bow2d'])], phys_median=float(np.median(gain['phys'])),
        pos_range=[min(pos), max(pos)], transfer_median=float(np.median(rec_ok)), transfer_n=len(rec_ok),
        transfer_n_all=len(rec),
        c_bias_x=[G['det6']['x']['const']['bias_0.3'], G['det6']['x']['bow2d']['bias_0.3']],
        c_bias_y=[G['det6']['y']['const']['bias_0.3'], G['det6']['y']['bow2d']['bias_0.3']])
    S['_recipe'] = ('depth = 0.5 k 1.5^2 per term (4 terms: X and Y view, kuu and kww); pooled = inverse-variance mean '
                    'of the bootstrap errors; sigma_stat = (sum 1/e^2)^-1/2; sigma_scat = std(4 terms, ddof=1)/2; '
                    'sigma_tot = hypot; s4 = single-term scatter. r_f errors: bootstrap of cu, B, k in quadrature.')
    json.dump(S, open(os.path.join(res, 'summary_paper.json'), 'w'), indent=1, default=float)
    return S


def plot(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, os.path.join(REPO, 'papers'))
    import paper_style as ps
    ps.use()
    S = summary(res)
    os.makedirs(FIGD, exist_ok=True)
    xx = np.linspace(0, 400, 81)
    u = (xx - 200) / 100
    f1, axs = ps.fig_for('full', height=4.3, nrows=2, ncols=3, sharex=True, sharey=True)
    rows = []
    for ax, d in zip(axs.ravel(), ORDER):
        x = X(res, d)
        k, e = terms(x)
        s = S[d]
        # the four fitted terms: height of the quadratic surface along each axis through the centre, minus the
        # rigid plane (h0, gu, gw); X strips solid, Y strips dashed; along x thick, along y thin
        for (lab, kk, ls, lw) in (('X strips, along x', k[0], '-', 1.5), ('X strips, along y', k[1], '-', 0.8),
                                  ('Y strips, along x', k[2], '--', 1.5), ('Y strips, along y', k[3], '--', 0.8)):
            h = 0.5 * kk * u ** 2
            ax.plot(xx, h, color=ps.color(d), ls=ls, lw=lw, alpha=0.75, zorder=3)
            rows += [dict(det=d, term=lab, pos_mm=a, h_minus_centre_mm=b) for a, b in zip(xx, h)]
        hp = s['depth_mm'] * (u / RIM) ** 2
        lo = (s['depth_mm'] - s['total']) * (u / RIM) ** 2
        hi = (s['depth_mm'] + s['total']) * (u / RIM) ** 2
        ax.fill_between(xx, lo, hi, color=ps.color(d), alpha=0.22, lw=0, zorder=1)
        ax.plot(xx, hp, color=ps.INK, lw=1.0, ls=':', zorder=4)
        ax.axhline(0, color=ps.MUTED, lw=0.5)
        ax.text(0.5, 0.93, f'{NAME[d]}   {s["depth_mm"]:.2f} ± {s["total"]:.2f} mm', transform=ax.transAxes,
                ha='center', va='top', fontsize=ps.BASE_PT * 0.9, color=ps.INK)
        ax.set_xlim(0, 400); ax.set_xticks([0, 100, 200, 300, 400])
        rows += [dict(det=d, term='pooled', pos_mm=a, h_minus_centre_mm=b, lo=l_, hi=h_) for a, b, l_, h_ in zip(xx, hp, lo, hi)]
    ax = axs.ravel()[5]
    ax.axis('off')
    hs = [Line2D([], [], color=ps.INK, ls='-', lw=1.5, label='X strips, along $x$'),
          Line2D([], [], color=ps.INK, ls='-', lw=0.8, label='X strips, along $y$'),
          Line2D([], [], color=ps.INK, ls='--', lw=1.5, label='Y strips, along $x$'),
          Line2D([], [], color=ps.INK, ls='--', lw=0.8, label='Y strips, along $y$'),
          Line2D([], [], color=ps.INK, ls=':', lw=1.0, label='pooled dish'),
          plt.Rectangle((0, 0), 1, 1, fc='0.75', ec='none', label='pooled $\\pm$ total error')]
    ax.legend(handles=hs, loc='center', frameon=False, fontsize=ps.BASE_PT * 0.9)
    for ax in list(axs[1, :2]) + [axs[0, 2]]:
        ax.set_xlabel('position along the axis [mm]')
    axs[0, 2].tick_params(labelbottom=True)
    for ax in axs[:, 0]:
        ax.set_ylabel(r'mesh height $h - h_{\rm centre}$ [mm]')
    axs[0, 0].set_ylim(-0.2, 3.8)
    ps.save(f1, os.path.join(FIGD, 'bow_profiles_2d'), data=pd.DataFrame(rows))

    f2, ax = ps.fig_for('half', height=3.0)
    xs = np.linspace(0, 10, 10)
    ax.plot(xs, xs, '--', color=ps.MODEL, lw=1)
    ax.plot(xs, 0.5 * xs, ':', color=ps.MODEL, lw=1)
    ax.text(6.3, 6.9, '$r_f = 1$', ha='right', va='bottom', fontsize=ps.BASE_PT * 0.85)
    ax.text(9.0, 4.1, '$r_f = 1/2$', ha='center', va='top', fontsize=ps.BASE_PT * 0.85)
    rows = []
    for d in ORDER:
        for p in 'xy':
            g = S[d]['rf_grad'][p]
            ax.errorbar(g['predicted'], g['measured'], xerr=g['predicted_err'], yerr=g['measured_err'], ls='none',
                        elinewidth=0.7, capsize=0, ms=5.5 if d == 'det3' else 4.5,
                        **ps.det(d, view=p, line=False, label=d if p == 'x' else None))
            rows.append(dict(det=d, view=p.upper(), **g))
    h = ps.det_handles(ORDER, line=False, ls='none')
    h += [Line2D([], [], ls='none', marker='o', color=ps.INK, label='X'),
          Line2D([], [], ls='none', marker='o', color=ps.INK, mfc='white', label='Y')]
    h = [h[i] for i in (0, 4, 1, 5, 2, 6, 3)]
    ps.legend_above(ax, h, ncol=4, columnspacing=0.8, handletextpad=0.2)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xlabel(r'predicted gradient [µm ns$^{-1}$ m$^{-1}$]')
    ax.set_ylabel(r'measured gradient [µm ns$^{-1}$ m$^{-1}$]')
    ps.save(f2, os.path.join(FIGD, 'bow_closure_2d'), data=pd.DataFrame(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('summary', 'plot'))
    ap.add_argument('--results', default=RES0)
    a = ap.parse_args()
    if a.mode == 'summary':
        S = summary(a.results)
        print(json.dumps({k: v for k, v in S.items() if k.startswith('_') and k != '_stability'}, indent=1, default=float))
        print({k: round(v['depth_mm'], 2) for k, v in S.items() if not k.startswith('_')})
    else:
        plot(a.results)
