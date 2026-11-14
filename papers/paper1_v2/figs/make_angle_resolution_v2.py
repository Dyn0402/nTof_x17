#!/usr/bin/env python3
"""Paper I v2, figure angle_resolution_v2: the bench angular resolution, three panels.

Chamber A only (2026-11-14, Dylan's feedback note; B and C-E were drawn until then and stay in the json):
(a) theta_fit against theta_ref, X and Y together, 0.5-deg cells, log colour (the 2D of the first version).
(b) residual theta_fit - theta_ref, X and Y, Runs 1 + 2 summed, log scale, s68 marked.
(c) s68 per 2-deg bin of theta_ref, X and Y.

Selection and s68 exactly as rerun/paper1_res/angle_resolution.py: the per_event.npz of mx_june_wft/03_angles
(interior_good, one good reference ray per event, reference recipe chi2 < 1 and NClus = 4), theta = atan(tan),
s68 = 68th percentile of |d - median(d)|, d = theta_fit - theta_ref, NO clip.

Inputs
  A (det3): two-track re-pass, x17rerun/paper1_eff_twotrack/angles/<run>/<subrun>/mx17_3/per_event.npz (Runs 1, 2)
  B (det2): pass 1, x17rerun/paper1_eff/angles/mx17_det2_det3_overnight_6-22-26/long_run/mx17_2/per_event.npz
            (fetched to $WORK_DATA/claude-jobs/v2_angres/B_pass1.npz)
  C, D, E and the B check: rerun/paper1_res/results/rows.csv (pass arm, the arm the paper text quotes)
  A bow-off check: the two-track calib bundles (w0, kw, v) applied to the stored transverse speed w.

    .venv/bin/python papers/paper1_v2/figs/make_angle_resolution_v2.py
Writes papers/paper1_v2/figs/out/angle_resolution_v2.{pdf,png,csv,json}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402

WD = os.environ.get('WORK_DATA', os.path.expanduser('~/work_data'))
TT = os.path.join(WD, 'claude-jobs', 'twotrack')
A_RUNS = {'run1': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V',
          'run2': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check'}
B_NPZ = os.path.join(WD, 'claude-jobs', 'v2_angres', 'B_pass1.npz')
ROWS = os.path.join(REPO, 'rerun', 'paper1_res', 'results', 'rows.csv')
HEAD = {'det2': 'c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2',
        'det6': 'g_det6_long', 'det7': 'g_det7_long', 'det4': 'g_det4'}
OUT = os.path.join(HERE, 'out', 'angle_resolution_v2')
LIM, BW, MIN_N, NBOOT = 20.0, 2.0, 50, 300
RNG = np.random.default_rng(17)


def s68(d):
    d = np.asarray(d, float)
    return float(np.percentile(np.abs(d - np.median(d)), 68))


def s68_clip5(d):
    d = np.asarray(d, float)
    d = d[np.abs(d) <= 5]
    return float(np.percentile(np.abs(d - np.median(d)), 68))


def boot(d):
    d = np.asarray(d, float)
    return float(np.std([s68(RNG.choice(d, len(d))) for _ in range(NBOOT)]))


def deg(t):
    return np.degrees(np.arctan(t))


def load_npz(path, run):
    z = np.load(path)
    return {p: pd.DataFrame(dict(run=run, ref=deg(z[f'{p}_tan_ref']), fit=deg(z[f'{p}_tan_fit']), w=z[f'{p}_w']))
            for p in 'xy'}


def load_A():
    parts = {'x': [], 'y': []}
    bowoff = {}
    for run, rel in A_RUNS.items():
        D = load_npz(os.path.join(TT, 'paper1_eff_twotrack', 'angles', rel, 'mx17_3', 'per_event.npz'), run)
        b = json.load(open(os.path.join(TT, 'Analysis_pass1_twotrack', rel, 'mx17_3', 'wft', 'calib_bundle',
                                        'bundle.json')))
        for p in 'xy':
            d = D[p]
            # bow off: the constant map (1e3 w - w0) / (kw v) of the same bundle (wft.reco.tan_const)
            d['fit_nobow'] = deg((d.w * 1e3 - b['w0'][p]) / (b['kw'][p] * b['v_drift']))
            parts[p].append(d)
        bowoff[run] = dict(bow_source_row=(b.get('bow') or {}).get('source_row'))
    return {p: pd.concat(v, ignore_index=True) for p, v in parts.items()}, bowoff


def binned(d):
    dd = (d.fit - d.ref).to_numpy()
    r = d.ref.to_numpy()
    be = np.arange(-LIM, LIM + 0.01, BW)
    out = []
    for lo, hi in zip(be[:-1], be[1:]):
        k = (r >= lo) & (r < hi)
        n = int(k.sum())
        out.append(dict(lo=float(lo), hi=float(hi), center=float(0.5 * (lo + hi)), n=n,
                        s68=s68(dd[k]) if n >= MIN_N else None, s68_err=boot(dd[k]) if n >= MIN_N else None))
    return out


def summary(d, by_run=True):
    dd = (d.fit - d.ref).to_numpy()
    o = dict(n=int(len(dd)), median=float(np.median(dd)), s68=s68(dd), s68_err=boot(dd),
             s68_clip5=s68_clip5(dd),
             frac_abs_gt3=float(np.mean(np.abs(dd) > 3)), frac_abs_gt5=float(np.mean(np.abs(dd) > 5)),
             frac_abs_gt10=float(np.mean(np.abs(dd) > 10)),
             frac_absdev_gt3=float(np.mean(np.abs(dd - np.median(dd)) > 3)),
             frac_absdev_gt5=float(np.mean(np.abs(dd - np.median(dd)) > 5)),
             ref_range=[float(d.ref.min()), float(d.ref.max())])
    if 'fit_nobow' in d:
        dn = (d.fit_nobow - d.ref).to_numpy()
        o['bow_off'] = dict(s68=s68(dn), s68_clip5=s68_clip5(dn), median=float(np.median(dn)))
    if by_run and d.run.nunique() > 1:
        o['by_run'] = {}
        for run, g in d.groupby('run'):
            x = (g.fit - g.ref).to_numpy()
            e = dict(n=int(len(x)), s68=s68(x), s68_err=boot(x), median=float(np.median(x)), s68_clip5=s68_clip5(x),
                     frac_abs_gt3=float(np.mean(np.abs(x) > 3)), frac_abs_gt5=float(np.mean(np.abs(x) > 5)))
            if 'fit_nobow' in g:
                xn = (g.fit_nobow - g.ref).to_numpy()
                e['bow_off'] = dict(s68=s68(xn), s68_clip5=s68_clip5(xn))
            o['by_run'][run] = e
    return o


def main():
    A, bowsrc = load_A()
    B = load_npz(B_NPZ, 'B')
    rows = pd.read_csv(ROWS)
    res = dict(definition='s68 = 68th percentile of |d - median(d)|, d = theta_fit - theta_ref [deg], no clip; '
                          'interior_good fiducial (common/mx17_fiducial.py); per-event tables of mx_june_wft/03_angles',
               sources=dict(A='two-track re-pass (Analysis_pass1_twotrack), Runs 1+2', B='pass 1 (Analysis_pass1), '
                            '22-23 June long run', CDE='rerun/paper1_res/results/rows.csv, pass arm'),
               bow_block_source=bowsrc, A={}, B={}, chambers={})
    for p in 'xy':
        res['A'][p] = summary(A[p]); res['A'][p]['by_ref_bin'] = binned(A[p])
        res['B'][p] = summary(B[p], by_run=False); res['B'][p]['by_ref_bin'] = binned(B[p])
    # (c): A from the per-event tables (summed), B-E headline rows of rows.csv, both arms recorded
    ch = {'det3': {p: dict(s68=res['A'][p]['s68'], err=res['A'][p]['s68_err'], n=res['A'][p]['n'], arm='pass',
                           row='Runs 1+2 summed') for p in 'xy'}}
    for d, key in HEAD.items():
        ch[d] = {}
        for p in 'xy':
            r = rows[(rows.key == key) & (rows.view == p)].iloc[0]
            ch[d][p] = dict(s68=float(r.ang_s68_pass), err=float(r.ang_s68_err_pass), n=int(r.ang_n_pass), arm='pass',
                            s68_vt=float(r.ang_s68_vt), row=key)
    res['chambers'] = {ps.NAME[d]: v for d, v in ch.items()}
    res['B_check_rows_csv_vs_per_event'] = {p: [ch['det2'][p]['s68'], res['B'][p]['s68']] for p in 'xy'}

    # ------------------------------------------------------------------ figure
    ps.use()
    # 2026-11-14 (Dylan, feedback note): chamber A only; the 2D theta_fit vs theta_ref of the first version is back
    # as (a). B and C-E stay in the json and the text, no longer drawn.
    # 2026-11-14 (second round): two rows, the 2D across the top; (b) linear with a Gaussian fit to the core
    # (the log scale did not say anything the text needs); (c) 0.2-deg ticks so 1.0 deg is a gridline.
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from scipy.optimize import curve_fit
    layout = os.environ.get('ANGRES_LAYOUT', 'top')
    if layout == 'top':
        fig = plt.figure(figsize=(ps.WIDTH['full'], 4.6), layout='constrained')
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
        a0 = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, 0]); a2 = fig.add_subplot(gs[1, 1])
    else:  # 'left': square 2D spanning both rows on the left, (b) and (c) stacked on the right
        fig = plt.figure(figsize=(ps.WIDTH['full'], 4.0), layout='constrained')
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
        a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
    cA = ps.color('det3')
    # (a) theta_fit against theta_ref, X and Y together, 0.5-deg cells, log colour (as rerun/paper1_res/angle_resolution.py)
    e2 = np.arange(-LIM, LIM + 0.01, 0.5)
    H = sum(np.histogram2d(A[p].ref.to_numpy(), A[p].fit.to_numpy(), [e2, e2])[0] for p in 'xy')
    cm = LinearSegmentedColormap.from_list('blues', ['#e8eef7', cA])
    im = a0.pcolormesh(e2, e2, np.ma.masked_equal(H, 0).T, cmap=cm, norm=LogNorm(vmin=1, vmax=H.max()), rasterized=True)
    a0.plot([-LIM, LIM], [-LIM, LIM], ls=(0, (3, 2)), color='#e07b00', lw=0.6, alpha=0.85)
    a0.set_xlim(-LIM, LIM); a0.set_ylim(-LIM, LIM)
    a0.set_xticks(np.arange(-20, 21, 5 if layout == 'top' else 10)); a0.set_yticks([-20, -10, 0, 10, 20])
    a0.set_xlabel(r'$\theta_{\rm ref}$ [deg]'); a0.set_ylabel(r'$\theta_{\rm fit}$ [deg]')
    a0.grid(False)
    cb = fig.colorbar(im, ax=a0, pad=0.01, fraction=0.025 if layout == 'top' else 0.05, aspect=15)
    cb.ax.tick_params(labelsize=7, length=2, pad=1); cb.set_label('tracks per cell', fontsize=7, labelpad=1)
    res['A']['n_beyond_2d_window'] = int(len(A['x']) + len(A['y']) - H.sum())
    # (b) residual, linear, 0.2-deg bins over +-6 deg; a Gaussian fitted to the core |d - median| < 2 s68
    # (counts-weighted least squares), drawn solid over the fit range and dashed beyond it
    GB, GW, GR = 0.2, 2.0, 6.0
    edges = np.arange(-GR, GR + 1e-6, GB)
    cen = 0.5 * (edges[1:] + edges[:-1])
    gauss = lambda x, N, mu, s: N * np.exp(-0.5 * ((x - mu) / s) ** 2)  # noqa: E731
    csv = []
    lines = []
    hmax = 0.0
    for p in 'xy':
        dd = (A[p].fit - A[p].ref).to_numpy()
        h, _ = np.histogram(dd, edges)
        s, m = res['A'][p]['s68'], res['A'][p]['median']
        k = np.abs(cen - m) < GW * s
        P, _ = curve_fit(gauss, cen[k], h[k], p0=[h.max(), m, s], sigma=np.sqrt(np.maximum(h[k], 1)))
        P[2] = abs(P[2])
        chi2 = float(np.sum((h[k] - gauss(cen[k], *P)) ** 2 / np.maximum(h[k], 1)))
        n_g = float(P[0] * P[2] * np.sqrt(2 * np.pi) / GB)
        res['A'][p]['gauss_core'] = dict(window=f'|d - median| < {GW} s68, {GB}-deg bins', mu=float(P[1]),
                                         sigma=float(P[2]), chi2=chi2, ndf=int(k.sum() - 3),
                                         frac_in_gauss=n_g / len(dd), s68_over_sigma=s / float(P[2]))
        kw = ps.det('det3', view=p, line=False)
        kw.pop('label'); kw.pop('zorder')
        a1.plot(cen, h / 1e3, ls='none', ms=2.6, mew=0.7, zorder=3, **kw)
        xf = np.linspace(-GR, GR, 400)
        inn = np.abs(xf - m) <= GW * s
        a1.plot(xf[inn], gauss(xf[inn], *P) / 1e3, color=cA, lw=1.0, ls=ps.VIEW_LS[p], zorder=2)
        a1.plot(xf, gauss(xf, *P) / 1e3, color=cA, lw=0.6, ls=':', alpha=0.7, zorder=1)
        g = res['A'][p]['gauss_core']
        lines.append(f'{p.upper()}: $\\sigma$ = {g["sigma"]:.2f}$^\\circ$, {100 * g["frac_in_gauss"]:.0f} % of tracks')
        hmax = max(hmax, h.max() / 1e3)
        csv += [dict(panel='b', view=p.upper(), x=c, lo=lo, hi=hi, n=int(n), gauss=float(gauss(c, *P)))
                for c, lo, hi, n in zip(cen, edges[:-1], edges[1:], h)]
        far = np.abs(dd) > 15
        res['A'][p]['n_outside_plot_15deg'] = int(far.sum())
        res['A'][p]['frac_outside_plot_15deg'] = float(far.mean())
        res['A'][p]['frac_outside_plot_6deg'] = float(np.mean(np.abs(dd) > GR))
        fa = np.abs(deg(np.tan(np.radians(A[p].fit.to_numpy()))))[np.abs(dd) > 10]
        res['A'][p]['far10_abs_theta_fit_p10_p50_p90'] = [float(v) for v in np.percentile(fa, [10, 50, 90])]
        res['A'][p]['max_abs_residual'] = float(np.abs(dd).max())
    a1.set_xlim(-GR, GR); a1.set_ylim(0, 1.42 * hmax)  # headroom for the fit box
    a1.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    a1.set_xlabel(r'$\theta_{\rm fit}-\theta_{\rm ref}$ [deg]')
    a1.set_ylabel(r'tracks per %.1f$^\circ$ [$10^3$]' % GB)
    a1.text(0.03, 0.97, 'Gaussian fit to the core\n' + '\n'.join(lines), transform=a1.transAxes, ha='left', va='top',
            fontsize=7.5, color=ps.INK, linespacing=1.3)
    # (c) s68 vs theta_ref
    for p in 'xy':
        bb = [b for b in res['A'][p]['by_ref_bin'] if b['s68'] is not None]
        kw = ps.det('det3', view=p)
        kw.pop('label'); kw.pop('zorder')
        kw.update(ms=3.4, elinewidth=0.8, lw=1.2)
        a2.errorbar([b['center'] for b in bb], [b['s68'] for b in bb], yerr=[b['s68_err'] for b in bb], zorder=5, **kw)
        csv += [dict(panel='c', chamber=ps.NAME['det3'], view=p.upper(), x=b['center'], n=b['n'], s68=b['s68'],
                     s68_err=b['s68_err']) for b in bb]
    a2.set_xlim(-LIM - 1, LIM + 1); a2.set_ylim(0.0, 1.5)
    a2.set_xticks([-20, -10, 0, 10, 20]); a2.set_yticks(np.arange(0, 1.41, 0.2))
    a2.set_xlabel(r'$\theta_{\rm ref}$ [deg]'); a2.set_ylabel(r'angular resolution $s_{68}$ [deg]')
    ps.panel_labels([a0, a1, a2])
    ps.legend_above(fig, ps.view_handles(color=cA), ncol=2, handlelength=2.2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ps.save(fig, OUT, data=pd.DataFrame(csv))
    # numbers read from other products (not recomputed here), for the text; see the docstrings of the cited files
    res['quoted_elsewhere'] = {
        'reference_plus_MS': dict(best_estimate_deg=0.05, measured_bound_deg=dict(x=0.107, y=0.140),
                                  chamberA_kink_regression_deg=0.055,
                                  src='rerun/model_audit/followups/gb_m3_angle/REPORT.md (sec. 1, 2c, 4)'),
        'drift_scan_A_490V': dict(E_V_cm='167-367', s68_x='0.92-0.97', s68_y='0.87-1.00', n_per_row='~650',
                                  at_100V_cm=[1.96, 2.04], at_33V_cm=[9.12, 6.96],
                                  src='rerun/paper1_res/results/rows.csv (pass 1, pass arm)'),
        'hv_scan_A_1000Vdrift': dict(mesh_V='425-505', s68_x='0.91-1.03', s68_y='0.80-0.96', at_515_525V='1.13-1.31',
                                     src='rerun/paper1_res/results/rows.csv'),
        'hv_scan_B': dict(mesh_V='450-510', s68_x='0.87-1.08', s68_y='0.92-1.19', at_515_525V='1.11-1.40'),
        'gain_slope': 'bench chambers double every 16-19 V (rerun/paper1_ops/README.md)',
        'hits_H0_vs_W': dict(H0=dict(run1=[2.06, 2.68], run2=[2.11, 2.61]), W=dict(run1=[1.04, 1.01], run2=[1.04, 1.01]),
                             footing='W there = bow OFF and s68 over |d|<=5 deg (tiers.py); reproduced here as '
                                     'A bow_off.s68_clip5; not the footing of the plotted 0.99/0.90',
                             src='papers/paper2/hits_daq/results/ana1106_tiers_{sat,p2}_det3.json'),
        'sps_E_25deg': dict(plotted=False,
                            candidates=dict(
                                gb_sps_compare=dict(run='63', gas='Ar/CF4/iso', view='Y (inclined)', s68=[0.96, 0.99],
                                                    s68_mount_scale=[1.04, 1.04], far10=0.20, clip='|d|<=5',
                                                    reco='pre-pass-1 (fe_unify2), angle scale 8 % low, ZS'),
                                hits_daq_W=dict(runs='57, 59', gas='Ar/CO2/iso 95/3/2', view='Y (inclined)',
                                                s68_clip5=[1.02, 1.10], s68_noclip=[1.83, 2.05], frac_in5=[0.77, 0.79],
                                                reco='pass 1; v set from the stage angle (W is the calibration)')),
                            why='clipped at 5 deg with 20 % beyond 10 deg vs 3.4 % on the bench; no single-track '
                                'selection; other gas; not on the bench footing')}
    json.dump(res, open(OUT + '.json', 'w'), indent=1)
    for p in 'xy':
        a = res['A'][p]
        print(p.upper(), 'A n', a['n'], 's68 %.3f +- %.3f' % (a['s68'], a['s68_err']),
              {k: round(v['s68'], 3) for k, v in a['by_run'].items()},
              'tails >3 %.2f%% >5 %.2f%% >10 %.2f%%' % (100 * a['frac_abs_gt3'], 100 * a['frac_abs_gt5'],
                                                       100 * a['frac_abs_gt10']),
              'clip5 %.3f bow-off %.3f (clip5 %.3f)' % (a['s68_clip5'], a['bow_off']['s68'], a['bow_off']['s68_clip5']))
        print('  B', res['B'][p]['n'], 's68 %.3f' % res['B'][p]['s68'],
              'tails >3 %.2f%% >5 %.2f%%' % (100 * res['B'][p]['frac_abs_gt3'], 100 * res['B'][p]['frac_abs_gt5']))
    for d in ps.DETS:
        print(ps.NAME[d], {p: (round(ch[d][p]['s68'], 3), round(ch[d][p].get('s68_vt', np.nan), 3)) for p in 'xy'})


if __name__ == '__main__':
    main()
