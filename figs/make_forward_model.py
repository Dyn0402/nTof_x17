#!/usr/bin/env python3
"""Paper IIa, figure forward_model (section 4): the production forward model on one real plane.

The model is the physical kernel of wft.model.build_matrix_rc (no fitted copy amplitudes): charge in 60 ns depth bins
along a straight track (p0, w, t0); each bin lands as a Gaussian of variance sigma0^2 + Dp^2 u (+ the bin's own width);
on the view along the resistive strips (Y) the landed charge keeps spreading as an RC line, sigma^2 += 2 D_RC s; each
strip's share is seen through the single-electron pulse response; the sum over bins is compared with the samples.

The plane is the Y view of the event of figure event_waveforms (same rule, read from out/event_waveforms.json), so
the two figures show the same track.  Everything drawn is evaluated with the pass's bundle and the stored fit.

(a) the fitted track and charge column: landing position against drift time, marker area = fitted bin charge
(b) lateral charge profile of one depth bin at landing, mesh end and cathode end, and after 300 ns of RC spreading
(c) the pulse response (template)
(d) seven strips: samples, the model, and the model's contributions from the mesh-end third and cathode-end third of
    the column: one strip receives charge from several depths.

    .venv/bin/python papers/paper2/figs/make_forward_model.py     (after make_event_waveforms.py)
Writes papers/paper2/figs/out/forward_model.{pdf,png,json}.
"""
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'papers'), REPO]
import paper_style as ps  # noqa: E402

WD = os.path.join(os.environ.get('WORK_DATA', os.path.expanduser('~/work_data')), 'paper2_hits')
OUT = os.path.join(HERE, 'out')
VIEW = 'y'
KEY = {'Run 1': 'sat_det3', 'Run 2': 'p2_det3'}
C_MESH, C_CATH, C_MOD, C_RC = '#D55E00', '#0072B2', '#000000', '#1b9e77'
L_GAP = 28.08


def main():
    from wft import model as wm
    from wft.calib import CalibrationBundle
    ev = json.load(open(os.path.join(OUT, 'event_waveforms.json')))
    cal = CalibrationBundle.load(os.path.join(WD, 'bundles', 'det3', 'calib_bundle'))
    wm.use_calibration(cal)
    hy = cal.hyper
    p = None
    for f in sorted(glob.glob(f"{WD}/windows/{KEY[ev['run']]}_p*.pkl")):
        for q in pickle.load(open(f, 'rb'))['planes']:
            if q['event_id'] == ev['event_id'] and q['view'] == VIEW:
                p = q
    pos, g = p['pos'].astype(float), p['gain'].astype(float)
    A = wm.build_matrix(VIEW, pos, p['p0'], p['w'], p['t0'], hy)            # (n_strip * NSAMP, K)
    qk = p['q'].astype(float)
    n, K = len(pos), wm.K
    comp = (A * qk[None, :]).reshape(n, wm.NSAMP, K) * g[:, None, None]
    M = comp.sum(axis=2)
    closure = float(np.max(np.abs(M - p['M'])) / np.max(p['M']))
    v = cal.v_drift / 1000.0
    kmax = int(np.ceil(L_GAP / v / wm.DT))                                  # bins inside the gap
    lo, hi = slice(0, kmax // 3), slice(kmax - kmax // 3, kmax)
    W = p['W'].astype(float)
    ts = wm.TS
    ps.use()
    fig = plt.figure(figsize=(6.1, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.42)
    g1 = gs[0].subgridspec(1, 3, wspace=0.50, width_ratios=[1.0, 1.1, 1.0])
    # (a) track and charge column
    ax = fig.add_subplot(g1[0])
    x0 = pos[np.argmax(W.max(axis=1))]
    for xs in pos:
        ax.axvline(xs - x0 - 0.39, c=ps.GRID, lw=0.6, zorder=0)
    u = wm.UK
    k = qk > 0.01 * qk.max()
    ax.plot(p['p0'] + p['w'] * np.array([0, L_GAP / v]) - x0, [0, L_GAP / v], '-', c=C_MOD, lw=1.0, zorder=2)
    ax.scatter(p['p0'] + p['w'] * u[k] - x0, u[k], s=90 * qk[k] / qk.max(), c=np.where(u[k] < u[kmax // 3], C_MESH,
               np.where(u[k] >= u[kmax - kmax // 3], C_CATH, '#888888')), zorder=3, lw=0)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-60, L_GAP / v + 150)
    ax.grid(False)
    ax.set_xlabel('position [mm]'); ax.set_ylabel('drift time $u$ [ns]')
    ax.text(0.96, 0.03, 'mesh', transform=ax.transAxes, fontsize=7.5, color=ps.MUTED, va='bottom', ha='right')
    ax.text(0.96, 0.97, 'cathode', transform=ax.transAxes, fontsize=7.5, color=ps.MUTED, va='top', ha='right')
    ax.annotate('', xy=(0.80, 0.22), xytext=(0.80, 0.52), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=ps.MUTED, lw=0.8))
    ax.text(0.80, 0.54, 'drift', transform=ax.transAxes, fontsize=7.5, color=ps.MUTED, ha='center', va='bottom')
    ax.text(0.04, 0.97, '(a)', transform=ax.transAxes, fontweight='bold', va='top')
    # (b) lateral profile of one bin
    ax = fig.add_subplot(g1[1])
    xx = np.linspace(-2.6, 2.6, 521)
    s0 = float(hy.get(f'sigma_p0_{VIEW}', hy['sigma_p0'])); Dp = hy['Dp']; D = float(hy.get(f'rc_D_{VIEW}', 0.0))
    half2 = (0.5 * abs(p['w']) * wm.DT) ** 2 / 3.0
    gau = lambda var: np.exp(-0.5 * xx ** 2 / var) / np.sqrt(2 * np.pi * var)
    u_m, u_c = u[0], u[kmax - 1]
    var_m, var_c = s0 ** 2 + Dp ** 2 * u_m + half2, s0 ** 2 + Dp ** 2 * u_c + half2
    S_RC = 300.0
    for e in np.arange(-3.5, 4) * wm.PITCH:
        ax.axvline(e, c=ps.GRID, lw=0.6, zorder=0)
    ax.plot(xx, gau(var_m), '-', c=C_MESH, label='mesh end')
    ax.plot(xx, gau(var_c), '-', c=C_CATH, label='cathode end')
    ax.plot(xx, gau(var_c + 2 * D * S_RC), '--', c=C_CATH, label=f'+ {S_RC:.0f} ns on the strips')
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(0, 2.3)
    ax.grid(False)
    ax.set_xlabel('offset from landing [mm]'); ax.set_ylabel('charge density [1/mm]')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.90), fontsize=6.6, handlelength=1.2, borderaxespad=0.0, labelspacing=0.25)
    ax.text(0.04, 0.97, '(b)', transform=ax.transAxes, fontweight='bold', va='top')
    # (c) template
    ax = fig.add_subplot(g1[2])
    tp = 'x' if hy.get('rc_tmpl', 'x') in ('x', 1, 1.0) else VIEW
    tm = wm.TMPL[tp] / np.max(wm.TMPL[tp])
    ax.plot(wm.TGRID, tm, '-', c=C_MOD)
    ax.axhline(0, c='#555555', lw=0.5)
    tsamp = np.arange(-120, 1300, wm.SNS)
    ax.plot(tsamp, np.interp(tsamp, wm.TGRID, tm), 'o', ms=2.6, c=C_MOD)
    ax.set_xlim(-200, 1300)
    ax.set_xlabel('time after landing [ns]'); ax.set_ylabel('pulse response')
    ax.text(0.90, 0.97, '(c)', transform=ax.transAxes, fontweight='bold', va='top', ha='right')
    # (d) seven strips around the brightest
    ic = int(np.argmax(W.max(axis=1)))
    o = np.argsort(pos)
    ic_o = int(np.flatnonzero(o == ic)[0])
    sel = o[max(0, ic_o - 3):ic_o + 4]
    g2 = gs[1].subgridspec(1, len(sel), wspace=0.08)
    ymax = 1.12 * max(W[sel].max(), M[sel].max())
    ymin = 1.25 * min(W[sel].min(), M[sel].min(), 0.0)
    first = None
    for j, i in enumerate(sel):
        ax = fig.add_subplot(g2[j], sharey=first)
        first = first or ax
        t = ts - p['t0']
        ax.fill_between(t, comp[i][:, lo].sum(axis=1), color=C_MESH, alpha=0.45, lw=0, label='model, mesh-end third of the column')
        ax.fill_between(t, comp[i][:, hi].sum(axis=1), color=C_CATH, alpha=0.45, lw=0, label='model, cathode-end third')
        ax.axhline(0, c='#555555', lw=0.5)
        ax.plot(t, M[i], '-', c=C_MOD, lw=1.1, label='model, all depths')
        ax.plot(t, W[i], 'o', ms=2.4, mfc='white', mec='#444444', mew=0.7, label='samples')
        ax.set_xlim(-250, 1650); ax.set_ylim(ymin, ymax)
        ax.set_xticks([0, 1000]); ax.tick_params(labelsize=7.5)
        ax.set_title(f'{pos[i] - x0:+.2f} mm', fontsize=7.5, loc='center', pad=2)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel('amplitude [ADC]')
            ax.text(0.62, 0.96, '(d)', transform=ax.transAxes, fontweight='bold', va='top')
        if j == len(sel) // 2:
            ax.set_xlabel('time $-$ $t_0$ [ns]')
            h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.035), fontsize=7.2, columnspacing=1.2,
               handlelength=1.4)
    ps.save(fig, os.path.join(OUT, 'forward_model'))
    num = dict(run=ev['run'], event_id=ev['event_id'], view=VIEW, p0=float(p['p0']), w=float(p['w']), t0=float(p['t0']),
               chi2=float(p['chi2']), dof=int(p['dof']), n_strips=int(n), depth_bins_in_gap=kmax, bin_ns=float(wm.DT),
               sigma0_mm=s0, Dp_mm_per_sqrt_ns=float(Dp), D_RC_mm2_per_ns=D, v_drift_um_ns=float(cal.v_drift),
               sigma_mesh_mm=float(np.sqrt(var_m)), sigma_cathode_mm=float(np.sqrt(var_c)),
               sigma_cathode_after_rc_mm=float(np.sqrt(var_c + 2 * D * S_RC)), rc_time_ns=S_RC,
               model_closure_vs_stored=closure, strips_shown_mm=[float(pos[i] - x0) for i in sel],
               hyper={k_: v_ for k_, v_ in hy.items() if isinstance(v_, (int, float, str))})
    json.dump(num, open(os.path.join(OUT, 'forward_model.json'), 'w'), indent=1)
    print(json.dumps(num, indent=1))


if __name__ == '__main__':
    main()
