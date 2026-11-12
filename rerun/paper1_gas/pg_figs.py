#!/usr/bin/env python3
"""pg_figs.py -- the figures of rerun/paper1_gas (results/gas.json from pg_gas.py) -> figures/*.png and the paper's
gas_vE_timing.pdf.

  v626.png       6-26 drift velocity against the clock: the anchored M3 ladder (k v_lad, attachment-blind) and the
                 timing stacks (gap / T), det6 and det7
  water626.png   the water each chamber carried through the night (anchored ladder through Magboltz)
  attach626.png  the visible fraction of the drift column T / T_true and its O2-equivalent attachment
  nights.png     water against time from the timing stacks on det4 6-23 -> 6-24 and det2 6-22
  gas_vE_timing  (Paper I §6) det3's 6-27 drift scan: timing v (gap / T) and angle-scale v (v_drift x mean kw, errors
                 from results/twov.json, pg_twov.py) on the Magboltz curve of the fitted water, with v / Magboltz
  gas_drying     (Paper I §6) 6-26 water against time for det6/det7 (anchored ladder), the shared time shape, and
                 det3's one water across the 6-27 fields
The two Paper I figures use papers/paper_style.py.  Run: pg_figs.py [name ...] (default: all).
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import gasinv as GI  # noqa: E402

D = json.load(open(os.path.join(HERE, 'results', 'gas.json')))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)
T626 = 600.0
COL = {'det2': '#4c72b0', 'det3': '#55a868', 'det4': '#c44e52', 'det6': '#8172b2', 'det7': '#dd8452'}


def v626():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for det in ('det6', 'det7'):
        A = D['att_626'][det]
        t = [w['t'] - T626 for w in A['water']]
        ax.plot(t, [w['v_true'] for w in A['water']], '-', color=COL[det], lw=1.5,
                label=f'{det} M3 ladder x k ({A["k"]:.3f})')
        P = A['points']
        ax.errorbar([p['t'] - T626 for p in P], [p['v_stack'] for p in P], yerr=[p['v_stack_err'] for p in P], fmt='o',
                    ms=4, color=COL[det], mfc='white', label=f'{det} timing stack gap/T')
    ax.set_xlabel('hours on 6-26'); ax.set_ylabel('drift velocity (µm/ns)')
    ax.set_title('6-26: the timing stack reads fast while the column is truncated (det6 before ~4 h)', fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'v626.png'), dpi=150); plt.close(fig)


def water626():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for det in ('det6', 'det7'):
        W = D['att_626'][det]['water']
        t = np.array([w['t'] - T626 for w in W])
        ax.plot(t, [w['water'] for w in W], '-o', ms=3, color=COL[det], label=det)
        lo = [w['water_lo'] for w in W]; hi = [w['water_hi'] for w in W]
        ax.fill_between(t, lo, hi, color=COL[det], alpha=0.2)
    S = D['att_626']['shared']
    ax.set_xlabel('hours on 6-26'); ax.set_ylabel('water (%) from Magboltz')
    ax.set_title(f'6-26 water: same time shape (r = {S["corr"]:.2f}), det6 excess = {S["excess_ratio_det6_over_det7"]:.2f} x det7; '
                 f'plateaus {S["plateau_water"]["det6"]:.2f} / {S["plateau_water"]["det7"]:.2f} %', fontsize=8.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'water626.png'), dpi=150); plt.close(fig)


def attach626():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    for det in ('det6', 'det7'):
        P = D['att_626'][det]['points']
        t = [p['t'] - T626 for p in P]
        a1.errorbar(t, [p['frac'] for p in P], yerr=[p['frac'] * p['v_stack_err'] / p['v_stack'] for p in P], fmt='o',
                    ms=4, color=COL[det], label=det)
        q = [(p['t'] - T626, p['o2_equiv_ppm']) for p in P if 'o2_equiv_ppm' in p and p['t'] - T626 < 5]
        if q:
            a2.plot(*zip(*q), 'o-', color=COL[det], label=det)
    a1.axhline(1, color='k', lw=0.8); a1.set_ylabel('T / T_true (visible column)'); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a2.set_ylabel('O2-equivalent (ppm)\n(tau = T / ln 2, x2 model unc.)'); a2.set_xlabel('hours on 6-26'); a2.grid(alpha=0.3)
    a2.set_yscale('log')
    fig.suptitle('6-26 attachment from the truncated drift column (T_true from the anchored ladder)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'attach626.png'), dpi=150); plt.close(fig)


def nights():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, det, lab in ((axs[0], 'det4', 'det4: 6-23 night -> 6-24 (g_det4), 197 V/cm'),
                         (axs[1], 'det2', 'det2: 6-22 night, 321 V/cm (near the v maximum)')):
        P = sorted((p for p in D['points'] if p['det'] == det and p['arm'].startswith('in30|t') and 'water' in p),
                   key=lambda p: p['t'])
        t0 = min(p['t'] for p in P)
        ax.errorbar([p['t'] - t0 for p in P], [p['water'] for p in P],
                    yerr=[[p['water'] - (p['water_lo'] if np.isfinite(p['water_lo']) else p['water']) for p in P],
                          [(p['water_hi'] if np.isfinite(p['water_hi']) else p['water']) - p['water'] for p in P]],
                    fmt='o', color=COL[det])
        for p in P:
            if p.get('ambiguous'):
                ax.annotate('2 branches', (p['t'] - t0, p['water']), fontsize=6)
        ax.set_title(lab, fontsize=9); ax.set_xlabel(f'hours after {t0:.1f}'); ax.set_ylabel('water (%)'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'nights.png'), dpi=150); plt.close(fig)


def r2(x):
    """two decimals, half up, after removing the grid's float noise (0.94499999 -> 0.95, as the README quotes it)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(round(float(x), 4))).quantize(Decimal('0.01'), ROUND_HALF_UP))


def _ps():
    sys.path.insert(0, os.path.join(REPO, 'papers'))
    import paper_style as ps
    ps.use()
    return ps


def vE():
    """Paper I §6 drift scan (2026-11-02): det3's two velocities on one Magboltz curve.
    Timing v = gap / T (head-on stack, 4 fields); angle-scale v = v_drift x mean kw (the M3 ladder, 7 fields; its error
    is the ladder estimator's, from pg_twov.py -> results/twov.json).  The curve is the timing-v fit (one water).
    Checklist pass 2026-11-02: drawn at 4.3 in, no title, (a)/(b) labels, the fit interval / chi2 / gap in the caption."""
    import pandas as pd
    ps = _ps()
    sc = D['scan']
    gap = sc['fixed']['gap']
    w0, (wlo, whi) = sc['fixed']['water'], sc['fixed']['water_68']
    TW = json.load(open(os.path.join(HERE, 'results', 'twov.json')))
    c3 = ps.DET_COLOR['det3']
    fig, (ax, ar) = ps.fig_for(4.3, height=3.5, nrows=2, sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1]))
    E = np.linspace(25, 410, 160)
    vm = np.array([GI.v_of(w0, e) for e in E])
    lo = np.array([GI.v_of(wlo, e) for e in E]); hi = np.array([GI.v_of(whi, e) for e in E])
    ax.fill_between(E, np.minimum(lo, hi), np.maximum(lo, hi), color=ps.MODEL, alpha=0.12, lw=0, zorder=1)
    ax.plot(E, vm, '-', color=ps.MODEL, lw=1.1, zorder=2, label=f'Magboltz, {r2(w0)} % H$_2$O (fit)')
    ax.plot(E, [GI.v_of(0.0, e) for e in E], ':', color=ps.MUTED, lw=0.9, zorder=2, label='Magboltz, dry')
    rows = []
    La = sorted(TW['ladder'].values(), key=lambda L: L['drift_V'])
    xa = np.array([L['drift_V'] / (gap / 10) for L in La]); ya = np.array([L['v_ang'] for L in La])
    ea = np.array([L['v_ang_err'] for L in La])
    ax.errorbar(xa, ya, yerr=ea, fmt='s', ms=4, color=c3, mfc='white', mec=c3, zorder=4,
                label=r'angle-scale $v_\mathrm{d}$ (M3 ladder)')
    P = sorted((p for p in sc['points']), key=lambda p: p['E'])
    xt = np.array([p['E'] for p in P]); yt = np.array([p['v'] for p in P]); et = np.array([p['v_err'] for p in P])
    ax.errorbar(xt, yt, yerr=et, fmt='o', ms=4.5, color=c3, zorder=5, label=r'timing $v_\mathrm{d} = g/T$')
    for x, y, e in zip(xa, ya, ea):
        rows.append(dict(kind='angle_scale_v', E=x, v=y, v_err=e, magboltz=GI.v_of(w0, x)))
    for x, y, e in zip(xt, yt, et):
        rows.append(dict(kind='timing_v', E=x, v=y, v_err=e, magboltz=GI.v_of(w0, x)))
    ax.set_ylabel(r'drift velocity $v_\mathrm{d}$ (µm/ns)'); ax.set_ylim(0, 44)
    h, l = ax.get_legend_handles_labels()
    order = [3, 2, 0, 1]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc='lower right')
    mz = lambda x: np.array([GI.v_of(w0, e) for e in x])
    ar.axhline(1.0, color=ps.MODEL, lw=1.0)
    ar.fill_between(E, np.minimum(lo, hi) / vm, np.maximum(lo, hi) / vm, color=ps.MODEL, alpha=0.12, lw=0)
    ar.errorbar(xa, ya / mz(xa), yerr=ea / mz(xa), fmt='s', ms=4, color=c3, mfc='white', mec=c3, zorder=4)
    ar.errorbar(xt, yt / mz(xt), yerr=et / mz(xt), fmt='o', ms=4.5, color=c3, zorder=5)
    ar.set_ylim(0.9, 1.26); ar.set_xlim(20, 415)
    ar.set_xlabel(r'drift field $E = V/g$ (V/cm)'); ar.set_ylabel(r'$v_\mathrm{d}\,/\,v_{\mathrm{d}}^{\mathrm{Magboltz}}$')
    ps.panel_labels([ax, ar])
    ps.save(fig, os.path.join(OUT, 'gas_vE_timing'), data=pd.DataFrame(rows))


def drying():
    """Paper I §6 (2026-11-02): the bench gas dries over a night.  Water on 6-26 from the anchored M3 ladder of det6 and
    det7 (attachment-blind; k from the attachment-free rows) through Magboltz; (a) the anchored velocity k v_lad, (b) its water, (c) the excess over each chamber's
    afternoon plateau, det6's scaled by the fitted ratio, to show the one time shape.
    Checklist pass 2026-11-02: full width, no titles, (a)-(c) labels, one legend above; the grey band (golden rows),
    r and det3's 0.97 % are in the caption."""
    import pandas as pd
    from matplotlib.lines import Line2D
    ps = _ps()
    import matplotlib.pyplot as plt
    A, S = D['att_626'], D['att_626']['shared']
    # 2026-11-03 (Dylan): drop the det3 per-field panel; left = what is measured (anchored reference-ladder velocity)
    # over the inferred water (Magboltz inversion), same time axis; right = the excess over plateau, one time shape.
    fig = plt.figure(figsize=(6.1, 3.1), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.3, 1.0])
    axv = fig.add_subplot(gs[0, 0]); axw = fig.add_subplot(gs[1, 0], sharex=axv); axe = fig.add_subplot(gs[:, 1])
    rows = []
    for det in ('det6', 'det7'):
        W = A[det]['water']
        t = np.array([w['t'] - T626 for w in W]); y = np.array([w['water'] for w in W])
        el = y - np.array([w['water_lo'] for w in W]); eh = np.array([w['water_hi'] for w in W]) - y
        v = np.array([w['v_true'] for w in W]); ve = np.array([w['v_err'] for w in W])
        axv.errorbar(t, v, yerr=ve, fmt=ps.DET_MARKER[det], ms=2.8, lw=0.6, color=ps.color(det))
        axw.errorbar(t, y, yerr=[el, eh], fmt=ps.DET_MARKER[det], ms=2.8, lw=0.6, color=ps.color(det))
        axw.axhline(S['plateau_water'][det], color=ps.color(det), lw=0.7, ls='--', zorder=1)
        for i in range(len(t)):
            rows.append(dict(panel='a/b', det=det, hours_626=t[i], v_um_per_ns=v[i], v_err=ve[i], water=y[i],
                             lo=y[i] - el[i], hi=y[i] + eh[i], E_Vcm=A[det]['E']))
    axv.set_ylabel(r'drift velocity $v_\mathrm{d}$ (µm/ns)', fontsize=ps.BASE_PT * 0.85)
    axw.set_ylabel('H$_2$O fraction (%)', fontsize=ps.BASE_PT * 0.85); axw.set_yticks([0, 0.5, 1.0, 1.5]); axw.set_xlabel('time on 6-26 (h)'); axw.set_ylim(0, 1.62)
    axw.set_xlim(0.5, 15.5)
    plt.setp(axv.get_xticklabels(), visible=False)
    r = S['excess_ratio_det6_over_det7']
    for det, sc_ in (('det6', 1.0 / r), ('det7', 1.0)):
        W = A[det]['water']
        t = np.array([w['t'] - T626 for w in W])
        y = (np.array([w['water'] for w in W]) - S['plateau_water'][det]) * sc_
        e = 0.5 * (np.array([w['water_hi'] for w in W]) - np.array([w['water_lo'] for w in W])) * sc_
        axe.errorbar(t, y, yerr=e, fmt=ps.DET_MARKER[det], ms=2.8, lw=0.6, color=ps.color(det))
        for i in range(len(t)):
            rows.append(dict(panel='c', det=det, hours_626=t[i], excess_scaled=y[i], err=e[i]))
    axe.text(0.95, 0.93, f'det6 ÷ {r:.2f}', transform=axe.transAxes, ha='right', va='top', color=ps.color('det6'))
    axe.axhline(0, color=ps.MODEL, lw=0.8)
    axe.set_xlabel('time on 6-26 (h)'); axe.set_ylabel('H$_2$O above plateau (%)'); axe.set_xlim(0.5, 15.5)
    from matplotlib.patches import Patch
    hs = [Line2D([], [], ls='', marker=ps.DET_MARKER[d], color=ps.color(d), label=d) for d in ('det6', 'det7')]
    hs += [Line2D([], [], color=ps.MUTED, ls='--', lw=0.7, label='afternoon plateau')]
    ps.legend_above(fig, hs, ncol=3, borderaxespad=0.1)
    ps.panel_labels([axv, axw, axe])
    ps.save(fig, os.path.join(OUT, 'gas_drying'), data=pd.DataFrame(rows))


if __name__ == '__main__':
    only = sys.argv[1:]
    for f in (v626, water626, attach626, nights, vE, drying):
        if not only or f.__name__ in only:
            f(); print(f.__name__)
