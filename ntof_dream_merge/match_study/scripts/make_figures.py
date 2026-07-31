#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the DREAM<->n_TOF matching calibration.

These describe ONE method -- the calibrated one. Where a figure shows two
states it is the calibration's own two stages (global map, then the per-bunch
clock), never a comparison against a superseded analysis.

Inputs: data/window_scan_{fitarm,perbunch}.npz + summaries, data/perbunch_wp.npz,
data/bias_check_wp.{npz,json}, data/alignment.json, data/timebase.json.
Vector PDF, sized for a 16:9 beamer frame, no in-figure titles.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt         # noqa: E402
from matplotlib.colors import LogNorm   # noqa: E402

from study_common import DATA, FIGS, SUBRUNS   # noqa: E402

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.constrained_layout.use': True,
                     'axes.axisbelow': True, 'legend.framealpha': 0.9})

# The two stages of the calibration itself.
STAGE = ('fitarm', 'perbunch')
SLAB = {'fitarm': 'global map: $K$, $T_0$, per-arm offset',
        'perbunch': 'full calibration: $+$ per-bunch clock'}
SCOL = {'fitarm': 'C1', 'perbunch': 'C0'}
LEGNAME = {'wp': 'wall AND plastic', 'w': 'wall only'}
LEGCOL = {'wp': 'C0', 'w': 'C4'}
T_BINS = ((1, 3), (3, 10), (10, 20), (20, 40), (40, 80))
WIN = 25.0

Z = {t: np.load(DATA / f'window_scan_{t}.npz') for t in STAGE}
S = {t: json.loads((DATA / f'window_scan_summary_{t}.json').read_text())
     for t in STAGE}
P = Z['perbunch']
SP = S['perbunch']


def _excess(z, leg, key=''):
    return (z[f'{leg}/hist_sig{key}'] - z[f'{leg}/hist_ctl{key}']).astype(float)


def _halfwidth(c, d):
    """68 % half-width of a (possibly asymmetric) excess distribution."""
    d = np.clip(d, 0, None)
    if d.sum() <= 0:
        return np.nan
    cum = np.cumsum(d) / d.sum()
    return 0.5 * (np.interp(0.84, cum, c) - np.interp(0.16, cum, c))


def fig_resid_vs_time():
    """The two stages of the time-base calibration, on the same colour scale."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for a, tb in zip(ax, STAGE):
        h = Z[tb]['wp/h2'].T
        te, re = Z[tb]['wp/h2_tedges'] / 1e6, Z[tb]['wp/h2_redges']
        pc = a.pcolormesh(te, re, np.ma.masked_where(h == 0, h), norm=LogNorm(),
                          cmap='viridis', shading='flat')
        a.set_xscale('log')
        a.set_xlabel('time since $\\gamma$ flash [ms]')
        a.set_ylim(-150, 150)
        a.text(0.02, 0.92, SLAB[tb], transform=a.transAxes, fontsize=8.5,
               color='w', bbox=dict(fc='0.2', ec='none', alpha=0.6, pad=2))
        a.axhline(0, color='w', lw=0.6, alpha=0.5)
        for s in (-WIN, WIN):
            a.axhline(s, color='w', lw=0.9, ls='--', alpha=0.85)
    ax[1].text(1.3, WIN + 8, '$\\pm25\\,$ns accept', color='w', fontsize=7.5)
    ax[0].set_ylabel('match residual [ns]')
    fig.colorbar(pc, ax=ax[1], label='candidates')
    fig.savefig(FIGS / 'fig_resid_vs_time.pdf')
    plt.close(fig)


def fig_residuals():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    e = P['wp/edges']
    c = 0.5 * (e[1:] + e[:-1])
    m = np.abs(c) <= 200
    ax[0].step(c[m], _excess(P, 'wp')[m], where='mid', color=SCOL['perbunch'],
               lw=1.4, label='matched excess (wall AND plastic)')
    ax[0].step(c[m], P['wp/hist_ctl'][m], where='mid', color='0.4', lw=1.0,
               ls=':', label='accidental control ($\\pm100\\,\\mu$s shift)')
    ax[0].axvspan(-WIN, WIN, color='C2', alpha=0.12, lw=0)
    ax[0].axvline(-WIN, color='C2', lw=1.0, ls='-.')
    ax[0].axvline(WIN, color='C2', lw=1.0, ls='-.')
    ax[0].set_yscale('log')
    ax[0].set_ylim(0.5, None)
    ax[0].set_xlim(-200, 200)
    ax[0].set_xlabel('match residual [ns]')
    ax[0].set_ylabel('candidates / 2 ns')
    ax[0].legend(fontsize=7.5, loc='upper left')
    ax[0].text(0.62, 0.55, 'accept\n$\\pm25\\,$ns', transform=ax[0].transAxes,
               fontsize=8, color='C2', ha='center')

    for tb in STAGE:
        wid = [_halfwidth(c, _excess(Z[tb], 'wp', f'_{lo}_{hi}'))
               for lo, hi in T_BINS]
        x = [np.sqrt(lo * hi) for lo, hi in T_BINS]
        ax[1].plot(x, wid, 'o-', color=SCOL[tb], lw=1.5, ms=4.5, label=SLAB[tb])
        for xx, ww in zip(x, wid):
            ax[1].annotate(f'{ww:.0f}', (xx, ww), textcoords='offset points',
                           xytext=(0, 7 if tb == 'fitarm' else -13),
                           fontsize=6.5, color=SCOL[tb], ha='center')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(3, 90)
    ax[1].set_xlabel('time since $\\gamma$ flash [ms]')
    ax[1].set_ylabel('68 % half-width of the residual [ns]')
    ax[1].legend(fontsize=7.5, loc='upper left')
    fig.savefig(FIGS / 'fig_residuals.pdf')
    plt.close(fig)


def fig_window_scan():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, leg in zip(ax, ('wp', 'w')):
        w = P[f'{leg}/sym/w']
        a.plot(w, 100 * P[f'{leg}/sym/eff'], color=LEGCOL[leg], lw=1.8,
               label='matched (efficiency)')
        a.plot(w, 100 * P[f'{leg}/sym/false'], color=LEGCOL[leg], lw=1.3,
               ls='--', label='matched by accident')
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_ylim(0.003, 300)
        a.set_xlim(1, 2000)
        a.set_xlabel('accept half-width $|r| < w$ [ns]')
        a.axvline(WIN, color='C2', lw=1.1, ls='-.')
        a.text(0.97, 0.06, LEGNAME[leg], transform=a.transAxes, fontsize=9.5,
               color='0.3', ha='right')
        a.text(WIN * 1.15, 0.0045, 'accept $\\pm25\\,$ns', fontsize=7,
               color='C2', rotation=90, va='bottom')
        a.legend(fontsize=7.5, loc='upper left')
    ax[0].set_ylabel('fraction of DREAM triggers [%]')
    fig.savefig(FIGS / 'fig_window_scan.pdf')
    plt.close(fig)


def fig_window_tbins():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    cm = plt.cm.viridis(np.linspace(0.1, 0.88, len(T_BINS)))
    for a, leg in zip(ax, ('wp', 'w')):
        for (lo, hi), col in zip(T_BINS, cm):
            n = f'sym_t{lo}_{hi}'
            w = P[f'{leg}/{n}/w']
            a.plot(w, 100 * P[f'{leg}/{n}/eff'], color=col, lw=1.5,
                   label=f'{lo}-{hi} ms')
            a.plot(w, 100 * P[f'{leg}/{n}/false'], color=col, lw=1.1, ls='--')
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_ylim(0.003, 300)
        a.set_xlabel('accept half-width [ns]')
        a.axvline(WIN, color='C2', lw=1.1, ls='-.')
        a.text(0.03, 0.06, LEGNAME[leg], transform=a.transAxes, fontsize=9.5,
               color='0.3')
    ax[0].set_ylabel('efficiency (solid) / accidental (dashed) [%]')
    ax[0].legend(fontsize=7.5, title='time since flash', title_fontsize=7.5,
                 loc='upper left')
    fig.savefig(FIGS / 'fig_window_tbins.pdf')
    plt.close(fig)


def fig_roc():
    """Efficiency against accidental rate, with the window as the parameter."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for leg in ('wp', 'w'):
        f = P[f'{leg}/sym/false']
        e = P[f'{leg}/sym/eff']
        ok = f > 0
        ax.plot(100 * f[ok], 100 * e[ok], color=LEGCOL[leg], lw=1.7,
                label=LEGNAME[leg])
    for nm, lab, dx, dy in (('tight_15', '$\\pm15\\ns$', -36, -3),
                            ('tight_25', '$\\pm25\\ns$', 2, -16),
                            ('tight_50', '$\\pm50\\ns$', -8, 9),
                            ('sym_100', '$\\pm100\\ns$', 4, -14)):
        r = SP['legs']['wp']['points'][nm]
        big = nm == 'tight_25'
        ax.plot(100 * r['false'], 100 * r['eff'], 'o', color='C2' if big else 'k',
                ms=9 if big else 5, mfc='none', mew=2.0 if big else 1.2, zorder=5)
        ax.annotate(lab.replace('\\ns', '\\,\\mathrm{ns}'),
                    (100 * r['false'], 100 * r['eff']),
                    textcoords='offset points', xytext=(dx, dy), fontsize=8,
                    color='C2' if big else '0.25',
                    fontweight='bold' if big else 'normal')
    ax.set_xscale('log')
    ax.set_xlabel('accidental match rate [%]')
    ax.set_ylabel('efficiency [%]')
    ax.set_ylim(90, 100)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.text(0.03, 0.06, 'points: symmetric windows on the\nwall AND plastic leg',
            transform=ax.transAxes, fontsize=7.5, color='0.35')
    fig.savefig(FIGS / 'fig_roc.pdf')
    plt.close(fig)


def fig_perbunch():
    Q = np.load(DATA / 'perbunch_wp.npz')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for sub, col in zip(SUBRUNS, ('C0', 'C1')):
        a = Q[sub]
        ax[0].plot(a[:, 0], a[:, 2] * 1e6, '.', ms=2.0, color=col,
                   label=sub.replace('stat090_', 'sub-run '))
        ax[1].hist(a[:, 1], bins=60, range=(-25, 25), histtype='step',
                   color=col, lw=1.3, label=sub.replace('stat090_', 'sub-run '))
    ax[0].set_xlabel('n_TOF bunch number')
    ax[0].set_ylabel('per-bunch clock error $\\delta k$ [ppm]')
    ax[0].legend(fontsize=8, markerscale=4)
    ax[1].set_xlabel('per-bunch offset $\\delta a$ [ns]')
    ax[1].set_ylabel('bunches')
    ax[1].legend(fontsize=8)
    fig.savefig(FIGS / 'fig_perbunch.pdf')
    plt.close(fig)


def fig_bias():
    """The per-bunch fit is not fitting noise, and does not manufacture matches."""
    B = np.load(DATA / 'bias_check_wp.npz')
    J = json.loads((DATA / 'bias_check_wp.json').read_text())
    fig, ax = plt.subplots(1, 4, figsize=(13.6, 3.2))

    # [1] statistics per bunch
    n = np.concatenate([B[f'{s}/n'] for s in SUBRUNS])
    sc = np.concatenate([B[f'{s}/sig_corr'] for s in SUBRUNS])
    ax[0].hist(n, bins=40, color='C0', alpha=0.85)
    ax[0].axvline(np.median(n), color='k', lw=1.2)
    ax[0].axvline(20, color='C3', lw=1.2, ls='--')
    ax[0].set_xlabel('matched triggers used per bunch')
    ax[0].set_ylabel('bunches')
    ax[0].text(0.96, 0.93, f'median {np.median(n):.0f}\nminimum {n.min():.0f}\n'
                           f'(2 parameters)', transform=ax[0].transAxes,
               fontsize=8, ha='right', va='top')
    ax[0].text(22, ax[0].get_ylim()[1] * 0.35, 'fit floor', fontsize=7,
               color='C3', rotation=90)

    # [2] split-half reproducibility of dk
    k0 = np.concatenate([B[f'{s}/k0'] for s in SUBRUNS]) * 1e6
    k1 = np.concatenate([B[f'{s}/k1'] for s in SUBRUNS]) * 1e6
    rho = np.corrcoef(k0, k1)[0, 1]
    rn = np.concatenate([B[f'{s}/rho_null'] for s in SUBRUNS])
    lim = np.percentile(np.abs(np.r_[k0, k1]), 99.5) * 1.15
    ax[1].plot(k0, k1, '.', ms=1.8, color='C0', alpha=0.6)
    ax[1].plot([-lim, lim], [-lim, lim], 'k-', lw=0.8, alpha=0.6)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel('$\\delta k$ from odd triggers [ppm]')
    ax[1].set_ylabel('$\\delta k$ from even triggers [ppm]')
    j = J[SUBRUNS[0]]['split_half']
    ax[1].text(0.04, 0.95, f'$\\rho = {rho:+.3f}$\n'
                           f'shuffled: $|\\rho| < {np.abs(rn).max():.02f}$\n'
                           f'drift {j["drift_rms_k_ppm"]:.2f} ppm\n'
                           f'fit noise {j["noise_rms_k_ppm"]:.2f} ppm',
               transform=ax[1].transAxes, fontsize=8, va='top')

    # [3] in-sample vs cross-validated
    x = [np.sqrt(lo * hi) for lo, hi in T_BINS]
    for key, lab, col, ls in (('raw', 'before the per-bunch fit', 'C1', '-'),
                              ('in_sample', 'in sample', '0.45', '--'),
                              ('xval', 'cross-validated', 'C0', '-')):
        y = [np.mean([J[s]['widths'][f'{lo}-{hi}'][key] for s in SUBRUNS])
             for lo, hi in T_BINS]
        ax[2].plot(x, y, 'o' + ls, color=col, lw=1.5, ms=4.5, label=lab)
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_ylim(3, 90)
    ax[2].set_xlabel('time since $\\gamma$ flash [ms]')
    ax[2].set_ylabel('68 % half-width [ns]')
    ax[2].legend(fontsize=7.5, loc='upper left')

    # [4] where the efficiency at +-25 ns comes from, and the wrong-bunch null
    w_ = np.array([J[s]['n_events'] for s in SUBRUNS], float)
    w_ /= w_.sum()

    def mix(key):
        return 100 * float(np.dot(w_, [J[s]['tight_window'][key]
                                       for s in SUBRUNS]))
    bars = [('parameters from the\nWRONG bunch', mix('shuffled_sig'), 'C3'),
            ('global map only', mix('none_sig'), '0.6'),
            ('$+\\,\\delta a_b$ (offset)', mix('offset_only_sig'), 'C1'),
            ('$+\\,\\delta k_b$ (rate)\n= the calibration', mix('pb_sig'), 'C0')]
    y = np.arange(len(bars))
    ax[3].barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62)
    ax[3].set_yticks(y)
    ax[3].set_yticklabels([b[0] for b in bars], fontsize=7.5)
    ax[3].set_xlim(0, 118)
    ax[3].set_xlabel('efficiency at $\\pm25\\,$ns [%]')
    ax[3].grid(axis='y', alpha=0)
    for yy, (_, v, _c) in zip(y, bars):
        ax[3].text(v + 2, yy, f'{v:.1f}', va='center', fontsize=8)

    for a, lab in zip(ax, ('[1] statistics per bunch',
                           '[2] split-half: real drift, not noise',
                           '[3] the cost of fitting in sample',
                           '[4] the parameters are bunch-specific')):
        a.text(0.0, 1.04, lab, transform=a.transAxes, fontsize=9, color='0.3')
    fig.savefig(FIGS / 'fig_bias.pdf')
    plt.close(fig)


def fig_arms():
    tbj = json.loads((DATA / 'timebase.json').read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    arms = list(tbj['per_arm'])
    x = np.arange(len(arms))
    for i, sub in enumerate(SUBRUNS):
        v = [tbj['per_arm'][a][sub]['a'] for a in arms]
        ax[0].plot(x + 0.06 * (i - 0.5), v, 'os'[i], ms=7, mfc='none', mew=1.8,
                   color=f'C{i}', ls='none',
                   label=sub.replace('stat090_', 'sub-run '))
    for i, a_ in enumerate(arms):
        v = [tbj['per_arm'][a_][s]['a'] for s in SUBRUNS]
        ax[0].plot([i, i], v, color='0.6', lw=1.0, zorder=0)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f'arm {a}' for a in arms])
    ax[0].set_xlim(-0.5, len(arms) - 0.5)
    ax[0].set_ylabel('per-arm offset [ns]')
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].legend(fontsize=8, loc='lower right')
    ax[0].text(0.02, 0.95, 'the two hours agree to $\\leq$2.6 ns',
               transform=ax[0].transAxes, fontsize=8, color='0.35', va='top')
    for a_, col in zip('ABCD', ('C0', 'C1', 'C2', 'C3')):
        n = f'sym_arm{a_}'
        w = P[f'wp/{n}/w']
        ax[1].plot(w, 100 * P[f'wp/{n}/eff'], color=col, lw=1.5, label=f'arm {a_}')
        ax[1].plot(w, 100 * P[f'wp/{n}/false'], color=col, lw=1.0, ls='--')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(0.003, 200)
    ax[1].set_xlabel('accept half-width [ns]')
    ax[1].set_ylabel('all DREAM triggers matched to this arm [%]')
    ax[1].axvline(WIN, color='C2', lw=1.1, ls='-.')
    ax[1].legend(fontsize=8, loc='lower right')
    ax[1].text(0.03, 0.93, 'solid: matched   dashed: accidental',
               transform=ax[1].transAxes, fontsize=7.5, color='0.3')
    fig.savefig(FIGS / 'fig_arms.pdf')
    plt.close(fig)


def fig_alignment():
    p = DATA / 'alignment.json'
    if not p.exists():
        print('  (no alignment.json -- run align_survey.py)')
        return
    A = json.loads(p.read_text())
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))

    trees = list(A['flash'])
    v = np.array([A['flash'][t]['median'] for t in trees])
    cal = np.array([A['calibration'].get(t, {}).get('C_ns', np.nan)
                    for t in trees])
    x = np.arange(len(trees))
    sg = np.array([A['flash'][t]['std_core'] for t in trees])
    ax[0].errorbar(x, v, yerr=sg, fmt='o', ms=5, color='k', ecolor='0.6',
                   capsize=2, label='measured on v12, run 224572', zorder=4)
    for fam, cc in (('WAL', 'C0'), ('PSS', 'C1'), ('LIQ', 'C2')):
        s_ = np.array([t.startswith(fam) for t in trees])
        ax[0].plot(x[s_], v[s_], 'o', ms=5, color=cc, zorder=5)
    ok = np.isfinite(cal)
    ax[0].plot(x[ok], cal[ok], 'kx', ms=7, mew=1.4,
               label='divert-off calibration')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(trees, rotation=90, fontsize=7)
    ax[0].set_ylim(-1740, -1620)
    ax[0].set_ylabel('$t_{\\rm flash}-t_{\\rm flash}$(PKUP) [ns]')
    ax[0].legend(fontsize=6.5, loc='upper left')

    ch = A['coincidence']['channel']
    keys = sorted(ch)
    ax[1].bar(range(len(keys)), [ch[k]['median'] for k in keys], color='C0')
    ax[1].set_xticks(range(len(keys)))
    ax[1].set_xticklabels(keys, rotation=90, fontsize=5.5)
    ax[1].set_ylabel('$t_{\\rm PSS}-t_{\\rm WAL}$ [ns]')
    ax[1].axhline(0, color='k', lw=0.8)

    tb = A['coincidence']['topbottom']
    labs, vals = [], []
    for arm in tb:
        for gseg in sorted(tb[arm], key=int):
            labs.append(f'{arm}{gseg}')
            vals.append(tb[arm][gseg]['peak'])
    liq = A['coincidence']['liq']
    labs += [f'LIQ{a}' for a in liq]
    vals += [liq[a]['peak'] for a in liq]
    ax[2].bar(range(len(labs)), vals,
              color=['C4'] * (len(labs) - len(liq)) + ['C2'] * len(liq))
    ax[2].set_xticks(range(len(labs)))
    ax[2].set_xticklabels(labs, rotation=90, fontsize=5.5)
    ax[2].set_ylabel('offset [ns]')
    ax[2].axhline(0, color='k', lw=0.8)
    for a, lab in zip(ax, ('[1] flash vs beam pickup',
                           '[2] wall vs plastic, per channel',
                           '[3] top$-$bottom (grey), LIQ$-$WAL (green)')):
        a.text(0.02, 1.03, lab, transform=a.transAxes, fontsize=8.5, color='0.3')
    fig.savefig(FIGS / 'fig_alignment.pdf')
    plt.close(fig)


if __name__ == '__main__':
    FIGS.mkdir(parents=True, exist_ok=True)
    for f in (fig_resid_vs_time, fig_residuals, fig_window_scan,
              fig_window_tbins, fig_roc, fig_perbunch, fig_bias, fig_arms,
              fig_alignment):
        f()
        print(f'  {f.__name__} ok')
    print(f'-> {FIGS}')
