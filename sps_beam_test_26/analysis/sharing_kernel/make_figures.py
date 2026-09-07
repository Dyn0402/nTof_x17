#!/usr/bin/env python3
"""make_figures.py -- figures for the sharing-kernel shape measurement."""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import forms
from fit_kernel import stacks_for, unpack, DLIST, LO, HI, PLATEAUS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

INK, MUTED, LINE = '#0b0b0b', '#52514e', '#8a8983'
COL = {'cascade': '#2a78d6', 'ladder': '#1f9e6e', 'delay': '#9E2B25',
       'geom': '#8a8983'}
FCOL = {243.0: '#2a78d6', 156.0: '#eb6834', 95.0: '#1f9e6e'}
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8, 'axes.grid': True,
    'grid.color': '#e6e5e1', 'grid.linewidth': 0.7, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.labelcolor': MUTED, 'text.color': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'legend.frameon': False})

Z = np.load(os.path.join(HERE, 'stacks_run71_raw.npz'))
J = json.load(open(os.path.join(HERE, 'fit_kernel.json')))
T = Z['t_rel']


# ------------------------------------------------------- 1. the stacks + fit
def fig_forms(view='y', lab='raw450'):
    W = stacks_for(Z, lab, view)
    n = len(W[0])
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 7.2))

    ax = axs[0, 0]
    for d, c, ls in ((0, INK, '-'), (1, '#2a78d6', '-'), (-1, '#2a78d6', '--'),
                     (2, '#eb6834', '-'), (-2, '#eb6834', '--')):
        ax.plot(T, W[d], ls, lw=1.6, color=c, label=f'd={d:+d}')
    ax.set_xlim(-600, 1800)
    ax.set_xlabel('time relative to the central strip peak  [ns]')
    ax.set_ylabel('amplitude / central peak')
    ax.set_title(f'the measured stacks  ({view.upper()} view, {lab})',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)

    # the cross-relation, both sides, for the winning and the shipped form
    for ax, d in zip((axs[0, 1], axs[1, 0]), (1, 2)):
        for form in ('cascade', 'delay'):
            p, q = unpack(form, J[view][lab][form]['x'])
            nn = forms.build_n(form, 2, q, p, n)
            a = np.convolve(nn[0], W[d])[:n]
            b = np.convolve(nn[d], W[0])[:n]
            if form == 'cascade':
                ax.plot(T, a, 'o', ms=3.4, color=INK,
                        label=r'data side  $n_0 \ast W_d$')
            ax.plot(T, b, '-', lw=1.9, color=COL[form],
                    label=rf'{form}:  $n_{{{d}}} \ast W_0$')
        ax.axhline(0, color=LINE, lw=1)
        ax.set_xlim(-600, 1800)
        ax.set_xlabel('time  [ns]')
        ax.set_ylabel('cross-relation  [arb]')
        ax.set_title(f'both sides of $n_0*W_d = n_d*W_0$,  d = +{d}',
                     fontsize=11)
        ax.legend(fontsize=8.5)

    ax = axs[1, 1]
    xs = np.arange(len(DLIST))
    wid = 0.2
    for i, form in enumerate(('cascade', 'ladder', 'delay', 'geom')):
        p, q = unpack(form, J[view][lab][form]['x'])
        nn = forms.build_n(form, 2, q, p, n)
        vals = []
        for d in DLIST:
            a = np.convolve(nn[0], W[d])[:n]
            b = np.convolve(nn[d], W[0])[:n]
            r = (a - b)[LO:HI]
            vals.append(100 * np.sqrt((r ** 2).mean()) / max(np.abs(b).max(), 1e-9))
        ax.bar(xs + (i - 1.5) * wid, vals, wid, color=COL[form],
               label=f'{form}  ({J[view][lab][form]["rms_pct"]:.1f} % overall)')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'd={d:+d}' for d in DLIST])
    ax.set_ylabel('cross-relation residual  [% of the trace]')
    ax.set_title('how well each kernel FORM can fit', fontsize=11)
    ax.legend(fontsize=8.5)
    fig.suptitle('The sharing kernel is a cascade of one-poles, not a '
                 'translated copy — run_71 RAW, head-on', fontsize=12.5,
                 y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f'forms_{view}.png'), dpi=160,
                bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------- 2. drift-field invariance
def fig_invariance():
    fig, axs = plt.subplots(1, 4, figsize=(16.4, 4.0))
    for view, mk, ls, col in (('y', 'o', '-', '#2a78d6'),
                              ('x', 's', '--', '#9E2B25')):
        E = [J[view][lab]['field_Vcm'] for lab, _ in PLATEAUS]
        for ax, key in ((axs[0], 'tau'), (axs[1], 'c')):
            v = [J[view][lab]['cascade']['par'][key] for lab, _ in PLATEAUS]
            e = [J[view][lab]['cascade']['err'].get(key, 0) for lab, _ in PLATEAUS]
            ax.errorbar(E, v, yerr=e, fmt=mk + ls, ms=6, lw=1.6, capsize=3,
                        color=col, label=f'{view.upper()} view')
        q1 = np.array([J[view][lab]['cascade']['par']['q1'] for lab, _ in PLATEAUS])
        q1m = np.array([J[view][lab]['cascade']['par']['q1m'] for lab, _ in PLATEAUS])
        axs[2].plot(E, 0.5 * (q1 + q1m), mk + ls, ms=6, lw=1.6, color=col,
                    label=f'{view.upper()} view')
        axs[3].plot(E, q1 / q1m, mk + ls, ms=6, lw=1.6, color=col,
                    label=f'{view.upper()} view')
    axs[0].set_ylabel(r'RC time constant $\tau$  [ns]')
    axs[0].set_title(r'$\tau$', fontsize=11)
    axs[1].set_ylabel('single-step amplitude $c$')
    axs[1].set_title('$c$', fontsize=11)
    axs[2].set_ylabel(r'geometric fraction $\bar q_{\pm 1}$')
    axs[2].set_title('$q$ — the cloud width\n(flat: diffusion is small here)',
                     fontsize=10.5)
    axs[3].axhline(1.0, color=LINE, lw=1.2)
    axs[3].set_ylabel(r'$q_{+1} / q_{-1}$')
    axs[3].set_title('the SYMMETRY of the cloud\n(X is not head-on)',
                     fontsize=10.5)
    for ax in axs:
        ax.set_xlabel('drift field  [V/cm]')
        ax.set_xlim(70, 270)
        ax.legend(fontsize=9)
    fig.suptitle('Y holds its constants to 4 % over a 2.6x range of drift '
                 'field; X does not hold them, and the last panel says why',
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'invariance.png'), dpi=160,
                bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------- 3. the ladder constraint
def fig_ladder():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for view, mk in (('y', 'o'), ('x', 's')):
        c, c2, e2 = [], [], []
        for lab, _ in PLATEAUS:
            p = J[view][lab]['ladder']['par']
            c.append(p['c'] ** 2)
            c2.append(p['c2'])
            e2.append(J[view][lab]['ladder']['err'].get('c2', 0))
        ax.errorbar(c, c2, yerr=e2, fmt=mk, ms=8, capsize=3,
                    color='#2a78d6' if view == 'y' else '#9E2B25',
                    label=f'{view.upper()} view, three drift fields')
    lim = [0, 0.55]
    ax.plot(lim, lim, '--', lw=1.4, color=LINE, label='the ladder:  $c_2=c_1^2$')
    ax.set_xlim(lim)
    ax.set_ylim(-0.03, 0.55)
    ax.set_xlabel(r'$c_1^2$   (the single-step amplitude, squared)')
    ax.set_ylabel(r'$c_2$   fitted FREE')
    ax.set_title('Is the $\\pm 2$ strip reached through the $\\pm 1$ strip?',
                 fontsize=11.5)
    ax.legend(fontsize=9, loc='upper left')
    ax.annotate('X: the +-2 term goes to ZERO —\nX has no second rung',
                xy=(0.085, 0.0), xytext=(0.16, 0.09), fontsize=8.5,
                color='#9E2B25',
                arrowprops=dict(arrowstyle='-|>', color='#9E2B25', lw=1.3))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ladder.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)





# ------------------------------------------ 4. systematics: the window walk
def fig_systematics():
    S = json.load(open(os.path.join(HERE, 'systematics.json')))
    w = S['window']
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.2))
    e = [r['end_ns'] for r in w]
    axs[0].plot(e, [r['tau'] for r in w], 'o-', lw=1.8, ms=6, color='#2a78d6')
    axs[0].set_ylabel(r'fitted $\tau$  [ns]')
    axs[0].set_title(r'$\tau$ is set by how much tail you look at'
                     '\n(so it is not a constant)', fontsize=11)
    axs[0].annotate('the bench window\nends here', xy=(720, 662),
                    xytext=(1050, 700), fontsize=8.5, color=MUTED,
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2))
    axs[1].plot(e, [r['rms_cascade'] for r in w], 'o-', lw=1.8, ms=6,
                color=COL['cascade'], label='cascade')
    axs[1].plot(e, [r['rms_delay'] for r in w], 's-', lw=1.8, ms=6,
                color=COL['delay'], label='delay (shipped)')
    axs[1].set_ylabel('cross-relation residual  [%]')
    axs[1].set_ylim(0, None)
    axs[1].set_title('the FORM ranking is stable at every window',
                     fontsize=11)
    axs[1].legend(fontsize=9)
    for ax in axs:
        ax.set_xlabel('end of the fit window, relative to the peak  [ns]')
    fig.suptitle('What survives the systematics, and what does not',
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'systematics.png'), dpi=160,
                bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------- 5. does it transfer?
def fig_transfer():
    import sys
    sys.path.insert(0, HERE)
    from bench_kernel import build as bbuild, trim_mean as btrim
    A, tb, nev = bbuild('y', 0.05, 12)
    Wb = {d: btrim(A[d]) for d in A}
    Wm = stacks_for(Z, 'raw450', 'y')
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True)
    for ax, (W, t, lab, n) in zip(
            axs, ((Wm, T, 'det4 at H4 — 120 GeV pions, head-on', 3471),
                  (Wb, tb, r'det3 on the bench — cosmics, $|\tan\theta|<0.05$',
                   nev))):
        for d, c, ls in ((0, INK, '-'), (1, '#2a78d6', '-'),
                         (-1, '#2a78d6', '--'), (2, '#eb6834', '-'),
                         (-2, '#eb6834', '--')):
            ax.plot(t, W[d], ls, lw=1.7, color=c, label=f'd={d:+d}')
        ax.axvspan(720, 1900, color='#f2f1ed', zorder=0)
        ax.axhline(0, color=LINE, lw=1)
        ax.set_xlim(-750, 1800)
        ax.set_xlabel('time relative to the central peak  [ns]')
        ax.set_title(f'{lab}\n{n} events', fontsize=10.5)
        ax.legend(fontsize=8, ncol=2)
    axs[0].set_ylabel('amplitude / central peak')
    axs[0].text(1300, 0.72, 'the bench window\nstops here', ha='center',
                fontsize=9, color=MUTED)
    fig.suptitle('The two detectors do not share the same amount, and the '
                 'bench cannot see the tail that sets the constants',
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'transfer.png'), dpi=160,
                bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------ 6. the bench verdict
BENCH = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
         'long_run_resist_490V_drift_1000V/mx17_3/wft/kernel_arms/'
         'ladder_bench.json')
ORDER = [('production', 'production\n(shipped, $c_2>c_1$)', '#9E2B25'),
         ('ratio0.45', '$c_2 = 0.45\,c_1$', '#7fb3e8'),
         ('ratio0.6', '$c_2 = 0.60\,c_1$', '#2a78d6'),
         ('ratio0.8', '$c_2 = 0.80\,c_1$', '#1a4f8f'),
         ('ladder_free', 'RC kernel,\nbench-fitted', '#1f9e6e'),
         ('ladder_pinY', 'RC kernel,\nbeam-pinned', '#8a8983')]


def fig_bench():
    B = json.load(open(BENCH))
    keys = [(k, lab, c) for k, lab, c in ORDER if k in B]
    xs = np.arange(len(keys))
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, plane in zip(axs, ('y', 'x')):
        s0 = B['production']['geo'][plane]['sig_theta']
        for i, (k, lab, c) in enumerate(keys):
            g = B[k]['geo'][plane]
            e = B[k].get('vs_production', {}).get(plane, {}).get('d_sig_err', 0)
            ax.bar(i, g['sig_theta'], 0.62, color=c,
                   yerr=e if e else None, capsize=3, ecolor=MUTED)
            ax.text(i, g['sig_theta'] + 0.04,
                    f"{g['sig_theta']:.3f}\nslope {g['slope']:.4f}",
                    ha='center', fontsize=8, color=INK)
        ax.axhline(s0, color='#9E2B25', ls='--', lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _k, lab, _c in keys], fontsize=8)
        ax.set_ylabel(r'$\sigma_\theta$ on held-out events  [deg]')
        ax.set_ylim(0, max(B[k]['geo'][plane]['sig_theta']
                           for k, _l, _c in keys) * 1.28)
        ax.set_title(f'{plane.upper()} plane', fontsize=11.5)
    fig.suptitle('det3 bench, 220 held-out cosmics vs the M3 reference: '
                 'slaving $c_2$ costs nothing and fixes the angle scale',
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'bench_verdict.png'), dpi=160,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_forms('y')
    fig_forms('x')
    fig_invariance()
    fig_ladder()
    fig_systematics()
    fig_transfer()
    fig_bench()
    print('wrote', sorted(os.listdir(FIG)))
