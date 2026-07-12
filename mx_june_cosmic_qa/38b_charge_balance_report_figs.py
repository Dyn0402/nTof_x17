#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
38b_charge_balance_report_figs.py

Standalone, paper-/talk-ready figures for the X/Y charge-balance analysis
(paper topic 4, PLAN_38).  Reads the per-event core tables cached by
`38_xy_charge_balance.py` (`charge_balance/charge_balance_events.csv` for det3
`sat_det3` and det2 `o22_long_det2`) and writes single-purpose figures plus two
physics schematics into `report_charge_balance/figs/`.

Run 38 for BOTH detectors first (it writes the event CSVs), then:
    ../.venv/bin/python 38b_charge_balance_report_figs.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Circle

from qa_config import get_config, setup_paths
setup_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'report_charge_balance', 'figs')
os.makedirs(FIGS, exist_ok=True)

ACTIVE = (0.0, 399.0)                 # active-area bounds [mm], identical det2/det3
C3, C2 = 'tab:blue', 'tab:orange'     # det3 / det2 colours
DETS = {
    'det3': dict(key='sat_det3', name='mx17\\_3', label='det3 (490 V)', c=C3),
    'det2': dict(key='o22_long_det2', name='mx17\\_2', label='det2 (525 V)', c=C2),
}


def event_csv(key):
    c = get_config(key)
    return os.path.join(c.OUT_BASE, 'alignment_tpc_veto50', 'charge_balance',
                        'charge_balance_events.csv')


def summary_csv(key):
    c = get_config(key)
    return os.path.join(c.OUT_BASE, 'alignment_tpc_veto50', 'charge_balance',
                        'xy_charge_balance.csv')


def load():
    for d in DETS.values():
        d['df'] = pd.read_csv(event_csv(d['key']))
        d['summ'] = pd.read_csv(summary_csv(d['key']))
    return DETS


def s68(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [16, 84])
    return 0.5 * (q[1] - q[0])


def profile(x, y, edges, min_n=60, stat='med'):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ctr, val, err = [], [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = (x >= b0) & (x < b1) & np.isfinite(y)
        n = int(m.sum())
        if n < min_n:
            continue
        ctr.append(0.5 * (b0 + b1))
        if stat == 'med':
            q = np.percentile(y[m], [16, 50, 84])
            val.append(q[1]); err.append(0.5 * (q[2] - q[0]) / np.sqrt(n))
        else:  # s68
            q = np.percentile(y[m], [16, 84])
            val.append(0.5 * (q[1] - q[0]))
            err.append(0.5 * (q[1] - q[0]) / np.sqrt(2 * n))
    return np.array(ctr), np.array(val), np.array(err)


def fmap(df, fmed, nb=14, min_bin=40):
    xe = np.linspace(*ACTIVE, nb + 1)
    ye = np.linspace(*ACTIVE, nb + 1)
    M = np.full((nb, nb), np.nan)
    f = df['f'].to_numpy()
    ix = np.clip(np.digitize(df['rx'], xe) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(df['ry'], ye) - 1, 0, nb - 1)
    exp = []
    fs = s68(f)
    for a in range(nb):
        for b in range(nb):
            m = (ix == a) & (iy == b)
            n = int(m.sum())
            if n >= min_bin:
                M[b, a] = np.median(f[m])
                if 1 <= a <= nb - 2 and 1 <= b <= nb - 2:
                    exp.append(1.2533 * fs / np.sqrt(n))
    inner = M[1:-1, 1:-1]
    fin = inner[np.isfinite(inner)]
    dstd = float(np.std(fin)) if len(fin) else np.nan
    estd = float(np.sqrt(np.mean(np.square(exp)))) if exp else np.nan
    return M, dstd, estd


# ============================================================ schematics
def fig_routing():
    """Physics: how the avalanche charge reaches the X and Y strip layers."""
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    # drift region
    ax.add_patch(Rectangle((1, 4.6), 8, 4.4, fc='#eef4fb', ec='0.6'))
    ax.text(9.05, 8.8, 'drift gap\n(30 mm, E$_{\\rm drift}$)', fontsize=9,
            ha='left', va='top', color='0.35')
    # incident muon
    ax.add_patch(FancyArrowPatch((2.0, 9.5), (5.1, 4.7), arrowstyle='-|>',
                 mutation_scale=16, lw=2.2, color='tab:red'))
    ax.text(1.7, 9.6, 'cosmic $\\mu$', color='tab:red', fontsize=11, ha='left')
    # primary electrons drifting down
    for xi in np.linspace(2.5, 4.9, 6):
        zi = 4.7 + (5.1 - xi) * (4.8 / 3.1) * 0.0 + np.interp(xi, [2.0, 5.1], [9.4, 4.7])
        ax.add_patch(FancyArrowPatch((xi, zi), (xi, 4.75), arrowstyle='-|>',
                     mutation_scale=8, lw=1.0, color='tab:blue', alpha=0.7))
    ax.text(6.2, 6.6, 'ionisation e$^-$\ndrift to mesh', color='tab:blue',
            fontsize=9, ha='left')

    # mesh + avalanche
    ax.plot([1, 9], [4.6, 4.6], color='k', lw=1.4)
    ax.text(9.05, 4.55, 'mesh', fontsize=9, ha='left', va='center')
    ax.add_patch(Circle((4.95, 4.35), 0.34, fc='gold', ec='orange', lw=1.5, zorder=5))
    ax.text(4.95, 4.35, '$\\times$10$^4$', fontsize=7.5, ha='center', va='center', zorder=6)
    ax.text(5.5, 3.95, 'avalanche', fontsize=9, ha='left', color='0.3')

    # readout stack: pixel top layer, X strips, Y strips
    ax.add_patch(Rectangle((1, 3.55), 8, 0.55, fc='#d9c6a5', ec='0.4'))
    for xi in np.linspace(1.35, 8.65, 16):
        ax.add_patch(Rectangle((xi - 0.16, 3.62), 0.32, 0.4, fc='#b9975b', ec='0.5', lw=0.5))
    ax.text(9.05, 3.82, 'pixelated resistive\ntop layer', fontsize=9, ha='left', va='center')

    ax.add_patch(Rectangle((1, 2.75), 8, 0.6, fc='tab:blue', alpha=0.35, ec='tab:blue'))
    ax.text(9.05, 3.05, 'X strips', fontsize=10, ha='left', va='center', color='tab:blue')
    ax.add_patch(Rectangle((1, 1.95), 8, 0.6, fc='tab:orange', alpha=0.4, ec='tab:orange'))
    ax.text(9.05, 2.25, 'Y strips', fontsize=10, ha='left', va='center', color='tab:orange')

    # charge routing arrows: avalanche -> X (q_X) and through to Y (q_Y)
    ax.add_patch(FancyArrowPatch((4.7, 3.55), (3.3, 3.05), arrowstyle='-|>',
                 mutation_scale=15, lw=2.2, color='tab:blue',
                 connectionstyle='arc3,rad=0.2'))
    ax.text(2.4, 3.15, '$q_X$', fontsize=13, color='tab:blue', ha='center', fontweight='bold')
    ax.add_patch(FancyArrowPatch((5.2, 3.55), (6.6, 2.25), arrowstyle='-|>',
                 mutation_scale=15, lw=2.2, color='tab:orange',
                 connectionstyle='arc3,rad=-0.2'))
    ax.text(7.2, 2.35, '$q_Y$', fontsize=13, color='tab:orange', ha='center', fontweight='bold')

    # definition box
    ax.add_patch(FancyBboxPatch((1.1, 0.35), 7.8, 1.05, boxstyle='round,pad=0.1',
                 fc='#fffef2', ec='0.5'))
    ax.text(5.0, 0.87,
            'balance fraction   $f=\\dfrac{q_X}{q_X+q_Y}$      '
            '($q_{X,Y}=\\sum_{\\rm strips}$ charge)',
            fontsize=13, ha='center', va='center')

    ax.set_title('How the avalanche charge is shared between the two strip layers',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'schematic_routing.png'), dpi=160)
    plt.close(fig)


def fig_metric():
    """What the metrics mean: charge per plane, and what f's width/map tell us."""
    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.7))

    # (a) per-plane charge = sum over the strip cluster; 3 charge proxies
    ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title('(a) per-plane charge $q$ = $\\Sigma$ over the cluster', fontsize=11)
    xs = np.array([2.0, 3.1, 4.2, 5.3, 6.4])
    hs = np.array([0.35, 0.7, 1.0, 0.6, 0.3]) * 3.0
    for x, h in zip(xs, hs):
        ax.add_patch(Rectangle((x - 0.28, 3.0), 0.56, h, fc='tab:blue', alpha=0.5, ec='tab:blue'))
    ax.text(4.2, 3.0 - 0.5, 'strips (one plane)', ha='center', fontsize=9, color='0.3')
    # a pulse on the peak strip, annotating peak vs area
    t = np.linspace(0, 3, 100)
    pk = 2.6 * np.exp(-((t - 1.1) / 0.55) ** 2)
    ax.plot(7.6 + t * 0.6, 6.2 + pk, color='k', lw=1.3)
    ax.annotate('peak\namplitude', xy=(7.6 + 1.1 * 0.6, 6.2 + 2.6), xytext=(8.4, 9.2),
                fontsize=8, ha='center', arrowprops=dict(arrowstyle='->'))
    ax.fill_between(7.6 + t * 0.6, 6.2, 6.2 + pk, color='tab:green', alpha=0.25)
    ax.text(7.9, 6.4, 'area\n(integral)', fontsize=8, color='tab:green')
    ax.text(0.3, 1.4,
            'three independent charge weights,\nall give the same balance:\n'
            '  • peak amplitude (sat.-corrected)\n  • pulse integral (area)\n'
            '  • clustered unshared $\\sum$amp',
            fontsize=8.6, va='top', family='monospace')

    # (b) narrow vs wide f  ->  routing quality
    ax = axs[1]
    xx = np.linspace(0, 1, 400)
    def g(mu, sig):
        return np.exp(-0.5 * ((xx - mu) / sig) ** 2)
    ax.fill_between(xx, 0, g(0.5, 0.07), color='tab:blue', alpha=0.55,
                    label='narrow $\\Rightarrow$ uniform routing (good)')
    ax.plot(xx, 0.62 * g(0.5, 0.2), color='crimson', lw=2, ls='--',
            label='wide $\\Rightarrow$ uneven coupling')
    ax.axvline(0.5, color='0.4', ls=':', lw=1)
    ax.set_xlabel('$f=q_X/(q_X+q_Y)$'); ax.set_yticks([])
    ax.set_title('(b) the WIDTH of $f$ is the design metric', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_xlim(0, 1)

    # (c) uniformity map cartoon — nearly flat (the real maps are Fig. f_maps)
    ax = axs[2]
    ax.set_title('(c) the MAP of median $f$ = pixel uniformity', fontsize=11)
    ph = np.linspace(0, 2 * np.pi, 8)
    grid = 0.49 + 0.0015 * np.subtract.outer(np.sin(ph), np.cos(ph))
    im = ax.imshow(grid, cmap='RdBu_r', vmin=0.44, vmax=0.54,
                   extent=[0, 400, 0, 400], origin='lower')
    ax.set_xlabel('detector x [mm]'); ax.set_ylabel('detector y [mm]')
    ax.text(200, 200, 'flat $\\Rightarrow$ position-\nindependent', ha='center',
            va='center', fontsize=9.5,
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='median $f$')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'schematic_metric.png'), dpi=160)
    plt.close(fig)


# ============================================================ data figures
def fig_correlation(D):
    d = D['det3']['df']
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    qm = np.percentile(d['qtot'], 99) * 0.72
    r = np.corrcoef(d['qx'], d['qy'])[0, 1]
    fmed = np.median(d['f'])
    h = ax.hist2d(d['qx'], d['qy'], bins=[85, 85], range=[[0, qm], [0, qm]],
                  norm=LogNorm(), cmap='viridis')
    ax.plot([0, qm], [0, qm], 'r--', lw=1.2, label='$q_X=q_Y$')
    ax.plot([0, qm], [0, qm * (1 - fmed) / fmed], color='w', lw=1.4,
            label=f'median $f={fmed:.3f}$')
    ax.set_xlabel('$q_X$  ($\\Sigma$ amplitude, X strips) [ADC]')
    ax.set_ylabel('$q_Y$  ($\\Sigma$ amplitude, Y strips) [ADC]')
    ax.set_title(f'det3: X vs Y layer charge  (Pearson $r={r:.3f}$)')
    ax.legend(fontsize=10, loc='upper left')
    fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04, label='events')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'qx_vs_qy.png'), dpi=160)
    plt.close(fig)


def fig_distribution(D):
    fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.2))
    bins = np.linspace(0.2, 0.8, 73)

    # (a) three charge measures give the same MEDIAN f (widths differ by
    #     construction — integral is intrinsically tighter — so a
    #     median±σ68 comparison is the honest view, not overlaid densities)
    ax = axs[0]
    methods = [('amplitude', 'peak amplitude\n(sat.-corrected)'),
               ('integral_unsat', 'pulse integral\n(area, unsat.)'),
               ('amp_sum', 'clustered $\\Sigma$amp\n(segment)')]
    yv = np.arange(len(methods))[::-1]
    for tag, dx in (('det3', -0.13), ('det2', +0.13)):
        s = D[tag]['summ']
        for y, (charge, _) in zip(yv, methods):
            row = s[s['charge'] == charge]
            if not len(row):
                continue
            m = float(row['f_median'].iloc[0]); sg = float(row['f_sigma68'].iloc[0])
            ax.plot([m - sg, m + sg], [y + dx, y + dx], '-', color=D[tag]['c'], lw=2, alpha=0.5)
            ax.plot(m, y + dx, 'o', color=D[tag]['c'], ms=8,
                    label=D[tag]['label'] if y == yv[0] else None)
            ax.text(m, y + dx + 0.11, f'{m:.3f}', ha='center', fontsize=8, color=D[tag]['c'])
    ax.axvline(0.5, color='0.4', ls=':', lw=1)
    ax.set_yticks(yv); ax.set_yticklabels([lab for _, lab in methods], fontsize=9)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.set_xlabel('median $f$   (bar = event-by-event $\\sigma_{68}$)')
    ax.set_title('(a) three charge measures $\\to$ same balance')
    ax.legend(fontsize=9.5, loc='lower right'); ax.grid(alpha=0.25, axis='x')

    # (b) det3 vs det2 full distributions
    ax = axs[1]
    for tag in ('det3', 'det2'):
        d = D[tag]['df']; f = d['f'].to_numpy()
        ax.hist(f, bins=bins, density=True, histtype='step', lw=2.2, color=D[tag]['c'],
                label=f'{D[tag]["label"]}: med {np.median(f):.3f}, $\\sigma_{{68}}$ {s68(f):.3f}')
    ax.axvline(0.5, color='0.4', ls=':', lw=1)
    ax.set_xlabel('$f=q_X/(q_X+q_Y)$'); ax.set_ylabel('normalised')
    ax.set_title('(b) det3 vs det2 — narrow, small chamber offset')
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'f_distribution.png'), dpi=160)
    plt.close(fig)


def fig_maps(D):
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for ax, tag in zip(axs, ('det3', 'det2')):
        d = D[tag]['df']
        fmed = np.median(d['f'])
        M, dstd, estd = fmap(d, fmed)
        im = ax.imshow(M, origin='lower', extent=[*ACTIVE, *ACTIVE], aspect='auto',
                       cmap='RdBu_r', vmin=fmed - 0.08, vmax=fmed + 0.08)
        ax.set_xlabel('detector x [mm]'); ax.set_ylabel('detector y [mm]')
        ax.set_title(f'{D[tag]["label"]}: median-$f$ map\n'
                     f'inner std {dstd:.3f} (stat {estd:.3f})')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='median $f$')
    fig.suptitle('Position uniformity of the charge balance '
                 '(flat = position-independent routing)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGS, 'f_maps.png'), dpi=160)
    plt.close(fig)


def fig_stability(D):
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.8))
    d = D['det3']['df']
    f = d['f'].to_numpy()
    fmed = np.median(f)

    # f vs |tan theta|
    ax = axs[0]
    for tag in ('det3', 'det2'):
        dd = D[tag]['df']
        ctr, med, err = profile(dd['tan_sp'], dd['f'], np.arange(0, 0.91, 0.1), 80, 'med')
        ax.errorbar(ctr, med, yerr=err, fmt='o-', ms=5, color=D[tag]['c'], label=D[tag]['label'])
    ax.set_xlabel('$|\\tan\\theta_{\\rm ref}|$  (track inclination)')
    ax.set_ylabel('median $f$'); ax.set_title('(a) vs track angle — flat')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # f vs q_tot, sat split (det3)
    ax = axs[1]
    qedge = np.percentile(d['qtot'], np.linspace(0, 98, 16))
    cln, stp = d[~d['sat']], d[d['sat']]
    c1, m1, e1 = profile(cln['qtot'], cln['f'], qedge, 40, 'med')
    ax.errorbar(c1, m1, yerr=e1, fmt='o-', ms=5, color=C3, label='no saturated strip')
    c2_, m2, e2 = profile(stp['qtot'], stp['f'],
                          np.percentile(stp['qtot'], np.linspace(0, 98, 12)), 40, 'med')
    ax.errorbar(c2_, m2, yerr=e2, fmt='s--', ms=5, color='crimson', label='$\\geq$1 saturated strip')
    ax.axhline(fmed, color='0.5', ls=':', lw=1)
    ax.set_xlabel('total charge $q_X+q_Y$ [ADC]'); ax.set_ylabel('median $f$')
    ax.set_title('(b) det3 vs total charge (saturation check)')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # sigma68(f) vs q_tot
    ax = axs[2]
    ct, sg, er = profile(d['qtot'], f, qedge, 60, 's68')
    ax.errorbar(ct, sg, yerr=er, fmt='o-', ms=5, color='tab:purple')
    ax.set_xlabel('total charge $q_X+q_Y$ [ADC]')
    ax.set_ylabel('$\\sigma_{68}(f)$  (sharing fluctuation)')
    ax.set_title('(c) det3 spread shrinks with charge')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'f_stability.png'), dpi=160)
    plt.close(fig)


def main():
    D = load()
    fig_routing()
    fig_metric()
    fig_correlation(D)
    fig_distribution(D)
    fig_maps(D)
    fig_stability(D)
    print(f'Figures written to {FIGS}')
    for f in sorted(os.listdir(FIGS)):
        print('  ', f)


if __name__ == '__main__':
    main()
