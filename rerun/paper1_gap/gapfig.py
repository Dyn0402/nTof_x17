#!/usr/bin/env python3
"""gapfig.py -- figures of the chamber-A gap maps from results/compare.json (gapcompare.py).

  figures/gap_profiles.{pdf,png,csv}  Paper I candidate: the gap along x and along y through the interior, both runs
                                      combined, three methods, with the flat-cathode-on-the-dish prediction.
  figures/gap_maps.png                the 4x4 cell maps of every method (report)
  figures/gap_runs.png                Run 1 against Run 2, every method and cell (report)
  figures/gap_signature.png           dln v against dln T per cell: gap change against gas change (report)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402

GAP0 = 28.19
C = json.load(open(f'{HERE}/results/compare.json'))
OUT = f'{HERE}/figures'
os.makedirs(OUT, exist_ok=True)
A = ps.DET_COLOR['det3']
STY = {'G': dict(marker='o', color=A, mfc=A, label='geometric, X strips'),
       'GY': dict(marker='o', color=A, mfc='white', label='geometric, Y strips'),
       'VT': dict(marker='s', color=ps.ACCENT, mfc=ps.ACCENT, label=r'$v_\mathrm{ref}\times T$ (timing)'),
       'T': dict(marker='^', color=ps.MUTED, mfc='white', label=r'$T$ only ($\beta=0.68$)')}


def profiles():
    ps.use()
    fig, axs = plt.subplots(1, 2, figsize=ps.WIDE, sharey=True)
    rows = []
    for ax, grid, axis, lab in ((axs[0], 'x5', 0, r'$x$ at mid-gap [mm]'), (axs[1], 'y5', 1, r'$y$ at mid-gap [mm]')):
        G = C['grids'][grid]
        off = {'G': -6, 'GY': -2, 'VT': 2, 'T': 6}
        for m, st in STY.items():
            t = np.array(G[m]['comb'])
            ok = np.isfinite(t[:, 2])
            ax.errorbar(t[ok, axis] + off[m], GAP0 + t[ok, 2], yerr=t[ok, 3], ls='none', ms=4, lw=0.9, capsize=1.5,
                        **st, zorder=4 if m in ('G', 'VT') else 3)
            rows += [dict(axis=grid, method=m, pos=float(r[axis]), gap=float(GAP0 + r[2]), err=float(r[3])) for r in t[ok]]
        # flat cathode on the measured dish of the board (curvature only), through the chamber mean
        b = C['grids']['4x4']['bow']['surface']['c']
        s = np.linspace(30, 370, 100)
        u = (s - 200) / 100
        prof = (b[3] * u * u) if axis == 0 else (b[4] * u * u)
        ref = np.mean(prof[(s > 60) & (s < 340)])
        ax.plot(s, GAP0 + prof - ref, color=ps.MODEL, ls='--', lw=1.0, label='flat cathode on the dished board')
        ax.axhline(GAP0, color=ps.GRID, lw=0.8, zorder=0)
        ax.set_xlabel(lab)
        ax.set_xlim(30, 370)
    axs[0].set_ylabel('drift gap [mm]')
    axs[0].text(0.03, 0.95, '(a)', transform=axs[0].transAxes, va='top')
    axs[1].text(0.03, 0.95, '(b)', transform=axs[1].transAxes, va='top')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    import pandas as pd
    ps.save(fig, os.path.join(OUT, 'gap_profiles'), data=pd.DataFrame(rows))


def maps():
    ps.use()
    ms = ('G', 'GY', 'VT', 'T', 'V', 'bow')
    title = {'G': 'geometric, X strips', 'GY': 'geometric, Y strips', 'VT': r'$v_\mathrm{ref}\times T$',
             'T': r'$T$ only', 'V': r'$v_\mathrm{ref}$ only', 'bow': 'flat cathode on the dish'}
    fig, axs = plt.subplots(2, 3, figsize=(6.1, 4.4))
    for ax, m in zip(axs.flat, ms):
        t = np.array(C['grids']['4x4'][m]['comb'])
        ex = np.unique(t[:, 0]); ey = np.unique(t[:, 1])
        Z = np.full((len(ey), len(ex)), np.nan)
        for r in t:
            Z[np.searchsorted(ey, r[1]), np.searchsorted(ex, r[0])] = r[2]
        im = ax.imshow(Z, origin='lower', cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto',
                       extent=(30, 368.6, 48, 349.9))
        for r in t:
            if np.isfinite(r[2]):
                ax.text(r[0], r[1], f'{r[2]:+.1f}', ha='center', va='center', fontsize=6)
        ax.set_title(title[m], fontsize=8)
        ax.set_xticks([100, 200, 300]); ax.set_yticks([100, 200, 300])
    for ax in axs[1]:
        ax.set_xlabel('x [mm]')
    for ax in axs[:, 0]:
        ax.set_ylabel('y [mm]')
    fig.colorbar(im, ax=axs, shrink=0.8, label='gap change [mm]')
    ps.publicise(fig) if hasattr(ps, 'publicise') else None
    fig.savefig(os.path.join(OUT, 'gap_maps.png'), dpi=200, bbox_inches='tight')


def runs():
    ps.use()
    fig, ax = plt.subplots(figsize=ps.HALF)
    for m, st in STY.items():
        r1 = np.array(C['grids']['4x4'][m]['run1']); r2 = np.array(C['grids']['4x4'][m]['run2'])
        ok = np.isfinite(r1[:, 2]) & np.isfinite(r2[:, 2])
        ax.errorbar(r1[ok, 2], r2[ok, 2], xerr=r1[ok, 3], yerr=r2[ok, 3], ls='none', ms=3.5, lw=0.7, **st)
    ax.plot([-2, 2], [-2, 2], color=ps.MODEL, lw=0.8)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_xlabel('Run 1 gap change [mm]'); ax.set_ylabel('Run 2 gap change [mm]')
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gap_runs.png'), dpi=200)


def signature():
    ps.use()
    work = sys.argv[1]
    R = json.load(open(f'{work}/gapmap.json'))
    fig, ax = plt.subplots(figsize=ps.HALF)
    for run, mk in (('run1', 'o'), ('run2', 's')):
        for k, c in R['runs'][run]['cells'].items():
            if k.startswith('4x4:'):
                ax.errorbar(100 * np.log(c['T_rel']), 100 * np.log(c['v_rel']), xerr=100 * c['T_err'],
                            yerr=100 * c['v_err'], marker=mk, color=A, mfc=A if run == 'run1' else 'white', ms=3,
                            ls='none', lw=0.6)
    x = np.linspace(-6, 6, 10)
    k = C['grids']['4x4']['k']
    ax.plot(x, k['pure_gap'] * x, color=ps.MODEL, ls='--', lw=0.9, label=r'gap change ($-\beta/(1+\beta)$)')
    ax.plot(x, -x, color=ps.MUTED, ls=':', lw=0.9, label='gas change (slope $-1$)')
    ax.plot(x, k['k'] * x, color=ps.ACCENT, lw=0.9, label=f"fit {k['k']:.2f} $\\pm$ {k['err']:.2f}")
    ax.set_xlabel(r'$\Delta\ln T$ [%]'); ax.set_ylabel(r'$\Delta\ln v_\mathrm{ref}$ [%]')
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'gap_signature.png'), dpi=200)


def surfaces():
    """Paper I appendix figure: the gap along x and y with the joint two-surface model, and the angle resolution
    against the cell size of a cross-validated drift-tilt + velocity map (track_grid.py)."""
    ps.use()
    J = json.load(open(f'{HERE}/results/joint.json'))['arms']['all']['params']
    TG = json.load(open(f'{HERE}/results/track_grid.json'))
    sys.path.insert(0, HERE)
    import joint as JT
    qh = [J['h_' + s] for s in JT.SURF]; qc = [J['c_' + s] for s in JT.SURF]
    gU, gW = np.meshgrid(np.linspace(-1.7, 1.7, 69), np.linspace(-1.51, 1.51, 61), indexing='ij')
    G = J['g0'] + JT.q(qc, gU, gW) - JT.q(qh, gU, gW)
    fig, axs = plt.subplots(1, 3, figsize=(6.1, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    rows = []
    for ax, grid, axis, lab in ((axs[0], 'x5', 0, r'$x$ at mid-gap [mm]'), (axs[1], 'y5', 1, r'$y$ at mid-gap [mm]')):
        Gd = C['grids'][grid]
        off = {'G': -5, 'GY': 0, 'VT': 5}
        for m in ('G', 'GY', 'VT'):
            t = np.array(Gd[m]['comb']); ok = np.isfinite(t[:, 2])
            st = dict(STY[m]); st['label'] = st['label'] if axis == 0 else None
            ax.errorbar(t[ok, axis] + off[m], GAP0 + t[ok, 2], yerr=t[ok, 3], ls='none', ms=3.5, lw=0.8, capsize=1.2, **st)
            rows += [dict(axis=grid, method=m, pos=float(r[axis]), gap=float(GAP0 + r[2]), err=float(r[3])) for r in t[ok]]
        prof = G.mean(axis=1 - axis)
        coord = (np.linspace(-1.7, 1.7, 69) if axis == 0 else np.linspace(-1.51, 1.51, 61)) * 100 + 200
        ax.plot(coord, GAP0 + prof - G.mean(), color=ps.MODEL, lw=1.0, label='mesh + cathode model' if axis == 0 else None)
        ax.axhline(GAP0, color=ps.GRID, lw=0.8, zorder=0)
        ax.set_xlabel(lab); ax.set_xlim(30, 370)
    axs[0].set_ylabel('drift gap [mm]'); axs[1].set_yticklabels([])
    axs[1].set_ylim(axs[0].get_ylim())
    ax = axs[2]
    sizes = {4: 85, 6: 57, 8: 42, 10: 34}
    for p, ls in (('x', '-'), ('y', '--')):
        for run, mk in (('run1', 'o'), ('run2', 's')):
            R = TG[run][p]
            xs = [0] + [sizes[n] for n in (4, 6, 8, 10)]
            ys = [R['pass1']['s68']] + [R[f'{n}x{n} w0+v']['s68'] for n in (4, 6, 8, 10)]
            ax.plot(xs[1:], ys[1:], ls=ls, marker=mk, ms=3, lw=0.9, color=A, mfc=A if run == 'run1' else 'white',
                    label=f'{p.upper()}, Run {run[-1]}')
            ax.plot([120], [ys[0]], marker=mk, ms=3, color=ps.ACCENT, mfc=ps.ACCENT if run == 'run1' else 'white')
    ax.set_xlim(132, 24); ax.set_xticks([120, 85, 57, 42, 30])
    ax.set_xticklabels(['pass 1', '85', '57', '42', ''])
    ax.set_xlabel('map cell size [mm]'); ax.set_ylabel(r'$\sigma_\theta$ (s68) [$^\circ$]')
    ax.legend(fontsize=5.5, frameon=False, ncol=2, loc='lower left')
    for a, t, xx in zip(axs, '(a) (b) (c)'.split(), (0.04, 0.04, 0.45)):
        a.text(xx, 0.95, t, transform=a.transAxes, va='top')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.38, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    import pandas as pd
    ps.save(fig, os.path.join(OUT, 'gap_surfaces'), data=pd.DataFrame(rows))


if __name__ == '__main__':
    profiles(); maps(); runs(); signature(); surfaces()
    print('figures in', OUT)
