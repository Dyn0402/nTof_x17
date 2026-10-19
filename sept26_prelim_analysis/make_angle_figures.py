#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_angle_figures.py -- the four figures for the opening-angle page.

  angle_physics      what a pair is born with: the X17 peak, the IPC continuum
                     and the band across its modelling assumptions, with the
                     full Geant4 truth overlaid as the toy's validation.
  angle_acceptance   what the geometry lets through, per topology.  The dip
                     sits exactly where the signal is.
  angle_spectrum     the measured spectrum per topology against the folded
                     expectations and against the event-mixed shape.
  angle_summary      the model-light test: the fraction above 109 deg, per
                     topology, data against every model.

    python -m sept26_prelim_analysis.make_angle_figures --run run_145
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402

TOPO = ('intra', 'perpendicular', 'opposing')
TOPO_LABEL = {'intra': 'intra-chamber\nboth legs in one chamber',
              'perpendicular': 'perpendicular\nneighbouring chambers',
              'opposing': 'opposing\nA–C  (B–D is lost with B)'}
TOPO_COLOR = {'intra': fs.INK, 'perpendicular': fs.COPPER,
              'opposing': fs.ACCENT}
X17_MIN = 109.0


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs.use()
    return plt


def _scaled(plt, f=0.72):
    return plt.rc_context({'font.size': fs.BASE_PT * f,
                           'axes.labelsize': fs.BASE_PT * f,
                           'axes.titlesize': fs.BASE_PT * f * 1.05,
                           'xtick.labelsize': fs.BASE_PT * f * 0.86,
                           'ytick.labelsize': fs.BASE_PT * f * 0.86})


def fig_physics(S, V, geant, out):
    plt = _plt()
    with _scaled(plt, 0.76):
        fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.72, 3.53),
                               constrained_layout=True)
        th = S.index.to_numpy()
        ipc = [c for c in S.columns if c.startswith('IPC')]
        band = S[ipc].to_numpy()
        ax.fill_between(th, band.min(axis=1), band.max(axis=1),
                        color=fs.COPPER, alpha=0.25, lw=0,
                        label='IPC continuum — band over the modelling\n'
                              'assumptions (mass spectrum, γ* polarisation)')
        ax.plot(th, S['IPC · geant (1/M, isotropic)'], '-', color=fs.COPPER,
                lw=2.2, label='IPC, the simulation’s own assumption')
        ax.plot(th, S['X17'], '-', color=fs.ACCENT, lw=2.8,
                label='X17 at 16.8 MeV')
        if geant:
            for k, col, ls in (('IPC', fs.COPPER, (0, (2, 2))),
                               ('X17', fs.ACCENT, (0, (2, 2)))):
                h, e = np.histogram(geant[k], bins=np.arange(0, 181, 3.0),
                                    density=True)
                ax.plot(0.5 * (e[:-1] + e[1:]), h, ls=ls, color=col, lw=1.6,
                        zorder=6)
            ax.plot([], [], ls=(0, (2, 2)), color=fs.INK, lw=1.6,
                    label='full Geant4 truth (validation)')
        ax.axvline(X17_MIN, color=fs.INK, lw=1.4, ls=':')
        ax.annotate('109°, the X17 kinematic minimum', (X17_MIN, 0.030),
                    textcoords='offset points', xytext=(7, 0),
                    fontsize=fs.BASE_PT * 0.68, color=fs.INK)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 0.12)
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_xlabel('opening angle at birth  [deg]')
        ax.set_ylabel('normalised  dN/d$\\theta$')
        ax.set_title('What a pair is born with — and the toy reproduces the '
                     'full simulation')
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.68, loc='lower left')
        if len(V):
            ax.annotate('KS vs Geant4:  '
                        + ',  '.join(f'{r.channel} {r.ks:.3f}'
                                     for r in V.itertuples()),
                        (0.985, 0.965), xycoords='axes fraction', ha='right',
                        va='top', fontsize=fs.BASE_PT * 0.64, color=fs.MUTED)
        fs.preliminary(ax, loc='upper left')
        fs.save(fig, out / 'angle_physics', data=S.reset_index())


def fig_acceptance(A, out):
    plt = _plt()
    with _scaled(plt, 0.76):
        fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.72, 3.53),
                               constrained_layout=True)
        rows = []
        for t in TOPO:
            g = A[A.group == t].sort_values('theta')
            ax.plot(g.theta, 100 * g.acc, '-', lw=2.6, color=TOPO_COLOR[t],
                    label=TOPO_LABEL[t].replace('\n', ' — '))
            rows.append(g.assign(topology=t))
        ax.axvspan(110, 140, color=fs.ACCENT, alpha=0.10, lw=0, zorder=0)
        ax.annotate('X17 signal region\n110–140°', (125, 0.30), ha='center',
                    color=fs.ACCENT, fontsize=fs.BASE_PT * 0.69,
                    linespacing=1.4)
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_xlabel('opening angle  [deg]')
        ax.set_ylabel('acceptance  [%]')
        ax.set_title('The acceptance collapses across the middle, and the '
                     'signal region sits on its rising edge',
                     fontsize=fs.BASE_PT * 0.91)
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.69,
                  loc='upper center', bbox_to_anchor=(0.42, 1.0))
        fs.preliminary(ax, loc='upper right')
        fs.save(fig, out / 'angle_acceptance', data=pd.concat(rows))


def fig_spectrum(real, mixed, E, out, bins=np.arange(0, 181, 15.0)):
    plt = _plt()
    mid = 0.5 * (bins[:-1] + bins[1:])
    with _scaled(plt, 0.66):
        fig, axes = plt.subplots(1, 3, figsize=(fs.WIDE[0], 3.38),
                                 constrained_layout=True)
        rows = []
        for ax, t in zip(axes, TOPO):
            r = real[real.topo == t]
            x = mixed[mixed.topo == t]
            obs, _ = np.histogram(r.open_deg, bins=bins)
            mx, _ = np.histogram(x.open_deg, bins=bins)
            n = obs.sum()
            g = E[E.topology == t]
            ipc = g[g.model.str.startswith('IPC')]
            if len(ipc):
                P = ipc.pivot(index='theta', columns='model', values='frac')
                lo = P.min(axis=1).to_numpy() * n
                hi = P.max(axis=1).to_numpy() * n
                ax.fill_between(P.index, lo, hi, color=fs.COPPER, alpha=0.28,
                                lw=0, step='mid',
                                label='IPC × acceptance (band)')
            gx = g[g.model == 'X17']
            if len(gx):
                ax.step(gx.theta, gx.sort_values('theta').frac * n, where='mid',
                        color=fs.ACCENT, lw=2.0, ls='--',
                        label='X17 × acceptance')
            if mx.sum():
                ax.step(mid, mx / mx.sum() * n, where='mid', color=fs.MUTED,
                        lw=2.0, label='event-mixed (accidentals)')
            ax.errorbar(mid, obs, yerr=np.sqrt(np.clip(obs, 1, None)),
                        fmt='o', ms=6, color=TOPO_COLOR[t], lw=1.8, capsize=3,
                        zorder=6, label=f'measured  (n={n})')
            ax.axvline(X17_MIN, color=fs.INK, lw=1.2, ls=':')
            ax.set_title(TOPO_LABEL[t], color=TOPO_COLOR[t], fontweight='600',
                         fontsize=fs.BASE_PT * 0.77)
            ax.set_xlabel('opening angle  [deg]')
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 181, 45))
            ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.59,
                      loc='upper left')
            for i, v in enumerate(obs):
                rows.append(dict(topology=t, theta=mid[i], n_obs=int(v),
                                 n_mixed=int(mx[i])))
        axes[0].set_ylabel('pairs per 15° bin')
        fig.suptitle('The measured pairs follow the accidental shape, '
                     'not a pair spectrum', fontsize=fs.BASE_PT * 0.98)
        fs.preliminary(axes[-1], loc='upper right')
        fs.save(fig, out / 'angle_spectrum', data=pd.DataFrame(rows))


def fig_summary(R, C, out):
    """Fraction above 109 deg: data against every model, per topology."""
    plt = _plt()
    with _scaled(plt, 0.76):
        fig, ax = plt.subplots(figsize=(fs.WIDE[0] * 0.72, 3.38),
                               constrained_layout=True)
        cols = [c for c in R.columns if c.startswith('frac_')
                and c not in ('frac_obs',)]
        rows = []
        for i, t in enumerate(TOPO):
            g = R[R.topology == t]
            if g.empty:
                continue
            r = g.iloc[0]
            ipc = [float(r[c]) for c in cols if 'IPC' in c]
            if ipc:
                ax.fill_between([i - 0.28, i + 0.28], min(ipc), max(ipc),
                                color=fs.COPPER, alpha=0.30, lw=0)
            mx = C[(C.topology == t) & (C.model.str.startswith('event-mixed'))]
            if len(mx):
                v = float(mx.frac_above_x17_mixed.iloc[0])
                ax.plot([i - 0.28, i + 0.28], [v, v], '-', color=fs.MUTED,
                        lw=2.4)
            ax.errorbar([i], [r.frac_obs], yerr=[r.err], fmt='o', ms=12,
                        color=TOPO_COLOR[t], lw=2.4, capsize=6, zorder=6)
            rows.append(dict(topology=t, frac_obs=r.frac_obs, err=r.err,
                             ipc_lo=min(ipc) if ipc else np.nan,
                             ipc_hi=max(ipc) if ipc else np.nan))
        ax.plot([], [], '-', color=fs.MUTED, lw=2.4,
                label='event-mixed (accidentals)')
        ax.fill_between([], [], [], color=fs.COPPER, alpha=0.30,
                        label='IPC × acceptance, across the model band')
        ax.set_xticks(range(len(TOPO)))
        ax.set_xticklabels([t.replace('perpendicular', 'perpen-\ndicular')
                            for t in TOPO])
        ax.set_xlim(-0.5, len(TOPO) - 0.5)
        ax.set_ylim(-0.05, 1.12)
        ax.set_ylabel('fraction of pairs above 109°')
        ax.set_title('The model-light test — and the accidentals sit on top '
                     'of the data')
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.72, loc='center left')
        fs.preliminary(ax, loc='upper left')
        fs.save(fig, out / 'angle_summary', data=pd.DataFrame(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    a = ap.parse_args()
    d = paths.out('angle')
    od = paths.out('angle', 'figures')

    S = pd.read_csv(paths.require(d / 'physics_shapes.csv', 'physics shapes'),
                    index_col=0)
    V = pd.read_csv(paths.require(d / 'physics_validation.csv', 'validation'))
    gp = d / 'geant_truth.npz'
    geant = dict(np.load(gp)) if os.path.exists(gp) else {}
    A = pd.read_csv(paths.require(d / f'acceptance_{a.run}.csv', 'acceptance'))
    E = pd.read_csv(paths.require(d / f'expected_{a.run}.csv', 'expected'))
    C = pd.read_csv(paths.require(d / f'compare_{a.run}.csv', 'compare'))
    R = pd.read_csv(paths.require(d / f'ratio_{a.run}.csv', 'ratio'))
    real = pd.read_parquet(d / f'pairs_{a.run}.parquet')
    mixed = pd.read_parquet(d / f'pairs_mixed_{a.run}.parquet')

    fig_physics(S, V, geant, od)
    fig_acceptance(A, od)
    fig_spectrum(real, mixed, E, od)
    fig_summary(R, C, od)
    print(f'wrote 4 figures (+ CSVs) to {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
