#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_imaging_figures.py -- the four figures for the source-imaging page.

  pointing_bands   the measurement itself: median(tan) against the lever arm,
                   one panel per chamber, with the fitted line and the zero
                   crossing marked.  Dead lever ranges shaded.
  source_map       the transverse plane, looking down the beam: the capsule
                   bore, the X constraint from A and C, the Z constraint from
                   B and D, and where they cross.
  source_y         target_y against a forward model of the real gas polycone
                   through the real trigger acceptance, per chamber.
  vertex_null      two-track vertices against their event-mixed control.

    python -m sept26_prelim_analysis.make_imaging_figures --run run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import source_imaging as SI  # noqa: E402
from sept26_prelim_analysis import k_arm as K  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')


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


def bands(run, subruns, merged):
    """The pointing-coincident sample per arm, pooled over sub-runs."""
    out = {}
    for arm in ARMS:
        xs, ts = [], []
        for sub in subruns:
            S = K.coincident_tracks(run, sub, arm, merged)
            xs.append(S['xl'] - S['foot_x'])
            ts.append(S['tx'])
        out[arm] = (np.concatenate(xs), np.concatenate(ts))
    return out


def fig_pointing_bands(B, T, DT, out):
    plt = _plt()
    with _scaled(plt, 0.68):
        fig, axes = plt.subplots(1, 4, figsize=(fs.WIDE[0], 4.7), sharey=True,
                                 constrained_layout=True)
        rows = []
        edges = np.arange(-140, 141, 20.0)
        for ax, arm in zip(axes, ARMS):
            lev, tx = B[arm]
            g = T[(T.arm == arm) & (T.variant == 'baseline')]
            x0l = float(np.nanmean(g.x0_local)) if len(g) else np.nan
            sl = float(np.nanmean(g.slope)) if len(g) else np.nan
            from ntof_tracking import run145_target_imaging as TI
            foot = TI.PINWHEEL[arm]
            ax.hexbin(lev, tx, gridsize=32, extent=(-150, 150, -0.6, 0.6),
                      mincnt=1, cmap='Blues', linewidths=0, zorder=1)
            mid, med, err = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (lev >= lo) & (lev < hi)
                if m.sum() < 25:
                    continue
                mid.append(0.5 * (lo + hi))
                med.append(float(np.median(tx[m])))
                err.append(float(1.4826 * np.median(np.abs(tx[m] - med[-1]))
                                 / np.sqrt(m.sum())))
                rows.append(dict(arm=arm, lever=mid[-1], tan=med[-1],
                                 err=err[-1], n=int(m.sum())))
            ax.errorbar(mid, med, yerr=err, fmt='o', ms=5,
                        color=fs.DET_COLOR[arm], lw=1.6, capsize=3, zorder=5)
            if np.isfinite(sl):
                xx = np.array([-130., 130.])
                ax.plot(xx, sl * (xx - (x0l - foot)), '-',
                        color=fs.DET_COLOR[arm], lw=2.2, zorder=6)
                ax.axvline(x0l - foot, color=fs.ACCENT, lw=2, ls='--', zorder=7)
            for a, b in eval(DT.loc[DT.arm == arm, 'lever_spans'].iloc[0]):
                ax.axvspan(a, b, color=fs.BAND_DEAD, alpha=0.16, lw=0, zorder=0)
            for s in (-1, 1):
                ax.axvspan(s * 0, s * 30, color=fs.MUTED, alpha=0.07, lw=0)
            ax.set_title(f'chamber {arm}', color=fs.DET_COLOR[arm],
                         fontweight='600')
            ax.set_xlabel('lever arm from the foot  [mm]')
            ax.set_xlim(-150, 150)
            ax.set_ylim(-0.6, 0.6)
            ax.axhline(0, color=fs.MUTED, lw=0.9, ls=':')
        axes[0].set_ylabel('reconstructed  tan $\\theta$')
        fig.suptitle('The pointing band, and the crossing that locates the '
                     'source — no angle scale anywhere in it',
                     fontsize=fs.BASE_PT * 0.9)
        fs.preliminary(axes[-1], loc='upper left')
        fs.save(fig, out / 'pointing_bands', data=pd.DataFrame(rows))


def fig_source_map(C, meta, out):
    """Looking down the beam: two constraints, one intersection."""
    plt = _plt()
    from ntof_tracking.reco import geometry as G
    with _scaled(plt, 0.78):
        fig, ax = plt.subplots(figsize=(fs.QUARTER[0] * 1.05, 5.2),
                               constrained_layout=True)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(G.HE3_R_MAX * np.cos(th), G.HE3_R_MAX * np.sin(th), '-',
                color=fs.INK, lw=2, zorder=6)
        ax.annotate('He-3 capsule bore, r = 10 mm', (0, G.HE3_R_MAX),
                    textcoords='offset points', xytext=(0, 7), ha='center',
                    fontsize=fs.BASE_PT * 0.55, color=fs.INK)
        ax.plot([0], [0], '+', color=fs.INK, ms=14, mew=2, zorder=6)
        rows = []
        seen = {'X': 0, 'Z': 0}
        for _, r in C.iterrows():
            col = fs.DET_COLOR[r.arm]
            e = max(r.err_stat, 0.01)
            if r.axis == 'X':
                ax.axvline(r.mm, color=col, lw=2, zorder=4)
                ax.axvspan(r.mm - e, r.mm + e, color=col, alpha=0.20, lw=0)
                ax.annotate(f'{r.arm}', (r.mm, 27 - 4.5 * seen['X']),
                            ha='center',
                            color=col, fontweight='bold',
                            fontsize=fs.BASE_PT * 0.65)
            else:
                ax.axhline(r.mm, color=col, lw=2, zorder=4)
                ax.axhspan(r.mm - e, r.mm + e, color=col, alpha=0.20, lw=0)
                ax.annotate(f'{r.arm}', (27 - 5.0 * seen['Z'], r.mm),
                            va='center',
                            color=col, fontweight='bold',
                            fontsize=fs.BASE_PT * 0.65)
            seen[r.axis] += 1
            rows.append(dict(arm=r.arm, axis=r.axis, mm=r.mm,
                             err_stat=r.err_stat, err_repro=r.err_repro))
        v = {d['axis']: d for d in meta['verdict']}
        if 'X' in v and 'Z' in v:
            sx, sz = v['X']['source_mm'], v['Z']['source_mm']
            ax.plot([sx], [sz], '*', ms=22, color=fs.ACCENT, zorder=8,
                    mec='white', mew=1.2)
            ax.annotate(f'({sx:+.1f}, {sz:+.1f}) mm', (sx, sz),
                        textcoords='offset points', xytext=(13, -24),
                        color=fs.ACCENT, fontweight='bold',
                        fontsize=fs.BASE_PT * 0.62)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.set_xlabel('global X  [mm]   ← A and C measure this')
        ax.set_ylabel('global Z  [mm]   ← B and D measure this')
        ax.set_title('The source sits inside the bore, and every chamber\n'
                     'agrees to about a millimetre',
                     fontsize=fs.BASE_PT * 0.78)
        fs.preliminary(ax, loc='lower left')
        fs.save(fig, out / 'source_map', data=pd.DataFrame(rows))


def fig_source_y(curves, Y, out):
    plt = _plt()
    arms = [a for a in ARMS if a in curves]
    with _scaled(plt, 0.72):
        fig, axes = plt.subplots(1, len(arms), figsize=(fs.WIDE[0] * 0.82, 4.4),
                                 sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        bins = np.arange(-300, 301, 15.0)
        rows = []
        for ax, arm in zip(axes, arms):
            obs, pred = curves[arm]
            r = Y[Y.arm == arm].iloc[0]
            ax.hist(pred, bins=bins, density=True, histtype='stepfilled',
                    color=fs.MUTED, alpha=0.28, lw=0,
                    label='forward model\n(gas polycone × trigger)')
            ax.hist(obs, bins=bins, density=True, histtype='step', lw=2.4,
                    color=fs.DET_COLOR[arm], label='measured')
            ax.axvline(0, color=fs.MUTED, lw=1, ls=':')
            ax.axvline(r.obs_median, color=fs.DET_COLOR[arm], lw=2, ls='--')
            ax.set_title(f'chamber {arm}', color=fs.DET_COLOR[arm],
                         fontweight='600')
            ax.annotate(f'offset {r.offset_mm:+.0f} mm\n'
                        f'width {r.width_ratio:.1f}× model\n'
                        f'$\\sigma_y$ ≈ {r.implied_sigma_mm:.0f} mm',
                        (0.03, 0.96), xycoords='axes fraction', va='top',
                        fontsize=fs.BASE_PT * 0.52, linespacing=1.5)
            far = float(np.mean(np.abs(obs) > 190))
            ax.annotate(f'{far:.0%} beyond ±190 mm', (0.97, 0.62),
                        xycoords='axes fraction', ha='right',
                        fontsize=fs.BASE_PT * 0.48, color=fs.MUTED)
            ax.set_xlabel('y at closest approach  [mm]')
            ax.set_xlim(-300, 300)
            rows.append(dict(arm=arm, **{k: r[k] for k in
                                         ('n', 'obs_median', 'pred_median',
                                          'offset_mm', 'width_ratio',
                                          'implied_sigma_mm')}))
        axes[0].set_ylabel('normalised')
        axes[0].legend(frameon=False, fontsize=fs.BASE_PT * 0.5,
                       loc='center right')
        fig.suptitle('y has no zero crossing, so it is a distribution against '
                     'a model — and the model is much narrower',
                     fontsize=fs.BASE_PT * 0.86)
        # Not on the last panel: chamber D's distribution is broad enough to
        # run underneath the badge there.
        fs.preliminary(axes[min(1, len(axes) - 1)], loc='upper right')
        fs.save(fig, out / 'source_y', data=pd.DataFrame(rows))


def fig_vertex_null(real, mixed, VS, out):
    plt = _plt()
    with _scaled(plt, 0.74):
        fig, axes = plt.subplots(1, 2, figsize=(fs.WIDE[0] * 0.75, 4.4),
                                 constrained_layout=True)
        rows = []
        for ax, topo in zip(axes, ('intra', 'inter')):
            r = real[real.topology == topo]
            x = mixed[mixed.topology == topo]
            bins = np.arange(0, 201, 10.0)
            ax.hist(x.v_r, bins=bins, density=True, histtype='stepfilled',
                    color=fs.MUTED, alpha=0.3, lw=0,
                    label=f'event-mixed  (n={len(x)})')
            ax.hist(r.v_r, bins=bins, density=True, histtype='step', lw=2.4,
                    color=fs.ACCENT, label=f'same trigger  (n={len(r)})')
            ax.axvline(10, color=fs.INK, lw=1.6, ls='--')
            ax.annotate('capsule bore', (10, ax.get_ylim()[1] * 0.92),
                        textcoords='offset points', xytext=(6, 0),
                        fontsize=fs.BASE_PT * 0.52, color=fs.INK)
            g = VS[VS.topology == topo]
            if len(g):
                s = g.iloc[0]
                ax.annotate(f'lift {s.lift:.2f}×\n'
                            f'excess {s.excess_sigma:+.1f}$\\sigma$',
                            (0.97, 0.62), xycoords='axes fraction', ha='right',
                            fontsize=fs.BASE_PT * 0.56, linespacing=1.5)
            ax.set_title(f'{topo}-chamber pairs')
            ax.set_xlabel('vertex distance from the beam axis  [mm]')
            ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.52)
            rows.append(dict(topology=topo, n_real=len(r), n_mixed=len(x)))
        axes[0].set_ylabel('normalised')
        fig.suptitle('Two-track vertices do not yet concentrate on the capsule '
                     'more than chance does', fontsize=fs.BASE_PT * 0.88)
        fs.preliminary(axes[-1], loc='upper right')
        fs.save(fig, out / 'vertex_null', data=pd.DataFrame(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]
    d = paths.out('imaging')
    od = paths.out('imaging', 'figures')
    merged = str(paths.out('fullpass') / a.run)

    T = pd.read_csv(paths.require(d / f'crossings_{a.run}.csv', 'crossings'))
    C = pd.read_csv(paths.require(d / f'transverse_{a.run}.csv', 'transverse'))
    Y = pd.read_csv(paths.require(d / f'y_compare_{a.run}.csv', 'y compare'))
    DT = pd.read_csv(paths.require(d / f'dead_{a.run}.csv', 'dead table'))
    VS = pd.read_csv(paths.require(d / f'vertex_summary_{a.run}.csv',
                                   'vertex summary'))
    meta = json.load(open(paths.require(d / f'imaging_{a.run}.meta.json',
                                        'imaging meta')))
    real = pd.read_parquet(d / f'vertices_{a.run}.parquet')
    mixed = pd.read_parquet(d / f'vertices_mixed_{a.run}.parquet')
    z = np.load(d / f'y_curves_{a.run}.npz')
    curves = {k.split('_')[0]: (z[f'{k.split("_")[0]}_obs'],
                                z[f'{k.split("_")[0]}_pred'])
              for k in z.files if k.endswith('_obs')}

    fig_pointing_bands(bands(a.run, subs, merged), T, DT, od)
    fig_source_map(C, meta, od)
    fig_source_y(curves, Y, od)
    fig_vertex_null(real, mixed, VS, od)
    print(f'wrote 4 figures (+ CSVs) to {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
