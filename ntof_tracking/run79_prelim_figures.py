#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run79_prelim_figures.py -- the four figures of the preliminary run_79
waveform-track / n_TOF merge.

  1 target_pointing  the point-source correlation between reconstructed
                     position and reconstructed angle (internal check)
  2 wall_pointing    predicted position at the SiPM wall, split by the wall
                     segment that actually fired (external check)
  3 rates            reconstructed-track rate vs time since the gamma flash,
                     with and without an arm-A scintillator tag
  4 quality          the reconstruction's own diagnostics

Usage:
    python -m ntof_tracking.run79_prelim_figures --merged <merged_prelim.parquet>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ntof_tracking.run79_merge_prelim import (
    TARGET_TO_STRIPS, STRIPS_TO_WALL,
    N_WALL_SEG, wall_segment_u, plastic_bar_u, plastic_u_at_wall,
    plastic_at_strips, N_POINTING_BINS, MIN_PER_BIN, TAN_SANE)

# Okabe-Ito: fixed order, assigned to wall segment 0..3 by identity (never by
# rank, so a segment keeps its colour whatever the selection).
SEG_COLOR = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
INK, MUTED = '#222222', '#888888'
PLASTIC_INK = '#4D4D4D'      # the trigger's AND partner, deliberately neutral

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': '#DDDDDD', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
})


def _binned_median(x, y, bins, min_n=MIN_PER_BIN):
    ib = np.digitize(x, bins) - 1
    bx, by, be = [], [], []
    for i in range(len(bins) - 1):
        s = ib == i
        if s.sum() < min_n:
            continue
        bx.append(0.5 * (bins[i] + bins[i + 1]))
        by.append(float(np.median(y[s])))
        be.append(1.253 * float(np.std(y[s])) / np.sqrt(s.sum()))  # median s.e.
    return np.array(bx), np.array(by), np.array(be)


def _plastic_vlines(ax, arm, mapping, coord='u', label=True):
    """Where the plastics -- the limiting element of the trigger -- sit on a
    mesh-coordinate axis. The wall is wider than they are once both are mapped
    back here, so these lines ARE the trigger acceptance."""
    if coord == 'u':
        e = sorted(x for dn in (1, 2) for x in plastic_at_strips(arm, dn, mapping))
        marks, gap = (e[0], e[-1]), 0.5 * (e[1] + e[2])
    else:
        h = plastic_at_strips(arm)
        marks, gap = (-h, h), None
    for i, x in enumerate(marks):
        ax.axvline(x, color=PLASTIC_INK, lw=1.2, ls='--', alpha=0.8, zorder=1,
                   label=('plastic acceptance, for a target ray'
                          if label and not i else None))
    if gap is not None:      # the 3 mm gap between the two bars
        ax.axvline(gap, color=PLASTIC_INK, lw=0.8, ls=':', alpha=0.5, zorder=1)


def fig_target_pointing(d, arm, out, mapping='descending'):
    L = TARGET_TO_STRIPS[arm]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for ax, (plane, u, label) in zip(axes, (
            ('x', 'u_mm', 'X plane  (u, across the arm)'),
            ('y', 'v_mm', 'Y plane  (v, along the beam)'))):
        m = (d[f'{plane}_ok'] & d[f'{plane}_quality_ok']
             & (d[f'{plane}_tan_theta'].abs() < TAN_SANE))
        x = d.loc[m, u].to_numpy()
        y = d.loc[m, f'{plane}_tan_theta'].to_numpy()
        ax.hist2d(x, y, bins=[np.linspace(-200, 200, 41), np.linspace(-1, 1, 41)],
                  cmap='Blues', cmin=1)
        bx, by, be = _binned_median(x, y, np.linspace(-150, 150, N_POINTING_BINS + 1))
        ax.errorbar(bx, by, yerr=be, fmt='o', ms=6, lw=2, color=INK,
                    label='binned median')
        uu = np.linspace(-190, 190, 2)
        ax.plot(uu, -uu / L, lw=2, ls='--', color='#D55E00',
                label=f'target at origin  (tan = -u/{L:.0f})')
        if len(bx) >= 3:
            sl = np.polyfit(bx, by, 1)[0]
            ax.plot(uu, np.polyval(np.polyfit(bx, by, 1), uu), lw=2,
                    color='#0072B2', label=f'measured  ({abs(sl) * L:.2f}x expected)')
        ax.set_xlabel(f'reconstructed position at the mesh [mm]')
        ax.set_ylabel('reconstructed tan(theta)')
        ax.set_title(label, loc='left', color=INK)
        ax.set_ylim(-1, 1)
        _plastic_vlines(ax, arm, mapping, coord='u' if plane == 'x' else 'v')
        ax.legend(frameon=False, fontsize=8, loc='upper right')

    # Panel 3: the dilution test. A shallow slope on ALL events can mean the
    # angle scale is low OR that many tracks simply do not come from the
    # target. Splitting on the n_TOF tag separates the two: events with a wall
    # AND plastic hit in THIS arm are through-going here; events whose trigger
    # came from another arm are the null sample.
    ax = axes[2]
    m_all = d['x_ok'] & d['x_quality_ok'] & (d['x_tan_theta'].abs() < TAN_SANE)
    tag = np.isfinite(d['wal_dt']) & np.isfinite(d['pss_dt'])
    other = ~d[f'wal_hit_{arm}'] & d[[f'wal_hit_{x}' for x in 'ABCD']].any(axis=1)
    bins = np.linspace(-150, 150, N_POINTING_BINS + 1)
    for sel, c, lab in ((m_all, '#888888', 'all reconstructed'),
                        (m_all & tag, '#0072B2', f'arm-{arm} wall + plastic tag'),
                        (m_all & other, '#D55E00', 'triggered by another arm')):
        if sel.sum() < 100:
            continue
        x = d.loc[sel, 'u_mm'].to_numpy()
        y = d.loc[sel, 'x_tan_theta'].to_numpy()
        bx, by, be = _binned_median(x, y, bins)
        if len(bx) < 3:
            continue
        sl = np.polyfit(bx, by, 1)[0]
        ax.errorbar(bx, by, yerr=be, fmt='o-', ms=6, lw=2, color=c,
                    label=f'{lab}  ({abs(sl) * L:.2f}x, n = {int(sel.sum()):,})')
    uu = np.linspace(-160, 160, 2)
    ax.plot(uu, -uu / L, lw=2, ls='--', color=INK, label='target at origin')
    ax.set_xlabel('reconstructed position at the mesh [mm]')
    ax.set_ylabel('reconstructed tan(theta)')
    ax.set_title('X plane, split by the n_TOF tag', loc='left', color=INK)
    _plastic_vlines(ax, arm, mapping)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')

    fig.suptitle(f'run_79 / mx17_{arm}: does the track point back at the target?  '
                 '[PRELIMINARY]', ha='left', x=0.01, fontsize=10)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def fig_wall_pointing(d, arm, sign_tag, out, mapping='descending'):
    """`mapping` is which bar group each n_TOF detn pair reads; the wall
    read-out order is an open item, so the bands drawn are the order the data
    selected in run79_merge_prelim.wall_pointing().

    Both panels also carry the two plastic bars, because the trigger is a wall
    AND plastic coincidence: outside their footprint nothing can fire at all.
    They sit further out than the wall, so what is drawn is their span mapped
    onto the wall plane along target-pointing rays (plastic_u_at_wall)."""
    col = f'u_wall_{sign_tag}'
    seg = ((d['wal_detn'] - 1) // 2)

    def band(g):
        return wall_segment_u(N_WALL_SEG - 1 - g if mapping == 'descending' else g)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    bins = np.linspace(-300, 300, 49)
    for g in range(N_WALL_SEG):
        m = (seg == g) & np.isfinite(d[col]) & d['x_ok'] & d['x_quality_ok']
        if m.sum() < 10:
            continue
        ax.hist(d.loc[m, col], bins=bins, histtype='step', lw=2,
                color=SEG_COLOR[g], label=f'segment {g}  (n = {int(m.sum()):,})')
        ax.axvline(float(np.median(d.loc[m, col])), color=SEG_COLOR[g], lw=1,
                   ls=':')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.78)
    # Two band rows, stacked the way the particle meets them: the plastics are
    # the far layer, so they go on top. They are the other half of the
    # coincidence -- no plastic, no trigger -- and sit 87-112 mm BEHIND the
    # wall, so their 200 mm bars are drawn where a target-pointing track through
    # them crosses THIS plane.
    for detn in (1, 2):
        lo, hi = plastic_u_at_wall(plastic_bar_u(detn, arm, mapping), arm)
        ax.axvspan(lo, hi, ymin=0.905, ymax=0.935, color=PLASTIC_INK, alpha=0.28)
        ax.text(0.5 * (lo + hi), 0.920, f'PSS {detn}',
                transform=ax.get_xaxis_transform(), ha='center', va='center',
                fontsize=7, color=PLASTIC_INK)
    for g in range(N_WALL_SEG):
        lo, hi = band(g)
        ax.axvspan(lo, hi, ymin=0.815, ymax=0.845, color=SEG_COLOR[g], alpha=0.55)
    ax.text(0.01, 0.995, 'the plastics behind the wall — the trigger\'s AND '
                         'partner, mapped onto this plane '
                         f'(x{plastic_u_at_wall(1.0, arm):.2f})',
            transform=ax.transAxes, va='top', fontsize=7.5, color=PLASTIC_INK)
    ax.text(0.01, 0.895, f'SiPM bar group each segment reads ({mapping} order)',
            transform=ax.transAxes, va='top', fontsize=7.5, color=MUTED)
    ax.set_xlabel('track extrapolated to the wall plane, u [mm]')
    ax.set_ylabel('events')
    ax.set_title(f'which wall segment fired vs where the track points  '
                 f'({STRIPS_TO_WALL:.0f} mm lever arm)', loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper left',
              bbox_to_anchor=(0.0, 0.795))

    ax = axes[1]
    pts = []
    for g in range(N_WALL_SEG):
        m = (seg == g) & np.isfinite(d[col]) & d['x_ok'] & d['x_quality_ok']
        if m.sum() < 10:
            continue
        u = d.loc[m, col].to_numpy()
        pts.append((g, np.median(u), np.percentile(u, 25), np.percentile(u, 75)))
        lo, hi = band(g)
        ax.plot([g, g], [lo, hi], lw=9, color=SEG_COLOR[g], alpha=0.3,
                solid_capstyle='butt')
    if pts:
        g = np.array([p[0] for p in pts])
        med = np.array([p[1] for p in pts])
        ax.errorbar(g, med, yerr=[med - np.array([p[2] for p in pts]),
                                  np.array([p[3] for p in pts]) - med],
                    fmt='o', ms=7, lw=2, color=INK, label='track prediction (median, IQR)')
        if len(g) >= 3:
            ax.set_title(f'ordering correlation {np.corrcoef(g, med)[0, 1]:+.2f}',
                         loc='left', fontsize=9)
    # The plastic coverage, as its own column: the wall says WHICH 100 mm group
    # fired, the plastic says only that the track was inside one of two 200 mm
    # bars -- so it belongs beside the segments, not among them. Same mapping
    # onto the wall plane as the left panel.
    px = N_WALL_SEG + 0.6
    pl = [plastic_u_at_wall(plastic_bar_u(dn, arm, mapping), arm) for dn in (1, 2)]
    ax.axvline(N_WALL_SEG - 0.4, color='#DDDDDD', lw=1)
    for dn, (lo, hi) in zip((1, 2), pl):
        ax.plot([px, px], [lo, hi], lw=9, color=PLASTIC_INK, alpha=0.28,
                solid_capstyle='butt')
        ax.text(px + 0.22, 0.5 * (lo + hi), f'PSS {dn}', ha='left', va='center',
                fontsize=7, color=PLASTIC_INK)
    ax.hlines([min(p[0] for p in pl), max(p[1] for p in pl)], -0.5,
              N_WALL_SEG - 0.4, color=PLASTIC_INK, lw=1, ls=':', alpha=0.6)
    ax.set_xlim(-0.5, px + 0.85)
    ylo, yhi = ax.get_ylim()          # headroom for the legend text above
    ax.set_ylim(ylo, yhi + 0.30 * (yhi - ylo))
    ax.set_xticks(list(range(N_WALL_SEG)) + [px])
    ax.set_xticklabels([str(g) for g in range(N_WALL_SEG)] + ['plastics'])
    ax.set_xlabel('n_TOF wall segment (detn pair)')
    ax.set_ylabel('u at the wall [mm]')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.text(0.99, 0.98, f'coloured bands = bar groups, {mapping} order\n'
                        '(the order the data selected)\n'
                        'grey = the plastics behind them (dotted = their edges)',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=MUTED)
    fig.suptitle(f'run_79 / mx17_{arm} x n_TOF v12: external pointing check  '
                 '[PRELIMINARY]', ha='left', x=0.01, fontsize=10)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def fig_rates(d, ev_all, arm, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
    edges = np.logspace(np.log10(2e3), np.log10(1e8), 45)          # 2 us .. 100 ms
    w = np.diff(edges) / 1e9                                        # s per bin
    have = np.isfinite(d['wal_dt'])
    for lab, sel, color in (
            ('reconstructed tracks', np.ones(len(d), bool), '#0072B2'),
            (f'... with an arm-{arm} wall+plastic tag', have.to_numpy(), '#D55E00')):
        t = d.loc[sel, 't_since_flash_ns'].to_numpy()
        n, _ = np.histogram(t, bins=edges)
        ax.step(edges[:-1], n / w, where='post', lw=2, color=color, label=lab)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('time since the gamma flash [ns]')
    ax.set_ylabel('tracks per second of flight time')
    ax.set_title(f'run_79 / mx17_{arm}: track rate vs time in the pulse  '
                 '[PRELIMINARY]', loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def fig_quality(d, arm, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    ax = axes[0]
    for p, c, lab in (('x', '#0072B2', 'X'), ('y', '#E69F00', 'Y')):
        ax.hist(d.loc[d[f'{p}_ok'], f'{p}_p0'], bins=np.linspace(0, 400, 41),
                histtype='step', lw=2, color=c, label=lab)
    ax.set_xlabel('fitted position at the mesh [mm]')
    ax.set_ylabel('events')
    ax.set_title('occupancy across the plane', loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for p, c, lab in (('x', '#0072B2', 'X'), ('y', '#E69F00', 'Y')):
        v = (d[f'{p}_chi2'] / d[f'{p}_dof'].clip(lower=1))[d[f'{p}_ok']]
        ax.hist(np.clip(v, 0, 60), bins=np.linspace(0, 60, 41), histtype='step',
                lw=2, color=c, label=lab)
    ax.set_xlabel('chi2 / dof')
    ax.set_title('fit quality', loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for p, c, lab in (('x', '#0072B2', 'X'), ('y', '#E69F00', 'Y')):
        ax.hist(d.loc[d[f'{p}_ok'], f'{p}_q_uend'], bins=np.linspace(0, 1200, 41),
                histtype='step', lw=2, color=c, label=lab)
    ax.axvline(1200, color=MUTED, lw=1, ls='--')
    ax.text(1195, ax.get_ylim()[1] * 0.9, 'end of the 20-sample window',
            ha='right', fontsize=7.5, color=MUTED)
    ax.set_xlabel('charge-column end after t0 [ns]')
    ax.set_title('drift column length (feeds v_drift)', loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'run_79 / mx17_{arm}: reconstruction diagnostics  [PRELIMINARY]',
                 ha='left', x=0.01, fontsize=10)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged', required=True)
    ap.add_argument('--arm', default='A')
    ap.add_argument('--sign', default=None, help='extrapolation sign tag (m/p); '
                                                 'default: read the summary json')
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()
    d = pd.read_parquet(a.merged)
    import json
    sj = Path(a.merged).with_name('merged_prelim.summary.json')
    summ = json.load(open(sj)) if sj.exists() else {}
    sign = a.sign or summ.get('extrapolation_sign', 'm')
    mapping = (summ.get('wall_pointing') or {}).get('mapping', 'descending')
    out = Path(a.outdir or Path(a.merged).parent / 'figures')
    out.mkdir(parents=True, exist_ok=True)
    fig_target_pointing(d, a.arm, out / f'01_target_pointing_{a.arm}.png',
                        mapping=mapping)
    fig_wall_pointing(d, a.arm, sign, out / f'02_wall_pointing_{a.arm}.png',
                      mapping=mapping)
    fig_rates(d, None, a.arm, out / f'03_track_rate_{a.arm}.png')
    fig_quality(d, a.arm, out / f'04_quality_{a.arm}.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
