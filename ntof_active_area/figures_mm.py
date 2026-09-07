#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figures_mm.py -- figures for the chamber active-area measurement.

Reads `profiles.npz` + `results_mm.json` written by `mm_edges.measure()`.
    .venv/bin/python -m ntof_active_area.figures_mm
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import mx17_active_area as JUNE
from .clusters import BENCH_ALIAS, CHAMBERS, N_STRIPS, PITCH_MM, STRIP_MAX_MM
from .mm_edges import (OUT, FIG, hot_strip_mask, select_pairs, span_profile)

C_TRACK, C_JUNE, C_NOM = '#1f77b4', '#d62728', '#7f7f7f'


def _clean(pairs):
    sel = select_pairs(pairs)
    if not len(sel):
        return sel
    hu, hv = hot_strip_mask(sel[:, 0]), hot_strip_mask(sel[:, 1])
    iu = np.clip(np.rint(sel[:, 0]).astype(int), 0, N_STRIPS - 1)
    iv = np.clip(np.rint(sel[:, 1]).astype(int), 0, N_STRIPS - 1)
    return sel[~(hu[iu] | hv[iv])]


def _june_band(ch):
    return JUNE.TRUE_ACTIVE_BY_DET[BENCH_ALIAS[ch]]


def fig_maps(data, res):
    """2-D track maps with the June telescope outline drawn on top."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    for ax, ch in zip(axes, CHAMBERS):
        sel = _clean(data[f'pairs_{ch}'])
        if not len(sel):
            ax.set_title(f'{ch}: no tracks')
            continue
        u, v = sel[:, 0] * PITCH_MM, sel[:, 1] * PITCH_MM
        b = np.arange(0, STRIP_MAX_MM + 4, 4)
        h = ax.hist2d(u, v, bins=[b, b], cmap='magma',
                      norm=matplotlib.colors.PowerNorm(0.5))
        box = _june_band(ch)
        ax.add_patch(plt.Rectangle((box['x'][0], box['y'][0]),
                                   box['x'][1] - box['x'][0],
                                   box['y'][1] - box['y'][0],
                                   fill=False, ec=C_JUNE, lw=1.6, ls='-'))
        ax.add_patch(plt.Rectangle((0, 0), STRIP_MAX_MM, STRIP_MAX_MM,
                                   fill=False, ec=C_NOM, lw=1.1, ls='--'))
        ax.set_aspect('equal')
        ax.set_title(f'{ch} ({BENCH_ALIAS[ch]})  {len(sel)} paired tracks')
        ax.set_xlabel('u  (x plane, tangential) [mm]')
        ax.set_ylabel('v  (y plane, along beam) [mm]')
        plt.colorbar(h[3], ax=ax, fraction=0.046)
    fig.suptitle('run_79 paired tracks, per chamber.  red = June cosmic-bench '
                 'telescope active area,  grey dashed = metallised strip region',
                 y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / 'mm_maps.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def fig_profiles(data, res):
    """Strip-participation profiles with the measured and June edges."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    s = np.arange(N_STRIPS) * PITCH_MM
    for i, ch in enumerate(CHAMBERS):
        sel = _clean(data[f'pairs_{ch}'])
        box = _june_band(ch)
        for j, plane in enumerate(('u', 'v')):
            ax = axes[i, j]
            if len(sel):
                ax.step(s, span_profile(sel, plane), where='mid', color=C_TRACK,
                        lw=0.9, label='strips in a paired track')
            pe = res['chambers'][ch]['planes'][plane]
            for k, key in enumerate(('live_lo_mm', 'live_hi_mm')):
                if pe[key] is not None:
                    ax.axvline(pe[key], color='k', lw=1.4,
                               label='measured edge' if k == 0 else None)
            jb = box['x' if plane == 'u' else 'y']
            for k, e in enumerate(jb):
                ax.axvline(e, color=C_JUNE, ls=':', lw=1.6,
                           label='June telescope' if k == 0 else None)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{ch} — {plane} ({pe['strip_plane']} plane)", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            if i == 3:
                ax.set_xlabel('detector-local position [mm]')
    fig.suptitle('Which strips take part in a paired track — the edges of these '
                 'profiles are the active area', y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / 'mm_profiles.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def fig_edges_zoom(data, res):
    """The four edge regions, magnified, so the step can be seen to be a step."""
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.5))
    s = np.arange(N_STRIPS) * PITCH_MM
    for j, ch in enumerate(CHAMBERS):
        sel = _clean(data[f'pairs_{ch}'])
        for i, plane in enumerate(('u', 'v')):
            ax = axes[i, j]
            if len(sel):
                sp = span_profile(sel, plane)
                for lo, hi, off in ((0, 45, 0), (STRIP_MAX_MM - 45, STRIP_MAX_MM, 45)):
                    m = (s >= lo) & (s <= hi)
                    ax.step(s[m] - lo + off, sp[m], where='mid', color=C_TRACK, lw=1.0)
            pe = res['chambers'][ch]['planes'][plane]
            box = _june_band(ch)['x' if plane == 'u' else 'y']
            # only draw an edge that actually falls inside the window it belongs
            # to -- A's u high edge sits at 349 mm, outside the 354-399 zoom
            if pe['live_lo_mm'] is not None and pe['live_lo_mm'] <= 45:
                ax.axvline(pe['live_lo_mm'], color='k', lw=1.3)
            if pe['live_hi_mm'] is not None and pe['live_hi_mm'] >= STRIP_MAX_MM - 45:
                ax.axvline(pe['live_hi_mm'] - (STRIP_MAX_MM - 45) + 45, color='k', lw=1.3)
            ax.axvline(box[0], color=C_JUNE, ls=':', lw=1.5)
            ax.axvline(box[1] - (STRIP_MAX_MM - 45) + 45, color=C_JUNE, ls=':', lw=1.5)
            ax.axvline(45, color='0.7', lw=3)
            ax.set_xticks([0, 20, 40, 50, 70, 90])
            ax.set_xticklabels(['0', '20', '40', '354', '374', '394'], fontsize=8)
            ax.set_ylim(bottom=0)
            ax.set_title(f'{ch} — {plane}', fontsize=10)
            if j == 0:
                ax.set_ylabel('tracks using this strip')
            ax.set_xlabel('position [mm]  (low edge | high edge)', fontsize=8)
    fig.suptitle('Both ends of every plane, magnified.  black = measured last '
                 'live strip, red dotted = June telescope 50 % point', y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / 'mm_edges_zoom.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def fig_connectors(res):
    """Readout health: which 64-strip connectors were live in this run."""
    health = res['connector_health']
    keys = [f'{c}{p}' for c in CHAMBERS for p in ('x', 'y')]
    m = np.array([health[k] for k in keys])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(m, cmap='RdYlGn', vmin=0, vmax=1.4, aspect='auto')
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'{i+1}\n{i*64*PITCH_MM:.0f}–{((i+1)*64-1)*PITCH_MM:.0f}mm'
                        for i in range(8)], fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_xlabel('detector connector (strip range)')
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            ax.text(b, a, f'{m[a, b]:.2f}', ha='center', va='center', fontsize=7)
    ax.set_title('run_79 cluster occupancy per connector, relative to the plane '
                 'interior', fontsize=10)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / 'mm_connectors.png', dpi=110)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    data = np.load(OUT / 'profiles.npz')
    res = json.loads((OUT / 'results_mm.json').read_text())
    fig_maps(data, res)
    fig_profiles(data, res)
    fig_edges_zoom(data, res)
    fig_connectors(res)
    print('figures ->', FIG)


if __name__ == '__main__':
    main()
