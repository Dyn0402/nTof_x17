#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_microtpc.py -- the micro-TPC operation figure.

    ../.venv/bin/python make_microtpc.py [--theme light|dark|both] [--draft]
                                         [--angle 32] [--seed 7] [--no-ladder]

Left: one muon crossing the 30 mm drift gap, every primary cluster drifting
down to the mesh, coloured by arrival time.  Right: what the chamber actually
records -- first arrival time per strip -- with a straight-line fit whose slope
is the track angle.  The right panel is the point of the left one.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib import cm, colors as mcolors       # noqa: E402
from PIL import Image                              # noqa: E402

import style as S                                  # noqa: E402
import annotate as A                               # noqa: E402
import scenes_microtpc as T                        # noqa: E402

FIG = os.path.join(HERE, 'figures')
CMAP = 'plasma'

VIEW = dict(pos=(52, -78, 40), focal=(0, 0, 13), up=(0, 0, 1), angle=34.0)


def render_3d(theme, size, ssaa, out, angle, seed):
    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=2.0)
    T.add_chamber(p, theme)
    a, b, clusters = T.make_event(angle_deg=angle, seed=seed)
    hits = T.add_event(p, a, b, clusters, cmap=CMAP)
    S.add_light_rig(p, np.array([0, 0, 14]), 34.0, theme=theme, shadows=False,
                    up='z')
    p.camera.position = VIEW['pos']
    p.camera.focal_point = VIEW['focal']
    p.camera.up = VIEW['up']
    p.camera.view_angle = VIEW['angle']
    p.renderer.reset_camera_clipping_range()
    S.finish(p, out)
    return clusters, hits


def compose(png, clusters, hits, out_base, theme, angle, dpi=300,
            with_ladder=True):
    """Render on the left, the strip-time ladder on the right."""
    img = np.asarray(Image.open(png).convert('RGB'))
    h, w = img.shape[:2]

    ink = '#f2f5f9' if theme == 'dark' else '#141b24'
    muted = '#9aa7b6' if theme == 'dark' else '#5d6874'
    page = '#0a0d13' if theme == 'dark' else '#ffffff'
    grid = '#2a3440' if theme == 'dark' else '#e3e8ee'

    lad_w = int(w * (0.62 if with_ladder else 0.0))
    head = int(0.12 * h)
    foot = int(0.24 * h)
    W, H = w + lad_w, h + head + foot

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=page)
    imax = fig.add_axes([0, foot / H, w / W, h / H])
    imax.imshow(img, interpolation='lanczos')
    imax.axis('off')

    fs = A.TEXT_FRAC * W * 72.0 / dpi
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')

    ax.text(0.026 * W, head * 0.38, 'Micro-TPC operation',
            ha='left', va='center', fontsize=fs * 1.95, color=ink,
            fontweight='bold', **A.FONT)
    ax.text(0.026 * W, head * 0.76,
            'One MX17 chamber measures a track angle from a single plane',
            ha='left', va='center', fontsize=fs * 0.95, color=muted, **A.FONT)

    if with_ladder:
        xs, ts = T.strip_ladder(hits)
        lx0 = (w + 0.085 * lad_w) / W
        pax = fig.add_axes([lx0, (foot + 0.30 * h) / H,
                            0.80 * lad_w / W, 0.60 * h / H], facecolor='none')
        pax.scatter(xs, ts, s=fs * 2.2, c=ts, cmap=CMAP,
                    vmin=0, vmax=T.drift_time_ns(T.DRIFT_MM),
                    edgecolors='none', zorder=3)
        if len(xs) > 2:
            m, c = np.polyfit(xs, ts, 1)
            xx = np.array([xs.min(), xs.max()])
            pax.plot(xx, m * xx + c, color=ink, lw=fs * 0.09, alpha=0.75,
                     zorder=2)
            # dt/dx = 1/(v * tan(theta))  ->  theta = atan(1/(v * dt/dx))
            v_mm_ns = T.V_DRIFT_UM_NS / 1000.0
            th = np.degrees(np.arctan(1.0 / abs(m * v_mm_ns))) if m else 90.0
            # the ladder runs top-left to bottom-right, so the box goes
            # bottom-left where there is nothing to cover
            pax.text(0.04, 0.06,
                     f'slope {m:+.1f} ns/mm\n'
                     f'$v_{{drift}}$ = {T.V_DRIFT_UM_NS:.0f} µm/ns\n'
                     f'→ θ = {th:.1f}°   (true {angle:.1f}°)',
                     transform=pax.transAxes, ha='left', va='bottom',
                     fontsize=fs * 0.80, color=ink, linespacing=1.5, **A.FONT)
        pax.set_xlabel('strip position  x [mm]', fontsize=fs * 0.85,
                       color=muted, **A.FONT)
        pax.set_ylabel('first arrival time [ns]', fontsize=fs * 0.85,
                       color=muted, **A.FONT)
        pax.tick_params(labelsize=fs * 0.72, colors=muted)
        for sp in pax.spines.values():
            sp.set_color(grid)
        pax.grid(True, color=grid, lw=fs * 0.04, alpha=0.8)
        pax.set_facecolor('none')

        # colour bar: the same scale the 3-D drift columns use
        cax = fig.add_axes([lx0, (foot + 0.135 * h) / H,
                            0.80 * lad_w / W, 0.020 * h / H])
        norm = mcolors.Normalize(0, T.drift_time_ns(T.DRIFT_MM))
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                     orientation='horizontal')
        cax.tick_params(labelsize=fs * 0.66, colors=muted)
        cax.set_xlabel('drift time to the mesh [ns]  =  depth in the gap',
                       fontsize=fs * 0.78, color=muted, **A.FONT)
        for sp in cax.spines.values():
            sp.set_color(grid)

    cap = (
        f'A muon crosses the {T.DRIFT_MM:.0f} mm drift gap at {angle:.0f}° and '
        f'leaves {len(clusters)} primary ionisation clusters '
        f'(~{T.CLUSTERS_PER_CM:.0f}/cm in Ar/isobutane).  Each drifts straight '
        f'down at v = {T.V_DRIFT_UM_NS:.0f} µm/ns, so its arrival time at the '
        f'mesh measures the depth it was created at: {T.drift_time_ns(T.DRIFT_MM):.0f} ns '
        f'across the full gap.  Transverse diffusion spreads each cloud by '
        f'σ_T ≈ {T.sigma_t_mm(T.DRIFT_MM):.1f} mm over the full drift, which is '
        f'why a cluster lights more than one {T.STRIP_PITCH_MM:.3f} mm strip.  '
        f'v_drift and σ_T are the Garfield++/Magboltz values for the mixture '
        f'the bench runs (Ar/iso 95/5 + ~1% H₂O at {T.E_DRIFT_V_CM:.0f} V/cm); '
        f'the gap, pitch and strip count are the detector\'s own.  The fit '
        f'uses the FIRST arrival on each strip, which is a deliberately simple '
        f'estimator and carries a small bias -- the real reconstruction fits '
        f'the waveforms forward (wft/) for exactly that reason.')
    cfs = fs * 0.66
    import textwrap
    pad = 0.026 * W
    ncols = max(60, int((W - 2 * pad) / (cfs * dpi / 72.0 * 0.52)))
    ax.text(pad, H - foot * 0.52, textwrap.fill(cap, ncols), ha='left',
            va='center', fontsize=cfs, color=muted, linespacing=1.6, **A.FONT)

    fig.savefig(out_base + '.png', dpi=dpi, facecolor=page)
    fig.savefig(out_base + '.pdf', facecolor=page)
    plt.close(fig)
    print(f'  wrote {out_base}.png/.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--angle', type=float, default=T.TRACK_ANGLE_DEG)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--no-ladder', action='store_true')
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()

    size = (1000, 820) if args.draft else (2100, 1720)
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    for theme in themes:
        out = os.path.join(FIG, f'microtpc_{theme}.png')
        clusters, hits = render_3d(theme, size, not args.draft, out,
                                   args.angle, args.seed)
        compose(out, clusters, hits,
                os.path.join(FIG, f'microtpc_{theme}_labelled'), theme,
                args.angle, with_ladder=not args.no_ladder)


if __name__ == '__main__':
    main()
