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
import shutil
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
SLIDE_IMG = os.path.join(HERE, 'slides', 'assets', 'img')

VIEW = dict(pos=(52, -78, 40), focal=(0, 0, 13), up=(0, 0, 1), angle=34.0)


def render_3d(theme, size, ssaa, out, angle, seed):
    cmap = S.microtpc_cmap(theme)
    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=2.0)
    T.add_chamber(p, theme)
    a, b, clusters = T.make_event(angle_deg=angle, seed=seed)
    hits = T.add_event(p, a, b, clusters, cmap=cmap)
    S.add_light_rig(p, np.array([0, 0, 14]), 34.0, theme=theme, shadows=False,
                    up='z')
    p.camera.position = VIEW['pos']
    p.camera.focal_point = VIEW['focal']
    p.camera.up = VIEW['up']
    p.camera.view_angle = VIEW['angle']
    p.renderer.reset_camera_clipping_range()
    S.finish(p, out)
    return clusters, hits


def draw_waveforms(fig, rect, clusters, fs, ink, muted, grid, cmap):
    """Stacked per-strip waveforms -- what the DAQ actually records.

    Each trace is one strip: the primaries that land on it, each folded with
    the MEASURED single-electron response of the plane, sampled at the bench's
    own 32 x 60 ns.  The pulse walking steadily across strips is the micro-TPC
    signature; no fit, just the raw signals.
    """
    r = T.strip_waveforms(clusters)
    if r is None:
        return False
    ks, xs, wf, ts = r
    ax = fig.add_axes(rect, facecolor='none')

    tmax = T.drift_time_ns(T.DRIFT_MM)
    peak = np.max(wf, axis=1)
    scale = 0.85 * T.STRIP_PITCH_MM * 6.0 / max(peak.max(), 1e-9)
    for i in np.argsort(xs):
        t_pk = ts[int(np.argmax(wf[i]))]
        col = cmap(np.clip(t_pk / tmax, 0, 1))
        base = xs[i]
        # lw 0.055 -> 0.085 (2026-08-18, Dylan): this panel is projected, and
        # a hairline trace in a dark colour is the first thing a projector
        # loses.  The traces overlap at ~6 strip pitches of offset and still
        # read as separate at this weight.
        ax.plot(ts, base + wf[i] * scale, color=col, lw=fs * 0.085,
                zorder=3, solid_capstyle='round')
        ax.axhline(base, color=grid, lw=fs * 0.018, zorder=1)
    # Trim the dead end of the window: the last few hundred ns of the 32-sample
    # window carry nothing but baseline, and on a slide that is width the pulses
    # want.  The cut is data-driven (last sample above 2 % of the biggest peak),
    # so it cannot silently hide a late pulse.
    thr = 0.02 * peak.max()
    live = np.where(np.max(wf, axis=0) > thr)[0]
    if len(live):
        ax.set_xlim(ts[0], min(ts[-1], ts[min(live[-1] + 2, len(ts) - 1)]))
    ax.set_xlabel(f'time  [ns]   ({T.N_SAMPLES:d} samples x '
                  f'{T.SAMPLE_NS:.0f} ns)', fontsize=fs * 0.85,
                  color=muted, **A.FONT)
    ax.set_ylabel('strip position  x [mm]   (traces offset)',
                  fontsize=fs * 0.85, color=muted, **A.FONT)
    ax.tick_params(labelsize=fs * 0.72, colors=muted)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.grid(True, axis='x', color=grid, lw=fs * 0.04, alpha=0.8)
    # "measured response (det3)" came off 2026-08-20 (Dylan): the slide now
    # says in HTML that the whole figure is a simulation, and a line claiming a
    # MEASURED response inside a simulated event is the one thing on it that can
    # be misread as data.
    ax.text(0.0, 1.012,
            'each trace = one strip',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=fs * 0.74, color=ink, **A.FONT)
    return True


def compose(png, clusters, hits, out_base, theme, angle, dpi=300,
            with_ladder=True, right='ladder', bare=False):
    """Render on the left, and on the right either the strip-time ladder or
    the stacked waveforms.

    ``bare`` is the DECK COPY (2026-08-17, Dylan: "remove both the figure
    caption and footer to make the visualization larger").  The title band and
    the caption paragraph together were 36 % of this figure's height, and on the
    slide they are both redundant: the slide carries its own title in HTML type,
    and a six-line caption on a projected slide is text nobody reads while the
    speaker is talking.  What the caption was actually load-bearing for -- the
    operating point, and that v_drift is MEASURED rather than assumed -- is
    burned onto the render instead, in two lines.  The report still gets the
    fully titled and captioned version.
    """
    img = np.asarray(Image.open(png).convert('RGB'))
    h, w = img.shape[:2]

    cmap = S.microtpc_cmap(theme)
    ink = '#f2f5f9' if theme == 'dark' else '#141b24'
    muted = '#9aa7b6' if theme == 'dark' else '#5d6874'
    page = '#0a0d13' if theme == 'dark' else '#ffffff'
    grid = '#2a3440' if theme == 'dark' else '#e3e8ee'

    with_ladder = right in ('ladder', 'waveforms')
    lad_w = int(w * (0.62 if with_ladder else 0.0))
    head = int((0.012 if bare else 0.12) * h)
    # bare still needs a sliver of foot: the colour bar's own axis label hangs
    # below the bar, and at head=foot=0 it is clipped by the page edge.
    foot = int((0.075 if bare else 0.24) * h)
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

    if not bare:
        ax.text(0.026 * W, head * 0.38, 'Micro-TPC operation',
                ha='left', va='center', fontsize=fs * 1.95, color=ink,
                fontweight='bold', **A.FONT)
        ax.text(0.026 * W, head * 0.76,
                'Simulated event, measured detector constants — one MX17 '
                'chamber measures a track angle from a single plane',
                ha='left', va='center', fontsize=fs * 0.95, color=muted,
                **A.FONT)
    else:
        # the operating point, on the render, where the caption used to say it.
        # Top-left of the 3-D panel is empty at this camera (the chamber sits
        # centre-right), so this costs the picture nothing.  Kept to three SHORT
        # lines on purpose: the muon enters the frame at ~27 % of the panel
        # width, and a long line runs straight into it.  NOTE ax is in pixels
        # from the TOP (ylim is inverted), so the render's top edge is ``head``,
        # not ``foot`` -- getting that wrong pushes the block down onto the track.
        ax.text(0.030 * W, head + 0.030 * h,
                f'{T.DRIFT_MM:.0f} mm gap  ·  {T.E_DRIFT_V_CM:.0f} V/cm\n'
                f'v = {T.V_DRIFT_UM_NS:.1f} µm/ns\n'
                f'{T.drift_time_ns(T.DRIFT_MM):.0f} ns full transit',
                ha='left', va='top', fontsize=fs * 0.90, color=muted,
                linespacing=1.7, **A.FONT)

    # The right panel's own x-label needs room ABOVE the colour bar; the two
    # collided until 2026-08-17 (the label was drawn onto the bar's top edge).
    cb_y, pan_y = 0.045, (0.20 if bare else 0.235)
    pan_h = (0.975 if bare else 0.90) - pan_y

    if right == 'waveforms':
        lx0 = (w + 0.085 * lad_w) / W
        draw_waveforms(fig, [lx0, (foot + pan_y * h) / H,
                             0.80 * lad_w / W, pan_h * h / H],
                       clusters, fs, ink, muted, grid, cmap)
        cax = fig.add_axes([lx0, (foot + cb_y * h) / H,
                            0.80 * lad_w / W, 0.018 * h / H])
        norm = mcolors.Normalize(0, T.drift_time_ns(T.DRIFT_MM))
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                     orientation='horizontal')
        cax.tick_params(labelsize=fs * 0.66, colors=muted)
        cax.set_xlabel('drift time to the mesh [ns]  =  depth in the gap',
                       fontsize=fs * 0.78, color=muted, **A.FONT)
        for sp in cax.spines.values():
            sp.set_color(grid)

    elif with_ladder:
        xs, ts = T.strip_ladder(hits)
        lx0 = (w + 0.085 * lad_w) / W
        pax = fig.add_axes([lx0, (foot + 0.30 * h) / H,
                            0.80 * lad_w / W, 0.60 * h / H], facecolor='none')
        pax.scatter(xs, ts, s=fs * 2.2, c=ts, cmap=cmap,
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
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                     orientation='horizontal')
        cax.tick_params(labelsize=fs * 0.66, colors=muted)
        cax.set_xlabel('drift time to the mesh [ns]  =  depth in the gap',
                       fontsize=fs * 0.78, color=muted, **A.FONT)
        for sp in cax.spines.values():
            sp.set_color(grid)

    if bare:
        fig.savefig(out_base + '.png', dpi=dpi, facecolor=page)
        fig.savefig(out_base + '.pdf', facecolor=page)
        plt.close(fig)
        print(f'  wrote {out_base}.png/.pdf')
        return

    cap = (
        f'A muon crosses the {T.DRIFT_MM:.0f} mm drift gap at {angle:.0f}° and '
        f'leaves {len(clusters)} primary ionisation clusters '
        f'(~{T.CLUSTERS_PER_CM:.0f}/cm in Ar/isobutane).  Each drifts straight '
        f'down at v = {T.V_DRIFT_UM_NS:.1f} µm/ns, so its arrival time at the '
        f'mesh measures the depth it was created at: {T.drift_time_ns(T.DRIFT_MM):.0f} ns '
        f'across the full gap.  Transverse diffusion spreads each cloud by '
        f'σ_T ≈ {T.sigma_t_mm(T.DRIFT_MM):.1f} mm over the full drift, which is '
        f'why a cluster lights more than one {T.STRIP_PITCH_MM:.3f} mm strip.  '
        f'v_drift is the value MEASURED for this detector (det3, wft '
        f'calibration bundle for the 6-22 reference run); σ_T is the '
        f'Garfield++/Magboltz value for the mixture the bench runs '
        f'(Ar/iso 95/5 + ~1% H₂O at {T.E_DRIFT_V_CM:.0f} V/cm), whose own '
        f'v_drift of ~34 µm/ns agrees with the measurement.  The gap, pitch '
        f'and strip count are the detector\'s own.')
    if right == 'waveforms':
        cap += (
            '  On the right, every strip\'s signal: each primary\'s charge '
            'share folded with the MEASURED single-electron response of this '
            'plane (det3, wft calibration bundle) and sampled at the bench\'s '
            f'own {T.N_SAMPLES:d} x {T.SAMPLE_NS:.0f} ns.  The pulse walking '
            'steadily across strips is the micro-TPC signature -- no fit, just '
            'the raw signals.')
    else:
        cap += (
            '  The fit uses the FIRST arrival on each strip, which is a '
            'deliberately simple estimator and carries a small bias -- the '
            'real reconstruction fits the waveforms forward (wft/) for exactly '
            'that reason.')
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
    ap.add_argument('--right', default='ladder',
                    choices=['ladder', 'waveforms', 'none'],
                    help="what goes beside the render: the strip-time ladder, "
                         "the stacked per-strip waveforms, or nothing")
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--no-slide', action='store_true',
                    help='skip the deck copy (light theme, --right waveforms)')
    args = ap.parse_args()

    size = (1000, 820) if args.draft else (2100, 1720)
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    for theme in themes:
        tag = '' if args.right == 'ladder' else f'_{args.right}'
        out = os.path.join(FIG, f'microtpc{tag}_{theme}.png')
        clusters, hits = render_3d(theme, size, not args.draft, out,
                                   args.angle, args.seed)
        compose(out, clusters, hits,
                os.path.join(FIG, f'microtpc{tag}_{theme}_labelled'), theme,
                args.angle, right=args.right)

        # the deck copy: the WAVEFORM variant, no type bands.  The deck asks for
        # the raw signals rather than the ladder fit (2026-08-17) -- the ladder
        # is the estimator the forward fit exists to replace, so showing it as
        # "what the chamber records" undersells the next three slides.
        if (theme != 'light' or args.right != 'waveforms'
                or args.no_slide or args.draft):
            continue
        bare = os.path.join(FIG, 'microtpc_slide')
        compose(out, clusters, hits, bare, theme, args.angle,
                right=args.right, bare=True)
        os.makedirs(SLIDE_IMG, exist_ok=True)
        dst = os.path.join(SLIDE_IMG, 'microtpc.png')
        shutil.copyfile(bare + '.png', dst)
        print(f'  wrote {dst}')


if __name__ == '__main__':
    main()
