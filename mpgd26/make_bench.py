#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_bench.py -- render the Saclay cosmic test bench.

    ../.venv/bin/python make_bench.py [--theme light|dark|both]
                                      [--views hero,side,cut]
                                      [--slots mx17,mx17 | p2,mx17 | ...]
                                      [--size W H] [--draft]

``--slots`` says what sits in the two test levels, lower (P1) first:
``mx17``, ``p2`` or ``none``.  ``p2,mx17`` is the configuration that
``mx17_det3_p2_det1_overnight_6-27-26`` actually ran.

Writes figures/bench_<view>_<slots>_<theme>.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geometry as G          # noqa: E402
import style as S             # noqa: E402
import scenes_bench as B      # noqa: E402
import scenes_sps as SPS      # noqa: E402
import annotate as A          # noqa: E402

FIG = os.path.join(HERE, 'figures')
CENTER = np.array([0.0, 0.0, 640.0])

VIEWS = {
    # three-quarter hero, camera a little above the mid-plane of the stack
    'hero': dict(pos=(4250, -2150, 2550), focal=(0, 0, 640), up=(0, 0, 1),
                 angle=23.0),
    # near-orthographic elevation: the stacking figure
    'side': dict(pos=(-1500, -7400, 2150), focal=(30, 0, 660), up=(0, 0, 1),
                 angle=16.4),
    # low three-quarter, looking up the stack
    'low': dict(pos=(2500, -2300, 260), focal=(0, 0, 700), up=(0, 0, 1),
                angle=30.0),
}


# Everything the bench scene can show.  ``build(show=...)`` takes a subset,
# which is what the build-up sequence steps through.
PARTS = ('structure', 'scint', 'm3', 'dut', 'tracks')


def build(theme='light', slots=('mx17', 'mx17'), tracks=True,
          size=(2200, 2600), ssaa=True, shadows=True, structure=True,
          show=PARTS, align=None, rays=None, n_tracks=7):
    """``align`` maps 'P1'/'P2' to an alignment.json path -- the chamber is then
    drawn where the fit says it is, not at nominal.  ``rays`` is an
    ``m3_tracking_root*`` directory; if given, the muons are real reconstructed
    tracks instead of sampled ones."""
    align = align or {}
    show = set(show)
    if not structure:
        show.discard('structure')
    if not tracks:
        show.discard('tracks')

    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=60.0)

    anchors, outlines = {}, []
    structure = 'structure' in show

    if structure:
        B.add_structure(p, theme)
        B.add_level_rails(p)

    for side, z in G.BENCH_SCINT_Z.items():
        if 'scint' not in show:
            continue
        parts = B.add_scintillator(p, z)   # both PMTs on -y
        anchors[f'scint_{side}'] = (0.0, -G.SCINT_MM / 2, z)
        outlines.append(parts['outline'])

    for name, z in G.BENCH_M3_Z.items():
        if 'm3' not in show:
            continue
        parts = B.add_m3(p, z)
        outlines.append(parts['outline'])
    if 'm3' in show:
        anchors['m3_top'] = (0.0, -G.M3_FRAME_MM / 2,
                             (G.BENCH_M3_Z['m3_top_bot']
                              + G.BENCH_M3_Z['m3_top_top']) / 2)
        anchors['m3_bot'] = (0.0, -G.M3_FRAME_MM / 2,
                             (G.BENCH_M3_Z['m3_bot_bot']
                              + G.BENCH_M3_Z['m3_bot_top']) / 2)

    pads_lab = sectors = None
    for slot, kind in zip(('P1', 'P2'), slots):
        z = G.BENCH_DUT_Z[slot]
        if kind == 'none':
            continue
        if 'dut' not in show:
            continue

        # measured position, if an alignment was handed in
        dx = dy = dth = 0.0
        if slot in align and kind != 'mx17':
            # alignment.json is fitted for an MX17 against the M3 reference --
            # detector-local strip coordinates, an MX17 active-area centre, an
            # MX17 z.  None of that transfers to a P2 fan sitting in the same
            # slot, so it is refused rather than silently applied.
            print(f'  {slot}: ignoring alignment -- it is an MX17 fit and this '
                  f'slot holds a {kind}')
        elif slot in align:
            a = G.load_bench_alignment(align[slot])
            dx, dy, dth = a['x'], a['y'], a['theta_deg']
            print(f'  {slot}: measured offset ({dx:+.1f}, {dy:+.1f}) mm, '
                  f'in-plane {dth:.2f} deg')
            # An alignment belongs to one chamber in one slot.  The fit's own z
            # says which slot that was, so a file from the other slot is caught
            # here rather than silently drawing the wrong chamber's offset.
            if a['z'] is not None:
                if abs(a['z'] - z) > 60.0:
                    print(f'    WARNING: this fit put the chamber at '
                          f'z = {a["z"]:.0f} mm, but slot {slot} is at '
                          f'z = {z:.0f} mm -- alignment from the other slot?')
                else:
                    print(f'    (fit z = {a["z"]:.0f} mm vs configured '
                          f'{z:.0f} mm -- the known origin offset)')

        if kind == 'mx17':
            parts = B.add_mx17(p, z, x=dx, y=dy, theta_deg=dth)
            anchors[slot] = (dx, dy - (G.MX17_PCB_MM / 2 + G.MX17_FRAME_MM), z)
        elif kind == 'p2':
            if pads_lab is None:
                pads_lab, sectors, _ = SPS.load_pads_lab()
            parts = B.add_p2_flat(p, z, pads_lab, sectors,
                                  x=dx, y=dy, theta_deg=dth)
            anchors[slot] = (dx, dy - 280.0, z)
        else:
            continue
        outlines.append(parts['outline'])

    if shadows:
        S.add_ground_shadows(p, outlines, B.FLOOR_Z, plane_axis='z', up='z',
                             theme=theme, opacity=0.045)

    if 'tracks' in show:
        trk = B.real_tracks(rays, n=n_tracks) if rays \
            else B.cosmic_tracks(n=n_tracks)
        B.add_tracks(p, trk, radius=4.6)

    S.add_light_rig(p, CENTER, 900.0, theme=theme, shadows=False, up='z')
    return p, anchors


def render(view, theme, out, **kw):
    p, anchors = build(theme=theme, **kw)
    v = VIEWS[view]
    p.camera.position = v['pos']
    p.camera.focal_point = v['focal']
    p.camera.up = v['up']
    p.camera.view_angle = v['angle']
    # VTK keeps the clipping range from the auto-framed camera, so a manual
    # position far outside it renders an empty frame
    p.renderer.reset_camera_clipping_range()
    px = A.project(p, anchors)
    S.finish(p, out)
    return px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--views', default='hero,side')
    ap.add_argument('--slots', default='mx17,mx17',
                    help='lower,upper: mx17 | p2 | none')
    ap.add_argument('--no-tracks', action='store_true')
    ap.add_argument('--no-structure', action='store_true')
    ap.add_argument('--no-shadows', action='store_true')
    ap.add_argument('--align', action='append', default=[],
                    metavar='SLOT=PATH',
                    help='e.g. --align P2=/path/to/alignment.json -- draw that '
                         'chamber where the fit says it is')
    ap.add_argument('--rays', default=None, metavar='DIR',
                    help='an m3_tracking_root* directory: draw REAL '
                         'reconstructed cosmic tracks instead of sampled ones')
    ap.add_argument('--n-tracks', type=int, default=7)
    ap.add_argument('--reference', action='store_true',
                    help='use geometry.BENCH_REFERENCE: the one June run that '
                         'carries both slots\' alignments AND its own M3 rays, '
                         'so the figure comes from a single dataset')
    ap.add_argument('--size', nargs=2, type=int, default=[2200, 2600])
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()

    slots = tuple(s.strip() for s in args.slots.split(','))
    align = dict(a.split('=', 1) for a in args.align)
    rays = args.rays
    if args.reference:
        ref = G.bench_reference_paths()
        if ref is None:
            print('  --reference: bench data disk not mounted; '
                  'falling back to nominal positions and sampled muons')
        else:
            align = {**ref['align'], **align}      # explicit --align still wins
            rays = rays or ref['rays']
            print(f"  reference dataset: {G.BENCH_REFERENCE['run']}"
                  f"/{G.BENCH_REFERENCE['sub_run']}")
    size = (900, 1050) if args.draft else tuple(args.size)
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]

    for theme in themes:
        for view in args.views.split(','):
            tag = '-'.join(slots)
            out = os.path.join(FIG, f'bench_{view}_{tag}_{theme}.png')
            render(view, theme, out, slots=slots,
                   tracks=not args.no_tracks,
                   structure=not args.no_structure,
                   shadows=not args.no_shadows,
                   align=align, rays=rays, n_tracks=args.n_tracks,
                   size=size, ssaa=not args.draft)


if __name__ == '__main__':
    main()
