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
          show=PARTS):
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
        if structure:
            B.add_shelf(p, z, G.SCINT_MM / 2)
        if 'scint' not in show:
            continue
        parts = B.add_scintillator(p, z, pmt_side=+1 if side == 'top' else -1)
        anchors[f'scint_{side}'] = (0.0, -G.SCINT_MM / 2, z)
        outlines.append(parts['outline'])

    for name, z in G.BENCH_M3_Z.items():
        if structure:
            B.add_shelf(p, z, G.M3_FRAME_MM / 2)
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
        if structure:
            B.add_shelf(p, z, G.MX17_PCB_MM / 2)
        if 'dut' not in show:
            continue
        if kind == 'mx17':
            parts = B.add_mx17(p, z)
            anchors[slot] = (0.0, -(G.MX17_PCB_MM / 2 + G.MX17_FRAME_MM), z)
        elif kind == 'p2':
            if pads_lab is None:
                pads_lab, sectors, _ = SPS.load_pads_lab()
            parts = B.add_p2_flat(p, z, pads_lab, sectors)
            anchors[slot] = (0.0, -280.0, z)
        else:
            continue
        outlines.append(parts['outline'])

    if shadows:
        S.add_ground_shadows(p, outlines, B.FLOOR_Z, plane_axis='z', up='z',
                             theme=theme, opacity=0.045)

    if 'tracks' in show:
        B.add_tracks(p, B.cosmic_tracks(n=7), radius=4.6)

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
    ap.add_argument('--size', nargs=2, type=int, default=[2200, 2600])
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()

    slots = tuple(s.strip() for s in args.slots.split(','))
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
                   size=size, ssaa=not args.draft)


if __name__ == '__main__':
    main()
