#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sps.py -- render the SPS H4 beam-test setup.

    ../.venv/bin/python make_sps.py [--theme light|dark] [--views hero,side,beam]
                                    [--no-mx17] [--no-spot] [--size W H]
                                    [--draft]

Writes figures/sps_<view>_<theme>.png (and the annotated .pdf if --annotate).
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
import scenes_sps as X        # noqa: E402
import annotate as A          # noqa: E402

FIG = os.path.join(HERE, 'figures')

# Camera presets: (position, focal_point, up, view_angle)
CENTER = np.array([0.0, 300.0, 685.0])

# The readout planes face UPSTREAM (into the beam), so every camera that is
# meant to show a readout face has to sit upstream too -- i.e. at negative z,
# looking downstream.  The beam then runs away from the viewer, which is also
# the right way round for a figure read left to right.
VIEWS = {
    # three-quarter hero from upstream-left and above
    'hero': dict(pos=(-2300, 1450, -1700), focal=(-20, 330, 620), up=(0, 1, 0),
                 angle=27.0),
    # down the rail from upstream, the stations nested inside each other
    'beam': dict(pos=(-300, 470, -2700), focal=(0, 330, 700), up=(0, 1, 0),
                 angle=24.0),
    # near-orthographic side elevation: the spacing figure
    'side': dict(pos=(-4900, 1000, -1150), focal=(0, 330, 660), up=(0, 1, 0),
                 angle=17.0),
}


# Everything the SPS scene can show.  ``build(show=...)`` takes a subset, which
# is what the build-up sequence steps through.
PARTS = ('table', 'urwell', 'p2', 'mx17', 'tracks')


def build(theme='light', mx17=False, spot=True, tracks=True, envelope=False,
          size=(2600, 1750), ssaa=True, shadows=True, show=PARTS,
          real_tracks=True):
    show = set(show)
    if not mx17:
        show.discard('mx17')
    if not tracks:
        show.discard('tracks')

    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=70.0)

    pads_lab, sectors, pad_map = X.load_pads_lab()
    lo, hi = G.P2_INSTRUMENTED_SECTORS
    live = (sectors >= lo) & (sectors <= hi)

    illum = None
    if spot:
        s = X.load_illumination('P2_MID')
        if s is not None:
            illum = s.reindex(pad_map.channel_id.values).fillna(0.0).values[live]
            illum = np.clip(illum / max(illum.max(), 1e-9), 0, 1)

    if 'table' in show:
        X.add_table(p, theme, legs=False)

    anchors, outlines = {}, []
    for st in G.SPS_STATIONS:
        if st.kind not in show:
            continue
        if st.kind == 'p2':
            parts = X.add_p2(
                p, st.z, pads_lab, sectors, x=st.x, y=st.y,
                spot=illum if (st.name == 'P2_MID' and illum is not None)
                else None)
            anchors[st.name] = (st.x,
                                G.P2_APEX_HEIGHT - G.P2_R_ACTIVE[1] - 40 + st.y,
                                st.z)
        elif st.kind == 'urwell':
            parts = X.add_urwell(p, st.z, x=st.x, y=st.y)
            anchors[st.name] = (st.x,
                                G.SPS_BEAM_HEIGHT + G.URW_PCB_MM / 2 + 40 + st.y,
                                st.z)
        else:
            parts = X.add_mx17(p, st.z, yaw=st.yaw, x=st.x, y=st.y)
            anchors[st.name] = (st.x,
                                G.SPS_BEAM_HEIGHT + G.MX17_PCB_MM / 2
                                + G.MX17_FRAME_MM + 20 + st.y, st.z)
        outlines.append(parts['outline'])

    if shadows:
        S.add_ground_shadows(p, outlines, 0.0, plane_axis='y', up='y',
                             theme=theme)

    if envelope:
        X.add_beam_envelope(p)
    if 'tracks' in show:
        # real measured tracks when the extraction has been run, the sampled
        # beam otherwise -- so the figure builds either way
        trk = X.real_beam_tracks(n=11) if real_tracks else None
        X.add_tracks(p, trk if trk is not None else X.beam_tracks(n=11),
                     radius=1.9)

    S.add_light_rig(p, CENTER, 900.0, theme=theme, shadows=False)
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
    ap.add_argument('--theme', default='light', choices=['light', 'dark', 'both'])
    ap.add_argument('--views', default='hero,beam,side')
    ap.add_argument('--mx17', action='store_true',
                    help='include MX17 "Detector E" at z = 1155 mm')
    ap.add_argument('--no-spot', action='store_true')
    ap.add_argument('--no-tracks', action='store_true')
    ap.add_argument('--envelope', action='store_true',
                    help='draw the trigger-acceptance envelope')
    ap.add_argument('--no-shadows', action='store_true')
    ap.add_argument('--size', nargs=2, type=int, default=[2600, 1750])
    ap.add_argument('--draft', action='store_true',
                    help='fast, small, no anti-aliasing')
    args = ap.parse_args()

    size = (1100, 750) if args.draft else tuple(args.size)
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]

    for theme in themes:
        for view in args.views.split(','):
            tag = '_mx17' if args.mx17 else ''
            out = os.path.join(FIG, f'sps_{view}{tag}_{theme}.png')
            render(view, theme, out,
                   mx17=args.mx17, spot=not args.no_spot,
                   tracks=not args.no_tracks,
                   envelope=args.envelope,
                   shadows=not args.no_shadows,
                   size=size, ssaa=not args.draft)


if __name__ == '__main__':
    main()
