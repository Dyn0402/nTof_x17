#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chamber.py -- the exploded MX17 chamber figure.

    ../.venv/bin/python make_chamber.py [--theme light|dark] [--draft]

Writes figures/chamber_exploded_<theme>.png and the labelled PNG/PDF.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geometry as G           # noqa: E402
import style as S              # noqa: E402
import annotate as A           # noqa: E402
import scenes_chamber as C     # noqa: E402

FIG = os.path.join(HERE, 'figures')

# tall, narrow stack: a portrait frame wastes far less than a landscape one
VIEW = dict(pos=(178, -205, 138), focal=(0, 0, 56), up=(0, 0, 1), angle=33.0)

CAPTION = (
    'One MX17 chamber, layers separated along the drift axis.  Strip count and '
    'pitch (512 strips, 0.7785 mm) are from the strip map; the 30 mm drift gap '
    'and the resistive-strip readout are from the run config; the 150 um '
    'amplification gap is the value the Garfield++ work uses '
    '(garfield_sim/mm_config.py).  The window drawn is 30 mm of a 400 mm '
    'chamber, at the real strip pitch.  Layer separations and layer thicknesses '
    'are drawn for legibility -- a 150 um gap under a 30 mm one is otherwise '
    'invisible -- so do not read a scale off the vertical axis.')


def render(theme, size, ssaa, out):
    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=6.0)
    anchors = C.build(p)
    S.add_light_rig(p, np.array([0, 0, 56]), 70.0, theme=theme,
                    shadows=False, up='z')
    for k, v in VIEW.items():
        if k == 'pos':
            p.camera.position = v
        elif k == 'focal':
            p.camera.focal_point = v
        elif k == 'up':
            p.camera.up = v
        else:
            p.camera.view_angle = v
    p.renderer.reset_camera_clipping_range()
    px = A.project(p, anchors)
    S.finish(p, out)
    return px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()

    size = (800, 1000) if args.draft else (1750, 2200)
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]

    for theme in themes:
        out = os.path.join(FIG, f'chamber_exploded_{theme}.png')
        px = render(theme, size, not args.draft, out)

        text = {n: lab for n, z, t, lab in C.layers()}
        order = ['cathode', 'gas', 'mesh', 'resist', 'strips_x', 'strips_y',
                 'pcb']
        items = [(k, text[k]) for k in order if k in px]
        labels = A.column_labels(px, items, size, side='right', gutter=0.44)

        A.compose(out, labels, os.path.join(FIG,
                                            f'chamber_exploded_{theme}_labelled'),
                  title='MX17 resistive micro-TPC chamber',
                  subtitle='Layer stack, exploded along the drift axis, with a '
                           'muon and its drifting ionisation',
                  caption=CAPTION, theme=theme, gutter=0.44,
                  header=0.11, footer=0.17)


if __name__ == '__main__':
    main()
