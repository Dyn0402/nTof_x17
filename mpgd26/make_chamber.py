#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chamber.py -- the exploded MX17 chamber figure.

    ../.venv/bin/python make_chamber.py [--theme light|dark] [--draft]

Writes figures/chamber_exploded_<theme>.png, the labelled PNG/PDF, and (light
theme only) the deck copy figures/chamber_exploded_slide.png ->
slides/assets/img/chamber_exploded.png.

LANDSCAPE since 2026-08-17 (Dylan).  It used to be a tall portrait frame with a
square window on the chamber and a label column in a gutter to the right of the
render.  On the slide that shape fought the layout twice: the slide gives this
figure ~62 % of the page width and rather less of its height, so a portrait
figure is height-limited -- it pays for width it cannot use and the layers come
out small -- and the label gutter then spends a third of what is left on white
space.  Now:

  * the window on the chamber is a RECTANGLE (scenes_chamber.WIN_MM), so the
    layers run the width of the frame,
  * the labels sit ON the render, down the LEFT side, next to the layer they
    name (annotate.side_labels, the same treatment the EAR2 and target figures
    use),
  * and the deck copy drops the title/subtitle/caption bands, because the slide
    carries its own title and the caption belongs in the speaker's mouth.

The report still gets the fully titled and captioned version.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geometry as G           # noqa: E402
import style as S              # noqa: E402
import annotate as A           # noqa: E402
import scenes_chamber as C     # noqa: E402

FIG = os.path.join(HERE, 'figures')
SLIDE_IMG = os.path.join(HERE, 'slides', 'assets', 'img')

# Landscape view of a wide window.  The focal point is pushed to -x so the
# stack sits right of centre and leaves the left third of the frame clear for
# the labels -- that empty band is the layout, not slack.
# angle/focal retuned 2026-08-17 with the 60 -> 44 mm zoom (scenes_chamber.
# WIN_MM): a narrower window at the same view angle would simply have shrunk
# the stack inside an unchanged frame, which is not what "zoom in" means.  17.8
# -> 16.6 deg keeps the stack the same size across the frame; the focal point
# came down 36 -> 32 because the drift box, whose 30 mm is REAL and does not
# scale with the window, is now a larger share of the object's height.
VIEW = dict(pos=(78, -218, 130), focal=(-7, 0, 32), up=(0, 0, 1), angle=16.6)

SIZE = (2400, 1980)
DRAFT_SIZE = (900, 743)

# The turntable animation (make_anim.py 'turn_chamber') needs its own camera:
# VIEW deliberately puts the stack off-centre to leave the left of the frame for
# the labels, and an off-centre focal point makes a spinning object swing across
# the frame.  It also needs a landscape frame now that the window is a rectangle.
ANIM_VIEW = dict(VIEW, focal=(0, 0, 36), angle=25.0)
ANIM_SIZE = (1300, 1000)

# label text size as a fraction of the canvas width; the on-render layout can
# afford more than the gutter one could
STYLE = dict(text=0.0165)

CAPTION = (
    'One MX17 chamber, layers separated along the drift axis.  The readout side '
    'is the as-built board, re-sourced from MX17_Geant (shared/'
    'MX17ModuleGeometry.hh and the gerbers): 0.68 mm L4 pads on a 0.78 mm grid '
    'under 0.5 mm L5/L6 strips of the same pitch, and over them the screen-'
    'printed ESL resistive film -- 550 um strips with 250 um gaps, so a 0.80 mm '
    'pitch of its own, not the readout pitch.  The 30 mm drift gap and the '
    'resistive-strip readout are from the run config; the 150 um amplification '
    'gap is the Garfield++ value (garfield_sim/mm_config.py); the mesh weave is '
    'a placeholder.  The window drawn is %.0f x %.0f mm of a 400 mm chamber, at '
    'the real pitch -- everything in the plane is real.  Thicknesses along the '
    'drift axis are exaggerated, and not by a common factor (a 10 um film under '
    'a 30 mm gap is otherwise invisible), so no scale can be read off the '
    'vertical axis.  Colours and layer numbers are those of the board-peel '
    'figure this sits beside.'
    % C.WIN_MM)

ORDER = ['cathode', 'gas', 'mesh', 'resist', 'pads', 'strips_y',
         'strips_x', 'pcb']


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


def laid_out(px, size):
    """Every label on the left of the render, next to its own layer."""
    A.TEXT_FRAC = STYLE['text']
    text = {n: lab for n, z, t, lab in C.layers()}
    items = [(k, text[k], 'left') for k in ORDER if k in px]
    return list(A.side_labels(px, items, size, x_pad=0.012).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--no-slide', action='store_true')
    args = ap.parse_args()

    size = DRAFT_SIZE if args.draft else SIZE
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]

    for theme in themes:
        out = os.path.join(FIG, f'chamber_exploded_{theme}.png')
        px = render(theme, size, not args.draft, out)
        labels = laid_out(px, size)

        A.compose(out, labels,
                  os.path.join(FIG, f'chamber_exploded_{theme}_labelled'),
                  title='MX17 resistive micro-TPC chamber',
                  subtitle='Layer stack, exploded along the drift axis, with a '
                           'muon and its drifting ionisation',
                  caption=CAPTION, theme=theme, gutter=0.0,
                  header=0.13, footer=0.20)

        if theme != 'light' or args.no_slide or args.draft:
            continue
        # the deck copy: same render, same labels, no type bands
        bare = os.path.join(FIG, 'chamber_exploded_slide')
        A.compose(out, labels, bare, title=None, subtitle=None, caption=None,
                  theme=theme, gutter=0.0, header=0.012, footer=0.012)
        os.makedirs(SLIDE_IMG, exist_ok=True)
        dst = os.path.join(SLIDE_IMG, 'chamber_exploded.png')
        shutil.copyfile(bare + '.png', dst)
        print(f'  wrote {dst}')


if __name__ == '__main__':
    main()
