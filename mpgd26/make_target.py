#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_target.py -- the two Target #3 detail figures, for the backup slides.

    ../.venv/bin/python make_target.py                  # both views
    ../.venv/bin/python make_target.py --only cooling
    ../.venv/bin/python make_target.py --draft          # small and fast

Writes, per view:

  figures/target3_<view>.png                    the bare render
  figures/target3_<view>_labelled.png/.pdf      titled, captioned, labelled
  slides/assets/img/target3_<view>.png          labels only, no title or caption
                                                bands -- what the slides use

Both views use the ON-FIGURE label layout (``annotate.side_labels``): these are
landscape figures with the subject in the middle, so there is background either
side to write on, and a gutter would be dead width.

Unlike ``make_ear2.py`` these are NOT a build-up -- they are two different
pictures of one object, at two scales, so nothing has to overlay anything.  Each
is laid out on its own.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import style as S              # noqa: E402
import annotate as A           # noqa: E402
import scenes_ear2 as E        # noqa: E402
import scenes_target as T      # noqa: E402

FIG = os.path.join(HERE, 'figures')
SLIDE_IMG = os.path.join(HERE, 'slides', 'assets', 'img')

# Azimuth in (0, 90) for the same two reasons as the facility figure: +X reads
# screen-right only for n_z > 0, and the cutaway keeps the half with n.p < 0, so
# the upstream proton beam survives only for n_x > 0.  See make_ear2.py.
VIEWS = {
    'layers': dict(elev=13.0, azim=58.0, dist=6200.0,
                   size=(1900, 1350), margin=1.14,
                   center_scale=T.layers_center_scale,
                   build=T.build_layers),
    # Lower and squarer on the plate: the mechanism is a groove running up the
    # face, and from a high angle you look at the top of the plate instead.
    'cooling': dict(elev=13.0, azim=26.0, dist=5200.0,
                    size=(1900, 1250), margin=1.16,
                    center_scale=T.cooling_center_scale,
                    build=T.build_cooling),
}

LABELS = {
    'layers': [
        ('neutrons', 'neutrons to EAR2, 20 m up'),
        ('vacwin', 'hemispherical Al\nvacuum window'),
        ('moderator', 'EAR2 moderator\n40 mm of water\nin an Al can'),
        ('n2_out', 'warm N₂ out'),
        ('pbplate', '50 mm lead plate'),
        ('nwin', '4 mm steel\nneutron window'),
        ('ear1', 'EAR1 moderator\n(the larger one)'),
        ('plates', '9.85 mm Al anti-creep\nplates — the N₂ channels\nare machined in these'),
        ('thick', '6th slice: 150 mm'),
        ('pwin', 'vessel thinned to\n3 mm: proton window'),
        ('slices', 'slices 1–5: 50 mm each\n600 × 600 mm, pure Pb'),
        ('protons', '20 GeV/c protons'),
        ('vessel', 'AISI 316L vessel\n0.5 bar of N₂'),
        ('yaw', 'target yawed 10°\nto the beam'),
        ('cradle', 'Al cradle: two arteries,\nplenums, flow deflectors'),
        ('n2_in', 'N₂ in at 20 °C\n780 Nm³/h'),
    ],
    'cooling': [
        ('plate', 'anti-creep plate, 9.85 ± 0.05 mm\nEN AW-6082 T6 aluminium'),
        ('flow', 'flow is highest where the\nbeam is: < 40 m/s there,\nup to 87 m/s at the edges'),
        ('channel', 'N₂ up the channels,\n3 mm deep, milled in\nthe aluminium'),
        ('lead_up', 'lead slice, 600 × 600 mm\n(exploded off the plate)'),
        ('wedge', 'a wedge throttles the outer\nchannels, so the flow goes\nwhere the beam is'),
        ('creep', 'and the plate stops the lead\nflowing into the grooves:\n0.64 of the 3 mm, in 2 lifetimes'),
        ('tc', '1 of 6 thermocouples,\ntouching the lead'),
    ],
}

SIDES = {
    'layers': dict(protons='left', pwin='left', yaw='left', vessel='left',
                   n2_in='left', slices='left', pbplate='left',
                   neutrons='right', vacwin='right', moderator='right',
                   nwin='right', plates='right', thick='right', ear1='right',
                   cradle='right', n2_out='right'),
    'cooling': dict(plate='left', lead_up='left', wedge='left', tc='left',
                    flow='right', channel='right', creep='right'),
}

TITLE = {
    'layers': 'The n_TOF spallation target, layer by layer',
    'cooling': 'How 2.7 kW leaves a block of lead that creeps at 135 °C',
}
SUBTITLE = {
    'layers': 'Target #3, installed in Long Shutdown 2 — six lead slices, gas '
              'cooled, with a purpose-built moderator stack facing EAR2',
    'cooling': 'One anti-creep plate between two slices: the channels are '
               'machined in the aluminium, not in the lead',
}

STYLE = dict(text=0.0132)


def caption(view):
    return ('  '.join(T.ASSUMPTIONS) + '  Everything else is from '
            + T.CITATION + '.')


def camera(v):
    e, a = np.radians(v['elev']), np.radians(v['azim'])
    focal, _ = v['center_scale']()
    focal = np.array(focal, float)
    pos = focal + v['dist'] * np.array([np.cos(e) * np.cos(a), np.sin(e),
                                        np.cos(e) * np.sin(a)])
    return pos, focal


def render(name, theme, size, ssaa, out):
    v = VIEWS[name]
    pos, focal = camera(v)
    look = pos - focal
    _, scale = v['center_scale']()
    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=None)
    # the moderator water inside its can inside the vessel is three nested
    # translucent shells -- exactly what VTK's back-to-front ordering gets wrong
    p.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
    anchors = v['build'](p, cut_normal=(look[0], 0.0, look[2]))
    S.add_light_rig(p, focal, scale, theme=theme, shadows=False, up='y')
    p.camera.position = tuple(pos)
    p.camera.focal_point = tuple(focal)
    p.camera.up = (0, 1, 0)
    p.camera.view_angle = 2.0 * np.degrees(
        np.arctan(scale * v['margin'] / 2.0 / v['dist']))
    p.renderer.reset_camera_clipping_range()
    px = A.project(p, anchors)
    S.finish(p, out)
    return px


def laid_out(name, px, size):
    A.TEXT_FRAC = STYLE['text']
    items = [(k, t, SIDES[name][k]) for k, t in LABELS[name]
             if k in px and k in SIDES[name]]
    return list(A.side_labels(px, items, size).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--only', default=None, help='layers | cooling')
    ap.add_argument('--no-slide', action='store_true')
    args = ap.parse_args()

    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    want = None if args.only is None else set(args.only.split(','))

    for theme in themes:
        for name, v in VIEWS.items():
            if want and name not in want:
                continue
            size = (620, 430) if args.draft else v['size']
            base = f'target3_{name}' + ('' if theme == 'light'
                                        else f'_{theme}')
            print(f'{name}  [{len(LABELS[name])} labels]')
            out = os.path.join(FIG, base + '.png')
            px = render(name, theme, size, not args.draft, out)

            A.compose(out, laid_out(name, px, size),
                      os.path.join(FIG, base + '_labelled'),
                      title=TITLE[name], subtitle=SUBTITLE[name],
                      caption=caption(name), theme=theme,
                      gutter=0.0, header=0.150, footer=0.300)

            if theme != 'light' or args.no_slide or args.draft:
                continue
            # the slide copy: the same labels, no title or caption bands, since
            # the slide carries its own
            bare = os.path.join(FIG, base + '_slide')
            A.compose(out, laid_out(name, px, size), bare, title=None,
                      subtitle=None, caption=None, theme=theme, gutter=0.0,
                      header=0.012, footer=0.012)
            os.makedirs(SLIDE_IMG, exist_ok=True)
            dst = os.path.join(SLIDE_IMG, base + '.png')
            shutil.copyfile(bare + '.png', dst)
            print(f'  wrote {dst}')


if __name__ == '__main__':
    main()
