#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ear2.py -- the n_TOF EAR2 vertical beam line, as a build-up sequence.

    ../.venv/bin/python make_ear2.py                    # all three frames
    ../.venv/bin/python make_ear2.py --only full        # just the last one
    ../.venv/bin/python make_ear2.py --draft            # small and fast
    ../.venv/bin/python make_ear2.py --theme dark
    ../.venv/bin/python make_ear2.py --no-slide         # skip the slide copies
    ../.venv/bin/python make_ear2.py --no-onfig         # skip the on-figure layout

Writes, per theme, one set per frame of ``scenes_ear2.STAGE_PARTS``:

  figures/ear2_build_<n>_<tag>_<theme>.png            the bare render
  figures/ear2_build_<n>_<tag>_<theme>_labelled.png   titled, captioned, labelled
                                              .pdf   ... with live text
  slides/assets/img/ear2_beamline_<n>_<tag>.png       the same labels, no title
                                                      or caption bands (light
                                                      theme only) -- the GUTTER
                                                      layout, used on the backup
                                                      slide
  figures/ear2_onfig_<n>_<tag>.png/.pdf               the ON-FIGURE layout: the
  slides/assets/img/ear2_onfig_<n>_<tag>.png          same short labels, but on
                                                      the drawing's own
                                                      background left and right,
                                                      on a wider canvas.  This is
                                                      what slides 5-7 use, since
                                                      they carry no caption and a
                                                      gutter would be dead width

plus the last frame under the names the deck and README have always used:
``figures/ear2_beamline_<theme>{,_labelled}.png`` (the standalone figure; there is
no ``slides/assets/img/ear2_beamline.png`` alias any more -- see the end of
``main``).

The three frames are **interchangeable stills of one picture**: the camera, the
lens, the light rig, the canvas and the drawn scale are all fixed, the parts are
strict subsets (``scenes_ear2.STAGE_PARTS``), and the label column is laid out
ONCE from the full set and then filtered -- so a label never moves between
frames either.  Drop them on three consecutive slides and the beam line builds
itself while you talk.

The camera is a long lens (a ~10 deg vertical field at 34 m) on a
just-under-six-metre-tall subject, so the top of the line does not converge away
from the bottom, and it sits just off the proton axis so that protons enter from
the left.  The scene is cut open on a vertical plane through the beam axis with
the near half removed -- the same cutaway idiom scenes_ntof uses on the He-3
capsule -- so the neutron flux inside the pipe, the lead disks and the
collimator's bore are all visible.
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

FIG = os.path.join(HERE, 'figures')
SLIDE_IMG = os.path.join(HERE, 'slides', 'assets', 'img')

# Elevation / azimuth about the beam axis, azimuth measured from +X towards +Z.
# The azimuth has to satisfy two conditions at once, which is why it is +70 and
# not -70.  With ``n`` the unit vector from the scene towards the camera, VTK's
# screen-right is ``y_hat x n = (n_z, 0, -n_x)``, so **+X is screen-right only
# for n_z > 0, i.e. azimuth in (0, 180)** -- that is what puts the protons on the
# left of the frame travelling right.  And the cutaway removes the half with
# ``n . p > 0``, so the proton beam (upstream, at -X) survives only for
# ``n_x > 0``, i.e. azimuth in (-90, 90).  The overlap is (0, 90).
# The lens is long (a ~10 deg vertical field at 34 m) so that the top of a
# just-under-six-metre-tall subject does not converge away from the bottom.
ELEV, AZIM, DIST = 6.0, 70.0, 34000.0
MARGIN = 1.07                       # of the drawn height, top and bottom


def view_angle():
    """The vertical field that just contains the drawn line, plus a margin."""
    return 2.0 * np.degrees(np.arctan(E.DRAWN_H * MARGIN / 2.0 / DIST))

# Two label sets, because the two outputs are read at completely different
# sizes.  The standalone figure is read at full size (or as the live-text PDF),
# so it carries sentences.  The slide copy is squeezed into ~a quarter of a
# 16:9 frame -- roughly a 3.7x downscale -- so anything set at the standalone
# figure's type size lands at ~6 px and is unreadable from the room.  The slide
# copy therefore gets short tags set 1.7x larger, and lets the slide's own
# bullets carry the sentences.
LABELS = [
    ('to_dump', 'and back into a pipe, on up to the\n'
                'beam dump — through the bunker\n'
                'ceiling at 23.66 m to its entrance at\n'
                '24.73 m, on the roof, above this frame'),
    ('detectors', 'Micromegas micro-TPCs around it — four in\n'
                  'a pinwheel, two of them drawn, in section.\n'
                  '400 mm strip boards, 30 mm of drift gas\n'
                  'facing the sample (support frame not drawn)'),
    ('sample', '³He sample — 19.95 m\n'
               'the beam is ≈ 3 cm FWHM here'),
    ('pipe_end', 'the beam pipe ENDS here, 19.16 m —\n'
                 '≈ 1 m above the floor. Above it the\n'
                 'hall is open and the beam is in air'),
    ('shield', 'polyethylene shielding around the\n'
               'floor penetration, and the lead-disk\n'
               'chamber inside it (from the photo)'),
    ('floor', 'EAR2 floor, 18.16 m above the target'),
    ('collimator', '2nd collimator, 15.0 – 18.0 m\n'
                   'iron + borated PE, bore 70 → 21.8 mm,\n'
                   'then lead disks in the hall'),
    ('gap_lo', '≈ 20 m of vertical flight path\n'
               'the drawing is broken here, and so is\n'
               'everything in it: 1st collimator 7.4 m\n'
               '(1 m of iron, 200 mm bore), sweeping\n'
               'magnet 10.4 m, filter station 11.4 m'),
    ('neutrons', 'Neutrons leave at 90° to the proton\n'
                 'beam and fill the 317 mm pipe'),
    ('moderator', '4 cm of water, over a 5 cm lead plate —\n'
                  'the EAR2 moderator, which sets the\n'
                  'energy resolution of this beam line'),
    ('target', 'Lead spallation target — Target #3\n'
               'six slices, 5 × 50 mm + 150 mm, 600 × 600 mm\n'
               '~300 neutrons per proton'),
    ('protons', '20 GeV/c protons from the CERN PS\n7 ns rms, up to 0.8 Hz'),
]

LABELS_SLIDE = [
    ('to_dump', 'back into a pipe, on\nup to the beam dump\n— 24.73 m'),
    # Deliberately GENERIC (Dylan, 2026-08-11).  This is slide 9 of a 32-slide
    # deck and the experiment has not been introduced yet: "Micromegas trackers"
    # and "³He sample" both name things the audience has not met, and they make
    # the facility slide look like the setup slide.  The words for what these are
    # belong to the setup section, seven slides later; here they only have to say
    # that something is in the beam and something else is watching it.  The
    # standalone figure's own LABELS still carry the full description.
    ('detectors', 'detectors'),
    ('sample', 'sample'),
    ('pipe_end', 'beam pipe ends here\n≈ 1 m above the floor'),
    ('shield', 'PE shielding on the\nfloor + lead disks'),
    ('floor', 'EAR2 floor, 18.16 m'),
    ('collimator', '2nd collimator\n+ lead disks'),
    # Keep every line in this list to ~21 characters. The slide copy's label
    # column is narrow (SLIDE_STYLE['gutter']), and annotate does not wrap: a
    # longer line is silently CLIPPED at the edge of the PNG, which is how the
    # first version of this label lost the words "flight path". The bullet on
    # slide 6 carries the detail; this only has to name the thing.
    ('gap_lo', 'break — ≈ 20 m of\npipe, and the 1st\ncollimator at 7.4 m'),
    ('neutrons', 'neutrons, 90° to p'),
    ('moderator', 'water moderator\n+ lead plate'),
    ('target', 'Lead target'),
    ('protons', '20 GeV/c protons'),
]

# gutter, and label cap height as a fraction of the canvas width, per output
FULL_STYLE = dict(gutter=0.62, text=0.0155)
SLIDE_STYLE = dict(gutter=0.50, text=0.026)

# --------------------------------------------------------------------------- #
# The on-figure variant (2026-08-11, Dylan asked for it to compare)
# --------------------------------------------------------------------------- #
# Same picture and the same short tags, but the labels sit on the figure's own
# background in two columns instead of in a gutter down one side.  Two things
# have to change together for that to work, and neither is optional:
#
#   * the render is WIDER (a gutter buys its own space; on-figure labels have to
#     be given some), so the canvas is ~1740 px against 940, which at the same
#     vertical field just adds empty hall left and right, and
#   * the type is sized against that wider canvas, so TEXT_FRAC comes down to
#     keep the same cap height in pixels as the gutter version has.
#
# Each side has ~1.7 m of drawn hall to write in, i.e. ~16 characters at this
# size -- close to the cap LABELS_SLIDE already works to, which is why this
# variant reuses that label set rather than defining a third one.
#
# text 0.022 -> 0.028 on 2026-08-26 (Dylan: "on all the figures in the
# motivation section ... make the text larger for presentation").  The frame
# goes on the slide 4.68 in wide, so a cap height of 0.022 x 1740 px = 38 px
# projects at ~10.6 pt and 0.028 at ~13.5 pt -- which is where the story beats
# next door now sit (scenes_x17.STORY_FS) and roughly where the slide's own
# bullets are.  This variant can afford it and the GUTTER one cannot: labels
# here run inward across empty hall and simply get closer to the beam line,
# where a gutter label that outgrows its column is silently CLIPPED at the edge
# of the PNG.  Checked frame by frame after the change -- the tightest is
# "20 GeV/c protons", which now reaches the tail of its own proton arrow.
ONFIG_SIZE = (1740, 2050)
ONFIG_STYLE = dict(gutter=0.0, text=0.028)

# Which side each label goes on.  Assigned by hand rather than from the sign of
# the projected anchor: the beam line is a narrow column near the middle of a
# wide frame, so nearly every anchor projects to the same side, and what actually
# matters is that neither column ends up carrying eight labels.  Bottom-to-top
# the left column takes the target end, the break and the end of the pipe; the
# right takes the beam, the collimation and the station.
# ``target`` is on the RIGHT (Dylan, 2026-08-11): its anchor is on the beam axis
# and ``protons`` comes in from the left, so with both on the left their two
# leader lines crossed over each other right under the target.
ONFIG_SIDES = dict(protons='left', gap_lo='left', shield='left',
                   pipe_end='left', detectors='left',
                   target='right', moderator='right', neutrons='right',
                   collimator='right', floor='right', sample='right',
                   to_dump='right')

TITLE = 'The n_TOF EAR2 vertical neutron beam line'
SUBTITLE = ('20 GeV/c protons on the lead spallation target, ~20 m up to the '
            'measuring station in the open beam above the end of the pipe')

# one line per build frame, for the standalone figure's subtitle and caption
STAGE_SUBTITLE = {
    'target': '1 / 5 — 20 GeV/c protons from the CERN PS onto Target #3, and '
              'the water moderator above it',
    'neutrons': '2 / 5 — ~300 neutrons per proton leave at 90° and fill the '
                'vertical pipe',
    'collimation': '3 / 5 — ~20 m up, through the second collimator and the '
                   'lead disks, and the pipe ends a metre above the floor',
    'dump': '4 / 5 — above the experimental space the beam goes back into a '
            'pipe, on up to the dump',
    'station': '5 / 5 — the ³He sample in the open beam at 19.95 m, with the '
               'Micromegas trackers around it',
}


def caption(tag=None):
    lead = ''
    if tag is not None:
        lead = ('Frame ' + STAGE_SUBTITLE[tag].replace('—', 'of the build-up:')
                + '.  The three frames are subsets of one picture: same camera, '
                'same scale, nothing moves.  ')
    return (lead + '  '.join(E.ASSUMPTIONS) + '  Positions, apertures and the '
            'heights quoted for what is above the frame are from ' + E.CITATION
            + '.')


def camera():
    e, a = np.radians(ELEV), np.radians(AZIM)
    focal = np.array(E.scene_center(), float)
    pos = focal + DIST * np.array([np.cos(e) * np.cos(a), np.sin(e),
                                   np.cos(e) * np.sin(a)])
    return pos, focal


def render(theme, size, ssaa, out, show=E.PARTS):
    pos, focal = camera()
    look = pos - focal
    # SSAO is OFF here, unlike the other scenes.  Its radius is in world units,
    # and at a ten-metre scene scale every radius large enough to seat anything
    # lays a diagonal hatch across the flat faces -- visible on the frame
    # uprights, which are plain boxes.  The cutaway does the seating instead.
    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=None)
    # nested translucent shells (the beam envelope inside the pipe inside the
    # shielding) are exactly the case VTK's back-to-front ordering gets wrong
    p.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
    anchors = E.build(p, show=show, cut_normal=(look[0], 0.0, look[2]))
    S.add_light_rig(p, E.scene_center(), E.scene_scale(), theme=theme,
                    shadows=False, up='y')
    p.camera.position = tuple(pos)
    p.camera.focal_point = tuple(focal)
    p.camera.up = (0, 1, 0)
    p.camera.view_angle = view_angle()
    p.renderer.reset_camera_clipping_range()
    px = A.project(p, anchors)
    S.finish(p, out)
    return px


def laid_out(px, labels, size, style, keys):
    """The label column, laid out from the WHOLE set, then filtered to ``keys``.

    This is what keeps the build honest.  ``annotate.column_labels`` pushes
    labels apart until they stop overlapping, so laying out only the labels a
    frame happens to carry would put "Lead target" in a different place on every
    frame -- the one thing a build-up must not do.  Instead the full column is
    solved once (the anchors and the camera are identical for every frame, so the
    solution is too) and each frame draws its subset of it.
    """
    A.TEXT_FRAC = style['text']
    items = [(k, t) for k, t in labels if k in px]
    full = A.column_labels(px, items, size, side='right',
                           gutter=style['gutter'])
    return [d for (k, _), d in zip(items, full) if k in keys]


def laid_out_onfig(px, labels, size, style, keys):
    """The same, for the on-figure two-column variant.  Same honesty rule."""
    A.TEXT_FRAC = style['text']
    items = [(k, t, ONFIG_SIDES[k]) for k, t in labels
             if k in px and k in ONFIG_SIDES]
    full = A.side_labels(px, items, size)
    return [d for k, d in full.items() if k in keys]


def onfig(theme, show, keys, oname, args):
    """Write one on-figure-labelled frame, PNG + PDF + the slide copy."""
    obare = os.path.join(FIG, oname + '_bare.png')
    opx = render(theme, ONFIG_SIZE, not args.draft, obare, show=show)
    A.compose(obare,
              laid_out_onfig(opx, LABELS_SLIDE, ONFIG_SIZE, ONFIG_STYLE, keys),
              os.path.join(FIG, oname), title=None, subtitle=None, caption=None,
              theme=theme, gutter=0.0, header=0.010, footer=0.010)
    os.makedirs(SLIDE_IMG, exist_ok=True)
    shutil.copyfile(os.path.join(FIG, oname + '.png'),
                    os.path.join(SLIDE_IMG, oname + '.png'))
    print(f'  wrote {os.path.join(SLIDE_IMG, oname + ".png")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--size', nargs=2, type=int, default=[940, 2050])
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--only', default=None,
                    help='comma-separated frame tags, e.g. "full" or '
                         '"target,station"')
    ap.add_argument('--no-slide', action='store_true',
                    help='do not write the slides/assets/img/ copies')
    ap.add_argument('--no-onfig', action='store_true',
                    help='skip the on-figure-labels variant (halves the render '
                         'time: it needs a second, wider pass per frame)')
    args = ap.parse_args()

    size = (400, 870) if args.draft else tuple(args.size)
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    want = None if args.only is None else set(args.only.split(','))

    for theme in themes:
        for i, (tag, _, _) in enumerate(E.STAGE_PARTS, start=1):
            if want and tag not in want:
                continue
            show = E.stage_parts(i)
            keys = E.stage_labels(i)
            base = f'ear2_build_{i}_{tag}_{theme}'
            print(f'{i}/{len(E.STAGE_PARTS)}  {tag}  '
                  f'[{len(show)} parts, {len(keys)} labels]')

            out = os.path.join(FIG, base + '.png')
            px = render(theme, size, not args.draft, out, show=show)

            A.compose(out, laid_out(px, LABELS, size, FULL_STYLE, keys),
                      os.path.join(FIG, base + '_labelled'),
                      title=TITLE, subtitle=STAGE_SUBTITLE[tag],
                      caption=caption(tag), theme=theme,
                      gutter=FULL_STYLE['gutter'], header=0.075, footer=0.145)

            # The last frame is also the standalone figure, under the names the
            # README and report have always used.
            if tag == 'station':
                shutil.copyfile(out, os.path.join(
                    FIG, f'ear2_beamline_{theme}.png'))
                for ext in ('.png', '.pdf'):
                    shutil.copyfile(
                        os.path.join(FIG, base + '_labelled' + ext),
                        os.path.join(FIG,
                                     f'ear2_beamline_{theme}_labelled' + ext))

            if theme != 'light' or args.no_slide or args.draft:
                continue

            # the on-figure variant: its own wider render, same parts, same
            # labels, laid out on the background instead of in a gutter
            if not args.no_onfig:
                onfig(theme, show, keys, f'ear2_onfig_{i}_{tag}', args)
                # The last frame goes out TWICE.  The default -- and what the
                # slide uses -- is TWO chambers in section (see
                # scenes_ear2.STATION_ARMS); the alternate is the real four-arm
                # pinwheel, kept because it is the true arrangement and because a
                # reader of ASSUMPTIONS should be able to see what was decided
                # against.  Same camera and same everything else, so the two are
                # interchangeable on the slide by editing one `img src`.
                if tag == 'station':
                    E.STATION_ARMS = 4
                    try:
                        onfig(theme, show, keys, f'ear2_onfig_{i}_{tag}_4arm',
                              args)
                    finally:
                        E.STATION_ARMS = 2

            # the slide has its own title, caption and bullets, so its copy
            # carries only short tags -- see LABELS_SLIDE
            bare = os.path.join(FIG, f'ear2_beamline_{i}_{tag}_slide')
            A.compose(out, laid_out(px, LABELS_SLIDE, size, SLIDE_STYLE, keys),
                      bare, title=None, subtitle=None, caption=None,
                      theme=theme, gutter=SLIDE_STYLE['gutter'],
                      header=0.012, footer=0.012)
            os.makedirs(SLIDE_IMG, exist_ok=True)
            dst = os.path.join(SLIDE_IMG, f'ear2_beamline_{i}_{tag}.png')
            shutil.copyfile(bare + '.png', dst)
            print(f'  wrote {dst}')
            # No `assets/img/ear2_beamline.png` alias any more (dropped
            # 2026-08-11): it existed for slides that still pointed at the
            # single-image name from before the build-up, no slide does, and an
            # unused copy of frame 3 under a name that sounds like the canonical
            # one is a trap.  The `figures/ear2_beamline_light*` aliases DO stay
            # -- README.md and report.html cite them as the standalone figure.


if __name__ == '__main__':
    main()
