#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_x17.py -- the X17 physics-case figure.

    ../.venv/bin/python make_x17.py                 # light theme
    ../.venv/bin/python make_x17.py --theme both
    ../.venv/bin/python make_x17.py --no-title      # bare, for a slide that
                                                    # already has a title

Writes ``figures/x17_signature_<theme>.png`` and ``.pdf``.  As everywhere else
in this package the type is set in matplotlib, so the PDF carries live text and
scales to any slide or page without going fuzzy -- drop the PDF into Beamer or
Keynote in preference to the PNG.

``--no-title`` drops the title, subtitle and footer caption and crops to the
diagram, which is what you want when the slide's own title bar says the same
thing.  The caption text is then yours to put in the speaker notes; it is
printed to the terminal so you can copy it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scenes_x17 as X                    # noqa: E402

FIG = os.path.join(HERE, 'figures')
# --slides also drops the light-theme PNG into the deck's asset directory, the
# way make_ear2/make_target/make_timeline do -- the deck copy is the same file
# under a shorter name (no _light), so there is nothing to keep in step by hand.
# (The older x17_signature.png / x17_story_capsule.png in there were copied by
# hand before this existed.)
SLIDES = os.path.join(HERE, 'slides', 'assets', 'img')


LAYOUTS = {
    'signature': dict(draw=lambda **kw: X.draw(**kw), name='x17_signature',
                      crop=(0, 0.75, 16.0, 6.85)),
    'story': dict(draw=lambda **kw: X.draw_story(part='all', **kw),
                  name='x17_story', opts=('capsule',)),
    # the same five beats split across two slides: 1-3 sets up the physics,
    # 4-5 derives the measurement from it
    'story1': dict(draw=lambda **kw: X.draw_story(part='top', **kw),
                   name='x17_story_1of2', opts=('capsule',)),
    'story2': dict(draw=lambda **kw: X.draw_story(part='bottom', **kw),
                   name='x17_story_2of2'),
}

# ...and the same five beats one file at a time (2026-08-16), for dropping into
# slides individually -- a build, a different deck, a poster.  Each is the same
# drawing cropped to its own beat, NOT a redrawn version of it: edit a beat and
# the compilation and the single file both change.  Beat 1 honours --capsule.
BEAT_LAYOUTS = {
    f'beat{n}': dict(draw=(lambda n: lambda **kw: X.draw_beat(n, **kw))(n),
                     name=f'x17_beat{n}_{tag}', bare_only=True,
                     opts=('capsule',) if n == '1' else ())
    for n, tag in X.BEAT_NAMES.items()
}
LAYOUTS.update(BEAT_LAYOUTS)

# ...and the two rows as BUILDS (2026-08-17): each row's beats revealed one at a
# time on the same canvas, for the two deck slides that page through them.  The
# frames are strict subsets of one picture -- same band, same coordinates -- so
# a beat appears in its final position and nothing already drawn moves.  Always
# bare: a build frame is a slide figure, and the slide's title bar is the title.
BUILD_TAGS = {'top': ('beam', 'capture', 'channels'),
              'bot': ('boost', 'spectrum')}
BUILD_LAYOUTS = {
    f'{row}{n}': dict(
        draw=(lambda part, n: lambda **kw: X.draw_story(part=part, upto=n,
                                                        **kw))(part, n),
        name=f'x17_story_{row}_{n}_{tag}', force_bare=True,
        opts=('capsule',) if row == 'top' else ())
    for row, part in (('top', 'top'), ('bot', 'bottom'))
    for n, tag in enumerate(BUILD_TAGS[row], start=1)
}
# The hand-over to the detector half of the talk: the pair leaving at its
# opening angle, through two micro-TPCs.  Since 2026-08-18 the deck's third
# bottom-row frame is the WHOLE ROW again -- the cartoon standing in beat 4's
# box, the spectrum untouched beside it (scenes_x17._story_detect) -- so it
# belongs with the other build frames and not on a canvas of its own.  The
# stand-alone canvas is still there as LAYOUTS['detect_solo'] for report.html
# and for any slide that wants the cartoon by itself.
BUILD_LAYOUTS['bot3'] = dict(
    draw=lambda **kw: X.draw_story(part='bottom', upto=2, detect=True, **kw),
    name='x17_story_bot_3_detect', force_bare=True)
LAYOUTS.update(BUILD_LAYOUTS)

LAYOUTS['detect_solo'] = dict(draw=lambda **kw: X.draw_detect(**kw),
                              name='x17_detect_solo', force_bare=True)

# The Summary slide's figure (2026-08-24): find the two-track events, then
# histogram their opening angle.  Same palette and the same kinematics as the
# physics-case beats, which is the point -- the closing figure is the opening
# one, answered.  Always bare: it sits under the Summary's own title.
LAYOUTS['outlook'] = dict(draw=lambda **kw: X.draw_outlook(**kw),
                          name='x17_outlook', force_bare=True)


def render(theme='light', title=True, dpi=300, layout='signature',
           capsule=False, tight=False, slides=False):
    spec = LAYOUTS[layout]
    kw = dict(theme=theme, dpi=dpi, title=title)
    if spec.get('bare_only'):
        # a single beat has no title/caption band to drop, so there is no
        # _bare variant of it either -- the file is the drawing
        kw.pop('title')
    if spec.get('force_bare'):
        # a build frame is only ever a slide figure: no title band, and no
        # _bare in the name because there is no other version of it
        kw['title'] = False
    if 'capsule' in spec.get('opts', ()):
        kw['capsule'] = capsule
    fig = spec['draw'](**kw)
    os.makedirs(FIG, exist_ok=True)
    name = spec['name'] + ('_capsule' if capsule and 'capsule' in spec.get('opts', ()) else '')
    if not title and not (spec.get('bare_only') or spec.get('force_bare')):
        name += '_bare'
    if tight and spec.get('bare_only'):
        name += '_tight'      # so it never silently overwrites the aligned one
    base = os.path.join(FIG, f'{name}_{theme}')
    page = X.palette(theme)['page']
    bbox = None
    if not title and 'crop' in spec:
        # the signature layout crops in inches; the story layouts crop by
        # narrowing their own canvas band, so they need nothing here
        bbox = fig.bbox_inches.from_bounds(*spec['crop'])
    if tight and spec.get('bare_only'):
        # single beats keep the row's full height by default so that beats
        # dropped one after another line up exactly as they do in the
        # compilation; --tight trims each to its own ink instead
        bbox = 'tight'
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', facecolor=page, bbox_inches=bbox,
                    pad_inches=0.06 if bbox == 'tight' else 0.0)
        print(f'  wrote {base}.{ext}')
    if slides and theme == 'light':
        os.makedirs(SLIDES, exist_ok=True)
        dest = os.path.join(SLIDES, f'{name}.png')
        shutil.copyfile(f'{base}.png', dest)
        print(f'  wrote {dest}')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--layout', default='signature',
                    choices=(['signature', 'story', 'story1', 'story2',
                              'split', 'beats', 'build', 'detect_solo',
                              'outlook', 'both']
                             + sorted(BEAT_LAYOUTS) + sorted(BUILD_LAYOUTS)),
                    help='signature: three panels on one row, the compact '
                         'version. story: five beats over two rows. '
                         'story1/story2: the same five beats split across two '
                         'slides (1-3 then 4-5); split does both of them. '
                         'beat1..beat5: ONE beat per file, for dropping into '
                         'slides individually; beats does all five. '
                         'top1..top3 / bot1..bot3: the two rows as BUILDS, one '
                         'more beat per frame on the same canvas; build does '
                         'all five frames. '
                         'outlook: the Summary slide figure -- two-track '
                         'search + the opening-angle spectrum it produces. '
                         'both: every layout.')
    ap.add_argument('--no-title', dest='title', action='store_false',
                    help='drop the title/caption bands and crop to the diagram')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--capsule', action='store_true',
                    help='story layout only: draw the real Geant4 3He vessel '
                         'in beat 1 instead of a generic group of nuclei. Use '
                         'once the target hardware has been introduced.')
    ap.add_argument('--tight', action='store_true',
                    help='beat layouts only: crop each file to its own ink '
                         'instead of keeping the row height that makes the '
                         'beats line up with each other')
    ap.add_argument('--slides', action='store_true',
                    help='also copy the light-theme PNG into '
                         'slides/assets/img/ under the same name without the '
                         'theme suffix, for the deck to reference')
    ap.add_argument('--validate', action='store_true',
                    help='cross-check the sampled X17 channel against the '
                         'analytic solution and report the IPC shape')
    args = ap.parse_args()

    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    if args.layout == 'both':
        layouts = list(LAYOUTS)
    elif args.layout == 'split':
        layouts = ['story1', 'story2']
    elif args.layout == 'beats':
        layouts = sorted(BEAT_LAYOUTS)
    elif args.layout == 'build':
        layouts = sorted(BUILD_LAYOUTS)
    else:
        layouts = [args.layout]
    for layout in layouts:
        for theme in themes:
            print(f'{LAYOUTS[layout]["name"]} [{theme}]')
            render(theme=theme, title=args.title, dpi=args.dpi,
                   layout=layout, capsule=args.capsule, tight=args.tight,
                   slides=args.slides)

    if args.validate:
        ana, samp, med, frac = X.validate()
        print(f'\nX17 kinematic minimum   analytic {ana:.2f} deg  vs  '
              f'MX17_Simulation sampler {samp:.2f} deg  '
              f'(delta {abs(ana - samp):.3f})')
        print(f'IPC opening angle       median {med:.1f} deg, '
              f'{frac * 100:.0f} % above 60 deg')
    else:
        print(f'\nkinematic minimum opening angle: '
              f'{X.opening_angle_pdf()[2]:.2f} deg '
              f'(m_X17 = {X.X17["m_x17"]} MeV, E = {X.X17["e_capture"]} MeV)')


if __name__ == '__main__':
    main()
