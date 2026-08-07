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
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scenes_x17 as X                    # noqa: E402

FIG = os.path.join(HERE, 'figures')


def render(theme='light', title=True, dpi=300, name='x17_signature'):
    fig = X.draw(theme=theme, dpi=dpi, title=title)
    os.makedirs(FIG, exist_ok=True)
    base = os.path.join(FIG, f'{name}_{theme}')
    page = X.palette(theme)['page']
    bbox = None
    if not title:
        # crop the header/footer bands away rather than leaving white space
        bbox = fig.bbox_inches.from_bounds(0, 0.75, 16.0, 6.85)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', facecolor=page, bbox_inches=bbox)
        print(f'  wrote {base}.{ext}')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--no-title', dest='title', action='store_false',
                    help='drop the title/caption bands and crop to the diagram')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--validate', action='store_true',
                    help='cross-check the sampled X17 channel against the '
                         'analytic solution and report the IPC shape')
    args = ap.parse_args()

    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    name = 'x17_signature' if args.title else 'x17_signature_bare'
    for theme in themes:
        print(f'{name} [{theme}]')
        render(theme=theme, title=args.title, dpi=args.dpi, name=name)

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
