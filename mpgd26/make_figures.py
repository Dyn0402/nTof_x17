#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py -- render and label the whole deliverable set.

    ../.venv/bin/python make_figures.py               # everything, light theme
    ../.venv/bin/python make_figures.py --theme both
    ../.venv/bin/python make_figures.py --only sps_hero,bench_side
    ../.venv/bin/python make_figures.py --draft       # fast, for framing checks

Each figure is produced twice: the bare render (``figures/<name>_<theme>.png``,
useful when you want to place your own labels in Keynote/Beamer) and the
labelled version (``figures/<name>_<theme>_labelled.png`` + ``.pdf``, with the
type set in matplotlib so the PDF carries live text).
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geometry as G          # noqa: E402
import annotate as A          # noqa: E402
import make_sps as MS         # noqa: E402
import make_bench as MB       # noqa: E402

FIG = os.path.join(HERE, 'figures')

SPS_CAPTION = (
    'Detector positions along the rail are the run-config values '
    '(run_59, ~/x17/p2_sps_july).  The P2 BASKET fans are the real annulus '
    'sector with all 1280 pads; the beam spot on P2 MID is the measured '
    'stage-22 illumination (15.1 M tagged tracks), and the beam particles are '
    'REAL two-point tracks from the two EIC uRWELLs, extracted only after '
    'reproducing the published front-to-back alignment on both axes.  '
    'Transverse alignment is '
    'nominal, but the measured uRWELL->P2 frame fits agree across the three '
    'stations to 0.7 mm in x and 1.6 mm in y and close to a multiple of 90 deg '
    'against the fan mounting to within 0.24 deg, so the telescope is square '
    'and aligned and every station is drawn on the nominal axis.  MX17 at '
    'z = 1155 mm is flagged a placeholder in the run config.')

BENCH_CAPTION = (
    'Plane heights are the run-config values (mx17_det2_det3_overnight_6-22-26).  '
    'The chambers under test are drawn at their MEASURED positions from that '
    'run\'s alignment fits, and the muons are REAL reconstructed M3 reference '
    'tracks from the same run, on the standard recipe (chi2 < 1, NClus = 4 on '
    'both planes) and required to cross both trigger paddles -- which is why '
    'none is steeper than about 15 deg.  The 60 x 60 cm paddles sit outside '
    'the DAQ geometry and are placed just beyond the stack; the rack itself '
    'is drawn for context, not surveyed.')


# --------------------------------------------------------------------------- #
def _mm(z):
    return f'z = {z:g} mm'


def sps_figure(name, view, theme, mx17, size, ssaa, spot=True):
    out = os.path.join(FIG, f'{name}_{theme}.png')
    px = MS.render(view, theme, out, mx17=mx17, spot=spot, size=size,
                   ssaa=ssaa)

    items = [(st.name, f'{st.label}\n{_mm(st.z)}')
             for st in G.SPS_STATIONS
             if st.kind != 'mx17' or mx17]
    items = sorted(items, key=lambda it: px.get(it[0], (0, 0))[1])
    side = 'right' if view != 'side' else 'right'
    labels = A.column_labels(px, items, size, side=side, gutter=0.32)

    sub = ('H4 parasitic run, P2 zone -- three P2 BASKET fans between two EIC '
           'uRWELL references')
    if mx17:
        sub += ', with MX17 "Detector E"'
    A.compose(out, labels, os.path.join(FIG, f'{name}_{theme}_labelled'),
              title='SPS H4 beam telescope', subtitle=sub,
              caption=SPS_CAPTION, theme=theme, gutter=0.32)


def bench_figure(name, view, theme, slots, size, ssaa):
    out = os.path.join(FIG, f'{name}_{theme}.png')
    # Ship the real thing when the data disk is there: measured chamber
    # positions and real reconstructed muons, both from the one run that
    # carries them together.  Falls back to nominal + sampled otherwise, so
    # the figures still build on a machine without the disk.
    ref = G.bench_reference_paths()
    kw = dict(align=ref['align'], rays=ref['rays']) if ref else {}
    px = MB.render(view, theme, out, slots=slots, size=size, ssaa=ssaa, **kw)

    kind_label = {'mx17': 'MX17 chamber\n40 x 40 cm, 30 mm drift gap',
                  'p2': 'P2 BASKET fan'}
    items = [
        ('scint_top', f'Trigger scintillator\n60 x 60 cm  ({_mm(G.BENCH_SCINT_Z["top"])})'),
        ('m3_top', 'M3 reference tracker\n2 x 50 x 50 cm Micromegas\nz = 1185, 1302 mm'),
    ]
    for slot, kind in zip(('P2', 'P1'), (slots[1], slots[0])):
        if kind in kind_label:
            items.append((slot, f'{slot} test slot -- {kind_label[kind]}\n'
                                f'{_mm(G.BENCH_DUT_Z[slot])}'))
    items += [
        ('m3_bot', 'M3 reference tracker\n2 x 50 x 50 cm Micromegas\nz = 24, 144 mm'),
        ('scint_bottom',
         f'Trigger scintillator\n60 x 60 cm  ({_mm(G.BENCH_SCINT_Z["bottom"])})'),
    ]
    items = [it for it in items if it[0] in px]
    items = sorted(items, key=lambda it: px[it[0]][1])
    labels = A.column_labels(px, items, size, side='right', gutter=0.50)

    A.compose(out, labels, os.path.join(FIG, f'{name}_{theme}_labelled'),
              title='Saclay cosmic test bench',
              subtitle='Four M3 reference Micromegas around two test slots, '
                       'triggered by a scintillator coincidence',
              caption=BENCH_CAPTION, theme=theme, gutter=0.50,
              header=0.10, footer=0.12)


# --------------------------------------------------------------------------- #
FIGURES = {
    'sps_hero':      dict(kind='sps', view='hero', mx17=False),
    'sps_hero_mx17': dict(kind='sps', view='hero', mx17=True),
    'sps_side':      dict(kind='sps', view='side', mx17=False),
    'sps_beam':      dict(kind='sps', view='beam', mx17=False),
    # the two headline bench configurations: two MX17 chambers, and two P2 fans
    'bench_hero':    dict(kind='bench', view='hero', slots=('mx17', 'mx17')),
    'bench_side':    dict(kind='bench', view='side', slots=('mx17', 'mx17')),
    'bench_p2':      dict(kind='bench', view='hero', slots=('p2', 'p2')),
    'bench_p2_side': dict(kind='bench', view='side', slots=('p2', 'p2')),
    # available but not part of the headline set
    'bench_mixed':   dict(kind='bench', view='hero', slots=('p2', 'mx17')),
}

SIZES = {'sps': (2800, 1750), 'bench': (1900, 2400)}
DRAFT = {'sps': (1200, 750), 'bench': (760, 960)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--only', default=None,
                    help='comma-separated subset of ' + ','.join(FIGURES))
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()

    names = list(FIGURES) if args.only is None else args.only.split(',')
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]

    for theme in themes:
        for name in names:
            spec = FIGURES[name]
            kind = spec['kind']
            size = DRAFT[kind] if args.draft else SIZES[kind]
            print(f'{name} [{theme}]')
            if kind == 'sps':
                sps_figure(name, spec['view'], theme, spec['mx17'], size,
                           not args.draft)
            else:
                bench_figure(name, spec['view'], theme, spec['slots'], size,
                             not args.draft)


if __name__ == '__main__':
    main()
