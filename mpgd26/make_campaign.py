#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_campaign.py -- nine months of the programme, and the six weeks of beam.

    ../.venv/bin/python make_campaign.py --slides
    ../.venv/bin/python make_campaign.py --slides --highlight-explode
    ../.venv/bin/python make_campaign.py --slides --timeline-only
    ../.venv/bin/python make_campaign.py --numbers

Writes ``figures/campaign_overview*.{png,pdf}`` and, with ``--slides``, drops
the PNG into ``slides/assets/img/``.  Three of them, one per frame of the
slide's build, all on the SAME canvas at the SAME axes rectangles so nothing
moves between frames:

  ``campaign_overview_timeline``   ``--timeline-only``: the timeline strip
                                   alone, the lower half of the canvas empty.
                                   Frame .1 -- the slide comment always said
                                   ".1 is the bare timeline" and until
                                   2026-08-26 it was not: it was the whole
                                   figure, census and all, so the build's first
                                   two frames differed by one outline and the
                                   numbers landed before the timeline had been
                                   read.
  ``campaign_overview``            the joined figure, no outline.  Not used by
                                   the deck any more; it is what report.html
                                   shows.
  ``campaign_overview_highlight``  ``--highlight-explode``: the same, with the
                                   July-August bar outlined.  Frames .2 / .3.

(``--timeline-only --highlight-explode`` is a legal fourth combination, if the
outline should ever get a frame of its own before the census lands.)

ONE FIGURE, TWO PANELS, AND THE SECOND IS INSIDE THE FIRST.  Dylan, 2026-08-19:
"miniaturize the timeline + remove almost all of the text, leave just the big
picture titles -- and work the plot/stats on slide 69 into this slide."

The two things being joined are the project timeline (``make_timeline.py``,
Nov 2025 -> Aug 2026) and the daily event census of the physics run
(``ntof_run_report/figures_local.py::events_collected``, 28 Jun -> 10 Aug).
Stacking them as two separate slide images would cost both of them: the deck's
figure hole is width-limited, so two pictures one above the other are each
capped at about 60 % of the slide's width (see slides/NOTES.md, the
width-limited-figure rule).  As one figure they share the width, and the
relationship between them -- the events panel IS the last bar of the timeline,
opened up -- can be drawn instead of asserted.

The connection is a pair of leaders from the ends of the July-August bar down
to the corners of the lower panel.  That is the whole reason to put them on one
canvas; without it this is just two plots that happen to be adjacent.

WHAT WENT
---------
Every campaign's paragraph.  The mini timeline carries the name, the place and
the dates, and nothing else -- the talk says the rest.  The full-text version
is still ``make_timeline.py`` and still builds ``project_timeline.png``, which
is what the backup slide uses.

SOURCES
-------
* the campaign list, verbatim from ``make_timeline.CAMPAIGNS`` -- imported, not
  copied, so the two figures cannot drift apart.
* ``../ntof_run_report/data/events_per_subrun.csv`` -- one row per (sub-run,
  file tag) with the entry count of that tag's ``nt`` tree, i.e. one entry per
  triggered event.  Produced by ``ntof_run_report/count_events.py`` against EOS.
  The timestamp is the DAQ's own file name, so the census does not depend on
  the n_TOF stream, on the clock fit, or on anything having been matched.

NB the slide says "events recorded", not "DREAM events recorded" (Dylan,
2026-08-19).  The axis label here says the same.  It is the same number either
way -- the census counts triggers of our own read-out -- but the acronym buys
the audience nothing at this point in the talk.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plotstyle as P            # noqa: E402


def _note(fig, text, x, y, chars):
    """P.note, but inside a canvas that is not going to be tight-cropped."""
    import textwrap
    fig.text(x, y, '\n'.join(textwrap.wrap(text, chars)), ha='left', va='top',
             fontsize=9.5, color=P.MUTED)
from make_timeline import CAMPAIGNS, _span   # noqa: E402

FIG = os.path.join(HERE, 'figures')
SLIDES = os.path.join(HERE, 'slides', 'assets', 'img')
CENSUS = os.path.normpath(os.path.join(
    HERE, '..', 'ntof_run_report', 'data', 'events_per_subrun.csv'))

# the two phase boundaries the run report marks, and what they are
EXPLODE_FROM = dt.date(2026, 7, 1)       # left edge of the exploded panel
SETUP_END = dt.date(2026, 7, 14)         # scintillator system complete
PRODUCTION_START = dt.date(2026, 7, 26)  # final configuration frozen

KINDS = ('neutrons', 'cosmics', 'pulser')
KIND_COLOUR = {'neutrons': P.ACCENT, 'cosmics': '#b9bfc6', 'pulser': P.COPPER}
KIND_LABEL = {'neutrons': 'beam', 'cosmics': 'cosmic reference (beam off)',
              'pulser': 'pulser'}


def census():
    per_day = defaultdict(lambda: dict.fromkeys(KINDS, 0))
    bad = 0
    for r in csv.DictReader(open(CENSUS)):
        if not r['events'] or not r['stamp']:
            bad += 1
            continue
        kind = r['beam_type'] if r['beam_type'] in KINDS else 'pulser'
        per_day[r['stamp'][:10]][kind] += int(r['events'])
    days = sorted(per_day)
    return days, per_day, bad


def numbers():
    days, per_day, bad = census()
    tot = {k: sum(per_day[d][k] for d in days) for k in KINDS}
    return dict(days=len(days), unreadable=bad, total=sum(tot.values()), **tot)


def draw(highlight_explode: bool = False, events: bool = True):
    """The campaign figure.

    ``events=False`` draws the timeline strip ALONE -- the build's first frame.
    Same canvas, same axes rectangle, so the strip does not move when the
    census lands underneath it; the lower half is left empty on purpose,
    because it is the hole the next frame fills.
    """
    P.use()
    # 2.95:1 -- the MEASURED aspect of this slide's figure hole with a
    # four-tile stat row and a one-line caption under it (probe render,
    # 2026-08-19).  Saved with bbox_inches=None so the canvas IS the
    # figure: a tight box would crop to the ink and change the ratio.
    fig = plt.figure(figsize=(14.8, 5.02))
    # the timeline strip is short on purpose: it is context, and the events
    # panel is the subject.  Left/right margins are shared so the leaders
    # between them are honest about which dates they connect.
    # margins have to hold BOTH y axes: events on the left, cumulative on
    # the right.  With bbox_inches naming the whole canvas there is no
    # tight box to rescue an axis label that falls off the edge.
    L, R = 0.070, 0.944
    ax_t = fig.add_axes([L, 0.775, R - L, 0.190])
    ax_e = fig.add_axes([L, 0.205, R - L, 0.425]) if events else None

    # ------------------------------------------------------- mini timeline
    t0, t1 = dt.date(2025, 10, 26), dt.date(2026, 8, 26)
    n0, n1 = mdates.date2num(t0), mdates.date2num(t1)
    ax_t.set_xlim(t0, t1)
    ax_t.set_ylim(0.0, 1.0)
    # spine low in the panel: everything above it is label, and the labels
    # need two tiers -- May, June and July-August are three weeks apart and
    # their names are three words each
    Y_SPINE, BAR_H = 0.235, 0.155
    ax_t.axhline(Y_SPINE, color=P.LINE, lw=1.3, zorder=1)

    month = dt.date(2025, 11, 1)
    while mdates.date2num(month) < n1:
        xm = mdates.date2num(month)
        ax_t.plot([xm, xm], [Y_SPINE - 0.055, Y_SPINE], color=P.LINE, lw=1.0,
                  zorder=1)
        lab = f'{month:%b}' if month.month != 1 else f'Jan {month.year}'
        if month == dt.date(2025, 11, 1):
            lab = f'Nov {month.year}'
        ax_t.text(xm, Y_SPINE - 0.085, lab, ha='center', va='top',
                  fontsize=9.5, color=P.MUTED)
        month = dt.date(month.year + month.month // 12,
                        month.month % 12 + 1, 1)

    # ONE tier (Dylan, 2026-08-19: "remove the gray build+bench and keep
    # labels on one line").  The two are the same edit: it was the Saclay
    # bench bar sitting three weeks from the physics run that forced the
    # labels into two rows.  Without it the four beam exposures are far
    # enough apart to label in a single line each.
    EAR2_ONLY = [c for c in CAMPAIGNS if c[3] != 'CEA Saclay']
    last_ab = None
    for i, (start, end, colour, where, name, _body) in enumerate(EAR2_ONLY):
        a, b = mdates.date2num(start), mdates.date2num(end)
        ax_t.add_patch(FancyBboxPatch(
            (a, Y_SPINE - BAR_H / 2), b - a, BAR_H,
            boxstyle='round,pad=0,rounding_size=0.004',
            facecolor=colour, edgecolor='none', zorder=3))
        mid = (a + b) / 2
        ax_t.text(mid, 0.98, name, ha='center', va='top', fontsize=12.5,
                  fontweight='bold', color=P.INK)
        ax_t.text(mid, 0.74, f'{where} · {_span(start, end)}',
                  ha='center', va='top', fontsize=9.5, color=colour)
        if start == dt.date(2026, 6, 28):
            last_ab = (mdates.date2num(EXPLODE_FROM), b)
            if highlight_explode:
                # Same bbox, stroke only, in COPPER not ACCENT: the bar's own fill
                # IS P.ACCENT, so an accent-coloured outline was invisible on it
                # (caught by reading the rendered PNG). An outline distinct from the fill so
                # the bar the wedge explodes from is visible as SUCH before
                # the wedge is emphasized, for the 17.2 build frame (Dylan,
                # 2026-08-23: "the explosion outline for the August test").
                ax_t.add_patch(FancyBboxPatch(
                    (a, Y_SPINE - BAR_H / 2), b - a, BAR_H,
                    boxstyle='round,pad=0,rounding_size=0.004',
                    facecolor='none', edgecolor=P.COPPER, linewidth=2.6,
                    zorder=4))

    # the SPS week is NOT on the mini timeline: it is a parasitic side test in
    # another hall, and this figure is down to big-picture titles only.  It is
    # still on the full timeline (make_timeline.py), which is in backup.

    for side in ('top', 'right', 'left', 'bottom'):
        ax_t.spines[side].set_visible(False)
    ax_t.set_xticks([]), ax_t.set_yticks([])

    if events:
        days, per_day, _bad = census()
        x = np.array([np.datetime64(d) for d in days])
        stacks = {k: np.array([per_day[d][k] for d in days], float) / 1e6
                  for k in KINDS}
        cum = np.cumsum(sum(stacks.values()))

        # --------------------------------------------------- events per day
        # The exploded panel starts on 1 JULY, not on the 28 June arrival
        # (Dylan, 2026-08-19: "it looks like the left side starts in June
        # rather than July").  The four days before it are the install: the
        # first recorded sub-run is 2 July, so nothing is lost, and the panel
        # now reads as the recording period rather than as the run with an
        # empty margin on it.  EXPLODE_FROM is also where the zoom wedge is
        # anchored, so the wedge and the panel describe the same interval.
        e0 = np.datetime64(EXPLODE_FROM.isoformat())
        e1 = np.datetime64('2026-08-11')
        bottom = np.zeros_like(cum)
        for k in KINDS:
            ax_e.bar(x, stacks[k], width=0.82, bottom=bottom,
                     color=KIND_COLOUR[k], label=KIND_LABEL[k], zorder=3)
            bottom = bottom + stacks[k]
        ax_e.set_xlim(e0, e1)
        ax_e.set_ylabel('events recorded per day\n[millions]')
        # '%-d' (no zero pad) is a glibc extension that raises on Windows,
        # and this deck gets built on both; do the day by hand instead.
        ax_e.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _pos: '{:d} {:%b}'.format(mdates.num2date(v).day,
                                                mdates.num2date(v))))
        ax_e.xaxis.set_major_locator(mdates.DayLocator(
            bymonthday=(1, 5, 9, 13, 17, 21, 25, 29)))
        ax_e.grid(axis='y', alpha=0.28)
        ax_e.set_axisbelow(True)
        P.strip(ax_e)

        for boundary, label in ((SETUP_END, 'scintillators complete'),
                                (PRODUCTION_START, 'final configuration')):
            xb = np.datetime64(boundary) - np.timedelta64(12, 'h')
            ax_e.axvline(xb, color=P.INK, lw=1.2, zorder=5)
            ax_e.text(xb, ax_e.get_ylim()[1] * 0.965, f' {label}',
                      fontsize=9.2, color=P.INK, ha='left', va='top',
                      zorder=6, bbox=dict(facecolor=P.SURFACE,
                                          edgecolor='none', pad=1.4))

        ax_c = ax_e.twinx()
        ax_c.plot(x, cum, color='#8a4b2a', lw=2.1, zorder=4)
        ax_c.set_ylabel('cumulative [millions]', color='#8a4b2a')
        ax_c.tick_params(axis='y', colors='#8a4b2a')
        ax_c.set_ylim(0, cum[-1] * 1.30)
        ax_c.set_xlim(e0, e1)
        # in the top-right corner rather than on the end of its own curve:
        # the last week's bars reach 90 % of the panel and the label was
        # inside them
        ax_c.text(0.992, 0.985, f'{cum[-1]:.1f} M events',
                  transform=ax_c.transAxes,
                  ha='right', va='top', fontsize=13.0, color='#8a4b2a',
                  fontweight='bold', zorder=7,
                  bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=2.4))
        ax_c.plot([], [], color='#8a4b2a', lw=2.1, label='cumulative')
        for side in ('top', 'left', 'bottom'):
            ax_c.spines[side].set_visible(False)

        h1, l1 = ax_e.get_legend_handles_labels()
        h2, l2 = ax_c.get_legend_handles_labels()
        # inside the panel, upper left: the first fortnight is near zero, and a
        # legend above the axes would be crossed by the zoom wedge
        ax_e.legend(h1 + h2, l1 + l2, loc='upper left', ncol=2, fontsize=9.5,
                    frameon=False, handlelength=1.5, columnspacing=1.4)

    # --------------------------------------------------- the two leaders
    # Drawn in FIGURE coordinates, from the ends of the July-August bar to the
    # two top corners of the events panel: the lower panel is that bar, opened
    # up.  This is the only thing that makes the two panels one figure.
    if events and last_ab:
        inv = fig.transFigure.inverted().transform
        pts = [inv(ax_t.transData.transform((last_ab[0], Y_SPINE - BAR_H / 2))),
               inv(ax_t.transData.transform((last_ab[1], Y_SPINE - BAR_H / 2))),
               inv(ax_e.transAxes.transform((1.0, 1.0))),
               inv(ax_e.transAxes.transform((0.0, 1.0)))]
        # zorder 0.5, not 0.  Figure-level artists are drawn BEFORE the axes
        # when the zorder ties (Figure.get_children puts .artists ahead of
        # .axes), so at zorder 0 the timeline panel's own opaque background
        # painted over the top third of the wedge -- the visible apex then
        # started level with the panel's bottom edge, three months to the LEFT
        # of the bar it is supposed to come out of, and the wedge read as
        # opening from May (Dylan, 2026-08-20: "the shaded region ... still
        # starts from May/June on the timeline").  Above the axes it is drawn
        # whole, from the underside of the July-August bar, and at 7.5 % it
        # tints the month labels it crosses without hiding them.
        fig.add_artist(matplotlib.patches.Polygon(
            pts, closed=True, facecolor=P.ACCENT, alpha=0.075,
            edgecolor='none', zorder=0.5))

    _note(fig, 'Every beam exposure of the programme (make_timeline.py; the '
                'annotated version is in backup)'
                + (', and one bar per day of entries in each sub-run’s own '
                   'decoded event tree, counted on EOS.' if events else '.'),
           0.07, 0.088, 168)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides', action='store_true')
    ap.add_argument('--numbers', action='store_true')
    ap.add_argument('--highlight-explode', action='store_true',
                     help='outline the July-August bar the events panel '
                          'explodes from (slide 26.2)')
    ap.add_argument('--timeline-only', action='store_true',
                     help='the timeline strip alone, no events panel, on the '
                          'same canvas (slide 26.1)')
    args = ap.parse_args()

    # the timeline strip does not read the census, so it also builds on a
    # machine that has no copy of it
    if not args.timeline_only:
        n = numbers()
        print(f'  {n["total"] / 1e6:.1f} M events over {n["days"]} days  '
              f'(beam {n["neutrons"] / 1e6:.1f} M, '
              f'cosmic {n["cosmics"] / 1e6:.1f} M, '
              f'pulser {n["pulser"] / 1e6:.2f} M; '
              f'{n["unreadable"]} unreadable tags)')
    if args.numbers:
        return

    os.makedirs(FIG, exist_ok=True)
    fig = draw(highlight_explode=args.highlight_explode,
               events=not args.timeline_only)
    name = ('campaign_overview'
            + ('_timeline' if args.timeline_only else '')
            + ('_highlight' if args.highlight_explode else ''))
    base = os.path.join(FIG, name)
    for ext in ('png', 'pdf'):
        # the rcParam is savefig.bbox='tight'; passing None falls back to it,
        # so the full canvas has to be named explicitly or the aspect drifts
        fig.savefig(f'{base}.{ext}', bbox_inches=fig.bbox_inches,
                    pad_inches=0.0)
    print(f'  -> {base}.png')
    if args.slides:
        import shutil
        os.makedirs(SLIDES, exist_ok=True)
        shutil.copyfile(f'{base}.png',
                        os.path.join(SLIDES, f'{name}.png'))
    plt.close(fig)


if __name__ == '__main__':
    main()
