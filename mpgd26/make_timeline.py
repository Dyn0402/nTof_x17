#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_timeline.py -- the project timeline figure for the MPGD2026 talk.

    ../.venv/bin/python make_timeline.py

Writes ``slides/assets/img/project_timeline.{png,pdf}`` -- the PDF is the one
to open if you want to read the small type.  One figure: every exposure of this
detector
programme from the first one, November 2025, to the end of the n_TOF physics
run on 10 August 2026.

Everything on the figure is a fact with a source, and the sources are all in
this repository or its companions.  Nothing here is re-derived from bulk data;
the counts below were read once and are hard-coded, so this script runs
anywhere and the numbers can be audited against the line comments:

  Nov 2025   86 run directories in
             /media/dylan/data/x17/nov_25_beam_test/dream_run/, first .fdf
             2025-11-27 11:40, last 2025-12-05 10:46.  Detectors, targets and
             the SiPM behaviour: the "nTOF-X17-Nov-BeamTest-Logbook" Google
             Doc and the "nTof X17 November Test Beam" deck (28 Nov 2025).
  Feb 2026   143 rows in /eos/experiment/ntof/data/x17/feb_beam/runs/
             run_table.csv, 2026-02-02 10:54 -> 2026-03-01 22:29.  Gases, drift
             gaps, frames and targets are that table's own columns.  The
             conclusions are MX17_Documentation/"X17_nTOF_Status_Compiled.docx"
             (Sections 1, 4, 5) and the 3/24 analysis-meeting deck.
  May 2026   72 rows in .../may_beam/runs/run_table.csv, 2026-05-09 -> 05-18;
             the test itself ran 6-18 May (commissioning before first beam).
             Conclusions: MX17_Documentation/"X17_nTOF_May2026_Status.docx" S1.2.
  Jun 2026   mx_june_cosmic_qa/ -- the bar is the bench-run span, 2026-06-06 to
             2026-06-27, over five chambers; the build of the four final
             chambers is the June to-do list in the May status document.  Headline numbers: JUNE_RESULTS_SUMMARY.md and the
             deck's own efficiency / resolution slides.
  Jul 2026   ntof_run_report/ -- 162 DREAM runs / 2 705 sub-runs / 17.9 TB, and
             41.8 M events from data/events_per_subrun.csv (2 511 sub-runs with
             decoded output).  Dates 2026-06-28 (arrival) -> 2026-08-10 09:10
             (last beam sub-run); first beam data 2 July.
  SPS       sps_beam_test_26/ -- detector E parasitic in the P2 uRWELL test
             beam at SPS H4, 31 July - 3 August 2026.

If any of those change, change them here and re-run; the slide caption in
slides/index.html quotes the same numbers and has to move with them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plotstyle as P  # noqa: E402

OUT = os.path.join(HERE, 'slides', 'assets', 'img')

D = dt.date

# Two roles, two colours.  Beam exposures at n_TOF EAR2 are the spine of the
# story and carry the deck accent; everything that happened away from EAR2
# (the build and characterization at Saclay, the parasitic SPS week) is
# secondary and recedes.
C_EAR2 = P.ACCENT
C_LAB = P.MUTED
C_SPS = P.COPPER

# (start, end, colour, where, name, what it settled)
#
# Layout note: the campaigns are weeks apart but their descriptions are
# paragraphs, so nothing readable fits *beside* a bar on a true date axis.
# The figure therefore separates the two jobs -- the spine carries duration
# and spacing, an equal-width panel strip underneath carries the words, and a
# leader ties each panel to its bar.  Order in this list is the panel order.
#
# Body text is wrapped to the column at draw time (WRAP_CHARS), so write it as
# one paragraph and do not hand-break it.
CAMPAIGNS = [
    (D(2025, 11, 27), D(2025, 12, 5), C_EAR2, 'n_TOF EAR2', 'First test',
     'Two prototype chambers and a SiPM wall on DREAM, '
     '86 runs. Read-out timed into the flash — and '
     'blinded by it; SiPM channels take ~10 ms to recover.'),
    (D(2026, 2, 2), D(2026, 3, 1), C_EAR2, 'n_TOF EAR2',
     'Prototype on beam',
     'A month of scans, 143 runs: 5 gases, 3 drift gaps, '
     '2 frames, 5 targets. DREAM saturates below the '
     'gain the chamber needs to see a pair.'),
    (D(2026, 5, 6), D(2026, 5, 18), C_EAR2, 'n_TOF EAR2',
     'Two chambers',
     'Al absorber, mesh charge injection, Ne/iso — none '
     'of the three fixed the flash. Coincidence trigger '
     'commissioned: the thermal window is the plan.'),
    (D(2026, 6, 6), D(2026, 6, 27), C_LAB, 'CEA Saclay',
     'Build + bench QA',
     'The four final chambers built, then the fleet '
     'measured on the cosmic bench: 93 % efficient, '
     'sub-mm position, ~1° on the track angle.'),
    (D(2026, 6, 28), D(2026, 8, 10), C_EAR2, 'n_TOF EAR2', 'The physics run',
     'Four TPCs around 500 bar of ³He, four-arm '
     'trigger and calorimetry. 162 runs, 41.8 M events, '
     '17.9 TB. Beam off 10 Aug, 09:10.'),
]

# The SPS week is drawn as its own short bar just under the spine: it overlaps
# the physics run in time and was a different beam in a different hall.
SPS = (D(2026, 7, 31), D(2026, 8, 3),
       'SPS H4 · a spare chamber, parasitic in the P2 test beam')


# Vertical layout, all in axes fractions.  The whole figure is drawn in this
# one coordinate system so the two halves (spine, panel strip) stay locked.
Y_SPINE = 0.905          # the date line and the campaign bars
BAR_H = 0.070
Y_DATES = 0.965          # date span, above its bar
Y_MONTHS = 0.775         # month labels; the ticks sit just under the spine
Y_PANEL_TOP = 0.635      # the panel strip's rule
Y_PANEL_NAME = 0.590
Y_PANEL_WHERE = 0.480
Y_PANEL_BODY = 0.420
PANEL_GAP = 0.014        # fraction of the axes width left between panels
WRAP_CHARS = 36          # measure of the body text, in characters


def timeline(out_dir: str) -> None:
    P.use()
    fig, ax = plt.subplots(figsize=(13.6, 4.1))

    x0, x1 = D(2025, 10, 26), D(2026, 8, 26)
    n0, n1 = mdates.date2num(x0), mdates.date2num(x1)
    ax.set_xlim(x0, x1)
    # y is read directly as an axes fraction; the window is cropped to the
    # band that is actually drawn, so the figure has no dead margin.
    ax.set_ylim(0.185, 1.0)

    def frac(day) -> float:
        """Axes fraction of a date — used to place the panel leaders."""
        return (mdates.date2num(day) - n0) / (n1 - n0)

    def xdata(f: float) -> float:
        """Inverse of `frac`, so panel geometry can be written in fractions."""
        return n0 + f * (n1 - n0)

    # ---------------------------------------------------------------- spine
    ax.axhline(Y_SPINE, color=P.LINE, lw=1.4, zorder=1)

    # Month ticks, drawn by hand: the real x-axis is at the bottom of the
    # frame and the spine is not, so the ticks have to travel with the spine.
    month = D(2025, 11, 1)
    while mdates.date2num(month) < n1:
        x = mdates.date2num(month)
        ax.plot([x, x], [Y_SPINE - 0.035, Y_SPINE - 0.050],
                color=P.LINE, lw=1.0, zorder=1)
        # The year rides on January (and on the first month drawn), which is
        # cheaper than a second label row and cannot be misread.
        label = f'{month:%b}' if month.month != 1 else f'{month:%b} %d' % month.year
        if month == D(2025, 11, 1):
            label = f'{month:%b} {month.year}'
        ax.text(x, Y_MONTHS, label, ha='center', va='top',
                fontsize=10, color=P.MUTED)
        month = D(month.year + month.month // 12, month.month % 12 + 1, 1)


    # ------------------------------------------------- bars + panel strip
    n = len(CAMPAIGNS)
    width = (1.0 - PANEL_GAP * (n - 1)) / n

    for i, (start, end, colour, where, name, body) in enumerate(CAMPAIGNS):
        a, b = mdates.date2num(start), mdates.date2num(end)
        ax.add_patch(FancyBboxPatch(
            (a, Y_SPINE - BAR_H / 2), b - a, BAR_H,
            boxstyle='round,pad=0,rounding_size=0.004',
            facecolor=colour, edgecolor='none', zorder=3))
        ax.text((a + b) / 2, Y_DATES, _span(start, end), ha='center',
                va='bottom', fontsize=10.5, fontweight='bold', color=colour)

        # Panel column, and the leader that ties it to its bar.
        left = i * (width + PANEL_GAP)
        centre = left + width / 2
        ax.plot([(a + b) / 2, xdata(centre)],
                [Y_SPINE - BAR_H / 2 - 0.020, Y_PANEL_TOP + 0.020],
                color=colour, lw=0.9, alpha=0.75, zorder=2)
        ax.plot([xdata(left), xdata(left + width)],
                [Y_PANEL_TOP, Y_PANEL_TOP], color=colour, lw=2.4, zorder=3,
                solid_capstyle='butt')
        ax.text(xdata(left), Y_PANEL_NAME, name, ha='left', va='top',
                fontsize=12.5, fontweight='bold', color=P.INK)
        ax.text(xdata(left), Y_PANEL_WHERE, where, ha='left', va='top',
                fontsize=9.5, color=colour, fontweight='bold')
        ax.text(xdata(left), Y_PANEL_BODY,
                '\n'.join(textwrap.wrap(body, WRAP_CHARS)),
                ha='left', va='top', fontsize=9.8, color=P.MUTED,
                linespacing=1.55)

    # ------------------------------------------------------------- SPS week
    s, e, label = SPS
    # Between the month ticks and the month labels: the only free band.
    y_sps = 0.830
    ax.add_patch(FancyBboxPatch(
        (mdates.date2num(s), y_sps - 0.020),
        mdates.date2num(e) - mdates.date2num(s), 0.040,
        boxstyle='round,pad=0,rounding_size=0.004',
        facecolor=C_SPS, edgecolor='none', zorder=3))
    # Labelled to the LEFT, on one line, inside the band between the month
    # ticks and the month labels: that band is empty, and to the right of the
    # bar there is only a fortnight of axis left.
    ax.annotate(label, xy=(mdates.date2num(s), y_sps),
                xytext=(-8, 0), textcoords='offset points',
                ha='right', va='center', fontsize=9.5, color=C_SPS,
                zorder=6,
                # Two panel leaders pass through this band; knock them out
                # behind the type rather than move the label somewhere worse.
                bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=2.0))

    # ------------------------------------------------------- frame cleanup
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.grid(False)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(out_dir, 'project_timeline.pdf'))
    P.save(fig, os.path.join(out_dir, 'project_timeline.png'))


def _span(a: D, b: D) -> str:
    """'27 Nov – 5 Dec' / '2 Feb – 1 Mar' — month repeated only when it changes."""
    if a.month == b.month:
        return f'{a.day}–{b.day} {b:%b}'
    return f'{a.day} {a:%b} – {b.day} {b:%b}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT, help='output directory')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    timeline(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
