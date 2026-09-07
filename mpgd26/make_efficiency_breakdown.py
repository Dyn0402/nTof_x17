#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_efficiency_breakdown.py -- the efficiency figures for the MPGD2026 talk.

    ../.venv/bin/python make_efficiency_breakdown.py
    ../.venv/bin/python make_efficiency_breakdown.py --only breakdown
    ../.venv/bin/python make_efficiency_breakdown.py --print   # numbers only

Three figures for the efficiency slide.  Since 2026-08-17 the SLIDE draws the
loss budget itself, in HTML, so the two below it are what the audience sees and
`efficiency_breakdown.png` is now the standalone/handoff copy of the same
numbers (slides/HANDOFF_efficiency.md quotes it).  It is still regenerated here
so that copy cannot go stale behind the slide's:

  efficiency_breakdown.png      det3's loss budget -- where every crossing muon
                                goes, in plain language, with an annotation box
                                whose every number is READ FROM THE JSON.  NOT
                                on a slide any more; the .bar-chart.loss markup
                                in slides/index.html is, and the two must agree
                                bar for bar.
  efficiency_residual_tail.png  the |r| distribution on a log scale with the
                                5 mm match circle marked -- the figure that
                                explains the "detected but >5 mm off track"
                                slice.  ONE panel since 2026-08-17.
  efficiency_map_2mm.png        efficiency across the chamber face at the tight
                                r < 2 mm criterion, from the 40x40 grid
                                mx_june_wft/report/make_maps_2mm.py writes.
                                Read its docstring before quoting a level off
                                it -- 2 mm is not the headline criterion.

Why this file exists
--------------------
It replaces `mx_june_cosmic_qa/engineer_package/make_efficiency_breakdown.py`
as the source of the deck's breakdown figure. That script had the headline
efficiency **hardcoded in its annotation string** ("...off the 88.8%"), which is
how the figure came to disagree with its own bars when the M3 recipe changed on
2026-07-14. Nothing here is hardcoded: every percentage, every count and the
10 mm recovery figure are parsed out of the analysis JSON, so the annotation
cannot drift from the bars again.

Input -- one committed-style reduction per chamber, written by the analysis:

    <OUT_BASE>/wft/efficiency/efficiency_breakdown.json        (waveform-first)
    <OUT_BASE>/wft/efficiency/efficiency_breakdown_hits.json   (hits chain)

regenerate with, from the repo root (~6 s per chamber, caches already on disk):

    for K in sat_det3 o22_long_det2 g_det4 g_det6_long g_det7_long; do
        .venv/bin/python mx_june_wft/02_efficiency.py $K --max-dropped -1
    done

`02_efficiency.py` is the single accounting for both reconstruction chains, so
the wft and hits columns below are directly comparable. Basis is the
waveform-first fit per `RECONSTRUCTION_BASIS.md` -- position may not come from
hit times on these detectors. Detection (`has_any`) stays hits-defined, on
purpose: whether the chamber fired is a property of the analyzer's trigger, not
of the fit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'mx_june_cosmic_qa'))

import plotstyle as P  # noqa: E402

OUT = os.path.join(HERE, 'slides', 'assets', 'img')

# Experiment letter -> the QA run key of that chamber's headline high-stats
# sub-run, in the A..E order of det_labels.py.  These are the five runs the
# fleet-state table of mx_june_wft/ANALYSIS_STATE_2026-07-31.md describes.
FLEET = [
    ('A', 'det3', 'sat_det3'),
    ('B', 'det2', 'o22_long_det2'),
    ('C', 'det6', 'g_det6_long'),
    ('D', 'det7', 'g_det7_long'),
    ('E', 'det4', 'g_det4'),
]
HEADLINE = 'g_det3_wknd'      # the chamber the main slide is about -- switched from sat_det3 (7,049 rays) to match the map's higher-statistics run (Dylan, 2026-08-23: "use the 22k set")

# Plain-language rows for the loss budget, best -> worst, keyed to the
# categories of 02_efficiency.py.  Colours: one green for the answer, copper
# for "our position, not their signal", accent for the discharge, grey for
# unusable, red reserved for genuine blindness.
#
# "DISCHARGE", not "spark", since 2026-08-17 (Dylan).  The analysis code's
# category is still `spark_cat` and stays that way -- renaming a JSON key to
# fix a slide would be the tail wagging the dog -- but the word on the page is
# the one an MPGD audience uses for what a resistive detector does.
ROWS = [
    ('within_R',    'Reconstructed within 5 mm\n(the efficiency)',        '#2e8b57'),
    ('reco_far',    'Detected, point >5 mm\noff the telescope track',      P.COPPER),
    ('spark_cat',   'Discharged during this muon\n(self-quenching, no dead time)', P.ACCENT),
    ('hit_no_reco', 'Fired, no valid X+Y point formed',                    P.MUTED),
    ('no_hit',      'Silent — no signal at all\n(genuine blindness)', '#b04a3a'),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def eff_json(run_key: str, source: str = 'wft') -> dict | None:
    """The efficiency reduction for one run key, or None if it is not on disk."""
    from qa_config import get_config
    cfg = get_config(run_key)
    tag = '' if source == 'wft' else '_hits'
    path = os.path.join(cfg.OUT_BASE, 'wft', 'efficiency',
                        f'efficiency_breakdown{tag}.json')
    if not os.path.exists(path):
        print(f'  ! missing {path}')
        return None
    with open(path) as f:
        d = json.load(f)
    d['_path'] = path
    d['_mtime'] = os.path.getmtime(path)
    return d


def counts(d: dict) -> dict:
    """Category counts, reconstructed from the percentages and the denominator.

    02_efficiency.py writes percentages; the counts are exact integers because
    n_rays is exact, so round-tripping is lossless at this precision.
    """
    n = d['n_rays']
    return {k: int(round(d[k] * n / 100.0)) for k, _, _ in ROWS}


def fleet_table(source: str = 'wft') -> list[dict]:
    rows = []
    for letter, det, key in FLEET:
        d = eff_json(key, source)
        if d is None:
            continue
        d.update(letter=letter, det=det)
        rows.append(d)
    return rows


# --------------------------------------------------------------------------- #
# Figure 1 -- the loss budget
# --------------------------------------------------------------------------- #

def fig_breakdown(d: dict) -> None:
    n = d['n_rays']
    cnt = counts(d)
    r10 = d['eff_vs_R']['10.0']

    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ypos = list(range(len(ROWS)))[::-1]
    for y, (key, label, color) in zip(ypos, ROWS):
        pct = d[key]
        ax.barh(y, pct, color=color, height=0.6, edgecolor=P.SURFACE,
                linewidth=0.8, zorder=3)
        vlabel = f'{pct:.1f}%   ({cnt[key]:,})'
        if pct > 60:
            ax.text(pct - 1.2, y, vlabel, va='center', ha='right', fontsize=11.5,
                    fontweight='bold', color='white', zorder=4)
        else:
            ax.text(pct + 1.2, y, vlabel, va='center', ha='left', fontsize=11.5,
                    fontweight='bold', color=P.INK, zorder=4)
        ax.text(-1.5, y, label, va='center', ha='right', fontsize=10.5,
                color=P.INK)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, len(ROWS) - 0.35)
    ax.set_yticks([])
    ax.set_xlabel('% of muons the telescope sent through the active area')
    P.title(ax, 'Detector A (det3): where do the crossing muons go?',
            f'{n:,} reference muons, 5 mm match, waveform-first reconstruction')
    P.strip(ax, left=False, bottom=True)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=P.LINE, lw=0.7, zorder=0)
    ax.grid(axis='y', visible=False)
    ax.margins(y=0)

    # Every number in this box is read from the JSON -- see the module docstring.
    note = (
        f'The chamber produces a signal for {d["has_any"]:.1f}% of crossings and '
        f'reconstructs a point for {d["reco_at_all"]:.1f}%.  Genuine blindness is '
        f'{d["no_hit"]:.2f}%.\n'
        f'Neither of the two losses off {d["within_R"]:.1f}% is the chamber failing to '
        f'see the muon: a {d["spark_cat"]:.1f}% discharge coincidence (self-quenching, no\n'
        f'dead time afterwards) and a {d["reco_far"]:.1f}% edge / near-miss position '
        f'tail — open the match to 10 mm and the efficiency recovers to {r10:.1f}%.')
    fig.text(0.5, 0.14, note, ha='center', va='top', fontsize=9.6,
             color=P.INK, linespacing=1.45,
             bbox=dict(boxstyle='round,pad=0.65', facecolor='#f4f2f7',
                       edgecolor=P.ACCENT, linewidth=1.0))

    fig.subplots_adjust(left=0.27, right=0.975, top=0.84, bottom=0.30)
    P.save(fig, os.path.join(OUT, 'efficiency_breakdown.png'))


# --------------------------------------------------------------------------- #
# Figure 2 -- the residual tail behind the "off track" slice
# --------------------------------------------------------------------------- #

def fig_tail(d: dict) -> None:
    """The |r| distribution, core and tail, on one axes.

    ONE panel since 2026-08-17 (Dylan).  It used to carry a second panel --
    efficiency against match radius -- which is a *re-plot of this histogram's
    own cumulative*, and it cost half the width to say something the tail
    already shows.  The two numbers worth having off it (93.3 % at 5 mm,
    94.6 % at 10 mm) are on the slide as type, where they can be read from the
    back of the room.

    NO burned-in title: the slide carries its heading in HTML type.
    """
    edges = np.asarray(d['r_hist_edges'], float)
    cts = np.asarray(d['r_hist_counts'], float)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    R = d['R_mm']
    r10 = d['eff_vs_R']['10.0']

    # SMALL ON PURPOSE (5.6 x 3.5 in, was 7.4 x 4.9).  On the slide this figure
    # gets about a fifth of the page width; a figure saved at 7.4 in and shown
    # at 2 in has 4-pixel tick labels.  Sizing the canvas near the size it is
    # displayed at is the only thing that makes matplotlib type legible from
    # the back of a room -- everything else here is unchanged.
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.step(ctr, np.where(cts > 0, cts, np.nan), where='mid',
            color=P.DET_COLOR['A'], lw=1.8)
    ax.fill_between(ctr, 1e-1, np.where(cts > 0, cts, 1e-1), step='mid',
                    color=P.DET_COLOR['A'], alpha=0.16, lw=0)
    ax.axvline(R, color=P.COPPER, lw=1.6, ls='--')
    ax.set_yscale('log')
    ax.set_xlim(0, 20)
    ax.set_ylim(0.7, max(cts.max() * 3.0, 10))
    ax.set_xlabel('|r|   reconstructed point − reference track   [mm]',
                  fontsize=12)
    ax.set_ylabel('muons per 0.25 mm', fontsize=12)
    ax.tick_params(labelsize=11.5)
    P.strip(ax)

    # the core, stated on the plot rather than in a caption nobody reads
    ax.annotate(f'core σ {d["core_sigma_mm"]:.2f} mm\nmedian '
                f'{d["median_r_mm"]:.2f} mm',
                xy=(0.6, cts.max() * 0.55), xytext=(3.2, cts.max() * 1.55),
                fontsize=12, color=P.DET_COLOR['A'], fontweight='bold',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=P.DET_COLOR['A'],
                                lw=1.1, shrinkA=2, shrinkB=6))
    ax.text(R + 0.55, cts.max() * 0.13,
            f'{R:.0f} mm match\n{d["within_R"]:.1f} % efficiency',
            fontsize=12, color=P.COPPER, va='top', ha='left',
            fontweight='bold', linespacing=1.35)
    n_tail = int(round(d['reco_far'] * d['n_rays'] / 100.0))
    # in the empty upper right, not along the bottom -- at 0.06 it sat on top
    # of the tail bins it is describing
    ax.text(0.985, 0.80, f'the tail: {n_tail:,} muons, {d["reco_far"]:.1f} %',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            color=P.MUTED)
    fig.tight_layout()
    P.save(fig, os.path.join(OUT, 'efficiency_residual_tail.png'))


# --------------------------------------------------------------------------- #
# Figure 3 -- where on the chamber, at the tight criterion
# --------------------------------------------------------------------------- #

def map_csv(run_key: str) -> str:
    from qa_config import get_config
    cfg = get_config(run_key)
    return os.path.join(cfg.OUT_BASE, 'wft', 'maps', 'Plot_Data',
                        'efficiency_r2mm_cut.csv')


def fig_map(run_key: str) -> None:
    """Efficiency across the chamber face at the r < 2 mm criterion.

    The input is the CSV written by mx_june_wft/report/make_maps_2mm.py --
    40x40 bins of ~12 mm, success = a reconstructed point within 2 mm of the
    reference track, bins with < 5 reference muons masked.  Nothing is
    re-derived here; this is the same grid the June report leads with, redrawn
    in the deck's inks.

    TWO THINGS TO KNOW BEFORE QUOTING A NUMBER OFF THIS MAP.

    * 2 mm is the TIGHT criterion, not the slide's headline.  Detector-wide the
      same reconstruction is 86.5 % within 2 mm and 93.3 % within 5 mm.  The
      map is here to show that the efficiency is FLAT across the face, which is
      a statement no single number can make; read the level off the bars.
    * the denominator is every M3 ray, including the ones that miss the
      chamber, so the bins outside the active area are genuinely 0 % and the
      chamber's own edge is visible in the map.  That is why the plotted range
      is cropped to the populated region rather than to a nominal 40 x 40 cm.
    """
    import pandas as pd
    path = map_csv(run_key)
    if not os.path.exists(path):
        print(f'  ! missing {path} -- run mx_june_wft/report/make_maps_2mm.py')
        return
    df = pd.read_csv(path)
    xs = np.sort(df.ref_x_mm.unique())
    ys = np.sort(df.ref_y_mm.unique())
    grid = (df.pivot(index='ref_y_mm', columns='ref_x_mm', values='efficiency')
              .reindex(index=ys, columns=xs).to_numpy())
    shown = df.dropna(subset=['efficiency'])

    dx = float(np.diff(xs).mean())
    dy = float(np.diff(ys).mean())
    xe = np.append(xs - dx / 2, xs[-1] + dx / 2)
    ye = np.append(ys - dy / 2, ys[-1] + dy / 2)

    # The FIDUCIAL crop, derived from the data rather than from a nominal size:
    # keep the rows and columns whose median populated bin is above 30 %.  The
    # M3 acceptance is wider than the chamber, so without this the frame ends
    # in a wall of dead bins that is not a property of the detector -- it is
    # the telescope pointing past its edge.  The threshold is far from either
    # population (inside runs 70-100 %, outside sits at 0), so nothing about
    # the picture depends on where between them it is put.
    with np.errstate(invalid='ignore'):
        col = np.nanmedian(np.where(np.isfinite(grid), grid, np.nan), axis=0)
        row = np.nanmedian(np.where(np.isfinite(grid), grid, np.nan), axis=1)
    jx = np.where(np.nan_to_num(col) > 0.30)[0]
    jy = np.where(np.nan_to_num(row) > 0.30)[0]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cmap = P.efficiency_cmap()
    cmap.set_bad('#e6eaef')                 # too few muons, not low efficiency
    pc = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid) * 100.0,
                       cmap=cmap, vmin=50, vmax=100, shading='flat')
    ax.set_aspect('equal')
    ax.set_facecolor('#e6eaef')
    ax.set_xlim(xe[jx[0]], xe[jx[-1] + 1])
    ax.set_ylim(ye[jy[0]], ye[jy[-1] + 1])
    ax.set_xlabel('reference x  [mm]')
    ax.set_ylabel('reference y  [mm]')
    ax.grid(False)
    P.strip(ax)

    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03,
                      ticks=[50, 60, 70, 80, 90, 100], extend='min')
    cb.set_label('efficiency in the bin,  r < 2 mm  [%]', fontsize=11)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)

    inside = np.isfinite(grid[np.ix_(jy, jx)])
    print(f'  map: {int(len(shown))} populated bins, '
          f'{int(inside.sum())} inside the fiducial crop, '
          f'~{dx:.0f} x {dy:.0f} mm each')
    fig.tight_layout()
    P.save(fig, os.path.join(OUT, 'efficiency_map_2mm.png'))


# --------------------------------------------------------------------------- #
# The numbers, for the handoff / slide text
# --------------------------------------------------------------------------- #

def print_fleet() -> None:
    wft = {d['run_key']: d for d in fleet_table('wft')}
    hits = {d['run_key']: d for d in fleet_table('hits')}
    print(f'\n{"det":8s} {"run key":16s} {"rays":>7s} {"wft<=5":>7s} '
          f'{"hits<=5":>8s} {"R=10":>7s} {"has_any":>8s} {"spark%":>7s} '
          f'{"coreSig":>8s} {"generated":>12s}')
    for letter, det, key in FLEET:
        d = wft.get(key)
        if d is None:
            continue
        h = hits.get(key)
        import datetime as _dt
        ts = _dt.datetime.fromtimestamp(d['_mtime']).strftime('%Y-%m-%d')
        print(f'{det+" ("+letter+")":8s} {key:16s} {d["n_rays"]:7d} '
              f'{d["within_R"]:7.2f} '
              f'{(h["within_R"] if h else float("nan")):8.2f} '
              f'{d["eff_vs_R"]["10.0"]:7.2f} {d["has_any"]:8.2f} '
              f'{d["spark_frac"]:7.2f} {d["core_sigma_mm"]:8.3f} {ts:>12s}')
    print('\n(wft<=5 is the number to quote; hits<=5 is the old hit-time chain '
          'through the same accounting. spark% is of all firing events.)')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=('breakdown', 'tail', 'map'), default=None)
    ap.add_argument('--print', dest='show', action='store_true',
                    help='print the fleet table and exit')
    args = ap.parse_args()

    if args.show:
        print_fleet()
        return

    P.use()
    d = eff_json(HEADLINE)
    if d is None:
        sys.exit(f'no efficiency reduction for {HEADLINE} -- run '
                 f'mx_june_wft/02_efficiency.py {HEADLINE} --max-dropped -1')
    if 'eff_vs_R' not in d:
        sys.exit(f'{d["_path"]} predates the eff_vs_R reduction -- re-run '
                 f'mx_june_wft/02_efficiency.py {HEADLINE} --max-dropped -1')
    if not args.only or args.only == 'breakdown':
        fig_breakdown(d)
    if not args.only or args.only == 'tail':
        fig_tail(d)
    if not args.only or args.only == 'map':
        fig_map(HEADLINE)
    print_fleet()


if __name__ == '__main__':
    main()
