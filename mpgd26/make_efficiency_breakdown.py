#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_efficiency_breakdown.py -- the efficiency figures for the MPGD2026 talk.

    ../.venv/bin/python make_efficiency_breakdown.py
    ../.venv/bin/python make_efficiency_breakdown.py --only breakdown
    ../.venv/bin/python make_efficiency_breakdown.py --print   # numbers only

Two figures, both for the efficiency slide:

  efficiency_breakdown.png      det3's loss budget -- where every crossing muon
                                goes, in plain language, with an annotation box
                                whose every number is READ FROM THE JSON.
  efficiency_residual_tail.png  the |r| distribution on a log scale with the
                                5 mm match circle marked, plus efficiency vs
                                match radius -- the panel that explains the
                                "detected but >5 mm off track" slice.

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
HEADLINE = 'sat_det3'          # the chamber the main slide is about

# Plain-language rows for the loss budget, best -> worst, keyed to the
# categories of 02_efficiency.py.  Colours: one green for the answer, copper
# for "our position, not their signal", accent for the discharge, grey for
# unusable, red reserved for genuine blindness.
ROWS = [
    ('within_R',    'Reconstructed within 5 mm\n(the efficiency)',        '#2e8b57'),
    ('reco_far',    'Detected, point >5 mm\noff the telescope track',      P.COPPER),
    ('spark_cat',   'Sparked during this muon\n(self-quenching, no dead time)', P.ACCENT),
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
        f'see the muon: a {d["spark_cat"]:.1f}% spark coincidence (self-quenching, no\n'
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
    edges = np.asarray(d['r_hist_edges'], float)
    cts = np.asarray(d['r_hist_counts'], float)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    R = d['R_mm']

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 3.9))

    # (a) the tail itself, log counts, so the 3.7 % is visible at all
    ax = axs[0]
    ax.step(ctr, np.where(cts > 0, cts, np.nan), where='mid',
            color=P.DET_COLOR['A'], lw=1.6)
    ax.fill_between(ctr, 1e-1, np.where(cts > 0, cts, 1e-1), step='mid',
                    color=P.DET_COLOR['A'], alpha=0.16, lw=0)
    ax.axvline(R, color=P.COPPER, lw=1.6, ls='--')
    ax.set_yscale('log')
    ax.set_xlim(0, edges[-1])
    ax.set_ylim(0.7, max(cts.max() * 2.2, 10))
    ax.set_xlabel('|r|  detector − reference track  [mm]')
    ax.set_ylabel('muons per 0.25 mm')
    P.title(ax, 'The position tail',
            f'core σ {d["core_sigma_mm"]:.2f} mm, median {d["median_r_mm"]:.2f} mm')
    P.strip(ax)
    ax.text(R + 0.7, ax.get_ylim()[1] / 3.0,
            f'{R:.0f} mm match\n→ {d["within_R"]:.1f}% efficiency',
            fontsize=10, color=P.COPPER, va='top', ha='left', fontweight='bold')

    # (b) the recovery curve: efficiency vs match radius, same denominator
    ax = axs[1]
    radii = sorted(float(k) for k in d['eff_vs_R'])
    vals = [d['eff_vs_R'][str(r)] for r in radii]
    ax.plot(radii, vals, marker=P.DET_MARKER['A'], color=P.DET_COLOR['A'],
            lw=2.0, ms=5)
    ax.axvline(R, color=P.COPPER, lw=1.6, ls='--')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 30])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('match radius R  [mm]')
    ax.set_ylabel('efficiency  [% of crossings]')
    plateau = d['eff_vs_R']['30.0']
    ax.set_ylim(min(vals) - 3, max(plateau + 3, 100))
    ax.axhline(d['reco_at_all'], color=P.MUTED, lw=1.2, ls=':')
    ax.text(30, d['reco_at_all'] + 0.6,
            f'reconstructed at all  {d["reco_at_all"]:.1f}%',
            fontsize=9.5, color=P.MUTED, ha='right', va='bottom')
    P.title(ax, 'Efficiency vs match radius',
            f'{d["within_R"]:.1f}% at {R:.0f} mm → '
            f'{d["eff_vs_R"]["10.0"]:.1f}% at 10 mm; '
            f'the tail is near-misses, not failures')
    P.strip(ax)

    P.note(fig, f'det3 / {d["run_key"]}, {d["n_rays"]:,} reference muons · '
                f'{d["basis"]} · mx_june_wft/02_efficiency.py')
    fig.tight_layout()
    P.save(fig, os.path.join(OUT, 'efficiency_residual_tail.png'))


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
    ap.add_argument('--only', choices=('breakdown', 'tail'), default=None)
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
    print_fleet()


if __name__ == '__main__':
    main()
