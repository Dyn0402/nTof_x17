#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pair_qa_figures.py -- what kind of object each pair is, per arm pairing.

The companion to `make_topology_figures.py`.  That one shows the opening angle;
this one shows the QUALITY of the pairs behind it, split exactly the same way --
three intra, two perpendicular, one opposing -- so a feature in a spectrum can
be traced to the sample that made it.  The question these answer is the one
that has to come before any physics: **is there filtering still to do, and on
what.**

EVERY PANEL CARRIES THE EVENT-MIXED NULL, and that is the point of the set.  A
mixed pair is two tracks from different triggers, so it is an accidental by
construction.  Where the real and mixed curves lie on top of each other, a cut
on that variable removes signal and background alike and buys nothing; where
they separate, there is something to cut on.  Reading these plots without the
grey curve would be reading half of each one.

THE PANELS SHARE THEIR LAYOUT AND COLOURS WITH `make_topology_figures`, which
is imported rather than copied: a chamber keeps its hue, an arm pair keeps its
label, and the three classes stay in the same order and the same places.

    python ntof_athens_26/make_pair_qa_figures.py
    python ntof_athens_26/make_pair_qa_figures.py --only chi2dof_worst
    X17_ROOT=D:/x17 python ntof_athens_26/make_pair_qa_figures.py
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(REPO / 'mpgd26'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths  # noqa: E402
import plotstyle as P  # noqa: E402

# The class split, the colours, the labels and the output helpers -- one
# definition, used by both figure sets.
from make_topology_figures import (  # noqa: E402
    CLASSES, INDEX_CSS, label_of, preliminary, save, series_color, thousands)

OUT = HERE / 'figures'
QA = paths.spell('out', 'pair_qa', 'pairs_qa_campaign.parquet')

#: Below this a histogram is noise wearing a shape, and on a shared axis its
#: spikes set the y limit for every other curve.  Such a series is named in the
#: key with its count instead of being drawn, so the reader learns the sample
#: is thin rather than being shown a shape that is not one.
MIN_N = 30

#: The provenance line is set at this width.  It matters: `savefig` runs with
#: `bbox_inches='tight'`, so a single long line of figure text extends the
#: SAVED CANVAS to fit it -- which silently produced a 4000 px image out of a
#: 13.8 in figure until the text was wrapped.
FOOT_COLS = 168

#: One entry per figure.  ``lo``/``hi`` are the drawn range: everything outside
#: it is counted into the end bins and the fraction is annotated, rather than
#: quietly dropped -- several of these have tails that run to 10^24 and an axis
#: that simply ignored them would be a lie about the sample.
METRICS = {
    'chi2dof_worst': dict(
        label='worst χ²/dof of the four track fits',
        scale='log', lo=1.0, hi=3000.0, bins=44,
        why='Nothing in this chain cuts on χ². The median pair has a worst-view '
            'χ²/dof near 40 and the tail runs past 10⁴.'),
    'n_strips_min': dict(
        label='fewest strips on any view of either leg',
        scale='linear', lo=10, hi=70, bins=60,
        why='The reconstruction needs ≥ 10, so the left edge is the existing '
            'cut and not a feature of the data.'),
    'q_total_min': dict(
        label='total charge of the quieter leg  [ADC]',
        scale='log', lo=1e2, hi=1e6, bins=44,
        why='A real pathology: q_total overflows on part of the sample. The '
            'median is ~1.8 × 10³ and the 99th percentile is ~10¹³.'),
    'dca_worst': dict(
        label='worse leg’s distance of closest approach to the beam axis  [mm]',
        scale='linear', lo=0.0, hi=30.0, bins=40,
        why='The 30 mm selection is the right-hand edge. The distribution '
            'rises INTO it, so the cut is not isolating a peak.'),
    'sep_mm': dict(
        label='closest approach of the two legs to each other  [mm]',
        scale='log', lo=0.5, hi=600.0, bins=44,
        why='For a pair born in a 20 mm capsule this should be small. The '
            'median is ~100 mm.'),
    'v_r': dict(
        label='reconstructed vertex radius from the beam axis  [mm]',
        scale='log', lo=1.0, hi=1000.0, bins=44,
        why='The capsule is r = 10 mm. The median vertex sits at ~67 mm.'),
    'dt_track_ns': dict(
        label='difference of the two legs’ fitted track times  [ns]',
        scale='linear', lo=-900.0, hi=900.0, bins=60,
        why='NOT a coincidence measurement: both legs share one trigger and a '
            'leg’s t0 moves with its depth in the 30 mm gap, so this is '
            'dominated by drift depth.'),
    'delta_t': dict(
        label='scintillator Δt between the two arms  [ns]',
        scale='linear', lo=-150.0, hi=150.0, bins=60,
        why='THE coincidence measurement, and it exists only where both arms '
            'are scintillator-tagged. A prompt peak on a broad pedestal — the '
            'pedestal is what a timing cut would remove. No mixed null here: '
            'tight_coincidence.py writes real pairs only.',
        absent='not defined for intra pairs:\none chamber, one arm time,\n'
               'so no arm-to-arm difference'),
    't_flash_ms': dict(
        label='neutron arrival time since the flash  [ms]',
        scale='log', lo=0.8, hi=100.0, bins=44,
        why='Per trigger, so a pair has exactly one. Whether the six arm pairs '
            'are drawn from the same neutron population is a real question.'),
}


def load(qa_path: Path) -> pd.DataFrame:
    """The campaign QA pair table, chamber B dropped, ready to plot.

    Both the real pairs and the event-mixed null are kept -- the null is half
    of every panel here, unlike the angle figures where the point was the data
    on its own.
    """
    d = pd.read_parquet(paths.require(
        qa_path, 'the pair QA table -- run '
                 'python -m sept26_prelim_analysis.pair_qa'))
    d = d[(d.arm1 != 'B') & (d.arm2 != 'B')].copy()
    # ns -> ms, so the axis reads in the units the flash veto is quoted in.
    d['t_flash_ms'] = d.t_flash_ns / 1e6
    return d


def _hist(v: np.ndarray, edges: np.ndarray):
    """Per-bin FRACTION over ``edges``.  Returns ``(y, err, n_finite, frac_outside)``.

    OUT-OF-RANGE VALUES ARE DROPPED, NOT PILED INTO THE END BINS, and the
    normalisation still divides by every finite value -- so the curve
    integrates to the in-range fraction and the rest is reported rather than
    drawn.  Clipping was the first version and it is wrong on a log axis: the
    end bins are the narrowest, dividing a 1 % pile by a 0.08-wide bin put a
    spike ten times the data's own peak at the left edge of half these
    figures.  A reader cannot tell that spike from a real population.

    NO DIVISION BY THE BIN WIDTH -- fixed 2026-09-12, and the bug it fixes had
    been in every log-scaled panel of this set.  This returned ``n / tot / w``
    with ``w = np.diff(edges)``, the bins' LINEAR widths, while `_edges` builds
    the log-scale metrics with ``np.geomspace`` and the axis is drawn
    log.  A log-spaced bin's linear width grows in proportion to x, so the
    plotted height was the per-bin fraction DIVIDED BY x: the left of the axis
    was lifted about 5x relative to the peak, and a small low-end shoulder was
    drawn as a mode of comparable height.  On ``chi2dof_worst`` that invented an
    apparent second population at chi2/dof ~ 2 on every arm pair containing
    chamber A, and the investigation of it is in ``chi2_bimodality/``.

    Why a plain fraction is the right answer for both scales rather than a
    per-scale width: the bins are UNIFORM IN THE PLOTTED COORDINATE either way
    (``linspace`` for linear, ``geomspace`` = uniform in log for log), so the
    per-bin fraction is already proportional to the density in the coordinate
    the reader is looking at.  It is also exactly what the y-axis label has
    said all along.  (Dividing by a width is not itself wrong -- it is correct
    and necessary when bins are UNEQUAL in the plotted coordinate, which is how
    ``chi2_bimodality``'s fine-binned linear view works.  The mistake was
    dividing by a linear width while binning geometrically.)
    """
    v = v[np.isfinite(v)]
    if not len(v):
        return None, None, 0, 0.0
    out = float(np.mean((v < edges[0]) | (v > edges[-1])))
    n, _ = np.histogram(v, bins=edges)
    tot = len(v)
    return n / tot, np.sqrt(n) / tot, tot, out


def _edges(spec) -> np.ndarray:
    if spec['scale'] == 'log':
        return np.geomspace(spec['lo'], spec['hi'], spec['bins'] + 1)
    return np.linspace(spec['lo'], spec['hi'], spec['bins'] + 1)


def figure_for(d: pd.DataFrame, metric: str, spec: dict):
    """One metric, three panels, the arm pairs overlaid and the null behind."""
    edges = _edges(spec)
    mid = np.sqrt(edges[:-1] * edges[1:]) if spec['scale'] == 'log' \
        else 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.9), squeeze=False)
    axes = axes[0]
    rows = []

    for ax, name in zip(axes, CLASSES):
        real = d[~d.mixed]
        mix = d[d.mixed]
        drawn = nulls = 0
        for a1, a2 in CLASSES[name]['pairs']:
            col = series_color(a1, a2)
            have = int(pd.to_numeric(
                real.loc[(real.arm1 == a1) & (real.arm2 == a2), metric],
                errors='coerce').notna().sum())
            if have < MIN_N:
                # n == 0 is "this metric does not exist here" and the panel
                # says so once, in the middle; repeating it per arm pair in
                # the key says the same thing three times.
                if have:
                    ax.plot([], [], color=col, lw=2.3,
                            label=f'{label_of(a1, a2)}   n = {have}  '
                                  f'— too few to plot')
                continue
            # Each arm pair gets ITS OWN null, in its own colour.  Pooling the
            # class's mixed pairs into one grey curve was the first attempt and
            # it is wrong here: A-A, C-C and D-D differ enough that the pooled
            # shape is nobody's null.  Dashed and thin, so the data still reads
            # first and the pairing of curve to null is by hue.
            g = mix[(mix.arm1 == a1) & (mix.arm2 == a2)]
            y, _, n, _ = _hist(
                pd.to_numeric(g[metric], errors='coerce').to_numpy(float),
                edges)
            if y is not None and n >= MIN_N:
                ax.stairs(y, edges, color=col, lw=1.2, ls=(0, (4, 2.5)),
                          alpha=0.85, zorder=3)
                nulls += 1
                for m, v in zip(mid, y):
                    rows.append(dict(topology=name,
                                     pair=f'{label_of(a1, a2)} (mixed)',
                                     metric=metric, x=m, value=v, err=np.nan))

            g = real[(real.arm1 == a1) & (real.arm2 == a2)]
            y, e, n, out = _hist(
                pd.to_numeric(g[metric], errors='coerce').to_numpy(float),
                edges)
            if y is None or not n:
                continue
            tail = f'   ({out * 100:.0f} % off-scale)' if out > 0.005 else ''
            ax.stairs(y, edges, color=col, lw=2.3, zorder=5,
                      label=f'{label_of(a1, a2)}   n = {thousands(n)}{tail}')
            ax.fill_between(mid, y - e, y + e, step='mid', color=col,
                            alpha=0.20, lw=0, zorder=4)
            drawn += 1
            for m, v, ev in zip(mid, y, e):
                rows.append(dict(topology=name, pair=label_of(a1, a2),
                                 metric=metric, x=m, value=v, err=ev))

        # Only claim a null when one was actually drawn.  `delta_t` has none:
        # `tight_coincidence.py` writes real pairs only, so the mixed column is
        # empty there and a legend entry for it would be a lie.
        if nulls:
            ax.plot([], [], color=P.MUTED, lw=1.2, ls=(0, (4, 2.5)),
                    label='dashed: the same pair, event-mixed')
        if not drawn:
            # An empty panel is a result, not a gap -- say which one.
            ax.text(0.5, 0.55, spec.get('absent', 'no pairs carry this '
                                        'measurement'),
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color=P.MUTED, style='italic',
                    linespacing=1.5, wrap=False)

        if spec['scale'] == 'log':
            ax.set_xscale('log')
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(name, loc='left', fontsize=14.5, fontweight='bold',
                     color=P.INK, pad=10)
        if ax.get_legend_handles_labels()[0]:
            # 'best', not a fixed corner: nine metrics put their peak in nine
            # different places, and one corner that suits chi2 sits on top of
            # the data in delta_t.
            ax.legend(loc='best', fontsize=9.5, handlelength=1.6,
                      handletextpad=0.7, labelspacing=0.45, borderaxespad=0.7,
                      labelcolor=P.INK)
        P.strip(ax)
        if not drawn:
            # A 0-1 density scale under an empty panel invites the reader to
            # compare it with the panels either side.  There is nothing to
            # compare, so the scale goes.
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)

    axes[0].set_ylabel('fraction of pairs per bin')
    # One axis label under the row.  These names are long, and three copies of
    # a long name collide with each other across the panel gaps.
    fig.supxlabel(spec['label'], y=0.175, fontsize=12.5, color=P.INK)
    fig.suptitle(spec['label'][0].upper() + spec['label'][1:],
                 x=0.007, y=0.988, ha='left', va='top', fontsize=16,
                 fontweight='bold', color=P.INK)
    fig.subplots_adjust(top=0.855, bottom=0.275, left=0.052, right=0.992,
                        wspace=0.16)
    # Kept to three wrapped lines.  The full argument for the null belongs on
    # qa_index.html, once, rather than under all nine figures.
    foot = (spec['why'] + '  Dashed is the same arm pair EVENT-MIXED — an '
            'accidental by construction.  Where it lies under the data a cut '
            'here removes both alike; where they separate, there is something '
            'to cut on.  Values off the axis are excluded from the curve but '
            'not from its normalisation, and the key gives the fraction.')
    fig.text(0.007, 0.012, textwrap.fill(foot, FOOT_COLS),
             ha='left', va='bottom', fontsize=9.5, color=P.MUTED,
             linespacing=1.5)
    return fig, pd.DataFrame(rows)


def write_index(built: list[tuple[str, str]]) -> None:
    rows = '\n'.join(
        f'<figure><img src="{n}.png" alt="{n}">'
        f'<figcaption><code>{n}.png</code> &middot; <code>{n}.pdf</code> '
        f'&middot; <code>{n}.csv</code><br>{cap}</figcaption></figure>'
        for n, cap in built)
    (OUT / 'qa_index.html').write_text(
        f'<!doctype html><meta charset="utf-8">'
        f'<title>Athens deck — pair QA</title>'
        f'<style>{INDEX_CSS}</style>'
        f'<h1>Pair quality, by arm pairing</h1>'
        f'<p class="lede">Built by <code>make_pair_qa_figures.py</code> from '
        f'<code>pair_qa.py</code>’s campaign table — the same pairs the '
        f'opening-angle figures are drawn from, verified pair for pair. '
        f'Grey dashed is the event-mixed null in every panel.</p>'
        f'{rows}', encoding='utf-8')
    print(f'  -> {OUT / "qa_index.html"}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--qa', default=str(QA), help='the pair QA table')
    ap.add_argument('--only', choices=sorted(METRICS),
                    help='build one metric instead of all')
    a = ap.parse_args()

    P.use()
    d = load(Path(a.qa))
    print(f'{int((~d.mixed).sum()):,} real pairs, '
          f'{int(d.mixed.sum()):,} mixed, chamber B excluded\n')

    built = []
    for metric in ([a.only] if a.only else list(METRICS)):
        spec = METRICS[metric]
        fig, S = figure_for(d, metric, spec)
        preliminary(fig)
        save(fig, f'qa_{metric}', S)
        built.append((f'qa_{metric}', spec['why']))
    write_index(built)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
