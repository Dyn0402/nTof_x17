#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_topology_figures.py -- the opening-angle topology split, for the Athens deck.

Two figures, and they are meant to be shown in this order:

  1. ``topology_split``   WHY the opening angle is split three ways.  A
     top-down map of the four chambers with one representative pair drawn per
     class, and under it the angular reach each class actually has in the data.
     The split is not a choice of binning -- with four chambers on a 90 deg
     pinwheel, *which two chambers a pair lands in* very nearly decides its
     opening angle, so a pooled spectrum is mostly a picture of the geometry.
  2. ``pairings``         the measured distributions, DATA ONLY, one panel per
     class with the individual arm pairs overlaid: three intra (A-A, C-C,
     D-D), two perpendicular (A-D, C-D), one opposing (A-C).

CHAMBER B IS NOT DRAWN INTO ANY PAIRING.  It has no field-shaping rings, so its
drift field is not uniform and it produces no time-to-depth ladder and therefore
no angle (``STATUS.md``, 2026-09-08).  It is drawn on the map, greyed, because
leaving it out would misrepresent the apparatus -- and because its absence is
what costs the B-D opposing channel, half the signal topology.

THE ANGLE SCALE IS PRELIMINARY and that is the leading caveat on figure 2.  The
per-arm scale ``k`` moves run to run by up to 25 % over one contiguous 48-hour
block, and the arm-A scintillator wall puts arm A's own scale 33 % out
(``STATUS.md``).  Every run here carries its OWN ``k`` -- this is the full pass,
not the borrowed-scale table -- but a real per-detector recalibration is
deferred to October.  So: the ordering of the three classes and the shape within
a class are robust; an absolute angle is not.  Both figures are badged.

Everything below the loader is drawing; the numbers come from one file, the
campaign pair table that ``campaign_angle.py`` writes.

    python ntof_athens_26/make_topology_figures.py
    python ntof_athens_26/make_topology_figures.py --only pairings
    python ntof_athens_26/make_topology_figures.py --norm counts --bin 15

On the Windows box the data disk is a drive letter, so point the tree at it::

    X17_ROOT=D:/x17 python ntof_athens_26/make_topology_figures.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(REPO / 'mpgd26')):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths  # noqa: E402

# The deck's house style, shared with the MPGD2026 talk this deck grew out of.
# A chamber that changes colour between two decks is a chamber the audience has
# to learn twice.
import plotstyle as P  # noqa: E402

# The chamber geometry, imported rather than re-typed.  `ntof_tracking.reco`
# resolves the July tree through `common.beam_july_paths`, which reads its own
# variable -- so bridge it from `paths` rather than asking the user to set two.
os.environ.setdefault('X17_BEAM_JULY', str(paths.spell('beam_july')))
from ntof_tracking.reco import geometry as G  # noqa: E402

OUT = HERE / 'figures'

#: The campaign pair table: every unordered pair of selected tracks in one
#: trigger, over 33 runs of the condor full pass, each run on its own `k`.
PAIRS = paths.spell('out', 'angle_campaign', 'pairs.parquet')

#: A 17 MeV boson from the 20.6 MeV M1 transition cannot make a pair below this.
X17_MIN_DEG = 109.0

#: Opposing and nearly collinear.  Imported rather than retyped so this figure
#: and the analysis can never disagree about where the category starts.
from sept26_prelim_analysis.tight_coincidence import (  # noqa: E402
    BACK_TO_BACK_DEG)

#: The arm pairs each class is built from, in the order they are drawn.  B is
#: absent by construction: see the module docstring.
CLASSES = {
    'intra': dict(
        pairs=[('A', 'A'), ('C', 'C'), ('D', 'D')],
        blurb='both legs in ONE chamber',
        # The map draws ONE representative pair.  All three are anchored on A so
        # the panels differ only in where the second leg goes, which is the
        # whole distinction being taught.
        draw=('A', 'A'), bisector=90.0),
    'perpendicular': dict(
        pairs=[('A', 'D'), ('C', 'D')],
        blurb='neighbouring chambers',
        # A sits at screen angle 90 deg, D at 180 (the top-down flip in
        # _arm_frame swapped D onto the left, where B used to be).  The two
        # legs are placed at bisector +/- theta/2 in THAT order, so the arm at
        # the larger screen angle (D, 180) has to be listed second here, not
        # A -- draw's order is "which arm gets +theta/2", not an electron/
        # positron assignment.
        draw=('D', 'A'), bisector=135.0),
    'opposing': dict(
        pairs=[('A', 'C')],
        blurb='facing chambers',
        draw=('A', 'C'), bisector=0.0),
}

#: Which chamber's colour identifies a curve.  Within a class the arm pairs are
#: told apart by the arm that is NOT shared -- A-D is "the A one", C-D "the C
#: one" -- so the hue always means a chamber and never a class.  A-C is alone in
#: its panel and takes the ink instead of borrowing a chamber's identity.
def series_color(a1: str, a2: str) -> str:
    if a1 == a2:
        return P.DET_COLOR[a1]
    if a1 == 'A' and a2 == 'C':
        return P.INK
    return P.DET_COLOR[a1 if a1 != 'D' else a2]


def label_of(a1: str, a2: str) -> str:
    return f'{a1}\u2013{a2}'


def thousands(n: int) -> str:
    """1 234, thin space.  A comma reads as a decimal point in half the room.

    Applied to the NUMBER and never to a whole sentence -- running ``.replace``
    over the sentence eats its punctuation along with the separator.
    """
    return f'{n:,}'.replace(',', '\u2009')


# --------------------------------------------------------------------------- #
# the data
# --------------------------------------------------------------------------- #
def load(pairs_path: Path, drop_b2b: bool = False) -> pd.DataFrame:
    """Real pairs only, chamber B dropped.  One row per pair.

    ``mixed`` is the event-mixed null that ``campaign_angle`` writes beside the
    data; these figures are data only, so it is dropped here rather than
    filtered at every use.

    THE BACK-TO-BACK PAIRS ARE IN BY DEFAULT.  They are opposing pairs above
    170 deg, and the reason to suspect them is real -- one charged particle
    crossing the target and punching through both facing chambers reads as a
    perfectly time-coincident pair at ~180 deg.  But that is a hypothesis about
    what they are, not a demonstration, and they sit inside the X17 region, so
    hiding them from the spectrum hides the biggest single decision in it.
    They are shown, marked, and the cut version is drawn beside them.
    """
    d = pd.read_parquet(paths.require(pairs_path, 'the campaign pair table -- '
                                      'run campaign_angle.py'))
    d = d[(~d.mixed) & (d.arm1 != 'B') & (d.arm2 != 'B')].copy()
    return d[~d.back_to_back] if drop_b2b else d


def reach(d: pd.DataFrame) -> pd.DataFrame:
    """Where each class actually lands: the 5-95 % span and the median."""
    rows = []
    for name in CLASSES:
        g = d[d.topo == name].open_deg.to_numpy()
        p5, p50, p95 = np.percentile(g, [5, 50, 95])
        rows.append(dict(topology=name, n=len(g), p5=p5, median_deg=p50, p95=p95,
                         frac_above_x17=float((g > X17_MIN_DEG).mean())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the map -- chamber geometry in the transverse (X, Z) plane, looking down
# --------------------------------------------------------------------------- #
def _arm_frame(arm: str):
    """(u_hat, w_hat, strip distance, pinwheel offset) in the 2-D (X, Z) plane.

    ``w_hat`` is the OUTWARD normal, so the strip plane sits at ``+d_strip*w``
    and the 30 mm drift gap is the slab just inside it.  The pinwheel offset
    slides each chamber along ``-u_hat``, which is what makes the four of them
    a pinwheel rather than a square.

    Screen y is physical +Z (arm A up, arm C down) and screen x is physical
    -X.  Plain ``+X`` here would put screen_x x screen_y = X x Z = -Y facing
    the viewer, i.e. the view from BELOW (``geometry.py``'s own top-down
    convention is screen_x = Z, screen_y = X, the transpose of this -- it
    trades which axis is horizontal instead of which way X points, and would
    swap A/C with D/B rather than just D with B).  Negating X instead gives
    (-X) x Z = +Y facing the viewer, the view from ABOVE, while leaving A and
    C exactly where they were -- it only swaps D and B left-for-right, which
    is the whole fix.
    """
    w = np.array([-G.W_HAT[arm][0], G.W_HAT[arm][2]], float)
    u = np.array([-G.U_HAT[arm][0], G.U_HAT[arm][2]], float)
    front = G.MM_DIST_Z if arm in ('A', 'C') else G.MM_DIST_X
    return u, w, front + G.W_STRIP, G.PINWHEEL[arm]


def _slab(arm: str) -> np.ndarray:
    """The drift gap of one arm as a polygon in (X, Z), at true scale."""
    u, w, d_strip, pin = _arm_frame(arm)
    half = G.MM_SIZE_U / 2.0
    c = -pin * u
    lo, hi = d_strip - G.DRIFT_GAP, d_strip
    return np.array([c - half * u + lo * w, c + half * u + lo * w,
                     c + half * u + hi * w, c - half * u + hi * w])


def _hit(alpha_deg: float, arm: str):
    """Where a leg leaving the target at ``alpha_deg`` crosses ``arm``'s strips.

    Returns ``(point, u_local)``; ``u_local`` is the in-plane coordinate the
    active area is defined in, so the caller can check the leg is drawable
    before drawing a track that misses the chamber.
    """
    u, w, d_strip, pin = _arm_frame(arm)
    dirn = np.array([np.cos(np.radians(alpha_deg)),
                     np.sin(np.radians(alpha_deg))])
    denom = float(dirn @ w)
    if denom <= 1e-9:
        return None, np.inf
    pt = dirn * (d_strip / denom)
    return pt, float(pt @ u) + pin


def draw_map(ax, name: str, theta_deg: float) -> None:
    """One class: the four chambers, and a pair that lands the way it must."""
    spec = CLASSES[name]
    a1, a2 = spec['draw']
    live = {a1, a2}

    ax.set_aspect('equal')
    ax.set_xlim(-272, 272)
    ax.set_ylim(-272, 272)
    ax.axis('off')

    for arm in ('A', 'B', 'C', 'D'):
        poly = _slab(arm)
        on = arm in live
        if arm == 'B':
            # Drawn, never used.  The hatch is the whole reason B-D is missing
            # from the opposing panel, so it has to be visible there.
            ax.add_patch(Polygon(poly, closed=True, facecolor='#eceff3',
                                 edgecolor=P.MUTED, lw=1.0, hatch='///',
                                 zorder=2, alpha=0.9))
        else:
            col = P.DET_COLOR[arm]
            ax.add_patch(Polygon(
                poly, closed=True,
                facecolor=col if on else '#e9ecf1',
                alpha=0.30 if on else 1.0,
                edgecolor=col if on else P.LINE,
                lw=2.0 if on else 1.0, zorder=3 if on else 2))

        # the letter, just outside the slab, on the chamber's own axis
        u, w, d_strip, pin = _arm_frame(arm)
        lab = (d_strip + 26) * w - pin * u
        ax.text(*lab, arm, ha='center', va='center',
                fontsize=15 if arm in live else 13,
                fontweight='bold',
                color=P.MUTED if arm == 'B' else
                      (P.DET_COLOR[arm] if arm in live else '#aab2bd'),
                zorder=6)

    # the target: the He-3 capsule, at true scale.  It is 20 mm across against
    # a 470 mm span, and that IS the point -- every pair starts from a spot.
    ax.add_patch(Circle((0, 0), G.HE3_R_MAX, facecolor='#c1841c',
                        edgecolor=P.INK, lw=0.9, zorder=7))

    # the pair.  Legs sit symmetrically about the class's own axis, opened to
    # the median the data actually shows, so the sketch is to scale in angle as
    # well as in length.
    bis = spec['bisector']
    # `side` is which way round the bisector the leg sits, and so which way to
    # push its label: outward, never into the arc drawn between the two.
    legs = [(bis + theta_deg / 2.0, a1, +1.0, '#a02c52', 'e$^{-}$'),
            (bis - theta_deg / 2.0, a2, -1.0, '#d6402c', 'e$^{+}$')]
    halo = [pe.withStroke(linewidth=2.6, foreground='white', alpha=0.95)]
    for alpha, arm, side, col, tag in legs:
        pt, u_local = _hit(alpha, arm)
        if pt is None or abs(u_local) > G.MM_SIZE_U / 2:
            raise ValueError(f'{name}: leg at {alpha:.1f} deg misses {arm} '
                             f'(u = {u_local:.0f} mm)')
        ax.add_patch(FancyArrowPatch(
            (0, 0), tuple(pt), arrowstyle='-|>', mutation_scale=13,
            lw=2.2, color=col, shrinkA=6, shrinkB=0, zorder=8))
        # far enough along the leg to clear the theta label, which sits on the
        # bisector between the two and has nowhere else to go
        n = np.array([-pt[1], pt[0]]) / np.linalg.norm(pt)
        ax.text(*(pt * 0.80 + n * 25 * side), tag, fontsize=12,
                fontweight='bold', color=col, ha='center', va='center',
                path_effects=halo, zorder=9)

    # the angle itself, as an arc between the legs
    r = 74.0
    t = np.radians(np.linspace(bis - theta_deg / 2, bis + theta_deg / 2, 120))
    ax.plot(r * np.cos(t), r * np.sin(t), color=P.INK, lw=1.3, ls='--',
            alpha=0.8, zorder=8)
    lx, ly = ((r + 46) * np.cos(np.radians(bis)),
              (r + 46) * np.sin(np.radians(bis)))
    ax.text(lx, ly, f'\u03b8 \u2248 {theta_deg:.0f}\u00b0', fontsize=14,
            fontweight='bold', color=P.INK, ha='center', va='center',
            path_effects=halo, zorder=10)


def draw_reach(ax, R: pd.DataFrame) -> None:
    """Where the three classes land on one 0-180 deg axis.  The punchline."""
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.80, len(CLASSES) - 0.30)
    ax.invert_yaxis()
    ax.axvspan(X17_MIN_DEG, 180, color=P.BAND_SIGNAL, alpha=0.10, lw=0,
               zorder=1)
    ax.axvline(X17_MIN_DEG, color=P.BAND_SIGNAL, lw=1.4, ls='--', zorder=4)
    ax.text(X17_MIN_DEG + 3, -0.74, 'X17 must land here  (\u03b8 \u2265 109\u00b0)',
            fontsize=11, fontweight='bold', color=P.BAND_SIGNAL,
            ha='left', va='top', zorder=5)

    for i, name in enumerate(CLASSES):
        row = R[R.topology == name].iloc[0]
        col = P.INK if name == 'opposing' else P.MUTED
        ax.plot([row.p5, row.p95], [i, i], lw=9, solid_capstyle='round',
                color=col, alpha=0.30, zorder=3)
        ax.plot([row.median_deg], [i], marker='|', ms=16, mew=2.6, color=col,
                zorder=4)
        # Both annotations hang OUTSIDE the axis in a fixed gutter.  Hung off
        # p5/p95 instead they step in and out with the data and the column of
        # labels reads as ragged noise.
        ax.text(-5, i, name, ha='right', va='center', fontsize=12.5,
                fontweight='bold', color=P.INK, zorder=5, clip_on=False)
        ax.text(185, i, f'{row.frac_above_x17 * 100:.0f} % above 109\u00b0',
                ha='left', va='center', fontsize=11, color=P.MUTED, zorder=5,
                clip_on=False)

    ax.set_yticks([])
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel('opening angle \u03b8 (deg)', labelpad=6)
    ax.grid(axis='y', visible=False)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# the figures
# --------------------------------------------------------------------------- #
def fig_topology_split(d: pd.DataFrame, with_reach: bool = True):
    R = reach(d)
    fig = plt.figure(figsize=(12.2, 7.1 if with_reach else 5.2))
    # The maps want the full width; the reach strip wants gutters either side
    # for its row labels.  Two independent placements rather than one grid, so
    # neither constrains the other.
    gs = fig.add_gridspec(1, 3, wspace=0.05, left=0.015, right=0.985,
                          top=0.815 if with_reach else 0.80,
                          bottom=0.37 if with_reach else 0.19)

    for j, name in enumerate(CLASSES):
        ax = fig.add_subplot(gs[0, j])
        med = float(R[R.topology == name].median_deg.iloc[0])
        draw_map(ax, name, med)
        pairs = '   '.join(label_of(*p) for p in CLASSES[name]['pairs'])
        ax.set_title(name, loc='center', fontsize=17, fontweight='bold',
                     color=P.INK, pad=30)
        ax.text(0.5, 1.045, CLASSES[name]['blurb'], transform=ax.transAxes,
                ha='center', va='bottom', fontsize=11.5, color=P.MUTED)
        ax.text(0.5, -0.02, pairs, transform=ax.transAxes, ha='center',
                va='top', fontsize=13.5, fontweight='bold', color=P.INK)

    if with_reach:
        draw_reach(fig.add_axes([0.085, 0.155, 0.755, 0.155]), R)

    fig.suptitle('Which two chambers a pair lands in nearly decides its '
                 'opening angle',
                 x=0.012, y=0.985, ha='left', va='top', fontsize=17,
                 fontweight='bold', color=P.INK)
    # Broken by hand rather than by `wrap=True`: wrapping measures against the
    # whole canvas and runs the last line straight under the badge.
    fig.text(0.012, 0.012,
             'Transverse section, true scale \u2014 each slab is one 30 mm drift '
             'gap, and the dot at the centre is the 20 mm \u00b3He capsule.  '
             '\u03b8 is drawn at each class\u2019s measured median'
             + ('; bars span 5\u201395 %.\n' if with_reach else '.\n')
             + 'Chamber B (hatched) has no field-shaping rings, so it measures '
               'no angle and enters no pairing \u2014 which is what costs the B\u2013D '
               'opposing channel.',
             ha='left', va='bottom', fontsize=9.5, color=P.MUTED,
             linespacing=1.5)
    return fig, R


def _spectrum(g: pd.DataFrame, edges: np.ndarray, norm: str):
    n, _ = np.histogram(g.open_deg.to_numpy(), bins=edges)
    w = np.diff(edges)
    if norm == 'density':
        tot = max(n.sum(), 1)
        return n / tot / w, np.sqrt(n) / tot / w, n
    return n.astype(float), np.sqrt(n), n


#: Where each panel's key can sit without landing on a curve.  Fixed by hand
#: because it depends on the SHAPE of the data, and the shape is not going to
#: change: intra and perpendicular leave the top right empty, opposing the top
#: left.  An automatic "label the peak" placement was tried first and collided
#: on every panel -- within a class the peaks sit almost on top of each other,
#: which is the very thing the figure is showing.
KEY_LOC = {'intra': 'upper right', 'perpendicular': 'upper right',
           'opposing': 'upper left'}


def fig_pairings(d: pd.DataFrame, d_cut: pd.DataFrame, norm: str = 'density',
                 binw: float = 5.0, only: str | None = None):
    """``d`` is drawn; ``d_cut`` is the back-to-back-removed version, ghosted.

    Only the opposing panel differs between the two, because back-to-back is
    an opposing-only category by construction.
    """
    edges = np.arange(0.0, 180.0 + binw, binw)
    mid = 0.5 * (edges[:-1] + edges[1:])
    names = [only] if only else list(CLASSES)

    fig, axes = plt.subplots(
        1, len(names), figsize=(4.5 * len(names) + 0.7, 5.2), squeeze=False)
    axes = axes[0]
    rows = []

    for ax, name in zip(axes, names):
        ax.axvspan(X17_MIN_DEG, 180, color=P.BAND_SIGNAL, alpha=0.09, lw=0,
                   zorder=1)
        ax.axvline(X17_MIN_DEG, color=P.BAND_SIGNAL, lw=1.3, ls='--', zorder=2)

        for a1, a2 in CLASSES[name]['pairs']:
            g = d[(d.arm1 == a1) & (d.arm2 == a2)]
            y, e, n = _spectrum(g, edges, norm)
            col = series_color(a1, a2)
            ax.stairs(y, edges, color=col, lw=2.3, zorder=5,
                      label=f'{label_of(a1, a2)}   n = {thousands(len(g))}')
            ax.fill_between(mid, y - e, y + e, step='mid', color=col,
                            alpha=0.20, lw=0, zorder=4)
            for m, v, ev, nv in zip(mid, y, e, n):
                rows.append(dict(topology=name, pair=label_of(a1, a2),
                                 theta_deg=m, value=v, err=ev, n=int(nv),
                                 norm=norm))

        # The back-to-back pairs are IN the solid curve.  What this panel adds
        # is where they are and what the spectrum looks like without them, so
        # the audience can judge the cut rather than be handed its result.
        if name == 'opposing':
            g = d_cut[(d_cut.arm1 == 'A') & (d_cut.arm2 == 'C')]
            full = d[(d.arm1 == 'A') & (d.arm2 == 'C')]
            nb2b = int(full.back_to_back.sum())
            ax.axvspan(BACK_TO_BACK_DEG, 180, color=P.COPPER, alpha=0.16, lw=0,
                       zorder=2)
            y, _, n = _spectrum(g, edges, norm)
            ax.stairs(y, edges, color=P.MUTED, lw=1.6, ls=(0, (4, 2.5)),
                      zorder=6,
                      label=f'without them   n = {thousands(len(g))}')
            # Kept narrow and low-left, the one corner of this panel the
            # spectrum never reaches: A-C has no pairs below ~90 deg.
            ax.annotate(
                f'shaded: {thousands(nb2b)} pairs at\n'
                f'\u03b8 \u2265 {BACK_TO_BACK_DEG:.0f}\u00b0.  One particle\n'
                f'through both chambers\n'
                f'would land there and is\n'
                f'not a pair \u2014 but that is\n'
                f'a hypothesis, not a cut\n'
                f'this figure has applied',
                xy=(0.035, 0.70), xycoords='axes fraction', ha='left',
                va='top', fontsize=9.5, color=P.MUTED, linespacing=1.5,
                zorder=7)
            for m, v, nv in zip(mid, y, n):
                rows.append(dict(topology=name, pair='A\u2013C (back-to-back cut)',
                                 theta_deg=m, value=v, err=np.nan, n=int(nv),
                                 norm=norm))

        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel('opening angle \u03b8 (deg)')
        ax.set_title(name, loc='left', fontsize=15, fontweight='bold',
                     color=P.INK, pad=12)
        # The key is the secondary encoding the palette check requires: the
        # line sample carries the hue, the text beside it carries the name, so
        # chamber D's low-contrast pink is never asked to identify a series on
        # its own.
        ax.legend(loc=KEY_LOC[name], fontsize=11, handlelength=1.6,
                  handletextpad=0.7, labelspacing=0.55, borderaxespad=0.9,
                  labelcolor=P.INK)
        P.strip(ax)

    ylab = (f'fraction of pairs per {binw:g}\u00b0' if norm == 'density'
            else f'pairs per {binw:g}\u00b0')
    axes[0].set_ylabel(ylab)

    if not only:
        fig.suptitle('Measured opening angle, by arm pair \u2014 data only, '
                     'no model, nothing subtracted, nothing cut',
                     x=0.008, y=0.988, ha='left', va='top', fontsize=16,
                     fontweight='bold', color=P.INK)
        fig.subplots_adjust(top=0.845, bottom=0.205, left=0.058, right=0.99,
                            wspace=0.17)
        fig.text(0.008, 0.012,
                 '33 runs of the n_TOF campaign on the condor full pass, each '
                 'run on its own angle scale.  Every unordered pair of '
                 'selected tracks in one trigger; chamber B carries no angle '
                 'and is excluded.\n'
                 + ('Each curve is normalised to unit area, so the panels '
                    'compare shape and not yield.  ' if norm == 'density'
                    else '')
                 + 'Bands are Poisson.  Purple: \u03b8 \u2265 109\u00b0, where an X17 pair '
                   'must land.  Copper: the back-to-back pairs, kept.',
                 ha='left', va='bottom', fontsize=9.5, color=P.MUTED,
                 linespacing=1.5)
    else:
        fig.subplots_adjust(top=0.90, bottom=0.145, left=0.155, right=0.97)

    return fig, pd.DataFrame(rows)


def preliminary(fig, x=0.988, y=0.985, va='top') -> None:
    """The badge PLAN.md sec 8 attaches to every reconstructed quantity.

    ``va='bottom'`` parks it beside the provenance line instead, for the
    figures whose headline runs the full width of the canvas.
    """
    fig.text(x, y, 'PRELIMINARY', ha='right', va=va, fontsize=10.5,
             fontweight='bold', color='#b04a3a', alpha=0.85,
             bbox=dict(boxstyle='round,pad=0.30', facecolor='#fdf1ef',
                       edgecolor='#e3bdb6', lw=0.8))


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def save(fig, name: str, data: pd.DataFrame | None) -> None:
    """PNG for the slide, PDF for live text, CSV so the figure can be checked."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}')
        print(f'  -> {OUT / f"{name}.{ext}"}')
    plt.close(fig)
    if data is not None:
        data.to_csv(OUT / f'{name}.csv', index=False)
        print(f'  -> {OUT / f"{name}.csv"}')


INDEX_CSS = """
body{margin:0;padding:32px 40px;background:#fbfcfe;color:#1b2430;
 font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
h1{font-size:22px;margin:0 0 4px}p.lede{color:#6a7583;margin:0 0 28px;max-width:62em}
figure{margin:0 0 40px}figure img{width:100%;max-width:1200px;display:block;
 border:1px solid #e4e8ee;border-radius:6px;background:#fff}
figcaption{color:#6a7583;font-size:13px;margin-top:8px;max-width:62em}
code{background:#eef1f5;padding:1px 5px;border-radius:3px;font-size:12.5px}
"""


def write_index(built: list[tuple[str, str]]) -> None:
    rows = '\n'.join(
        f'<figure><img src="{n}.png" alt="{n}">'
        f'<figcaption><code>{n}.png</code> &middot; <code>{n}.pdf</code> '
        f'&middot; <code>{n}.csv</code><br>{cap}</figcaption></figure>'
        for n, cap in built)
    (OUT / 'index.html').write_text(
        f'<!doctype html><meta charset="utf-8">'
        f'<title>Athens deck \u2014 topology figures</title>'
        f'<style>{INDEX_CSS}</style>'
        f'<h1>Opening-angle topology figures</h1>'
        f'<p class="lede">Built by <code>make_topology_figures.py</code> from '
        f'the campaign pair table. Preview only \u2014 the deck uses the PDFs.</p>'
        f'{rows}', encoding='utf-8')
    print(f'  -> {OUT / "index.html"}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--pairs', default=str(PAIRS),
                    help='campaign pair table (default: the staged one)')
    ap.add_argument('--only', choices=['topology', 'pairings'],
                    help='build one family instead of all')
    ap.add_argument('--norm', choices=['density', 'counts'], default='density',
                    help='overlay shapes (default) or raw counts')
    ap.add_argument('--bin', type=float, default=5.0, dest='binw',
                    help='histogram bin width in degrees (default 5)')
    ap.add_argument('--b2b', choices=['keep', 'drop'], default='keep',
                    help='back-to-back opposing pairs (>= 170 deg): show them '
                         '(default) or cut them')
    a = ap.parse_args()

    P.use()
    # `d` is what the figures draw; `d_cut` is the other choice, ghosted onto
    # the opposing panel so the cut is visible whichever way round it is made.
    d = load(Path(a.pairs), drop_b2b=(a.b2b == 'drop'))
    d_cut = load(Path(a.pairs), drop_b2b=(a.b2b != 'drop'))
    built = []

    if a.only in (None, 'topology'):
        print('topology split')
        fig, R = fig_topology_split(d, with_reach=True)
        preliminary(fig)
        save(fig, 'topology_split', R)
        built.append(('topology_split',
                      'The explainer: why the opening angle is split three '
                      'ways, and where each class lands in the data.'))

        fig, R = fig_topology_split(d, with_reach=False)
        save(fig, 'topology_map', R)
        built.append(('topology_map',
                      'The same maps without the reach strip \u2014 for a slide '
                      'that builds the strip in afterwards.'))
        print(R.to_string(index=False))

    if a.only in (None, 'pairings'):
        print('pairings')
        fig, S = fig_pairings(d, d_cut, norm=a.norm, binw=a.binw)
        preliminary(fig)
        save(fig, 'pairings', S)
        built.append(('pairings',
                      'The measured distributions, data only, one panel per '
                      'class with the arm pairs overlaid.'))

        for name in CLASSES:
            fig, s = fig_pairings(d, d_cut, norm=a.norm, binw=a.binw,
                                  only=name)
            preliminary(fig, x=0.975, y=0.975)
            save(fig, f'pairings_{name}', s)
            built.append((f'pairings_{name}',
                          f'The {name} panel alone, for a build.'))

        other = 'counts' if a.norm == 'density' else 'density'
        fig, S = fig_pairings(d, d_cut, norm=other, binw=a.binw)
        preliminary(fig)
        save(fig, f'pairings_{other}', S)
        built.append((f'pairings_{other}',
                      f'The same three panels as {other} \u2014 the alternative '
                      f'normalisation, in case the yield is the point.'))

    write_index(built)
    print(f'\n{len(d):,} pairs, B excluded, back-to-back {a.b2b}')
    print(d.groupby(['topo', 'arm1', 'arm2']).size().to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
