#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figstyle.py -- the house style for every figure in this package.

The aim is an ordinary, good-looking scientific figure: the kind that reads
well in the HTML report, drops into a note or a paper without being resized,
and can be pasted onto a slide if someone wants it there.  It is **not** built
for a projector, and nothing here is pinned to a slide's aspect ratio.  Sizing
figures for a 16:9 canvas with type large enough for the back of a room is what
made the earlier figures ugly: stretched panels, banner-sized bold titles, and
line weights that swamped the data.

The defaults are close to what a careful matplotlib user would choose by hand:

* **A normal figure shape.**  Roughly 3:2 for a single panel -- see the sizes
  below, which are suggestions rather than a rule.  Pick the shape the data
  wants (a map wants to be square, a long time series wants to be wide) and let
  the aspect follow from that, not from where the figure might be shown.
* **Type at document size.**  ``BASE_PT`` is 10.5 pt on a ~7 in wide figure,
  which is the size text is meant to be *read* at.  Nothing needs to be legible
  from three metres away.
* **Light ink, heavy data.**  Thin spines, a faint grid behind the data, muted
  tick labels, and a title that names the message without shouting it.

Two rules this module still makes hard to break, because they are about
correctness rather than looks:

* **Every figure ships its numbers.**  :func:`save` writes ``foo.png`` *and*
  ``foo.csv``, and refuses to write the PNG alone unless you say
  ``data=NO_DATA`` and mean it.  A figure that cannot be rebuilt from a CSV
  beside it is a figure nobody can check.
* **Anything touching the reconstruction is badged.**  :func:`preliminary`
  stamps it; the badge is the caveat that ``PLAN.md`` §8 attaches to every
  reconstructed quantity.

The four-chamber palette is the Okabe-Ito subset, colour-blind safe and shared
with ``mpgd26/plotstyle.py`` -- a chamber that changes colour between documents
is a chamber the reader has to re-learn.  :func:`det_style` hands back a marker
with every colour so identity never rests on hue alone (D's ``#CC79A7`` is low
contrast on white, so give it a label or a marker, never a bare fill).

    import figstyle as fs
    fs.use()
    fig, ax = fs.figure()
    ...
    fs.preliminary(ax)
    fs.save(fig, out / 'opening_angle', data=df)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Ink -- shared with mpgd26/plotstyle.py
# --------------------------------------------------------------------------- #
INK = '#1b2430'
MUTED = '#6a7583'
LINE = '#c9ced6'            # spines
GRID = '#e6e9ee'            # grid, lighter than the spines
SURFACE = '#ffffff'
ACCENT = '#8a3f8f'          # mx17 purple
COPPER = '#d18a44'          # annotation / caution
TRACK = '#ff4f36'           # one sharp highlight, used sparingly

#: Categorical, fixed order, never cycled.  Okabe-Ito, colour-blind safe.
DET_COLOR = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73', 'D': '#CC79A7'}
DET_MARKER = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D'}

#: Reserved status fills.  Never reused as a series colour.
BAND_SIGNAL = '#8a3f8f'     # the 110-140 deg X17 region
BAND_DEAD = '#b04a3a'       # masked / dead / vetoed
BAND_CONTROL = '#6a7583'    # the intra-chamber control region

# --------------------------------------------------------------------------- #
# Sizes -- suggestions, not a canvas
# --------------------------------------------------------------------------- #
# These are ordinary document figure sizes in inches.  They exist so that
# figures in the same report end up a consistent width, not because anything
# has to fit a particular frame.  Deviate whenever the data has a shape of its
# own: a hit map should be square, an occupancy ladder can be tall.
#
# The type scale does not change with them.  A matplotlib point is absolute
# (1/72 in), so a HALF figure and a FIG figure made with the same rcParams have
# type the same physical size -- which is what you want as long as each is
# placed at its natural size.  Scaling a saved PNG up or down in a document is
# what breaks the type scale, so prefer making the figure at the width it will
# be shown.
#
#: The default: one panel, about 3:2.  Fills a report column at ~100 % zoom.
FIG = (6.8, 4.2)
#: Wide: two panels side by side, or a long time axis.
WIDE = (9.6, 3.6)
#: Full: a multi-panel grid (2x2, 1x4) that needs the extra width.
FULL = (9.6, 5.4)
#: Short and wide: a timeline or a ladder.
BANNER = (9.6, 3.0)
#: Half width -- a small inset, or two independent figures on one row.
HALF = (4.8, 3.4)
#: Third width, for a row of three separate figures.
THIRD = (3.4, 3.0)
#: Small and short -- a supporting panel beside a table.
QUARTER = (4.8, 2.7)

#: Deprecated alias of :data:`FULL`, kept so older scripts keep running.
#: There is no slide canvas any more; use ``FULL`` (or a size of your own).
SLIDE = FULL

BASE_PT = 10.5              # document type size, not projection type size


def use() -> None:
    """Apply the house rcParams.  Call once, before plotting."""
    mpl.rcParams.update({
        'figure.figsize': FIG,
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        # A conventional type scale: the title a little larger than the body,
        # axis labels at body size, tick labels a notch below.
        'font.size': BASE_PT,
        'axes.titlesize': BASE_PT * 1.15,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.labelsize': BASE_PT,
        'xtick.labelsize': BASE_PT * 0.92,
        'ytick.labelsize': BASE_PT * 0.92,
        'legend.fontsize': BASE_PT * 0.92,
        'axes.labelcolor': INK,
        'axes.edgecolor': LINE,
        'axes.linewidth': 0.9,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.7,
        'grid.alpha': 1.0,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK,
        'ytick.labelcolor': INK,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'legend.frameon': False,
        'legend.handlelength': 1.8,
        'legend.borderaxespad': 0.4,
        'lines.linewidth': 1.6,
        'lines.markersize': 4.5,
        'lines.markeredgewidth': 0.0,
        'lines.solid_capstyle': 'round',
        'patch.linewidth': 0.8,
        'text.color': INK,
        'image.cmap': 'viridis',
        'figure.dpi': 110,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.06,
    })


def figure(figsize=None, **kw):
    """``plt.subplots`` at the house defaults, with the frame already stripped."""
    fig, ax = plt.subplots(figsize=figsize or FIG, **kw)
    for a in (ax.flat if hasattr(ax, 'flat') else [ax]):
        strip(a)
    return fig, ax


#: Deprecated alias of :func:`figure`, from when every figure was a slide.
slide = figure


def strip(ax, left=True, bottom=True) -> None:
    """Recessive frame: keep the two spines that carry a scale, drop the rest."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def det_style(det: str) -> dict:
    """Colour **and** marker for one chamber -- so a reader who cannot separate
    two hues still has the shape.  Never take the colour without the marker."""
    d = det.strip().upper()[-1]
    return dict(color=DET_COLOR[d], marker=DET_MARKER[d], label=f'chamber {d}')


def title(ax, headline: str, sub: str | None = None) -> None:
    """Left-aligned headline plus optional grey deck, on ONE axes.

    The headline is the figure's *message*, not its subject: "B and D truncate
    half their columns", not "column occupancy".  Keep it to one line -- a
    headline that wraps is a caption, and captions belong in the report.

    **Single-axes figures only.**  On a multi-panel figure use :func:`fig_title`
    instead, and read its docstring for why.
    """
    ax.set_title(headline, loc='left', color=INK, pad=16 if sub else 8)
    if sub:
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=BASE_PT * 0.92, color=MUTED)


def fig_title(fig, headline: str, sub: str | None = None,
              pad_in: float = 0.06) -> float:
    """Headline on the FIGURE, wrapped, without blowing out the saved aspect.

    THE TRAP THIS EXISTS FOR.  :func:`title` sets the headline on an axes with
    ``loc='left'``.  On a single full-width axes that is fine.  On a two- or
    three-panel figure the headline is several times wider than the panel it is
    anchored to, so it runs off the right-hand edge of the canvas -- and
    ``savefig(bbox_inches='tight')``, which the house rcParams turn on, then
    EXPANDS the saved image sideways to contain it.  Measured on the
    2026-09-10 reports: a two-panel figure that should have been 2.7:1 came out
    4.8:1, and a report that scales the image to its column width was left with
    panels a third of their intended height.

    So: anchor the text to the figure, wrap it to the figure's own width, and
    reserve the vertical space with ``subplots_adjust`` rather than by shrinking
    the axes with a ``tight_layout`` rect and letting the tight crop eat it.

    Lays the panels out as well, so call it LAST and do not call
    ``tight_layout`` yourself: one before this would be undone and one after it
    would undo the reserved space.

    Returns the ``top`` fraction the axes were held below, so a caller that
    wants to place its own figure-level legend knows where the free space ends.
    """
    import textwrap
    w_in, h_in = fig.get_size_inches()
    # Character budget from the real type size: a bold headline at 1.15x the
    # base advances about 0.55 em per glyph in this family.
    head_pt = BASE_PT * 1.15
    sub_pt = BASE_PT * 0.92
    ncols_head = max(int(w_in * 72.0 / (head_pt * 0.55)) - 2, 24)
    ncols_sub = max(int(w_in * 72.0 / (sub_pt * 0.55)) - 2, 32)
    head = textwrap.wrap(headline, ncols_head) or ['']
    deck = textwrap.wrap(sub, ncols_sub) if sub else []

    # Height of the block, in inches, with the leading the renderer will use.
    h_head = len(head) * head_pt * 1.22 / 72.0
    h_deck = len(deck) * sub_pt * 1.35 / 72.0
    block = pad_in + h_head + (0.03 + h_deck if deck else 0.0) + pad_in
    top = max(1.0 - block / h_in, 0.45)

    x = 0.008
    y = 1.0 - pad_in / h_in
    fig.text(x, y, '\n'.join(head), ha='left', va='top', color=INK,
             fontsize=head_pt, fontweight='bold', linespacing=1.22)
    if deck:
        fig.text(x, y - (h_head + 0.03) / h_in, '\n'.join(deck), ha='left',
                 va='top', color=MUTED, fontsize=sub_pt, linespacing=1.35)
    # Lays the panels out itself, so a caller never has to get the order right:
    # a `tight_layout` after this would undo the reserved space, and one before
    # it would be undone.  Call this LAST and do not call `tight_layout` at all.
    try:
        fig.tight_layout(rect=(0, 0, 1, top))
    except Exception:
        fig.subplots_adjust(top=top)
    return top


def end_label(ax, x, y, text, color, dx=0.0, dy=0.0, **kw) -> None:
    """Direct series label at the end of a line -- often clearer than a legend.

    For more than two series use :func:`end_labels`, which does the same thing
    but pushes overlapping labels apart first.
    """
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=BASE_PT * 0.92, va='center',
                ha=kw.pop('ha', 'left'), **kw)


def end_labels(ax, items, dx=0.0, min_gap=0.055, **kw) -> None:
    """Direct-label several series at once, nudged apart so none collide.

    ``items`` is ``[(x, y, text, color), ...]``.  Series that converge -- which
    is exactly what efficiency and angle curves do at the right-hand edge --
    would otherwise stack their labels on top of each other.

    ``min_gap`` is the minimum vertical separation in axes fraction.  Labels are
    resolved bottom-up against the data's own y-limits, and the leader line back
    to the true endpoint is drawn whenever a label had to move.

    Leave room for them: ``ax.set_xlim`` a little past the last point, or the
    tight crop will grow the figure sideways to fit the text.
    """
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    rows = sorted(((y, x, t, c) for x, y, t, c in items), key=lambda r: r[0])

    placed = []
    for y, x, text, color in rows:
        yf = (y - lo) / span
        if placed and yf - placed[-1][0] < min_gap:
            yf = placed[-1][0] + min_gap
        placed.append((yf, y, x, text, color))

    for yf, y, x, text, color in placed:
        y_lab = lo + yf * span
        moved = abs(y_lab - y) > 0.01 * span
        ax.annotate(
            text, xy=(x, y), xytext=(x + dx, y_lab), color=color,
            fontsize=BASE_PT * 0.92, va='center',
            ha=kw.pop('ha', 'left'), annotation_clip=False,
            arrowprops=dict(arrowstyle='-', color=color, alpha=0.45,
                            linewidth=0.8, shrinkA=2, shrinkB=2)
            if moved else None,
            **kw)


def preliminary(ax, loc: str = 'upper right') -> None:
    """Stamp the PRELIMINARY badge.

    Required on anything that touches the reconstruction.  ``PLAN.md`` §8 lists
    what this badge is standing in for -- no resolution measurement, no absolute
    position better than ~1 cm, B and D angles not quotable, no invariant mass.

    Deliberately small and grey-red rather than a watermark: it is a label on
    the figure, not a thing the figure is about.
    """
    xy = {'upper right': (0.99, 0.99, 'right', 'top'),
          'upper left': (0.01, 0.99, 'left', 'top'),
          'lower right': (0.99, 0.01, 'right', 'bottom'),
          'lower left': (0.01, 0.01, 'left', 'bottom')}[loc]
    ax.text(xy[0], xy[1], 'Preliminary', transform=ax.transAxes,
            ha=xy[2], va=xy[3], fontsize=BASE_PT * 0.85, style='italic',
            color=TRACK, alpha=0.9, zorder=100)


def note(fig, text: str, y: float = -0.01) -> None:
    """Provenance line under the plot -- run, bundle, commit, date.

    Small on purpose: it is there to be checked, not read.
    """
    fig.text(0.0, y, text, ha='left', va='top',
             fontsize=BASE_PT * 0.72, color=MUTED, wrap=True)


# --------------------------------------------------------------------------- #
# Saving -- the PNG and the numbers, together, always
# --------------------------------------------------------------------------- #
class _NoData:
    """Sentinel for the rare figure with no underlying table (a schematic)."""
    def __repr__(self):                      # pragma: no cover - debugging only
        return 'NO_DATA'


NO_DATA = _NoData()


def save(fig, path, data=None, index: bool = False) -> Path:
    """Write ``<path>.png`` and ``<path>.csv``, and return the PNG path.

    ``data`` is the table behind the figure: a DataFrame, or a ``{suffix:
    DataFrame}`` mapping for a multi-panel figure (writing ``<path>.<suffix>.csv``
    for each).  Pass ``data=NO_DATA`` **only** for a figure that genuinely has no
    numbers -- a geometry schematic -- and expect to justify it.

    Passing ``data=None`` raises.  That is the point: ``PLAN.md`` §7 requires the
    numbers beside every PNG so a figure can be rebuilt without rerunning the
    analysis, and a default that silently skipped the CSV would make the
    requirement advisory.
    """
    p = Path(path)
    if p.suffix.lower() == '.png':
        p = p.with_suffix('')
    p.parent.mkdir(parents=True, exist_ok=True)

    if data is None:
        raise ValueError(
            f'save({p.name}): no data given. Every figure ships its numbers -- '
            f'pass the DataFrame behind it, or data=figstyle.NO_DATA if this '
            f'figure genuinely has none (a schematic).')

    if data is not NO_DATA:
        tables = data if isinstance(data, dict) else {'': data}
        for suffix, df in tables.items():
            csv = p.with_suffix(f'.{suffix}.csv' if suffix else '.csv')
            df.to_csv(csv, index=index)
            print(f'  -> {csv}')

    png = p.with_suffix('.png')
    fig.savefig(png)
    plt.close(fig)
    print(f'  -> {png}')
    return png


if __name__ == '__main__':
    # Smoke test: renders the palette and the type scale at final size, so a
    # change here can be eyeballed rather than argued about.
    import numpy as np
    use()
    fig, ax = figure()
    x = np.linspace(0, 180, 40)
    labels = []
    for i, d in enumerate('ABCD'):
        st = det_style(d)
        y = np.exp(-((x - 60 - 25 * i) / 40) ** 2)
        ax.plot(x, y, **st, markevery=6)
        labels.append((x[-1], y[-1], f'chamber {d}', st['color']))
    # B and C converge to ~0 at the right edge -- the collision end_labels exists
    # to resolve, and the reason the smoke test uses four series and not two.
    end_labels(ax, labels, dx=4)
    ax.set_xlim(0, 215)
    ax.set_xlabel('opening angle  [deg]')
    ax.set_ylabel('arbitrary')
    title(ax, 'Four chambers, each with a marker and a direct label',
          'figstyle smoke test -- colour never carries identity alone')
    ax.axvspan(110, 140, color=BAND_SIGNAL, alpha=0.10, zorder=0)
    preliminary(ax)
    note(fig, 'figstyle.py self-test - no data, no physics')
    # Imported here, not at module scope: figstyle is a style module that every
    # figure script imports, so it must not drag the data tree in behind it.
    # Only this self-test needs somewhere to write.
    from sept26_prelim_analysis import paths
    out = paths.out() / 'figstyle_smoke'
    save(fig, out, data=NO_DATA)
