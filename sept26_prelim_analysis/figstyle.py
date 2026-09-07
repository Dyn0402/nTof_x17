#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figstyle.py -- the house style for every figure in the September deck.

A figure here is built for a **projected slide**, not a page.  That single
constraint drives everything below: one fixed 16:9 canvas, a type scale that is
readable from the back of a room, recessive axes, and one message per figure --
the message being the title.

Three rules this module makes hard to break:

* **Every figure ships its numbers.**  :func:`save` writes ``foo.png`` *and*
  ``foo.csv``, and refuses to write the PNG alone unless you say
  ``data=NO_DATA`` and mean it.  A figure that cannot be rebuilt from a CSV
  beside it is a figure nobody can check.
* **Anything touching the reconstruction is badged.**  :func:`preliminary`
  stamps it; the badge is not decoration, it is the caveat that ``PLAN.md`` §8
  attaches to every reconstructed quantity.
* **Identity never rests on hue alone.**  The four-detector palette is the
  Okabe-Ito subset, re-validated for this package on 2026-09-07 against the
  ``#fbfcfe`` surface: it returns ALL CHECKS PASS with two warnings, and
  :func:`det_style` discharges both by handing back a marker with every colour.

  * CVD separation D↔C is ΔE 7.6 (deutan), in the 6-8 floor band, legal *only*
    with secondary encoding -- hence the per-detector marker shape, and
    :func:`end_label` for direct labelling.
  * D's ``#CC79A7`` is 2.98:1 against the surface, just under 3:1 -- so it is
    only ever used with a visible label attached, never as a bare fill.

The palette is deliberately identical to ``mpgd26/plotstyle.py``: the two decks
show the same four chambers to overlapping audiences, and a detector that
changes colour between talks is a detector the audience has to re-learn.

    import figstyle as fs
    fs.use()
    fig, ax = fs.slide()
    ...
    fs.preliminary(ax)
    fs.save(fig, fs_out / 'opening_angle', data=df)
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
LINE = '#d4d9e0'
SURFACE = '#fbfcfe'
ACCENT = '#8a3f8f'          # mx17 purple
COPPER = '#d18a44'          # annotation / caution
TRACK = '#ff4f36'           # one sharp highlight, used sparingly

#: Categorical, fixed order, never cycled.  Validated -- see the module docstring.
DET_COLOR = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73', 'D': '#CC79A7'}
DET_MARKER = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D'}

#: Reserved status fills.  Never reused as a series colour.
BAND_SIGNAL = '#8a3f8f'     # the 110-140 deg X17 region
BAND_DEAD = '#b04a3a'       # masked / dead / vetoed
BAND_CONTROL = '#6a7583'    # the intra-chamber control region

# --------------------------------------------------------------------------- #
# Geometry -- one canvas, so every figure in the deck is the same size
# --------------------------------------------------------------------------- #
# --- the canvas, and why the other sizes are still in its units ------------- #
#
# SLIDE is the **reference canvas**, not a requirement: it is the full slide, and
# every other size below is a fraction of it.  Most figures are not full-slide
# and should not be -- a two-panel comparison or a small inset is often the
# clearer object.
#
# The rule that actually matters is that **the type scale never changes**.  A
# matplotlib point is absolute (1/72 in), so a HALF-width figure built with the
# same rcParams and dropped onto half a slide has type exactly the same physical
# size as a full-width one.  That holds only while the figure is placed at its
# natural size: rescaling a figure in the slide is what breaks it, and is why
# these are named sizes rather than a free parameter.
#
#: The full slide.  16:9 at 13.333 x 7.5 in -> 2133 x 1200 px at 160 dpi.  An
#: 18 pt label is 1.9 % of the frame width, about 5.6 cm on a 3 m screen.
SLIDE = (13.333, 7.5)
#: Half width, full height: two side by side fill one slide.
HALF = (6.667, 7.5)
#: Half width, half height: a 2 x 2 grid of separate figures.
QUARTER = (6.667, 3.75)
#: Third width, for a row of three.
THIRD = (4.444, 7.5)
#: Full width, short: a timeline or a ladder across the bottom of a slide.
BANNER = (13.333, 4.2)
#: Full width, two-thirds height -- a figure with a caption block beneath it.
WIDE = (13.333, 5.0)

BASE_PT = 18.0              # the floor PLAN.md 7 sets, at final size


def use() -> None:
    """Apply the house rcParams.  Call once, before plotting."""
    mpl.rcParams.update({
        'figure.figsize': SLIDE,
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        # The type scale. Nothing is below BASE_PT except the provenance line,
        # which is deliberately for the reader who walks up to the screen.
        'font.size': BASE_PT,
        'axes.titlesize': BASE_PT * 1.45,
        'axes.titleweight': 'bold',
        'axes.labelsize': BASE_PT * 1.1,
        'xtick.labelsize': BASE_PT,
        'ytick.labelsize': BASE_PT,
        'legend.fontsize': BASE_PT,
        'axes.labelcolor': INK,
        'axes.edgecolor': LINE,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': LINE,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'lines.linewidth': 2.4,
        'lines.markersize': 9,
        'text.color': INK,
        'figure.dpi': 160,
        'savefig.dpi': 160,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.22,
    })


def slide(figsize=SLIDE, **kw):
    """``plt.subplots`` on the house canvas, with the frame already stripped."""
    fig, ax = plt.subplots(figsize=figsize, **kw)
    for a in (ax.flat if hasattr(ax, 'flat') else [ax]):
        strip(a)
    return fig, ax


def strip(ax, left=True, bottom=True) -> None:
    """Recessive frame: keep the two spines that carry a scale, drop the rest."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def det_style(det: str) -> dict:
    """Colour **and** marker for one chamber -- the secondary encoding the
    palette validation requires.  Never take the colour without the marker."""
    d = det.strip().upper()[-1]
    return dict(color=DET_COLOR[d], marker=DET_MARKER[d], label=f'chamber {d}')


def title(ax, headline: str, sub: str | None = None) -> None:
    """Left-aligned headline plus optional grey deck.

    The headline is the figure's *message*, not its subject: "B and D truncate
    half their columns", not "column occupancy".
    """
    ax.set_title(headline, loc='left', color=INK, pad=26 if sub else 14)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=BASE_PT * 0.9, color=MUTED)


def end_label(ax, x, y, text, color, dx=0.0, dy=0.0, **kw) -> None:
    """Direct series label at the end of a line -- so identity is never hue alone.

    For more than two series use :func:`end_labels`, which does the same thing
    but pushes overlapping labels apart first.
    """
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=BASE_PT * 0.95, fontweight='bold', va='center',
                ha=kw.pop('ha', 'left'), **kw)


def end_labels(ax, items, dx=0.0, min_gap=0.055, **kw) -> None:
    """Direct-label several series at once, nudged apart so none collide.

    ``items`` is ``[(x, y, text, color), ...]``.  Series that converge -- which
    is exactly what efficiency and angle curves do at the right-hand edge --
    would otherwise stack their labels on top of each other and lose the
    secondary encoding the palette validation depends on.

    ``min_gap`` is the minimum vertical separation in axes fraction.  Labels are
    resolved bottom-up against the data's own y-limits, and the leader line back
    to the true endpoint is drawn whenever a label had to move.
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
            fontsize=BASE_PT * 0.95, fontweight='bold', va='center',
            ha=kw.pop('ha', 'left'), annotation_clip=False,
            arrowprops=dict(arrowstyle='-', color=color, alpha=0.45,
                            linewidth=1.0, shrinkA=2, shrinkB=2)
            if moved else None,
            **kw)


def preliminary(ax, loc: str = 'upper right') -> None:
    """Stamp the PRELIMINARY badge.

    Required on anything that touches the reconstruction.  ``PLAN.md`` §8 lists
    what this badge is standing in for -- no resolution measurement, no absolute
    position better than ~1 cm, B and D angles not quotable, no invariant mass.
    """
    xy = {'upper right': (0.985, 0.975, 'right', 'top'),
          'upper left': (0.015, 0.975, 'left', 'top'),
          'lower right': (0.985, 0.025, 'right', 'bottom'),
          'lower left': (0.015, 0.025, 'left', 'bottom')}[loc]
    ax.text(xy[0], xy[1], 'Preliminary', transform=ax.transAxes,
            ha=xy[2], va=xy[3], fontsize=BASE_PT * 1.05, fontweight='bold',
            color=TRACK, alpha=0.85, zorder=100)


def note(fig, text: str, y: float = -0.01) -> None:
    """Provenance line under the plot -- run, bundle, commit, date.

    Deliberately below BASE_PT: it is for the reader who walks up to the screen
    or opens the PNG, not for the back of the room.
    """
    fig.text(0.0, y, text, ha='left', va='top',
             fontsize=BASE_PT * 0.55, color=MUTED, wrap=True)


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
    fig, ax = slide()
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
    title(ax, 'figstyle smoke test -- four chambers, marker + direct label',
          'colour never carries identity alone; this is what the deck looks like')
    ax.axvspan(110, 140, color=BAND_SIGNAL, alpha=0.12, zorder=0)
    preliminary(ax)
    note(fig, 'figstyle.py self-test - no data, no physics')
    out = Path(os.environ.get('X17_SEPT26_OUT',
                              '/media/dylan/data/x17/sept26_prelim')) / 'figstyle_smoke'
    save(fig, out, data=NO_DATA)
