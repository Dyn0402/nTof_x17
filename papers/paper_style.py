#!/usr/bin/env python3
"""paper_style.py -- one look for every MX17 paper figure (Paper I and II). Dylan 2026-11-02: one colour per chamber.

Import from any producer:

    import sys, os; sys.path.insert(0, os.path.join(<repo>, 'papers'))
    import paper_style as ps
    ps.use()                                   # rcParams, before plotting
    ax.plot(x, y, **ps.det('det3'))            # colour, marker, zorder, line width for that chamber
    ps.save(fig, out_dir / 'name', data=df)    # name.pdf + name.png + name.csv

Rules it encodes:
* **One colour per chamber, everywhere** (Okabe-Ito, colour-blind safe). det3, the paper's chamber, is the darkest and
  is drawn on top with a heavier line; det4 (the defective chamber) is grey and drawn underneath.
* Views: X solid, Y dashed (``ps.VIEW_LS``). Gas: CO2 / CF4 mixtures have their own two colours (``ps.GAS``), never a
  chamber colour.
* Data vs model: data are markers (with error bars), models are lines (Magboltz, fits); a reference model is black.
* Labels are mathtext, never a raw underscore: ``ps.lbl('Q_x')`` -> ``$Q_{x}$``.
* Type at document size (JINST full width ~15.5 cm = 6.1 in; single panel 3:2).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
#: The palette (2026-11-02). Keyed by bench chamber; n_TOF arms map A=det3, B=det2, C=det6, D=det7.
DET_COLOR = {'det2': '#56B4E9', 'det3': '#0b3d91', 'det4': '#999999', 'det6': '#E69F00', 'det7': '#CC79A7'}
DET_MARKER = {'det2': 's', 'det3': 'o', 'det4': 'v', 'det6': '^', 'det7': 'D'}
DET_ZORDER = {'det2': 3, 'det3': 5, 'det4': 1, 'det6': 3, 'det7': 3}
DET_LW = {'det2': 1.2, 'det3': 1.8, 'det4': 1.0, 'det6': 1.2, 'det7': 1.2}
ARM_TO_DET = {'A': 'det3', 'B': 'det2', 'C': 'det6', 'D': 'det7'}
VIEW_LS = {'x': '-', 'y': '--', 'X': '-', 'Y': '--'}
VIEW_FILL = {'x': True, 'y': False, 'X': True, 'Y': False}     # filled markers X, open Y
GAS = {'CO2': '#b2182b', 'CF4': '#2166ac'}                      # SPS gas families (attachment etc.)
MODEL = '#000000'                                               # reference model / fit line
ACCENT = '#D55E00'                                              # one highlight (operating point, chosen cut)
INK, MUTED, GRID = '#1b2430', '#6a7583', '#e6e9ee'

FULL = (6.1, 3.8)      # full text width, one panel
WIDE = (6.1, 2.6)      # full width, two/three panels in a row
TALL = (6.1, 6.0)      # full width, stacked panels
HALF = (3.0, 2.4)      # half width
BASE_PT = 9.0


def use() -> None:
    """House rcParams. Call once before plotting."""
    mpl.rcParams.update({
        'figure.figsize': FULL, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'mathtext.default': 'regular', 'font.size': BASE_PT,
        'axes.titlesize': BASE_PT, 'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
        'axes.labelsize': BASE_PT, 'xtick.labelsize': BASE_PT * 0.9, 'ytick.labelsize': BASE_PT * 0.9,
        'legend.fontsize': BASE_PT * 0.85, 'legend.frameon': False, 'legend.handlelength': 1.8,
        'axes.edgecolor': '#555555', 'axes.linewidth': 0.7, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.2, 'lines.markersize': 4.0, 'errorbar.capsize': 0,
        'axes.prop_cycle': mpl.cycler(color=[DET_COLOR[d] for d in DETS]),
    })


def det(name, view=None, line=True, **kw) -> dict:
    """Keyword arguments for one chamber's series (plot / errorbar): colour, marker, z-order, width, label.
    ``view`` ('x'/'y') sets the line style and an open marker for Y."""
    d = str(name).lower().replace('mx17_', 'det')
    d = ARM_TO_DET.get(str(name), d)
    c = DET_COLOR[d]
    out = dict(color=c, marker=DET_MARKER[d], zorder=DET_ZORDER[d], label=d + (f' {view.upper()}' if view else ''))
    if line:
        out.update(lw=DET_LW[d], ls=VIEW_LS.get(view, '-'))
    if view is not None and not VIEW_FILL.get(view, True):
        out.update(mfc='white', mec=c)
    out.update(kw)
    return out


def color(name) -> str:
    d = ARM_TO_DET.get(str(name), str(name).lower().replace('mx17_', 'det'))
    return DET_COLOR[d]


def lbl(s: str) -> str:
    """'Q_x' -> '$Q_{x}$'; leaves strings that already contain $...$ alone."""
    if '$' in s or '_' not in s:
        return s
    return re.sub(r'([A-Za-z\\]+)_([A-Za-z0-9]+)', r'$\1_{\2}$', s)


def save(fig, path, data=None, png=True) -> Path:
    """Write <path>.pdf (+ .png, + .csv of the plotted numbers when ``data`` is a DataFrame or dict of them)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'))
    if png:
        fig.savefig(path.with_suffix('.png'), dpi=200)
    if data is not None:
        if isinstance(data, dict):
            for k, v in data.items():
                v.to_csv(path.parent / f'{path.name}.{k}.csv', index=False)
        else:
            data.to_csv(path.with_suffix('.csv'), index=False)
    plt.close(fig)
    return path.with_suffix('.pdf')


def legend_dets(ax, dets=DETS, **kw):
    """A chamber legend in the fixed order (handles from the palette, not from what was plotted)."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], **{k: v for k, v in det(d).items() if k != 'zorder'}) for d in dets]
    return ax.legend(handles=h, **kw)


if __name__ == '__main__':          # a swatch, to look at the palette
    use()
    fig, ax = plt.subplots(figsize=HALF)
    for i, d in enumerate(DETS):
        ax.plot([0, 1], [i, i], **det(d))
    legend_dets(ax, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yticks([]); ax.set_xticks([])
    save(fig, Path(os.environ.get('CLAUDE_JOB_DIR', '/tmp')) / 'tmp' / 'palette_swatch')
    print('ok')
