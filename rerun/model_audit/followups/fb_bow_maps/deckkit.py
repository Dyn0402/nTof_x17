"""deckkit.py -- small shared pieces for the two FB slide notes (chamber bow, chamber maps)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402
from slidedoc import BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT  # noqa: E402,F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import RES, DETS, DATASETS  # noqa: E402,F401

DCOL = {'det2': BLUE, 'det3': GREEN, 'det4': RED, 'det6': PURPLE, 'det7': GOLD}
XC, YC = BLUE, ORANGE
E5 = np.linspace(25, 375, 6)


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fin(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def hmap(M, title, vmin, vmax, cmap, w=300, h=300, unit='', fmt='{:+.2f}', E=E5, ticks=(100, 200, 300),
         xlabel='x [mm]', ylabel='y [mm]', errs=None, ns=None, margin=(16, 12, 64, 76)):
    """5x5 (or n x n) map M[i][j], i along x, j along y, cell edges E (mm)."""
    P = sd.Plot(w, h, x=(E[0], E[-1]), y=(E[0], E[-1]), xlabel=xlabel, ylabel=ylabel, margin=margin, title=title)
    cells = []
    M = np.asarray(M, float)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                cells.append((E[i], E[i + 1], E[j], E[j + 1], None, 'no data'))
                continue
            tip = f'x {E[i]:.0f}–{E[i + 1]:.0f} mm, y {E[j]:.0f}–{E[j + 1]:.0f} mm\n{fmt.format(v)} {unit}'
            if errs is not None and np.isfinite(errs[i][j]):
                tip += f' ± {errs[i][j]:.2f}'
            if ns is not None:
                tip += f'\nn = {int(ns[i][j])}'
            cells.append((E[i], E[i + 1], E[j], E[j + 1], float(v), tip))
    P.cells(cells, vmin, vmax, cmap=cmap, missing='#e4e1da', gap=1.0)
    P.xticks([(t, str(t)) for t in ticks]).yticks([(t, str(t)) for t in ticks])
    return P.svg(title)


def plain_ticks(lo, hi, step, fmt='{:g}'):
    n = int(np.floor((hi - lo) / step + 1e-9))
    return [(lo + k * step, fmt.format(lo + k * step)) for k in range(n + 1)]
