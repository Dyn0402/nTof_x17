#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hit_maps.py -- what each chamber surface actually sees, as a picture.

Efficiency numbers integrated over a chamber hide the thing acceptance needs:
*where* on the surface the response falls off, and whether the falloff has the
shape of geometry or the shape of a fault.  These are the maps that show it,
built so the same code runs over the whole campaign later.

FOUR VIEWS, each answering a different question:

  occupancy      Where do clusters land?  Raw counts in the chamber's own local
                 (x, y), so dead strips, hot regions and the beam spot are
                 visible directly.
  relative       The same, divided by the chamber's own mean -- structure
                 rather than rate, so chambers with very different occupancies
                 can be compared side by side.
  efficiency     Response divided by an MM-independent denominator: the
                 scintillator-tagged events of that arm.  Coarse in position,
                 because the wall localises to a 100 mm group and top/bottom
                 only -- but it is the only view whose denominator is not the
                 detector being measured.
  by angle       Occupancy split by reconstructed track angle, which is what an
                 opening-angle acceptance is differential in.  Chamber B is
                 excluded from this one by construction: no uniform drift
                 field, so no angle.

WHAT A MAP IS NOT.  Occupancy is illumination x efficiency, and the beam is not
uniform, so a dip in an occupancy map is *not* an inefficiency on its own.  The
relative and efficiency views exist because of that; the occupancy view is for
spotting faults, not for correcting acceptance.

    python -m sept26_prelim_analysis.hit_maps --run run_145
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis.build_tracks import (  # noqa: E402
    IN_PLANE_SIGN, IN_PLANE_SIGN_Y, STRIP_MAP_HALF)

ARMS = ('A', 'B', 'C', 'D')
#: The active area is 398.58 mm square (the strip map), so bin it in 10 mm.
EDGES = np.arange(-200.0, 201.0, 10.0)
#: Angle bands for the differential view, in |tan|. The X17 topology needs the
#: acceptance differential in angle, and these are the bands the data supports.
TAN_BANDS = ((0.0, 0.12), (0.12, 0.25), (0.25, 0.45), (0.45, 1.0))
HITS_ONLY = ('B',)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs.use()
    return plt


def load_positions(run: str, subruns, fullpass: str) -> pd.DataFrame:
    """Cluster positions in the chamber's own local frame, per arm.

    Uses every event the chamber SEEDED and fitted both planes on -- not only
    the gated tracks -- because chamber B has no tracks worth the name and the
    map has to mean the same thing for all four.
    """
    out = []
    for sub in subruns:
        for arm in ARMS:
            p = os.path.join(fullpass, sub, f'mx17_{arm}',
                             'events_prelim.parquet')
            if not os.path.exists(p):
                raise FileNotFoundError(f'no full pass for {arm}/{sub}: {p}')
            d = pd.read_parquet(p, columns=['event_id', 'x_ok', 'y_ok',
                                            'x_p0', 'y_p0', 'n_tracks',
                                            'x_tan_theta', 'x_q_sum'])
            m = d.x_ok.to_numpy() & d.y_ok.to_numpy()
            g = d[m]
            out.append(pd.DataFrame(dict(
                subrun=sub, arm=arm, event_id=g.event_id.to_numpy(),
                x=IN_PLANE_SIGN * (g.x_p0.to_numpy() - STRIP_MAP_HALF),
                y=IN_PLANE_SIGN_Y * (g.y_p0.to_numpy() - STRIP_MAP_HALF),
                tan=g.x_tan_theta.to_numpy(),
                q=g.x_q_sum.to_numpy(),
                tracked=g.n_tracks.to_numpy() > 0)))
    return pd.concat(out, ignore_index=True)


def _panel(ax, H, vmin, vmax, cmap, title, colour, log=False):
    # A few edge channels run 20x the bulk, so a linear scale shows the hot
    # spots and nothing else. Log keeps both visible, which is the whole point
    # of an occupancy map.
    norm = None
    if log:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(vmin, 0.5), vmax=max(vmax, 1.0))
        vmin = vmax = None
    im = ax.imshow(H.T, origin='lower', aspect='equal', cmap=cmap,
                   vmin=vmin, vmax=vmax, norm=norm,
                   extent=[EDGES[0], EDGES[-1], EDGES[0], EDGES[-1]])
    ax.set_title(title, color=colour, pad=6)
    ax.set_xlabel('x local  [mm]')
    ax.set_xticks([-200, -100, 0, 100, 200])
    ax.set_yticks([-200, -100, 0, 100, 200])
    return im


def fig_occupancy(P: pd.DataFrame, out, relative: bool = False):
    """Cluster occupancy per chamber, raw or divided by the chamber's mean."""
    plt = _plt()
    with plt.rc_context({'font.size': fs.BASE_PT * 0.7,
                         'axes.titlesize': fs.BASE_PT * 0.85,
                         'axes.labelsize': fs.BASE_PT * 0.72,
                         'xtick.labelsize': fs.BASE_PT * 0.6,
                         'ytick.labelsize': fs.BASE_PT * 0.6}):
        fig, axes = plt.subplots(1, 4, figsize=(fs.BANNER[0], 5.4),
                                 constrained_layout=True)
        rows = []
        for ax, a in zip(axes, ARMS):
            s = P[P.arm == a]
            H, _, _ = np.histogram2d(s.x, s.y, bins=[EDGES, EDGES])
            if relative:
                # median, not mean: a handful of hot edge bins drag a mean and
                # then every ordinary bin reads as under-responding.
                inside = H[H > 0]
                H = H / (np.median(inside) if len(inside) else 1.0)
                im = _panel(ax, H, 0, 2.0, 'RdBu_r',
                            f'chamber {a}', fs.DET_COLOR[a])
            else:
                im = _panel(ax, H, 0.5, H.max() if (H > 0).any() else 1,
                            'viridis', f'chamber {a}   n={len(s):,}',
                            fs.DET_COLOR[a], log=True)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            for i in range(len(EDGES) - 1):
                for j in range(len(EDGES) - 1):
                    rows.append(dict(arm=a, x=0.5 * (EDGES[i] + EDGES[i + 1]),
                                     y=0.5 * (EDGES[j] + EDGES[j + 1]),
                                     value=H[i, j]))
        axes[0].set_ylabel('y local  [mm]')
        if relative:
            fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01,
                         label='occupancy / chamber mean')
            fig.suptitle('Relative occupancy — structure, not rate. '
                         'Blue is under-responding, red over.',
                         fontsize=fs.BASE_PT * 0.9)
        else:
            fig.suptitle('Cluster occupancy on each chamber surface '
                         '(both planes fitted)', fontsize=fs.BASE_PT * 0.9)
        fs.preliminary(axes[0], loc='lower left')
        name = 'hitmap_relative' if relative else 'hitmap_occupancy'
        fs.save(fig, out / name, data=pd.DataFrame(rows))


def fig_by_angle(P: pd.DataFrame, out):
    """Occupancy in bands of |tan| -- what acceptance is differential in."""
    plt = _plt()
    arms = [a for a in ARMS if a not in HITS_ONLY]
    with plt.rc_context({'font.size': fs.BASE_PT * 0.62,
                         'axes.titlesize': fs.BASE_PT * 0.72,
                         'axes.labelsize': fs.BASE_PT * 0.62,
                         'xtick.labelsize': fs.BASE_PT * 0.52,
                         'ytick.labelsize': fs.BASE_PT * 0.52}):
        fig, axes = plt.subplots(len(arms), len(TAN_BANDS),
                                 figsize=(fs.BANNER[0], 3.1 * len(arms)),
                                 constrained_layout=True)
        rows = []
        for r, a in enumerate(arms):
            s = P[(P.arm == a) & P.tracked]
            t = np.abs(s.tan.to_numpy())
            for c, (lo, hi) in enumerate(TAN_BANDS):
                ax = axes[r, c]
                m = (t >= lo) & (t < hi)
                H, _, _ = np.histogram2d(s.x.to_numpy()[m], s.y.to_numpy()[m],
                                         bins=[EDGES, EDGES])
                ins = H[H > 0]
                Hn = H / (ins.mean() if len(ins) else 1.0)
                _panel(ax, Hn, 0, 2.0, 'RdBu_r',
                       f'{a}   |tan| {lo:.2f}–{hi:.2f}   n={int(m.sum()):,}',
                       fs.DET_COLOR[a])
                if c:
                    ax.set_yticklabels([])
                if r < len(arms) - 1:
                    ax.set_xlabel('')
                rows.append(dict(arm=a, tan_lo=lo, tan_hi=hi, n=int(m.sum())))
            axes[r, 0].set_ylabel('y local  [mm]')
        fig.suptitle('Occupancy by track angle, each panel relative to its own '
                     'mean — chamber B excluded (no drift field, no angle)',
                     fontsize=fs.BASE_PT * 0.8)
        fs.preliminary(axes[0, 0], loc='lower left')
        fs.save(fig, out / 'hitmap_by_angle', data=pd.DataFrame(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--fullpass',
                    default=str(paths.out('fullpass') / 'run_145'))
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    P = load_positions(a.run, subs, a.fullpass)
    print(f'{len(P):,} fitted clusters over {len(subs)} sub-runs')
    print(P.groupby('arm').agg(n=('x', 'size'),
                               tracked=('tracked', 'sum')).to_string())
    out = paths.figures('hitmaps')
    fig_occupancy(P, out, relative=False)
    fig_occupancy(P, out, relative=True)
    fig_by_angle(P, out)
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
