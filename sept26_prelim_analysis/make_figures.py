#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py -- the figures for the funnel report and the X17 board.

Four, each answering one question:

  k_scan       Does the angle-scale fit have a minimum, or is it flat?  One
               panel per chamber: the focus objective against k, the three
               point estimators marked on it, and the plateau shaded.  This is
               the figure that says why A and C are provisional and B is not
               calibrated at all -- B's curve simply has no minimum.
  k_summary    The measured drift velocity per chamber against the prior every
               bundle carries, with the plateau as the error bar.
  det_status   Where each chamber stands: the funnel as a rate, and the n_TOF
               confirmation against its own no-track control.  The lift is the
               number that says the tracking is doing work.
  det_evidence The exclusive n_TOF partition of the tracked events.

Every figure ships its CSV (figstyle.save refuses otherwise), so nothing here
is a picture without numbers behind it.

    python -m sept26_prelim_analysis.make_figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
VERDICT_COLOR = {'CALIBRATED': '#0072B2', 'PROVISIONAL': '#d18a44',
                 'NOT CALIBRATED': '#b04a3a', 'NO DATA': '#8a95a3'}


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs.use()
    return plt


def _scaled(plt, f: float = 0.72):
    """figstyle's type scale is sized for a FULL-SLIDE figure.  A multi-panel
    figure at the same scale fills its panels with text, so step the scale down
    for the duration of one figure -- here, not globally, so a single-panel
    figure elsewhere in the deck is unaffected."""
    return plt.rc_context({'font.size': fs.BASE_PT * f,
                           'axes.titlesize': fs.BASE_PT * f * 1.18,
                           'axes.labelsize': fs.BASE_PT * f * 1.08,
                           'xtick.labelsize': fs.BASE_PT * f * 0.97,
                           'ytick.labelsize': fs.BASE_PT * f * 0.97,
                           'legend.fontsize': fs.BASE_PT * f * 0.9})


# --------------------------------------------------------------------- k scan
def fig_k_scan(cal: dict, out):
    """The focus objective against k, one panel per chamber.

    Plotted as a FRACTION of each chamber's coincident sample, so the four
    panels share a y axis and the shapes are comparable; the absolute counts
    are in the CSV.  What the reader should see: A and C have a maximum, D has
    a broad one, and B is flat -- which is the whole verdict, visually.
    """
    plt = _plt()
    with _scaled(plt, 0.72):
        return _fig_k_scan(plt, cal, out)


def _fig_k_scan(plt, cal: dict, out):
    fig, axes = plt.subplots(1, 4, figsize=(fs.BANNER[0], 4.6), sharey=True,
                             constrained_layout=True)
    rows = []
    for ax, a in zip(axes, ARMS):
        v = cal['arms'][a]
        col = fs.DET_COLOR[a]
        scan = v.get('scan') or {}
        if not scan:
            ax.text(0.5, 0.5, 'not scanned', transform=ax.transAxes,
                    ha='center', va='center', color=fs.MUTED)
            ax.set_title(f'{a}', color=col)
            continue
        # sum the sub-runs: one curve per chamber, more stable than either
        grid = np.array(next(iter(scan.values()))['grid'], float)
        tot = np.zeros_like(grid)
        n = 0
        for s in scan.values():
            tot += np.array(s['counts']['30.0'], float)
            n += s['n']
        frac = tot / max(n, 1)

        pl = v.get('focus_plateau')
        if pl:
            ax.axvspan(pl[0], pl[1], color=col, alpha=0.13, lw=0,
                       label='plateau (within 5 % of peak)')
        ax.plot(grid, frac, color=col, lw=2.0, solid_capstyle='round')

        pe = v.get('per_estimator', {})
        # Stagger the three labels in y: they land within ~0.1 in k of each
        # other on a good chamber, and stacked at one height they overprint.
        marks = [('band', '^', 30), ('track', 'o', 17), ('focus', 's', 4)]
        ymax = frac.max() if len(frac) else 1.0
        for name, mk, dy in marks:
            x = pe.get(name)
            if x is None or not np.isfinite(x) or not (grid[0] <= x <= grid[-1]):
                continue
            y = np.interp(x, grid, frac)
            ax.plot([x], [y], mk, color=fs.INK, ms=6, mfc=fs.SURFACE,
                    mew=1.5, zorder=5)
            ax.annotate(name, (x, y), textcoords='offset points',
                        xytext=(7, dy), ha='left',
                        fontsize=fs.BASE_PT * 0.55, color=fs.MUTED,
                        arrowprops=dict(arrowstyle='-', lw=0.7,
                                        color=fs.LINE, shrinkA=0, shrinkB=2))
        vc = VERDICT_COLOR.get(v.get('verdict', 'NO DATA'))
        ax.set_title(f'chamber {a}', color=col, pad=8)
        ax.text(0.5, 0.04, v.get('verdict', ''), transform=ax.transAxes,
                ha='center', fontsize=fs.BASE_PT * 0.6, color=vc,
                fontweight='bold')
        ax.set_xlabel('angle scale $k$')
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, max(ymax * 1.42, 0.05))
        for k_, f_ in zip(grid, frac):
            rows.append(dict(arm=a, k=k_, frac_within_30mm=f_,
                             n_coincident=n))
    axes[0].set_ylabel('coincident tracks pointing\nwithin 30 mm of the axis')
    axes[0].legend(loc='lower left', fontsize=fs.BASE_PT * 0.55, frameon=False,
                   bbox_to_anchor=(0.0, 0.06))
    fig.suptitle('The angle scale peaks in A, C and D — and never turns over in B',
                 fontsize=fs.BASE_PT * 0.95)
    fs.preliminary(axes[3], loc='upper left')
    fs.save(fig, out / 'k_scan', data=pd.DataFrame(rows))


def fig_k_summary(cal: dict, out):
    """Measured drift velocity per chamber, against the prior."""
    plt = _plt()
    with _scaled(plt, 0.8):
        return _fig_k_summary(plt, cal, out)


def _fig_k_summary(plt, cal: dict, out):
    fig, ax = plt.subplots(figsize=(fs.QUARTER[0], 4.4),
                           constrained_layout=True)
    v_prior = cal['v_bundle']
    rows = []
    xs = np.arange(len(ARMS))
    for i, a in enumerate(ARMS):
        v = cal['arms'][a]
        k = v.get('k')
        if not k:
            continue
        pl = v.get('focus_plateau') or [k, k]
        # v = v_prior / k, so the plateau in k inverts into the error bar in v
        lo, hi = v_prior / pl[1], v_prior / pl[0]
        val = v_prior / k
        cert = v.get('verdict') in ('CALIBRATED', 'PROVISIONAL')
        ax.errorbar([i], [val], yerr=[[val - lo], [hi - val]],
                    fmt=fs.DET_MARKER[a], ms=11, color=fs.DET_COLOR[a],
                    mfc=fs.DET_COLOR[a] if cert else 'none', mew=2,
                    capsize=6, lw=2, zorder=4)
        ax.annotate(f'{val:.0f}', (i, val), textcoords='offset points',
                    xytext=(15, -4), fontsize=fs.BASE_PT * 0.8,
                    color=fs.DET_COLOR[a], fontweight='bold')
        rows.append(dict(arm=a, v_insitu_um_ns=val, v_lo=lo, v_hi=hi,
                         k=k, verdict=v.get('verdict')))
    ax.axhline(v_prior, color=fs.COPPER, lw=2, ls='--', zorder=2)
    ax.annotate(f'bundle prior {v_prior:.1f}', (len(ARMS) - 0.5, v_prior),
                textcoords='offset points', xytext=(-4, 8), ha='right',
                color=fs.COPPER, fontsize=fs.BASE_PT * 0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{a}' for a in ARMS])
    ax.set_xlim(-0.5, len(ARMS) - 0.5)
    ax.set_xlabel('chamber')
    ax.set_ylabel('drift velocity  [µm/ns]')
    ax.set_title('Every chamber drifts slower than the prior')
    ax.text(0.99, 0.04, 'open marker = not certified\nbar = focus plateau',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=fs.BASE_PT * 0.62, color=fs.MUTED)
    fs.preliminary(ax)
    fs.save(fig, out / 'k_summary', data=pd.DataFrame(rows))


# ---------------------------------------------------------------- det status
def fig_det_status(F: pd.DataFrame, out):
    """Two panels: the funnel as a rate, and n_TOF confirmation vs its control."""
    plt = _plt()
    with _scaled(plt, 0.74):
        return _fig_det_status(plt, F, out)


def _fig_det_status(plt, F: pd.DataFrame, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.WIDE[0], 5.2),
                                   constrained_layout=True)
    xs = np.arange(len(F))
    w = 0.38

    ax1.bar(xs - w / 2, F.seed_eff, w, label='seeded / triggers',
            color=[fs.DET_COLOR[a] for a in F.arm], alpha=0.45,
            edgecolor='none')
    ax1.bar(xs + w / 2, F.n_track_events / F.n_triggers, w,
            label='tracked / triggers',
            color=[fs.DET_COLOR[a] for a in F.arm], edgecolor='none')
    for i, (_, r) in enumerate(F.iterrows()):
        ax1.annotate(f'{100 * r.seed_eff:.0f}', (i - w / 2, r.seed_eff),
                     ha='center', va='bottom', fontsize=fs.BASE_PT * 0.68,
                     color=fs.MUTED)
        t = r.n_track_events / r.n_triggers
        ax1.annotate(f'{100 * t:.0f}', (i + w / 2, t), ha='center',
                     va='bottom', fontsize=fs.BASE_PT * 0.68, color=fs.INK)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(F.arm)
    ax1.set_ylabel('fraction of DAQ triggers')
    ax1.set_xlabel('chamber')
    ax1.set_title('D seeds twice as often as anyone else', pad=10)
    ax1.legend(frameon=False, loc='upper center', ncol=2)
    ax1.set_ylim(0, max(F.seed_eff.max() * 1.32, 0.1))

    ax2.bar(xs - w / 2, F.ctrl_coinc_frac, w, label='seeded, NO track (control)',
            color=fs.BAND_CONTROL, alpha=0.6, edgecolor='none')
    ax2.bar(xs + w / 2, F.coinc_frac, w, label='has a gated track',
            color=[fs.DET_COLOR[a] for a in F.arm], edgecolor='none')
    for i, (_, r) in enumerate(F.iterrows()):
        ax2.annotate(f'{r.lift:.2f}×', (i, max(r.coinc_frac,
                                               r.ctrl_coinc_frac)),
                     textcoords='offset points', xytext=(0, 12), ha='center',
                     fontsize=fs.BASE_PT * 0.78, fontweight='bold',
                     color=fs.DET_COLOR[r.arm])
    ax2.set_xticks(xs)
    ax2.set_xticklabels(F.arm)
    ax2.set_ylabel('wall AND plastic in time,\nsame chamber')
    ax2.set_xlabel('chamber')
    ax2.set_title('Tracked events confirm more often', pad=10)
    ax2.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax2.set_ylim(0, max(F.coinc_frac.max() * 1.45, 0.1))
    fs.preliminary(ax2, loc='upper right')
    fs.save(fig, out / 'det_status', data=F)


def fig_det_evidence(F: pd.DataFrame, out):
    """The exclusive n_TOF partition of the tracked events, per chamber."""
    plt = _plt()
    with _scaled(plt, 0.8):
        return _fig_det_evidence(plt, F, out)


def _fig_det_evidence(plt, F: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(fs.QUARTER[0], 4.4),
                           constrained_layout=True)
    keys = [('wal_and_pss', 'wall AND plastic', '#0072B2'),
            ('wal_only', 'wall only', '#56B4E9'),
            ('pss_only', 'plastic only', '#E69F00'),
            ('neither', 'neither', '#b8bfc9')]
    ys = np.arange(len(F))[::-1]
    left = np.zeros(len(F))
    tot = F.n_track_events.to_numpy(float)
    for key, lab, col in keys:
        vals = F[key].to_numpy(float) / tot
        ax.barh(ys, vals, left=left, height=0.62, color=col, label=lab,
                edgecolor=fs.SURFACE, lw=2)
        for y, v_, l_ in zip(ys, vals, left):
            if v_ > 0.075:
                ax.annotate(f'{100 * v_:.0f}', (l_ + v_ / 2, y), ha='center',
                            va='center', color='white', fontweight='bold',
                            fontsize=fs.BASE_PT * 0.72)
        left = left + vals
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{a}' for a in F.arm])
    ax.set_xlim(0, 1)
    ax.set_xlabel('fraction of events with a gated track  [%]')
    ax.set_ylabel('chamber')
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f'{int(100 * t)}' for t in np.linspace(0, 1, 6)])
    ax.set_title('What n_TOF saw, in the same chamber, in time')
    ax.legend(fontsize=fs.BASE_PT * 0.65, frameon=False, ncol=2,
              loc='lower center', bbox_to_anchor=(0.5, -0.42))
    fs.preliminary(ax)
    fs.save(fig, out / 'det_evidence',
            data=F[['arm', 'n_track_events'] + [k for k, _, _ in keys]])


def fig_opening(out):
    """Opening angle between two chambers' tracks -- the end-to-end geometry check.

    This figure does NOT claim a pair signal; the rate is null (see pairs.py).
    What it validates is the geometry: two tracks from the target into OPPOSING
    chambers must give a large opening angle and into PERPENDICULAR ones ~90
    deg.  If the strip maps, either in-plane sign, the pinwheel, the transforms
    or k were wrong, these distributions would not separate.
    """
    import json
    from ntof_tracking import run145_target_imaging as TI
    from ntof_tracking.reco import geometry as G
    from sept26_prelim_analysis.build_tracks import (
        IN_PLANE_SIGN, IN_PLANE_SIGN_Y, STRIP_MAP_HALF)

    base = str(paths.out('fullpass') / 'run_145')
    runs = str(paths.root('runs'))
    cfg = json.loads((paths.root('runs') / 'run_145' / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)
    cal = json.load(open(paths.out('kcal') / 'k_arm_run_145.json'))
    K = cal['apply']
    DCA = 50.0
    rows = []
    # Whatever sub-runs the merged pass actually holds -- hardcoding the list
    # meant a newly reconstructed sub-run was silently left out of this figure
    # while every other product picked it up.
    subs = sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)))
    for sub in subs:
        for a, k in K.items():
            f = os.path.join(base, sub, f'mx17_{a}', 'events_prelim.parquet')
            if not os.path.exists(f):
                continue
            df = pd.read_parquet(f)
            sel = (df.x_ok.to_numpy() & df.y_ok.to_numpy()
                   & (df.n_tracks.to_numpy() > 0))
            xl = IN_PLANE_SIGN * (df.x_p0.to_numpy() - STRIP_MAP_HALF)
            yl = IN_PLANE_SIGN_Y * (df.y_p0.to_numpy() - STRIP_MAP_HALF)
            tx = df.x_tan_theta.to_numpy() * k
            ty = df.y_tan_theta.to_numpy() * k
            tr = trs[f'mx17_{a}']
            P0 = tr.local_to_global(xl, yl, np.zeros_like(xl))
            P1 = tr.local_to_global(xl - tx * 30., yl - ty * 30.,
                                    np.full_like(xl, 30.))
            D = P1 - P0
            D = D / np.linalg.norm(D, axis=-1, keepdims=True)
            r, _, _ = TI.axis_approach(P0, D)
            rows.append(pd.DataFrame(dict(
                subrun=sub, event_id=df.event_id.to_numpy(), arm=a, dca=r,
                dx=D[:, 0], dy=D[:, 1], dz=D[:, 2]))[sel])
    T = pd.concat(rows, ignore_index=True)
    P = T[T.dca < DCA]
    m = P.merge(P, on=['subrun', 'event_id'], suffixes=('1', '2'))
    m = m[m.arm1 < m.arm2].copy()
    dot = (m.dx1 * m.dx2 + m.dy1 * m.dy2 + m.dz1 * m.dz2).clip(-1, 1)
    m['open_deg'] = np.degrees(np.arccos(dot))
    m['pair'] = m.arm1 + m.arm2

    plt = _plt()
    with _scaled(plt, 0.8):
        fig, ax = plt.subplots(figsize=(fs.WIDE[0], 4.8),
                               constrained_layout=True)
        bins = np.arange(0, 181, 7.5)
        # opposing pairs are the signal topology; perpendicular are the control
        opposing = {'AC', 'BD'}
        # The two perpendicular pairs are one category but must still be
        # separable from each other: same muted ink, different dash.
        dashes = {}
        for pair, g in sorted(m.groupby('pair')):
            opp = pair in opposing
            col = fs.DET_COLOR[pair[0]] if opp else fs.MUTED
            if not opp:
                dashes[pair] = (4, 2) if len(dashes) == 0 else (1.5, 2)
            ax.hist(g.open_deg, bins=bins, histtype='step',
                    lw=2.6 if opp else 1.7, color=col,
                    ls='-' if opp else (0, dashes[pair]), density=True,
                    label=f'{pair[0]}\u2013{pair[1]}  '
                          f'({"opposing" if opp else "perpendicular"}), '
                          f'n={len(g)}, med {g.open_deg.median():.0f}\u00b0')
        ax.axvline(109, color=fs.BAND_SIGNAL, lw=2, ls=':')
        ax.annotate('X17 minimum, 109\u00b0', (109, ax.get_ylim()[1] * 0.94),
                    textcoords='offset points', xytext=(7, 0),
                    color=fs.BAND_SIGNAL, fontsize=fs.BASE_PT * 0.6)
        ax.set_xlabel('opening angle between the two tracks  [deg]')
        ax.set_ylabel('normalised')
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_title('Geometry check: opposing chambers give large opening '
                     'angles, perpendicular ones ~90\u00b0')
        ax.legend(frameon=False, loc='upper left',
                  fontsize=fs.BASE_PT * 0.62)
        fs.preliminary(ax, loc='upper right')
        fs.save(fig, out / 'opening_angle',
                data=m[['subrun', 'event_id', 'pair', 'open_deg']])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    od = paths.out('funnel', 'figures') if a.out is None else a.out
    fdir = paths.out('funnel')
    F = pd.read_csv(paths.require(os.path.join(str(fdir), f'funnel_{a.run}.csv'),
                                  'funnel table -- run funnel.py first'))
    cal = json.load(open(paths.require(
        paths.out('kcal') / f'k_arm_{a.run}.json', 'angle calibration')))

    from pathlib import Path
    od = Path(od)
    fig_k_scan(cal, od)
    fig_k_summary(cal, od)
    fig_det_status(F, od)
    fig_det_evidence(F, od)
    fig_opening(od)
    print(f'wrote 5 figures (+ CSVs) to {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
