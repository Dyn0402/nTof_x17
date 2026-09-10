#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tracking_qa_figures.py -- figures for `tracking_qa.py`.

Five figures, each answering one question, in the order a reader asks them:

  1. `qa_reference`   WHAT NORMAL IS -- the campaign distribution of the four
     variables that matter most, four arms overlaid. Everything after this is
     read against it.
  2. `qa_by_run`      IS ANY RUN DIFFERENT -- the same distribution drawn once
     per run, one panel per arm. A run that sits off the pack is visible here
     and nowhere in a table of medians.
  3. `qa_timeline`    DOES ANYTHING DRIFT -- per-tag median with its inter-
     quartile band against wall-clock time, so a trend and a step look
     different from each other.
  4. `qa_outlier_map` WHERE, COMPACTLY -- runs down, variables across, robust
     z as colour. The index into the two figures above.
  5. `qa_pathology`   WHAT BROKE OUTRIGHT -- the fraction of tracks per run
     whose fit returned a value that cannot be right. A median cannot show
     this at all, which is the point.

Every panel carries its own n. The distributions are drawn from the histogram
tables `tracking_qa` wrote, not re-derived, so the figure and the CSV beside it
cannot disagree.

    python -m sept26_prelim_analysis.make_tracking_qa_figures
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

from sept26_prelim_analysis import paths           # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis.tracking_qa import VARS, ARMS  # noqa: E402

#: The four the eye should go to first: fit quality, cluster size, the angle
#: error that sets the opening-angle resolution, and the charge that carries
#: the gain.
HEADLINE = ('chi2dof_x', 'n_strips_x', 'tan_err_x', 'q_total')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs.use()
    return plt


def _hist(qa_dir, var: str) -> pd.DataFrame:
    p = paths.require(os.path.join(qa_dir, f'hist_{var}.csv'),
                      f'histogram table for {var}')
    return pd.read_csv(p)


def _set_x(ax, var: str):
    _col, label, logx, clip = VARS[var]
    if logx:
        ax.set_xscale('log')
    if clip:
        ax.set_xlim(*clip)
    ax.set_xlabel(label)


# --------------------------------------------------------------------------- #
def fig_reference(qa_dir, per_arm: pd.DataFrame, out):
    """The campaign distribution per arm: what every later panel is read against."""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(fs.SLIDE[0], 8.4),
                             constrained_layout=True)
    tables = {}
    for ax, var in zip(axes.flat, HEADLINE):
        fs.strip(ax)
        h = _hist(qa_dir, var)
        tables[var] = h[h.arm != 'ALL'].groupby(['arm', 'centre'],
                                                as_index=False)['count'].sum()
        for arm in ARMS:
            sub = tables[var][tables[var].arm == arm]
            if sub.empty:
                continue
            y = sub['count'].to_numpy(float)
            y = y / max(y.sum(), 1)
            ax.step(sub['centre'], y, where='mid', color=fs.DET_COLOR[arm],
                    lw=2.2, label=arm)
        _set_x(ax, var)
        ax.set_ylabel('fraction of tracks')
        ax.legend(frameon=False, fontsize=fs.BASE_PT * 0.55, ncol=4,
                  loc='upper right')
    n = {r['arm']: int(r['n_tracks']) for _, r in per_arm.iterrows()}
    fig.suptitle('What normal looks like, per chamber',
                 fontsize=fs.BASE_PT * 1.45, fontweight='bold')
    fs.note(fig, 'gated tracks, whole campaign:  '
                 + '   '.join(f'{a} {n.get(a, 0):,}' for a in ARMS))
    return fs.save(fig, os.path.join(out, 'qa_reference'),
                   data={k: v for k, v in tables.items()})


def fig_by_run(qa_dir, var: str, flagged: set, out):
    """One curve per run, one panel per arm. The outlier hunt, drawn."""
    plt = _plt()
    h = _hist(qa_dir, var)
    h = h[h.arm != 'ALL']
    fig, axes = plt.subplots(2, 2, figsize=(fs.SLIDE[0], 8.4),
                             constrained_layout=True)
    for ax, arm in zip(axes.flat, ARMS):
        fs.strip(ax)
        sub = h[h.arm == arm]
        runs = sorted(sub.run.unique(), key=lambda r: int(r.split('_')[1]))
        for run in runs:
            s = sub[sub.run == run]
            hot = (arm, run) in flagged
            ax.step(s.centre, s.density, where='mid',
                    color=fs.TRACK if hot else fs.MUTED,
                    lw=2.2 if hot else 0.9, alpha=1.0 if hot else 0.30,
                    zorder=3 if hot else 1)
        hot_here = sorted(r for a, r in flagged if a == arm)
        ax.set_title(f'{arm}   {len(runs)} runs', fontsize=fs.BASE_PT,
                     color=fs.DET_COLOR[arm], loc='left', fontweight='bold')
        if hot_here:
            ax.text(0.98, 0.94, '  '.join(x.replace('run_', '') for x in
                                          hot_here[:6]),
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=fs.BASE_PT * 0.5, color=fs.TRACK)
        _set_x(ax, var)
        ax.set_ylabel('fraction')
    fig.suptitle(f'{VARS[var][1]} — one curve per run',
                 fontsize=fs.BASE_PT * 1.3, fontweight='bold')
    fs.note(fig, 'red: a run this variable flags as an outlier for that arm '
                 '(robust z ≥ 3.5 and ≥ 0.15 IQR of shift)')
    return fs.save(fig, os.path.join(out, f'qa_by_run_{var}'), data=h)


def fig_timeline(per_tag: pd.DataFrame, specs, out, min_tracks=50):
    """Per-tag median and IQR band against wall-clock time.

    A trend and a step look different here, and neither looks like the scatter
    of a chamber that is simply noisy. That distinction is the whole reason the
    tag level exists.
    """
    plt = _plt()
    d = per_tag[per_tag.n_tracks >= min_tracks].copy()
    d['t'] = pd.to_datetime(d['t'])
    fig, axes = plt.subplots(len(specs), 1, figsize=(fs.SLIDE[0], 3.1 * len(specs)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    keep = []
    for ax, (arm, var) in zip(axes, specs):
        fs.strip(ax)
        s = d[d.arm == arm].sort_values('t')
        lo, mid, hi = f'{var}_p25', f'{var}_p50', f'{var}_p75'
        ax.fill_between(s['t'], s[lo], s[hi], color=fs.DET_COLOR[arm],
                        alpha=0.18, lw=0)
        ax.plot(s['t'], s[mid], color=fs.DET_COLOR[arm], lw=1.2)
        if VARS[var][2]:
            ax.set_yscale('log')
        ax.set_ylabel(VARS[var][1], fontsize=fs.BASE_PT * 0.8)
        ax.set_title(f'chamber {arm}', loc='left', fontsize=fs.BASE_PT * 0.85,
                     color=fs.DET_COLOR[arm], fontweight='bold')
        keep.append(s[['arm', 'run', 'tag', 't', lo, mid, hi, 'n_tracks']]
                    .assign(variable=var))
    # The 27 July access: everything before it is the other condition.
    for ax in axes:
        ax.axvline(pd.Timestamp('2026-07-27 12:00'), color=fs.COPPER,
                   lw=1.4, ls='--', zorder=0)
    axes[0].text(pd.Timestamp('2026-07-27 13:00'),
                 axes[0].get_ylim()[1], ' 27 Jul access', color=fs.COPPER,
                 va='top', fontsize=fs.BASE_PT * 0.55)
    axes[-1].set_xlabel('file tag timestamp')
    fig.suptitle('What drifts, tag by tag', fontsize=fs.BASE_PT * 1.3,
                 fontweight='bold')
    fs.note(fig, f'line: per-tag median.  band: p25–p75.  '
                 f'tags with ≥ {min_tracks} gated tracks only')
    return fs.save(fig, os.path.join(out, 'qa_timeline'),
                   data=pd.concat(keep, ignore_index=True))


def fig_outlier_map(per_run: pd.DataFrame, out, top: int = 14):
    """Runs down, variables across, robust z as colour: the index."""
    plt = _plt()
    cols = [f'{v}_p50' for v in VARS if f'{v}_p50' in per_run.columns]
    fig, axes = plt.subplots(1, 4, figsize=(fs.SLIDE[0], 7.2), sharey=False,
                             constrained_layout=True)
    keep = []
    for ax, arm in zip(axes, ARMS):
        sub = per_run[per_run.arm == arm].copy()
        sub = sub.sort_values('t_start')
        Z = []
        for c in cols:
            x = pd.to_numeric(sub[c], errors='coerce')
            med = x.median()
            mad = 1.4826 * (x - med).abs().median()
            Z.append((x - med) / mad if np.isfinite(mad) and mad > 0
                     else pd.Series(0.0, index=x.index))
        Z = np.vstack([z.to_numpy() for z in Z]).T          # runs x vars
        Z = np.clip(np.nan_to_num(Z), -8, 8)
        im = ax.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-8, vmax=8)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[:-4] for c in cols], rotation=90,
                           fontsize=fs.BASE_PT * 0.42)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([r.replace('run_', '') for r in sub.run],
                           fontsize=fs.BASE_PT * 0.40)
        ax.set_title(arm, color=fs.DET_COLOR[arm], fontweight='bold',
                     fontsize=fs.BASE_PT)
        ax.grid(False)
        keep.append(pd.DataFrame(Z, columns=[c[:-4] for c in cols])
                    .assign(arm=arm, run=list(sub.run)))
    fig.colorbar(im, ax=axes, shrink=0.55, label='robust z of the run median')
    fig.suptitle('Which run departs, and on what',
                 fontsize=fs.BASE_PT * 1.3, fontweight='bold')
    fs.note(fig, 'runs in time order, top to bottom.  z clipped at ±8.  '
                 'colour is departure from the ARM\'s own median, so a globally '
                 'poor chamber does not colour every row')
    return fs.save(fig, os.path.join(out, 'qa_outlier_map'),
                   data=pd.concat(keep, ignore_index=True))


def fig_pathology(per_run: pd.DataFrame, out):
    """The fractions a median cannot see: fits that returned an impossible number."""
    plt = _plt()
    keys = ['frac_chi2_gt_100', 'frac_chi2_gt_1000', 'frac_tanerr_gt_0p1',
            'frac_q_gt_1e6', 'frac_strips_ge_200']
    keys = [k for k in keys if k in per_run.columns]
    fig, axes = plt.subplots(len(keys), 1, figsize=(fs.SLIDE[0], 2.0 * len(keys)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    d = per_run.sort_values('t_start')
    order = [r for r in dict.fromkeys(d.run)]
    xi = {r: i for i, r in enumerate(order)}
    for ax, k in zip(axes, keys):
        fs.strip(ax)
        for arm in ARMS:
            s = d[d.arm == arm]
            ax.plot([xi[r] for r in s.run], s[k].to_numpy(float), 'o-',
                    color=fs.DET_COLOR[arm], ms=4.5, lw=1.3, label=arm)
        ax.set_ylabel(k.replace('frac_', ''), fontsize=fs.BASE_PT * 0.65)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, ncol=4, fontsize=fs.BASE_PT * 0.55,
                   loc='upper left')
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels([r.replace('run_', '') for r in order],
                             rotation=90, fontsize=fs.BASE_PT * 0.45)
    axes[-1].set_xlabel('run, in time order')
    fig.suptitle('Fits that returned a number that cannot be right',
                 fontsize=fs.BASE_PT * 1.3, fontweight='bold')
    fs.note(fig, 'fraction of that run\'s gated tracks.  a 12-bit DREAM cluster '
                 'cannot hold 1e6 ADC, and a 512-strip plane cannot give a '
                 '200-strip track a meaningful angle')
    return fs.save(fig, os.path.join(out, 'qa_pathology'), data=d[
        ['arm', 'run', 't_start', 'n_tracks'] + keys])


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--qa-dir', default=None, help='default <out>/tracking_qa')
    ap.add_argument('--out', default=None, help='default <qa-dir>/figures')
    ap.add_argument('--by-run-vars', default='chi2dof_x,n_strips_x,q_total,t0_x')
    a = ap.parse_args()

    qa = a.qa_dir or str(paths.out('tracking_qa'))
    out = a.out or os.path.join(qa, 'figures')
    os.makedirs(out, exist_ok=True)

    per_arm = pd.read_csv(paths.require(os.path.join(qa, 'per_arm.csv'), 'per_arm'))
    per_run = pd.read_csv(paths.require(os.path.join(qa, 'per_run.csv'), 'per_run'))
    per_tag = pd.read_csv(paths.require(os.path.join(qa, 'per_tag.csv'), 'per_tag'))
    outl = pd.read_csv(paths.require(os.path.join(qa, 'outliers.csv'), 'outliers'))
    drift = pd.read_csv(paths.require(os.path.join(qa, 'drift.csv'), 'drift'))

    print('[fig] reference')
    fig_reference(qa, per_arm, out)

    for v in [x for x in a.by_run_vars.split(',') if x]:
        if not os.path.exists(os.path.join(qa, f'hist_{v}.csv')):
            print(f'  ! no histogram table for {v}, skipped')
            continue
        flagged = {(r.arm, r.run) for r in outl.itertuples()
                   if r.variable.startswith(v)}
        print(f'[fig] by run: {v}  ({len(flagged)} flagged)')
        fig_by_run(qa, v, flagged, out)

    # The strongest trends the drift table found, one per arm at most, so the
    # timeline shows what actually moved rather than a fixed shopping list.
    # `n_cand_*` moves between 2 and 3 and nothing else, so a strong Spearman
    # rho on it is a step between two integers -- true, and unreadable as a
    # time series. The timeline wants a continuous variable.
    quantized = {'n_cand_x', 'n_cand_y'}
    specs, seen = [], set()
    for r in drift.sort_values('rho', key=lambda s: s.abs(), ascending=False) \
                   .itertuples():
        var = r.variable[:-4] if r.variable.endswith('_p50') else None
        if var is None or var not in VARS or var in quantized or r.arm in seen:
            continue
        seen.add(r.arm)
        specs.append((r.arm, var))
        if len(specs) == 4:
            break
    if specs:
        print(f'[fig] timeline: {specs}')
        fig_timeline(per_tag, specs, out)

    print('[fig] outlier map')
    fig_outlier_map(per_run, out)
    print('[fig] pathology')
    fig_pathology(per_run, out)
    print(f'\nfigures in {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
