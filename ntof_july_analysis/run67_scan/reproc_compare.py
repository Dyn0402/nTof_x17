#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What did the 2026-07-24 small-pulse reprocessing buy? Old cache vs new.

The runs were re-decoded on 2026-07-24 with a lower pulse-finding threshold, and
the run_67 reco cache was rebuilt on top (plus three reco fixes — see README
"Re-reco on the reprocessed hits"). The pre-reprocessing cache was kept at
`cache/run_67_preReprocess_20260723/`, so the two are directly comparable:
SAME sub-runs, SAME events, SAME reco code path, different input hits.

This script quantifies the change, because a 10x jump in a tracking efficiency
is exactly as consistent with "we now accept noise" as with "we now find the
tracks we were missing". The distinction is made on SEGMENT QUALITY, not on
yield:

  * genuine recovery -> segments gain strips and time extent, r2 goes UP,
    median amplitude goes DOWN (the recovered hits are the small ones), and the
    X/Y pair fraction rises (two orthogonal planes cannot agree by accident);
  * noise contamination -> r2 and occupancy fall, tspan shortens, and the pair
    fraction does NOT rise, because uncorrelated noise does not pair.

Det A (clean M1) is the reference; B/C/D sit on bad M1 cards and their
single-plane yields are noise-inflated by construction.

Outputs -> <OUT_BASE>/reproc_compare/
Run: .venv/bin/python ntof_july_analysis/run67_scan/reproc_compare.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import scan_lib as L  # noqa: E402
import stats as ST  # noqa: E402

OLD_CACHE = os.path.join(L.CACHE_DIR, 'run_67_preReprocess_20260723')
NEW_CACHE = os.path.join(L.CACHE_DIR, L.RUN)
OUT_DIR = os.path.join(L.OUT_BASE, 'reproc_compare')

QUAL_COLS = ['n_strips', 'pspan_mm', 'tspan_ns', 'r2', 'q_sum', 'a_max']


def _subruns(cache):
    return {f[:-len('_events.parquet')]
            for f in os.listdir(cache) if f.endswith('_events.parquet')}


def _hits_mtime(subrun):
    """Newest combined_hits mtime for a sub-run (0.0 if none found)."""
    import glob
    fs = glob.glob(os.path.join(L.io.BASE_PATH, L.RUN, subrun,
                                'combined_hits_root', '*_datrun_*.root'))
    return max((os.path.getmtime(f) for f in fs), default=0.0)


def common_subruns():
    """Sub-runs present in both caches AND genuinely re-reco'd.

    GUARD (this bit matters). `process.py --force` rewrites the new cache IN
    PLACE, so while a re-reco is in flight the new-cache directory still holds
    the PRE-reprocessing parquet for every sub-run not yet redone. Comparing
    those to the backup silently compares a file to itself and reports "the
    reprocessing changed nothing" for most of the grid. Freshness is therefore
    established per sub-run against the combined_hits it was built from, not
    from the file merely existing.
    """
    if not os.path.isdir(OLD_CACHE):
        sys.exit(f'no pre-reprocessing cache at {OLD_CACHE}')
    old, new = _subruns(OLD_CACHE), _subruns(NEW_CACHE)
    both = sorted(old & new)
    only_new = sorted(new - old)
    if only_new:
        print(f'  {len(only_new)} sub-run(s) only in the NEW cache — excluded')
    fresh, stale = [], []
    for s in both:
        ev_p = os.path.join(NEW_CACHE, f'{s}_events.parquet')
        (fresh if os.path.getmtime(ev_p) > _hits_mtime(s) else stale).append(s)
    if stale:
        print(f'  !! {len(stale)} sub-run(s) in the new cache are OLDER than '
              f'their combined_hits — NOT re-reco\'d yet, EXCLUDED. '
              f'Finish process.py --force before trusting this comparison.')
    print(f'  comparing {len(fresh)} genuinely re-reco\'d sub-run(s)')
    return fresh


def load_side(cache, subs):
    """Per-event and per-segment tables from one cache, flash-ok probes only."""
    evs, sgs = [], []
    for s in subs:
        ev = pd.read_parquet(os.path.join(cache, f'{s}_events.parquet'))
        evs.append(ev[ev.flash_ok & ~ev.is_leader])
        sg = pd.read_parquet(os.path.join(cache, f'{s}_segs.parquet'))
        if not sg.empty:
            sgs.append(sg)
    return (pd.concat(evs, ignore_index=True),
            pd.concat(sgs, ignore_index=True) if sgs else pd.DataFrame())


def eff_table(ev):
    """P(3D pair) and P(track segment) per (mip, drift, resist, det).

    NOTE the denominator here is every flash-ok probe event, NOT the
    FEU-readout-gated denominator of stats.per_cell_stats. That is deliberate:
    the presence table is rebuilt from the new combined_hits and so is not
    common to both sides, and this script only ever compares the two caches to
    EACH OTHER. Do not quote these as the analysis efficiencies — use
    slide_curves.csv / per_cell_stats for that.
    """
    rows = []
    for (mip, dr, r), g in ev.groupby(['mip', 'drift', 'resist']):
        n = len(g)
        for Ld in 'ABCD':
            rows.append({
                'mip': mip, 'drift': dr, 'resist': r, 'det': Ld, 'n': n,
                'p_pair': (g[f'n_pair_{Ld}'] > 0).mean(),
                'p_trk': (g[f'n_trkseg_{Ld}'] > 0).mean(),
            })
    return pd.DataFrame(rows)


def fig_efficiency(old_e, new_e):
    """New vs old efficiency, per det, coloured by threshold."""
    m = old_e.merge(new_e, on=['mip', 'drift', 'resist', 'det'],
                    suffixes=('_old', '_new'))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, Ld in zip(axes, 'ABCD'):
        d = m[m.det == Ld]
        for mip, g in d.groupby('mip'):
            ax.scatter(g.p_pair_old, g.p_pair_new, s=26, alpha=0.8,
                       color=L.MIP_COLOR[mip], label=L.MIP_LABEL[mip])
        hi = max(d.p_pair_old.max(), d.p_pair_new.max(), 1e-4) * 1.15
        ax.plot([0, hi], [0, hi], 'k--', lw=1, label='no change')
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel('P(3D pair) — before reprocessing')
        ax.set_title(f'Det {Ld}' + (' (clean M1)' if Ld == 'A' else ''),
                     fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('P(3D pair) — after')
    axes[0].legend(fontsize=8)
    fig.suptitle('run_67 — 3D x/y pair efficiency, small-pulse reprocessing + '
                 'reco fixes vs before (one point per HV cell)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT_DIR, 'efficiency_new_vs_old.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('  wrote', p)
    return m


def fig_gain_vs_hv(m):
    """Where the gain lands in HV space — the low-HV end is the acid test.

    beam_track_finding saw its 200->50 ADC recovery concentrate at LOW HV
    (under-amplified tracks). The same signature here is evidence the recovered
    hits are track hits rather than a flat noise pedestal.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, mip in zip(axes, sorted(m.mip.unique())):
        d = m[(m.mip == mip)]
        for Ld, col in ST.DET_COL.items():
            g = (d[d.det == Ld].groupby('resist')[['p_pair_old', 'p_pair_new']]
                 .mean().reset_index())
            ratio = g.p_pair_new / g.p_pair_old.replace(0, np.nan)
            ax.plot(g.resist, ratio, marker='o', ms=5, lw=1.5, color=col,
                    label=f'Det {Ld}')
        ax.axhline(1.0, color='k', ls='--', lw=1)
        ax.set_yscale('log')
        ax.set_xlabel('resist HV [V]')
        ax.set_title(L.MIP_LABEL[mip], fontsize=11)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('efficiency gain  (new / old)')
    axes[0].legend(fontsize=8)
    fig.suptitle('run_67 — where the small-pulse gain lands in HV space '
                 '(pooled over drift)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(OUT_DIR, 'gain_vs_hv.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('  wrote', p)


def fig_quality(old_s, new_s, det='A'):
    """Segment-quality distributions: the real/noise discriminator."""
    o = old_s[old_s.det == det]
    n = new_s[new_s.det == det]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), QUAL_COLS):
        lo = float(min(o[c].quantile(0.01), n[c].quantile(0.01)))
        hi = float(max(o[c].quantile(0.99), n[c].quantile(0.99)))
        if c == 'n_strips':
            # integer-valued: linspace bins straddle integers unevenly and draw
            # a comb that looks like structure. One bin per strip count.
            bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
        else:
            bins = np.linspace(lo, hi, 60)
        ax.hist(o[c], bins=bins, histtype='step', lw=1.8, density=True,
                color='0.45', label=f'before (n={len(o)})')
        ax.hist(n[c], bins=bins, histtype='step', lw=1.8, density=True,
                color='crimson', label=f'after (n={len(n)})')
        ax.axvline(o[c].median(), color='0.45', ls=':', lw=1.4)
        ax.axvline(n[c].median(), color='crimson', ls=':', lw=1.4)
        ax.set_xlabel(c)
        ax.set_ylabel('density')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'run_67 Det {det} track-segment quality — before vs after the '
                 f'small-pulse reprocessing\n'
                 f'genuine recovery: strips/tspan/r² UP, amplitude DOWN  |  '
                 f'noise contamination: r² DOWN', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(OUT_DIR, f'segment_quality_det{det}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('  wrote', p)


def summary(old_e, new_e, old_s, new_s, path):
    lines = ['# run_67 — small-pulse reprocessing: before vs after', '']
    lines.append('## 3D x/y pair efficiency (mean over HV cells, per det)')
    lines.append('')
    lines.append('| det | before | after | gain |')
    lines.append('|---|---|---|---|')
    for Ld in 'ABCD':
        a = old_e[old_e.det == Ld].p_pair.mean()
        b = new_e[new_e.det == Ld].p_pair.mean()
        lines.append(f'| {Ld} | {a:.4f} | {b:.4f} | '
                     f'{(b / a if a else np.nan):.1f}x |')
    lines.append('')
    lines.append('## Det A segment quality (median) — the real/noise test')
    lines.append('')
    lines.append('| quantity | before | after | expected if REAL |')
    lines.append('|---|---|---|---|')
    exp = {'n_strips': 'up', 'pspan_mm': 'flat/up', 'tspan_ns': 'up',
           'r2': 'up', 'q_sum': 'down', 'a_max': 'down'}
    oa, na = old_s[old_s.det == 'A'], new_s[new_s.det == 'A']
    for c in QUAL_COLS:
        lines.append(f'| {c} | {oa[c].median():.2f} | {na[c].median():.2f} | '
                     f'{exp[c]} |')
    lines.append(f'| in_pair fraction | {oa.in_pair.mean():.3f} | '
                 f'{na.in_pair.mean():.3f} | up |')
    lines.append('')
    lines.append('Uncorrelated noise cannot raise the X/Y pair fraction: the '
                 'two planes are orthogonal, so a coincidence has to come from '
                 'a real ionisation track.')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('  wrote', path)


def main(argv=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    subs = common_subruns()
    if not subs:
        sys.exit('no sub-runs in common')
    old_ev, old_sg = load_side(OLD_CACHE, subs)
    new_ev, new_sg = load_side(NEW_CACHE, subs)
    print(f'  old: {len(old_ev)} events / {len(old_sg)} segs')
    print(f'  new: {len(new_ev)} events / {len(new_sg)} segs')
    if len(old_ev) != len(new_ev):
        print('  NOTE: event counts differ between caches — the comparison is '
              'still per-cell, but cells are not event-identical')

    old_e, new_e = eff_table(old_ev), eff_table(new_ev)
    old_e.to_csv(os.path.join(OUT_DIR, 'eff_before.csv'), index=False)
    new_e.to_csv(os.path.join(OUT_DIR, 'eff_after.csv'), index=False)
    m = fig_efficiency(old_e, new_e)
    m.to_csv(os.path.join(OUT_DIR, 'eff_paired.csv'), index=False)
    fig_gain_vs_hv(m)
    for det in 'ABCD':
        fig_quality(old_sg, new_sg, det)
    summary(old_e, new_e, old_sg, new_sg,
            os.path.join(OUT_DIR, 'SUMMARY.md'))
    print('done ->', OUT_DIR)


if __name__ == '__main__':
    main()
