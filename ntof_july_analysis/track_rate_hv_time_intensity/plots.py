#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots.py — run_67 tracks vs HV vs time-since-flash, on CNS-reprocessed hits.

Track definitions follow the run_70 CNS study (detA_track_freq_run70/plots_clean.py):
  drift track   n_hits >= 5, time_span >= 0.10 us, |angle| < 80 deg
                (a genuine drift-time spread, not a same-time charge-sharing cluster)
  2D drift      an event with a drift track in BOTH the x and the y projection
Anything without the time-span / angle cut is dominated by residual same-time clusters,
so the loose numbers are reported only as an upper bound.

Rate is quoted per physics trigger (denominator = all decoded non-flash triggers).

Figures (figures/):
  angle_sanity.png      angle + time_span distributions per detector — is CNS clean?
  rate_vs_hv.png        2D-drift rate vs resist HV, one line per time-since-flash band
  hv_time_heatmap.png   resist x dt heatmap of the rate, per detector
  rate_vs_time.png      rate vs dt, one line per resist HV
  drift_panels.png      rate vs resist, separate panel per drift HV
Run: ~/PycharmProjects/nTof_x17/.venv/bin/python plots.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

_ap = argparse.ArgumentParser(description='HV x time-since-flash track-rate figures')
_ap.add_argument('--cache', default=None, help='cache dir (default ./cache)')
_ap.add_argument('--out', default=None, help='figure output dir (default ./figures)')
_args, _ = _ap.parse_known_args()

OUT = Path(__file__).resolve().parent
CACHE = Path(_args.cache) if _args.cache else OUT / 'cache'
FIG = Path(_args.out) if _args.out else OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

NHIT, TSPAN, ANG = 5, 0.10, 80.0
# time-since-flash bands. 1-4 ms is the saturated/deadtime-shadowed early window,
# 4-8 ms the GEANT thermal peak, then the late tail out to the 81 ms gate close.
BANDS = [(1, 4), (4, 8), (8, 20), (20, 81)]
BAND_C = ['#c4342b', '#e08214', '#4c72b0', '#2a7f62']
DETS = ['A', 'B', 'C', 'D']
INK, MUTED = '#1a1a1a', '#666666'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})

tr = pd.read_csv(CACHE / 'tracks.csv')
ev = pd.read_csv(CACHE / 'events.csv')
sr = pd.read_csv(CACHE / 'subruns.csv')
MIPS = sorted(ev['mip'].unique(), reverse=True)
DRIFTS = sorted(ev['drift'].unique(), reverse=True)
RESISTS = sorted(ev['resist'].unique())


def drift_tracks(t):
    return t[(t['n_hits'] >= NHIT) & (t['time_span'] >= TSPAN) & (t['angle_deg'].abs() < ANG)]


def twod_events(t):
    """event-level table (det, subrun, event_id, dt_ms, drift, resist, mip) with an
    x AND a y drift track.

    `trk_hits` = total hits carried by that event's tracks, a PILE-UP PROXY. The run_70
    CNS study found the loose 2D-drift selection is inflated by busy/pile-up events where
    residual structure fakes an X&Y coincidence (median occupancy 226 hits), so every
    result here is also quoted for the low-pile-up subset."""
    per = t.groupby(['det', 'subrun', 'event_id', 'projection']).size().unstack(fill_value=0)
    has = (per.get('x', 0) > 0) & (per.get('y', 0) > 0)
    g = has[has].reset_index()[['det', 'subrun', 'event_id']]
    meta = t.groupby(['det', 'subrun', 'event_id']).agg(
        dt_ms=('dt_ms', 'first'), trk_hits=('n_hits', 'sum')).reset_index()
    g = g.merge(meta, on=['det', 'subrun', 'event_id'], how='left')
    return g.merge(ev[['subrun', 'mip', 'drift', 'resist']].drop_duplicates(), on='subrun')


DR = drift_tracks(tr)
G2D = twod_events(DR)
# low-pile-up subset: a genuine single through-going track deposits a few strips per
# view, not hundreds. Cut chosen to match the run_70 "clean single track" definition.
MAX_TRK_HITS = 30
G2D_CLEAN = G2D[G2D['trk_hits'] <= MAX_TRK_HITS]


def _rate(num_df, den_df, keys):
    """percent rate + binomial error, grouped by `keys`."""
    num = num_df.groupby(keys).size()
    den = den_df.groupby(keys).size()
    idx = den.index
    n = num.reindex(idx, fill_value=0).astype(float)
    d = den.astype(float)
    return 100 * n / d, 100 * np.sqrt(np.maximum(n, 1)) / d, d


def _band(df):
    """add a `band` label column from dt_ms."""
    df = df.copy()
    lab = pd.Series(index=df.index, dtype=object)
    for lo, hi in BANDS:
        lab[(df['dt_ms'] >= lo) & (df['dt_ms'] < hi)] = f'{lo}-{hi} ms'
    df['band'] = lab
    return df[df['band'].notna()]


def fig_angle_sanity():
    """Is the CNS-reprocessed sample actually clean? The CNS-off pathology was a hard
    ±90 deg spike with time_span ~ 0 (whole plane firing in one 60 ns sample)."""
    t5 = tr[tr['n_hits'] >= NHIT]
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for j, d in enumerate(DETS):
        s = t5[t5['det'] == d]
        axs[0, j].hist(s['angle_deg'], bins=np.linspace(-90, 90, 61), color='#4c72b0')
        axs[0, j].axvspan(-90, -ANG, color='#c4342b', alpha=0.12)
        axs[0, j].axvspan(ANG, 90, color='#c4342b', alpha=0.12)
        frac = 100 * (s['angle_deg'].abs() >= ANG).mean() if len(s) else np.nan
        axs[0, j].set_title(f'Det {d}   n={len(s)}\n{frac:.0f}% at |angle|>{ANG:.0f}° (rejected)',
                            fontsize=9.5)
        axs[0, j].set_xlabel('angle [deg]')
        axs[1, j].hist(s['time_span'], bins=np.linspace(0, 1.0, 50), color='#e08214')
        axs[1, j].axvline(TSPAN, color='#c4342b', lw=1.5, ls='--')
        axs[1, j].set_xlabel('time_span [µs]')
        axs[1, j].set_yscale('log')
        for ax in (axs[0, j], axs[1, j]):
            ax.grid(alpha=0.2)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel('tracks')
    axs[1, 0].set_ylabel('tracks')
    fig.suptitle('run_67 CNS-reprocessed — track angle and drift-time spread per detector\n'
                 'red = the same-time (charge-sharing / residual common-mode) population '
                 'the drift cut removes', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = FIG / 'angle_sanity.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_rate_vs_hv():
    """The headline: 2D-drift-track rate vs resist HV, one line per dt band.
    Top row = all 2D-drift events, bottom row = the low-pile-up subset."""
    e = _band(ev)
    fig, axs = plt.subplots(2, 4, figsize=(17, 9), sharex=True)
    for i, (src, tag) in enumerate([(G2D, 'all 2D-drift'),
                                    (G2D_CLEAN, f'low pile-up (≤{MAX_TRK_HITS} hits)')]):
        g = _band(src)
        for j, d in enumerate(DETS):
            ax = axs[i, j]
            gd = g[g['det'] == d]
            for (lo, hi), c in zip(BANDS, BAND_C):
                lab = f'{lo}-{hi} ms'
                r, er, den = _rate(gd[gd['band'] == lab], e[e['band'] == lab], ['resist'])
                ax.errorbar(r.index, r.values, yerr=er.values, color=c, marker='o', ms=5,
                            lw=1.8, capsize=2, label=lab)
            if i == 0:
                ax.set_title(f'Det {d}', fontsize=11)
            if i == 1:
                ax.set_xlabel('resist (mesh) HV [V]')
            ax.grid(alpha=0.25)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        axs[i, 0].set_ylabel(f'{tag}\ntrack per trigger [%]', fontsize=9.5)
    axs[0, 0].legend(fontsize=8, title='time since flash', title_fontsize=8)
    fig.suptitle('run_67 — real (CNS) 2D-drift track rate vs resist HV, per time-since-flash band\n'
                 'drift HV and plastic threshold pooled; error bars binomial',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = FIG / 'rate_vs_hv.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_heatmap():
    """resist x dt map of the rate — where does the optimum sit and does it move?"""
    edges = np.array([1, 4, 8, 14, 20, 30, 45, 81], dtype=float)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for j, d in enumerate(DETS):
        gd = G2D[G2D['det'] == d]
        M = np.full((len(RESISTS), len(edges) - 1), np.nan)
        for i, rv in enumerate(RESISTS):
            num, _ = np.histogram(gd[gd['resist'] == rv]['dt_ms'], bins=edges)
            den, _ = np.histogram(ev[ev['resist'] == rv]['dt_ms'], bins=edges)
            M[i] = np.where(den > 0, 100 * num / den, np.nan)
        im = axs[j].pcolormesh(edges, np.arange(len(RESISTS) + 1), M, cmap='magma',
                               shading='flat')
        axs[j].set_yticks(np.arange(len(RESISTS)) + 0.5)
        axs[j].set_yticklabels(RESISTS)
        axs[j].set_xlabel('time since flash [ms]')
        axs[j].set_title(f'Det {d}', fontsize=11)
        # mark the per-column (per-time) best HV — this is the "does the optimum move?" test
        for k in range(M.shape[1]):
            col = M[:, k]
            if np.isfinite(col).any() and np.nanmax(col) > 0:
                axs[j].plot(ctr[k], np.nanargmax(col) + 0.5, marker='*', ms=11,
                            color='#00e5ff', mec='k', mew=0.5)
        fig.colorbar(im, ax=axs[j], label='rate [%]' if j == 3 else '')
    axs[0].set_ylabel('resist (mesh) HV [V]')
    fig.suptitle('run_67 — 2D-drift track rate: resist HV × time since flash  '
                 '(★ = best HV in that time slice)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p = FIG / 'hv_time_heatmap.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_rate_vs_time():
    edges = np.array([1, 4, 8, 14, 20, 30, 45, 81], dtype=float)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    cmap = {rv: c for rv, c in zip(RESISTS, plt.cm.viridis(np.linspace(0.05, 0.92, len(RESISTS))))}
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), sharex=True)
    for j, d in enumerate(DETS):
        gd = G2D[G2D['det'] == d]
        for rv in RESISTS:
            num, _ = np.histogram(gd[gd['resist'] == rv]['dt_ms'], bins=edges)
            den, _ = np.histogram(ev[ev['resist'] == rv]['dt_ms'], bins=edges)
            with np.errstate(divide='ignore', invalid='ignore'):
                fr = np.where(den > 0, 100 * num / den, np.nan)
            axs[j].plot(ctr, fr, 'o-', color=cmap[rv], lw=1.6, ms=4, label=f'{rv} V')
        axs[j].set_title(f'Det {d}', fontsize=11)
        axs[j].set_xlabel('time since flash [ms]')
        axs[j].grid(alpha=0.25)
        axs[j].axvspan(4, 8, color='#c4342b', alpha=0.07)
        for sp in ('top', 'right'):
            axs[j].spines[sp].set_visible(False)
    axs[0].set_ylabel('2D-drift track per trigger [%]')
    axs[0].legend(fontsize=7.5, title='resist HV', title_fontsize=8, ncol=2)
    fig.suptitle('run_67 — 2D-drift track rate vs time since flash, one line per resist HV\n'
                 'shaded = 4-8 ms GEANT thermal peak', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = FIG / 'rate_vs_time.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_drift_panels():
    """separate the drift-HV axis: rate vs resist for each drift setting."""
    g, e = _band(G2D), _band(ev)
    fig, axs = plt.subplots(len(DRIFTS), 4, figsize=(17, 3.6 * len(DRIFTS)),
                            sharex=True, squeeze=False)
    for i, dv in enumerate(DRIFTS):
        for j, d in enumerate(DETS):
            ax = axs[i, j]
            gd = g[(g['det'] == d) & (g['drift'] == dv)]
            ed = e[e['drift'] == dv]
            for (lo, hi), c in zip(BANDS, BAND_C):
                lab = f'{lo}-{hi} ms'
                if not (ed['band'] == lab).any():
                    continue
                r, er, den = _rate(gd[gd['band'] == lab], ed[ed['band'] == lab], ['resist'])
                ax.errorbar(r.index, r.values, yerr=er.values, color=c, marker='o', ms=4,
                            lw=1.5, capsize=2, label=lab)
            ax.grid(alpha=0.25)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(f'Det {d}', fontsize=11)
            if j == 0:
                ax.set_ylabel(f'drift {dv} V\n2D-drift track / trigger [%]', fontsize=9.5)
            if i == len(DRIFTS) - 1:
                ax.set_xlabel('resist HV [V]')
    axs[0, 0].legend(fontsize=7.5, title='time since flash', title_fontsize=8)
    fig.suptitle('run_67 — 2D-drift track rate vs resist HV, split by drift HV',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = FIG / 'drift_panels.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_threshold_check():
    """Does the HV x time structure depend on the plastic threshold? Compare the MIP
    settings on the drift values where more than one threshold has been reprocessed."""
    shared = [dv for dv in DRIFTS
              if ev[ev['drift'] == dv]['mip'].nunique() > 1]
    if not shared:
        return None
    g, e = _band(G2D), _band(ev)
    g, e = g[g['drift'].isin(shared)], e[e['drift'].isin(shared)]
    mips = sorted(e['mip'].unique(), reverse=True)
    ls = {m: s for m, s in zip(mips, ['-', '--', ':'])}
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), sharex=True)
    for j, d in enumerate(DETS):
        ax = axs[j]
        for m in mips:
            gd = g[(g['det'] == d) & (g['mip'] == m)]
            ed = e[e['mip'] == m]
            for (lo, hi), c in zip(BANDS, BAND_C):
                lab = f'{lo}-{hi} ms'
                r, er, den = _rate(gd[gd['band'] == lab], ed[ed['band'] == lab], ['resist'])
                ax.errorbar(r.index, r.values, yerr=er.values, color=c, ls=ls[m],
                            marker='o' if m == mips[0] else 's', ms=4, lw=1.5, capsize=2,
                            label=f'{lab}, {m:.2f} MIP')
        ax.set_title(f'Det {d}', fontsize=11)
        ax.set_xlabel('resist (mesh) HV [V]')
        ax.grid(alpha=0.25)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel('2D-drift track per trigger [%]')
    axs[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle('run_67 — does the HV × time structure move with plastic threshold?  '
                 f'(drift {", ".join(str(d) for d in shared)} V; solid = {mips[0]:.2f} MIP)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = FIG / 'threshold_check.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p


def table():
    """rate per (det, resist, band) -> CSV, plus the best HV per band."""
    g, gc, e = _band(G2D), _band(G2D_CLEAN), _band(ev)
    rows = []
    for d in DETS:
        for lo, hi in BANDS:
            lab = f'{lo}-{hi} ms'
            gd = g[(g['det'] == d) & (g['band'] == lab)]
            gcd = gc[(gc['det'] == d) & (gc['band'] == lab)]
            ed = e[e['band'] == lab]
            for rv in RESISTS:
                n = int((gd['resist'] == rv).sum())
                nc = int((gcd['resist'] == rv).sum())
                den = int((ed['resist'] == rv).sum())
                if den == 0:
                    continue
                rows.append(dict(det=d, band=lab, resist=rv, n_2d=n, n_clean=nc, n_trig=den,
                                 rate_pct=100 * n / den,
                                 err_pct=100 * np.sqrt(max(n, 1)) / den,
                                 clean_pct=100 * nc / den))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'rate_vs_hv_time.csv', index=False)
    best = (df.sort_values('rate_pct', ascending=False)
              .groupby(['det', 'band']).head(1)
              .sort_values(['det', 'band']))
    return df, best


def main():
    for f in (fig_angle_sanity, fig_rate_vs_hv, fig_heatmap, fig_rate_vs_time,
              fig_drift_panels, fig_threshold_check):
        p = f()
        print('wrote', p) if p else print(f'skipped {f.__name__} (needs >1 threshold '
                                          'on a shared drift setting)')
    df, best = table()
    print('\nbest resist HV per detector x time band (2D-drift rate):')
    print(best.to_string(index=False,
                         columns=['det', 'band', 'resist', 'rate_pct', 'err_pct', 'n_2d', 'n_trig'],
                         float_format=lambda x: f'{x:.3f}'))
    print('\nfull table -> rate_vs_hv_time.csv')


if __name__ == '__main__':
    main()
