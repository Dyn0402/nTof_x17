#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intensity_split.py — split the run_67 tracks-vs-HV-vs-time result by BEAM PULSE INTENSITY.

Follows the recipe in `nTof_x17/ntof_july_analysis/pulse_match.py` +
`run30_flash_intensity.py`: each DREAM event is matched to its PS pulse via the
beam_watcher per-pulse log (clock-offset fit), and the July pulses are bimodal
(~410e10 and ~850e10) so events split cleanly into LOW and HIGH at 600e10.

Why the burst is the natural unit here: run_67 is flash-anchored, one gamma flash per
PS pulse, so a "cluster" in pulse_match IS a burst IS a beam pulse. Every trigger in a
burst inherits that pulse's intensity.

The numerator (tracks) carries a combined-hits `event_id`, so pulse_match's per-event
map applies directly. The denominator (`cache/events.csv`) is built from the DECODED
trigger list and has no event ids — so it is re-derived here from the decoded times with
a burst index, and each physics trigger inherits its burst's intensity BY BURST ORDER.
The burst counts from the two paths are asserted equal before that mapping is used.

Outputs:
  cache/e10_tracks.csv     subrun, event_id, e10
  cache/e10_events.csv     subrun, dt_ms, e10   (row-aligned with cache/events.csv)
  figures/intensity_rate_vs_hv.png     rate vs resist HV, LOW vs HIGH, per dt band
  figures/intensity_heatmap.png        resist x dt map per intensity class
  figures/intensity_ratio.png          HIGH/LOW rate ratio vs HV and vs dt
  intensity_split.csv                  the numbers

Run: ~/PycharmProjects/nTof_x17/.venv/bin/python intensity_split.py
     (add --rebuild to refit the pulse matching)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / 'beam_july/analysis/flash_timing_threshold'))
import flash_timing_lib as FT  # noqa: E402

PM_DIR = Path.home() / 'PycharmProjects/nTof_x17/ntof_july_analysis'
sys.path.insert(0, str(PM_DIR))
import pulse_match as PM  # noqa: E402

import argparse

_ap = argparse.ArgumentParser(description='split the track-rate result by beam intensity')
_ap.add_argument('--run', default='run_67', help='run name (for pulse_match)')
_ap.add_argument('--cache', default=None, help='cache dir (default ./cache)')
_ap.add_argument('--out', default=None, help='figure output dir (default ./figures)')
_ap.add_argument('--rebuild', action='store_true', help='refit the pulse matching')
_args, _ = _ap.parse_known_args()

RUN = _args.run
OUT = Path(__file__).resolve().parent
CACHE = Path(_args.cache) if _args.cache else OUT / 'cache'
FIG = Path(_args.out) if _args.out else OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

E10_SPLIT = 600.0        # July pulses are bimodal ~410 vs ~850 e10
NHIT, TSPAN, ANG = 5, 0.10, 80.0
BANDS = [(1, 4), (4, 8), (8, 20), (20, 81)]
BAND_C = ['#c4342b', '#e08214', '#4c72b0', '#2a7f62']
DETS = ['A', 'B', 'C', 'D']
CLASSES = [('low', '#4c72b0'), ('high', '#c4342b')]
INK, MUTED = '#1a1a1a', '#666666'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})


# ---------------------------------------------------------------- pulse matching

def burst_e10_from_combined(run, sub, rebuild=False):
    """Per-BURST intensity, in time order, from pulse_match's per-event map."""
    pm = PM.match_subrun(run, sub, rebuild=rebuild)
    if pm is None:
        return None, None
    eid, t_rel, _ = PM._event_times(run, sub)
    if eid is None:
        return None, None
    starts = np.concatenate([[0], np.where(np.diff(t_rel) > PM.GAP_S)[0] + 1])
    e10 = np.array([pm['event_e10'].get(int(eid[s])) or np.nan for s in starts],
                   dtype=float)
    return e10, pm


def decoded_burst_index(run, sub):
    """(dt_ms, burst_idx) for every PHYSICS trigger in the decoded list — the same
    quantity and ordering `build_cache.py` wrote into cache/events.csv."""
    t = FT._event_times_ms(Path(FT.RUNS_DIR) / run / sub)
    is_flash = np.concatenate([[True], np.diff(t) > FT.GAP_MS])
    t0 = np.maximum.accumulate(np.where(is_flash, t, -1e18))
    bidx = np.cumsum(is_flash) - 1
    return (t - t0)[~is_flash], bidx[~is_flash], int(is_flash.sum())


def build(rebuild=False):
    sr = pd.read_csv(CACHE / 'subruns.csv')
    trk_rows, ev_rows, report = [], [], []
    for sub in sr['subrun']:
        e10_burst, pm = burst_e10_from_combined(RUN, sub, rebuild=rebuild)
        if e10_burst is None:
            print(f'  {sub}: NO pulse match — skipped', flush=True)
            continue
        dt, bidx, n_flash_dec = decoded_burst_index(RUN, sub)
        if n_flash_dec != len(e10_burst):
            # the two paths disagree on the burst decomposition -> the by-index
            # mapping would silently mis-assign intensities. Refuse it.
            print(f'  {sub}: BURST MISMATCH combined={len(e10_burst)} '
                  f'decoded={n_flash_dec} — skipped', flush=True)
            continue
        ev_rows.append(pd.DataFrame({'subrun': sub, 'dt_ms': dt,
                                     'e10': e10_burst[bidx]}))
        trk_rows.append(pd.DataFrame({'subrun': sub,
                                      'event_id': list(pm['event_e10'].keys()),
                                      'e10': list(pm['event_e10'].values())}))
        ok = np.isfinite(e10_burst)
        report.append(dict(subrun=sub, n_burst=len(e10_burst),
                           matched=int(ok.sum()),
                           frac_high=float((e10_burst[ok] >= E10_SPLIT).mean())
                           if ok.any() else np.nan))
        print(f'  {sub}: {ok.sum()}/{len(e10_burst)} pulses matched, '
              f'{100 * report[-1]["frac_high"]:.0f}% high', flush=True)
    pd.concat(ev_rows, ignore_index=True).to_csv(CACHE / 'e10_events.csv', index=False)
    pd.concat(trk_rows, ignore_index=True).to_csv(CACHE / 'e10_tracks.csv', index=False)
    return pd.DataFrame(report)


# ---------------------------------------------------------------- analysis

def load():
    tr = pd.read_csv(CACHE / 'tracks.csv')
    ev = pd.read_csv(CACHE / 'events.csv')
    e10t = pd.read_csv(CACHE / 'e10_tracks.csv')
    e10e = pd.read_csv(CACHE / 'e10_events.csv')

    tr = tr.merge(e10t, on=['subrun', 'event_id'], how='left')
    # events.csv and e10_events.csv are both in decoded order per sub-run; align on
    # (subrun, row-within-subrun) rather than dt_ms, which is not unique.
    ev = ev.copy()
    ev['_i'] = ev.groupby('subrun').cumcount()
    e10e = e10e.copy()
    e10e['_i'] = e10e.groupby('subrun').cumcount()
    ev = ev.merge(e10e[['subrun', '_i', 'e10']], on=['subrun', '_i'], how='left')
    return tr, ev


def cls_col(df):
    c = pd.Series(index=df.index, dtype=object)
    c[df['e10'] < E10_SPLIT] = 'low'
    c[df['e10'] >= E10_SPLIT] = 'high'
    df = df.copy()
    df['cls'] = c
    return df[df['cls'].notna()]


def band_col(df):
    lab = pd.Series(index=df.index, dtype=object)
    for lo, hi in BANDS:
        lab[(df['dt_ms'] >= lo) & (df['dt_ms'] < hi)] = f'{lo}-{hi} ms'
    df = df.copy()
    df['band'] = lab
    return df[df['band'].notna()]


def twod(tr):
    t = tr[(tr['n_hits'] >= NHIT) & (tr['time_span'] >= TSPAN)
           & (tr['angle_deg'].abs() < ANG)]
    per = t.groupby(['det', 'subrun', 'event_id', 'projection']).size().unstack(fill_value=0)
    has = (per.get('x', 0) > 0) & (per.get('y', 0) > 0)
    g = has[has].reset_index()[['det', 'subrun', 'event_id']]
    meta = t.groupby(['det', 'subrun', 'event_id'])[['dt_ms', 'e10']].first().reset_index()
    return g.merge(meta, on=['det', 'subrun', 'event_id'], how='left')


def _e10_is_stale():
    """True if the pulse-intensity cache is missing sub-runs that build_cache has since
    added — otherwise those triggers would silently carry no intensity."""
    ep = CACHE / 'e10_events.csv'
    if not ep.exists():
        return True
    have = set(pd.read_csv(ep, usecols=['subrun'])['subrun'].unique())
    want = set(pd.read_csv(CACHE / 'subruns.csv', usecols=['subrun'])['subrun'].unique())
    missing = want - have
    if missing:
        print(f'e10 cache missing {len(missing)} sub-run(s): '
              f'{sorted(missing)[:3]}{"..." if len(missing) > 3 else ""} — rematching')
    return bool(missing)


def main():
    rebuild = _args.rebuild
    if rebuild or _e10_is_stale():
        print('matching pulses...')
        rep = build(rebuild=rebuild)
        print(f'\n{len(rep)} sub-runs matched, median high-pulse fraction '
              f'{rep["frac_high"].median():.2f}\n')

    tr, ev = load()
    g2d = twod(tr)
    resists = sorted(ev['resist'].dropna().unique())

    # attach subrun HV to the track table
    meta = ev[['subrun', 'resist', 'drift', 'mip']].drop_duplicates()
    g2d = g2d.merge(meta, on='subrun', how='left')

    g, e = band_col(cls_col(g2d)), band_col(cls_col(ev))
    frac_matched = 100 * ev['e10'].notna().mean()
    print(f'{frac_matched:.1f}% of triggers carry a matched pulse intensity')
    print(f'LOW  ({E10_SPLIT:.0f}e10): {(e["cls"] == "low").sum()} triggers')
    print(f'HIGH ({E10_SPLIT:.0f}e10): {(e["cls"] == "high").sum()} triggers')

    # ---- fig 1: rate vs HV, low vs high
    fig, axs = plt.subplots(2, 4, figsize=(17, 9), sharex=True)
    for i, (cls, _) in enumerate(CLASSES):
        for j, d in enumerate(DETS):
            ax = axs[i, j]
            gd = g[(g['det'] == d) & (g['cls'] == cls)]
            ed = e[e['cls'] == cls]
            for (lo, hi), c in zip(BANDS, BAND_C):
                lab = f'{lo}-{hi} ms'
                num = gd[gd['band'] == lab].groupby('resist').size()
                den = ed[ed['band'] == lab].groupby('resist').size()
                n = num.reindex(den.index, fill_value=0).astype(float)
                r, er = 100 * n / den, 100 * np.sqrt(np.maximum(n, 1)) / den
                ax.errorbar(den.index, r.values, yerr=er.values, color=c, marker='o',
                            ms=5, lw=1.8, capsize=2, label=lab)
            if i == 0:
                ax.set_title(f'Det {d}', fontsize=11)
            if i == 1:
                ax.set_xlabel('resist (mesh) HV [V]')
            ax.grid(alpha=0.25)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        axs[i, 0].set_ylabel(f'{cls.upper()} intensity pulses\n2D-drift track / trigger [%]',
                             fontsize=9.5)
    axs[0, 0].legend(fontsize=8, title='time since flash', title_fontsize=8)
    fig.suptitle(f'run_67 — 2D-drift track rate vs resist HV, split by BEAM PULSE INTENSITY\n'
                 f'LOW < {E10_SPLIT:.0f}e10 (~410) vs HIGH ≥ {E10_SPLIT:.0f}e10 (~850)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / 'intensity_rate_vs_hv.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', FIG / 'intensity_rate_vs_hv.png')

    # ---- fig 2: heatmaps per class
    edges = np.array([1, 4, 8, 14, 20, 30, 45, 81], dtype=float)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    gc, ec = cls_col(g2d), cls_col(ev)
    fig, axs = plt.subplots(2, 4, figsize=(18, 8.4))
    for i, (cls, _) in enumerate(CLASSES):
        for j, d in enumerate(DETS):
            gd = gc[(gc['det'] == d) & (gc['cls'] == cls)]
            ed = ec[ec['cls'] == cls]
            M = np.full((len(resists), len(edges) - 1), np.nan)
            for k, rv in enumerate(resists):
                num, _ = np.histogram(gd[gd['resist'] == rv]['dt_ms'], bins=edges)
                den, _ = np.histogram(ed[ed['resist'] == rv]['dt_ms'], bins=edges)
                M[k] = np.where(den > 0, 100 * num / den, np.nan)
            im = axs[i, j].pcolormesh(edges, np.arange(len(resists) + 1), M,
                                      cmap='magma', shading='flat')
            axs[i, j].set_yticks(np.arange(len(resists)) + 0.5)
            axs[i, j].set_yticklabels(resists)
            for k in range(M.shape[1]):
                col = M[:, k]
                if np.isfinite(col).any() and np.nanmax(col) > 0:
                    axs[i, j].plot(ctr[k], np.nanargmax(col) + 0.5, marker='*', ms=11,
                                   color='#00e5ff', mec='k', mew=0.5)
            if i == 0:
                axs[i, j].set_title(f'Det {d}', fontsize=11)
            if i == 1:
                axs[i, j].set_xlabel('time since flash [ms]')
            fig.colorbar(im, ax=axs[i, j])
        axs[i, 0].set_ylabel(f'{cls.upper()} intensity\nresist HV [V]', fontsize=9.5)
    fig.suptitle('run_67 — resist HV × time since flash, per beam-pulse-intensity class '
                 '(★ = best HV in that time slice)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / 'intensity_heatmap.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', FIG / 'intensity_heatmap.png')

    # ---- fig 3: HIGH/LOW ratio
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), sharex=True)
    for j, d in enumerate(DETS):
        ax = axs[j]
        for (lo, hi), c in zip(BANDS, BAND_C):
            lab = f'{lo}-{hi} ms'
            rr, ee = [], []
            for rv in resists:
                vals = {}
                for cls, _ in CLASSES:
                    n = int(((g['det'] == d) & (g['cls'] == cls) & (g['band'] == lab)
                             & (g['resist'] == rv)).sum())
                    dd = int(((e['cls'] == cls) & (e['band'] == lab)
                              & (e['resist'] == rv)).sum())
                    vals[cls] = (n, dd)
                (nh, dh), (nl, dl) = vals['high'], vals['low']
                if dh == 0 or dl == 0 or nl == 0:
                    rr.append(np.nan)
                    ee.append(np.nan)
                    continue
                rh, rl = nh / dh, nl / dl
                rr.append(rh / rl)
                ee.append((rh / rl) * np.sqrt(1 / max(nh, 1) + 1 / max(nl, 1)))
            ax.errorbar(resists, rr, yerr=ee, color=c, marker='o', ms=5, lw=1.6,
                        capsize=2, label=lab)
        ax.axhline(1.0, color='0.5', ls='--', lw=1)
        ax.set_title(f'Det {d}', fontsize=11)
        ax.set_xlabel('resist (mesh) HV [V]')
        ax.set_yscale('log')
        ax.grid(alpha=0.25, which='both')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel('track rate  HIGH / LOW intensity')
    axs[0].legend(fontsize=8, title='time since flash', title_fontsize=8)
    fig.suptitle('run_67 — does beam intensity change the track rate per trigger?\n'
                 'ratio of HIGH-pulse to LOW-pulse rate; dashed = no dependence',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / 'intensity_ratio.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', FIG / 'intensity_ratio.png')

    # ---- table
    rows = []
    for d in DETS:
        for lo, hi in BANDS:
            lab = f'{lo}-{hi} ms'
            for cls, _ in CLASSES:
                for rv in resists:
                    n = int(((g['det'] == d) & (g['cls'] == cls) & (g['band'] == lab)
                             & (g['resist'] == rv)).sum())
                    dd = int(((e['cls'] == cls) & (e['band'] == lab)
                              & (e['resist'] == rv)).sum())
                    if dd == 0:
                        continue
                    rows.append(dict(det=d, band=lab, cls=cls, resist=rv, n_2d=n,
                                     n_trig=dd, rate_pct=100 * n / dd,
                                     err_pct=100 * np.sqrt(max(n, 1)) / dd))
    df = pd.DataFrame(rows)
    df.to_csv(FIG / 'intensity_split.csv', index=False)

    print('\nbest resist HV per detector × band × intensity class:')
    best = (df.sort_values('rate_pct', ascending=False)
              .groupby(['det', 'band', 'cls']).head(1)
              .sort_values(['det', 'band', 'cls']))
    print(best.to_string(index=False,
                         columns=['det', 'band', 'cls', 'resist', 'rate_pct', 'err_pct',
                                  'n_2d', 'n_trig'],
                         float_format=lambda x: f'{x:.3f}'))
    print('\nfull table ->', FIG / 'intensity_split.csv')


if __name__ == '__main__':
    main()
