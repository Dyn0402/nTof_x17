#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared statistics core for the run_67 analysis (plastic-threshold x HV scan).

The unit of the run_67 analysis is a cell
    (mip, drift, resist, det, window)
where `window` is a HAND-DEFINED time-since-flash bin (run_67 has no comb — see
scan_lib). Every plotting script builds on `per_cell_stats(ev, windows)`.

Efficiency metric (same as run_64): P(3D x/y pair) per recorded trigger is the
trustworthy one — the X/Y coincidence kills the residual common-mode fake
'tracks' that inflate the single-plane track-segment yield on the noisy B/C/D M1
cards. Det A (clean M1) is the reference. Both are reported.

Denominator = events this detector was READ OUT for (feu_presence.readout_*),
NOT its live_* (produced-hits) flag: a detector that is blind from the flash
produced no hits, and that blindness is the very inefficiency being measured, so
it must stay in the denominator. `blind_frac` reports 1 - live as an OBSERVABLE.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scan_lib as L  # noqa: E402
import feu_presence as FP  # noqa: E402

DET_COL = {'A': 'crimson', 'B': 'royalblue', 'C': 'seagreen', 'D': 'darkorange'}


def binom_err(k, n):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / np.maximum(n, 1))


def load(mesh='On'):
    """Flash-ok events with FEU-presence flags attached, plus the seg table."""
    ev, segs, _ = L.load_all(mesh=mesh)
    if ev.empty:
        sys.exit('no cached events yet — run process.py first')
    ev = ev[ev.flash_ok].copy()
    # GUARD: the presence table covers only the sub-runs cached when it was
    # built. Processing runs concurrently, so a sub-run cached since then would
    # left-join to NaN -> readout_*=False -> silently dropped from EVERY
    # efficiency denominator while still looking like a loaded sub-run. Drop
    # such sub-runs explicitly and say so, rather than letting them rot the
    # denominators. Fix by rebuilding: feu_presence.py --force.
    have = set(pd.read_parquet(FP.OUT_PATH, columns=['subrun'])['subrun'].unique())
    missing = sorted(set(ev.subrun.unique()) - have)
    if missing:
        print(f'  WARNING: {len(missing)} cached sub-run(s) absent from the FEU '
              f'presence table — EXCLUDED. Rebuild with feu_presence.py --force '
              f'to include them:')
        for s in missing:
            print(f'    - {s}')
        ev = ev[ev.subrun.isin(have)].copy()
        if not segs.empty:
            segs = segs[segs.subrun.isin(have)].copy()
    ev = FP.attach(ev)
    if not segs.empty:
        segs = segs[segs.flash_ok].copy()
    print(f'loaded {len(ev)} flash-ok events from {ev.subrun.nunique()} sub-runs '
          f'({ev.mip.nunique()} thr x {ev.drift.nunique()} drift x '
          f'{ev.resist.nunique()} resist pts)')
    print('  FEU readout fraction (denominator cut): '
          + ', '.join(f'{d}={ev[f"readout_{d}"].mean():.3f}' for d in 'ABCD'))
    print('  hit-producing ("live") fraction, an OBSERVABLE not a cut: '
          + ', '.join(f'{d}={ev[f"live_{d}"].mean():.3f}' for d in 'ABCD'))
    return ev, segs


def per_cell_stats(ev, windows):
    """Per (mip, drift, resist, det, window): yields, liveness, gain.

    `windows` = [(lo, hi), ...] in ms; a copy of `ev` is binned into them.
    """
    ev = L.add_window(ev, windows)
    win_order = [L.win_label(lo, hi) for lo, hi in windows]
    rows = []
    for (mip, dr, r), grp in ev.groupby(['mip', 'drift', 'resist']):
        for Ld in 'ABCD':
            live = grp[grp[f'readout_{Ld}']]
            for wlo, whi in windows:
                wl = L.win_label(wlo, whi)
                p = live[live.window == wl]
                n = len(p)
                k_seg = int((p[f'n_trkseg_{Ld}'] > 0).sum())
                k_pair = int((p[f'n_pair_{Ld}'] > 0).sum())
                p_seg, e_seg = binom_err(k_seg, n)
                p_pair, e_pair = binom_err(k_pair, n)
                rows.append({
                    'mip': mip, 'drift': dr, 'resist': r,
                    'resist_eff': L.resist_for_det(r, Ld),
                    'det': Ld, 'window': wl, 'win_lo': wlo, 'win_hi': whi,
                    'n': n, 'k_pair': k_pair, 'k_seg': k_seg,
                    'p_trk': float(p_seg), 'e_trk': float(e_seg),
                    'p_pair': float(p_pair), 'e_pair': float(e_pair),
                    'busy_frac': (((p[f'n_clean_strips_{Ld}'] > L.BUSY_CLEAN_STRIPS)
                                   | p['reco_skipped']).mean() if n else np.nan),
                    'blind_frac': (1.0 - p[f'live_{Ld}'].mean()) if n else np.nan,
                    'nhits_med': p[f'n_hits_{Ld}'].median() if n else np.nan,
                    'q_med': p[f'seg_q_{Ld}'].median() if n else np.nan,
                })
    st = pd.DataFrame(rows)
    st['window'] = pd.Categorical(st['window'], categories=win_order, ordered=True)
    return st


def balanced_grid(st, verbose=False):
    """Restrict to the complete rectangular drift x resist sub-grid.

    MUST be applied before marginalising over an HV axis. run_67's grid is
    ragged: drifts 500/600/700 carry the full 7-point resist ladder, but the
    drift-400 block was truncated after a SINGLE sub-run at resist 550 (the
    highest gain). Pooling over resist would then compare drift 400's best-case
    cell against the others' full-ladder average, which makes drift 400 look
    like the optimum purely from the company it keeps -- the first pass of this
    analysis reported exactly that spurious "best drift 400 V".

    Keeping only drifts that carry the full resist ladder removes the confound.
    Per-cell views (detA_2d) do NOT need this: a single cell is a legitimate
    measurement there and is annotated with its own n.
    """
    cnt = st.drop_duplicates(['drift', 'resist']).groupby('drift').size()
    full = cnt.max()
    keep = sorted(cnt[cnt == full].index)
    dropped = sorted(set(cnt.index) - set(keep))
    if verbose and dropped:
        print(f'  balanced grid: keeping drifts {keep}; dropping {dropped} '
              f'(incomplete resist ladder -> would bias the drift marginal)')
    return st[st.drift.isin(keep)]


def agg_yield(st, xcol, metric='p_pair', window=None, mip=None, balanced=True):
    """Pool the yield over the OTHER axes (pooled binomial), grouped by det+xcol.

    `balanced=True` (default) first restricts to the complete rectangular
    sub-grid -- required for an unbiased marginal, see balanced_grid().
    Optionally restrict to one window and/or one mip first.
    """
    s = st.copy()
    if balanced:
        s = balanced_grid(s)
    if window is not None:
        s = s[s.window == window]
    if mip is not None:
        s = s[s.mip == mip]
    s['k'] = np.round(s[metric] * s['n']).astype('int64')
    g = s.groupby(['det', xcol], observed=True).agg(
        k=('k', 'sum'), n=('n', 'sum')).reset_index()
    p, e = binom_err(g.k.to_numpy(float), g.n.to_numpy(float))
    g['p'], g['e'] = p, e
    return g


# A threshold block is a 3 drift x 7 resist = 21-cell grid. Figures are only
# drawn for blocks with at least this many cells, so a partially-processed
# threshold cannot masquerade as a measured HV surface.
MIN_CELLS = 15


def grid_cells(st):
    """{mip: number of distinct (drift, resist) cells present}."""
    return (st.drop_duplicates(['mip', 'drift', 'resist'])
            .groupby('mip').size().to_dict())


def complete_mips(st, min_cells=MIN_CELLS, verbose=True):
    """Thresholds whose HV grid is complete enough to plot; others are skipped
    loudly rather than drawn sparse."""
    cells = grid_cells(st)
    ok = [m for m in sorted(cells, key=lambda m: -cells[m]) if cells[m] >= min_cells]
    if verbose:
        for m in sorted(cells):
            tag = 'OK' if cells[m] >= min_cells else 'SKIPPED (incomplete)'
            print(f'  grid completeness: mip {m} -> {cells[m]}/21 cells  {tag}')
    return [m for m in [141, 113, 90] if m in ok]


def flash_burst_counts(ev):
    """Number of confirmed-flash spills (bursts) per (mip, drift, resist).

    The per-spill event rate denominator: one row per (subrun) since each
    (mip, drift, resist) is a single sub-run in run_67's grid.
    """
    lead = ev[ev.is_leader & ev.flash_ok]
    return (lead.groupby(['mip', 'drift', 'resist', 'subrun'])
            .size().rename('n_spill').reset_index())
