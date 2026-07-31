#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
liq_coincidence.py -- are there liquid-scintillator hits coincident with
DREAM<->n_TOF matched events?

Consumes the per-event npz written by mm_activity_crosscheck.py --out (events
matched to exactly one arm, with the matched n_TOF wall time), reads the LIQ*
trees of a candidate processing, and histograms t_LIQ - t_wall per (matched
arm, liquid). A real coincidence shows as a narrow excess over the accidental
floor; the floor is measured on the same events with the wall time shifted.

Traps honoured (FINDINGS_2026-07-29_signed_decoding.md, which supersedes the
pre-ship findings this file first followed):
  * saturated hits are dropped as `satuflag` OR amp > ceiling, via
    ntof_io.saturated(). The samples are signed int16, so the LIQ/PSS ceiling is
    ~63 800, not the ~31 000 baseline. Both tests are needed on the liquids:
    satuflag leaves 8.9-15 % of over-ceiling hits unflagged, and the amp cut
    misses the ~4 000 hits per tree that are flagged with an extrapolated amp
    back inside the range;
  * there is no ADC wrap. The old amp > 31 000 cut sat mid-range and threw away
    ordinary half-scale pulses (74 % of what it cut on LIQA);
  * offsets between LIQ and WAL time bases are REPORTED, not assumed zero.

Usage:
    python liq_coincidence.py <parts-dir-or-file> <match.npz> [--win 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

ARMS = ('A', 'B', 'C', 'D')
SHIFT_NS = 100_000.0
# The saturation rule lives in ntof_io.saturated() so walls, plastics and liquids
# all get their own ceiling from one place.

# Bunch/time keys are packed into one sorted float64 so the whole per-event
# window search is two searchsorted calls instead of a Python loop over bunches
# and events. |t_since_flash| stays under ~2e7 ns (a 20 ms window) and the
# +100 us control shift is small, so 1e9 keeps every bunch's keys disjoint with
# room to spare; bunch <= 3018 puts the largest key at ~3e12, well inside
# float64's exact-integer range.
KEY_SCALE = 1e9


def window_residuals(hit_bunch, hit_t, ev_bunch, ev_t, win):
    """All (hit_t - ev_t) for hits in the same bunch within +-win of an event.

    Vectorised replacement for the per-bunch/per-event double loop. `hit_*` must
    be sorted by (bunch, t) -- the caller's lexsort guarantees it.
    """
    if hit_t.size == 0 or ev_t.size == 0:
        return np.array([])
    # Guard the packing rather than trust it: if a time ever exceeded half the
    # scale, one bunch's keys would run into the next bunch's and hits would be
    # matched across bunches with no error raised.
    reach = max(np.abs(hit_t).max(), np.abs(ev_t).max()) + win
    if reach >= 0.5 * KEY_SCALE:
        raise ValueError(f'time base reaches {reach:.3g} ns, too large for '
                         f'KEY_SCALE={KEY_SCALE:.3g}: raise KEY_SCALE')
    key_hits = hit_bunch.astype(np.float64) * KEY_SCALE + hit_t
    lo = np.searchsorted(key_hits,
                         ev_bunch.astype(np.float64) * KEY_SCALE + (ev_t - win),
                         side='left')
    hi = np.searchsorted(key_hits,
                         ev_bunch.astype(np.float64) * KEY_SCALE + (ev_t + win),
                         side='right')
    n = hi - lo
    total = int(n.sum())
    if total == 0:
        return np.array([])
    # ragged gather: expand each [lo_i, hi_i) into flat indices in one shot
    starts = np.repeat(lo, n)
    within = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    return hit_t[starts + within] - np.repeat(ev_t, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('target')
    ap.add_argument('npz')
    ap.add_argument('--ntof-run', type=int, default=224572)
    ap.add_argument('--win', type=float, default=2000.0,
                    help='half-width of the search window [ns]')
    ap.add_argument('--coinc', type=float, default=100.0,
                    help='half-width of the coincidence window [ns]')
    ap.add_argument('--amp-min', type=float, default=0.0)
    ap.add_argument('--out-hist', default=None,
                    help='npz with the residual histograms')
    args = ap.parse_args()

    import ntof_dream_merge.ntof_io as ntof_io
    import ntof_dream_merge.tflash_repair as rep

    p = Path(args.target).resolve()
    files = (sorted(p.glob(f'run{args.ntof_run}_[0-9]*.root'),
                    key=lambda q: int(q.stem.split('_')[-1]))
             if p.is_dir() else [p])
    ntof_io.ntof_paths = lambda r: files          # type: ignore
    ntof_io.ntof_path = lambda r: files[0]        # type: ignore
    # Persistent, per-variant and fingerprinted: same isolation as the mkdtemp
    # this replaces, but the bunch index survives between runs (~7 s/tree to
    # rebuild over 16 partials, so ~30 s for the four LIQ trees).
    rep.CACHE_DIR = ntof_io.CACHE_DIR = ntof_io.variant_cache(p, files)
    ntof_io._TFLASH_FIX_CACHE.clear()

    d = np.load(args.npz)
    ok = (d['arm'] >= 0) & np.isfinite(d['t_ntof_ns'])
    bunch, tw, arm = d['bunch'][ok], d['t_ntof_ns'][ok], d['arm'][ok]
    print(f'{ok.sum():,} exclusively-matched DREAM events, '
          f'bunches {bunch.min()}-{bunch.max()}')

    bunches = np.unique(bunch)
    hists = {}
    edges = np.arange(-args.win, args.win + 10.0, 10.0)
    for liq in ARMS:
        t = ntof_io.read_bunches(args.ntof_run, f'LIQ{liq}', bunches,
                                 branches=('BunchNumber', 'amp', 'satuflag'),
                                 repair_tflash=False)
        lt, lb, la = t['t_since_flash_ns'], t['BunchNumber'], t['amp']
        sf = t['satuflag'].astype(bool)
        over = la > ntof_io.saturation_ceiling(f'LIQ{liq}')
        sat = ntof_io.saturated(f'LIQ{liq}', la, sf)
        keep = ~sat & (la >= args.amp_min)
        lt, lb = lt[keep], lb[keep]
        o = np.lexsort((lt, lb))
        lt, lb = lt[o], lb[o]
        print(f'LIQ{liq}: {keep.sum():,} hits in {bunches.size} bunches '
              f'({int(sat.sum())} saturated dropped: {int(sf.sum())} flagged, '
              f'{int((over & ~sf).sum())} over ceiling but unflagged)')
        for a_i, a in enumerate(ARMS):
            m = arm == a_i
            eb, et = bunch[m], tw[m]
            for lab, shift in (('sig', 0.0), ('ctl', SHIFT_NS)):
                res = window_residuals(lb, lt, eb, et + shift, args.win)
                hists[f'{lab}_{a}_LIQ{liq}'] = np.histogram(res, bins=edges)[0]

    print(f'\ncoincident LIQ hits per matched event, +-{args.coinc:.0f} ns '
          f'around the residual PEAK (sig) vs flat floor (ctl):')
    print('  matched arm     ' + '     '.join(f'LIQ{q}' for q in ARMS))
    c = (edges[:-1] + edges[1:]) / 2
    for a_i, a in enumerate(ARMS):
        n_ev = int((arm == a_i).sum())
        if n_ev == 0:
            continue
        cells = []
        for liq in ARMS:
            hs = hists[f'sig_{a}_LIQ{liq}']
            hc = hists[f'ctl_{a}_LIQ{liq}']
            pk = c[hs.argmax()] if hs.sum() else 0.0
            w = np.abs(c - pk) <= args.coinc
            sig, ctl = hs[w].sum() / n_ev, hc[w].sum() / n_ev
            cells.append(f'{sig:.3f}/{ctl:.3f}@{pk:+5.0f}')
        print(f'    {a} (n={n_ev:5d})  ' + '  '.join(cells))
    print('\n  cell = sig/ctl per event @ peak-residual ns; '
          'sig >> ctl at a stable offset = real coincidence')

    if args.out_hist:
        np.savez(args.out_hist, edges=edges, **hists)
        print(f'histograms -> {args.out_hist}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
