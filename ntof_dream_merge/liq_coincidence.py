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

Traps honoured (FINDINGS_2026-07-29_pre_ship_tests.md):
  * amp > ~31 000 is the ADC wrap, not a big pulse -> dropped (counted);
  * `satuflag` is not consulted (it is not usable);
  * offsets between LIQ and WAL time bases are REPORTED, not assumed zero.

Usage:
    python liq_coincidence.py <parts-dir-or-file> <match.npz> [--win 2000]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

ARMS = ('A', 'B', 'C', 'D')
WRAP_AMP = 31_000.0
SHIFT_NS = 100_000.0


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
    tmp = Path(tempfile.mkdtemp(prefix=f'liqc_{args.ntof_run}_'))
    ntof_io.ntof_paths = lambda r: files          # type: ignore
    ntof_io.ntof_path = lambda r: files[0]        # type: ignore
    rep.CACHE_DIR = ntof_io.CACHE_DIR = tmp
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
                                 branches=('BunchNumber', 'amp'),
                                 repair_tflash=False)
        lt, lb, la = t['t_since_flash_ns'], t['BunchNumber'], t['amp']
        wrap = la > WRAP_AMP
        keep = ~wrap & (la >= args.amp_min)
        lt, lb = lt[keep], lb[keep]
        o = np.lexsort((lt, lb))
        lt, lb = lt[o], lb[o]
        print(f'LIQ{liq}: {keep.sum():,} hits in {bunches.size} bunches '
              f'({int(wrap.sum())} wrapped dropped)')
        for a_i, a in enumerate(ARMS):
            for lab, shift in (('sig', 0.0), ('ctl', SHIFT_NS)):
                m = arm == a_i
                res = []
                for b in np.unique(bunch[m]):
                    s, e = np.searchsorted(lb, [b, b + 1])
                    tt = lt[s:e]
                    if tt.size == 0:
                        continue
                    for t0 in tw[m][bunch[m] == b] + shift:
                        rr = tt[np.searchsorted(tt, t0 - args.win):
                                np.searchsorted(tt, t0 + args.win)] - t0
                        res.append(rr)
                res = (np.concatenate(res) if res else np.array([]))
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
