#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_reprocessing.py -- grade a (re)processed n_TOF hit ROOT file.

This is the acceptance test for the UserInput iteration loop described in
archive/HANDOFF_2026-07-28_ntof_processing.md. Point it at a candidate runXXXXXX.root
(produced by RunProcessing.sh with a new UserInput) and it prints the three
detector-timing health checks that the 2026-07-27/28 investigation showed are
what actually matter, each with a PASS/FAIL against the acceptance target:

  1. gamma-flash identification per tree
       fraction of bunches whose stored tflash deviates >150 ns from the
       tree's per-run mode. Broken official processing: PSS 37-85 %.
       TARGET: < 2 % on every tree.
  2. cross-detector flash consistency (per arm, vs the same arm's wall)
       the prompt-coincidence peak position of LARGE (amp>1000) PSS hits and
       of LIQ hits relative to wall hits, after removing each tree's modal
       tflash. Broken official processing: -375/+25/-325/-325 ns (A/B/C/D)
       because WALA/C/D time the flash on a different waveform feature than
       WALB/PSS/LIQ. TARGET: |peak| < 25 ns everywhere.
  3. prompt-coincidence capture
       fraction of wall hits (late, t>20 ms) with a large plastic hit within
       +-40 ns, and the median amplitude of those partners. This is what makes
       the wall AND plastic trigger emulation work. No hard target -- compare
       against the same number from the previous processing and from the
       hardware scaler ratios (sector/plastic ~ 15-21 %).

It also prints per-tree hit counts on the sampled bunches so a UserInput change
that silently drops real hits is caught (counts should not fall by more than
the artifact populations you intend to remove).

USAGE
    .venv/bin/python ntof_processing/validate_reprocessing.py <run> [path.root]

If a path is given it is validated IN PLACE (bunch-index and tflash caches go to
a directory private to that variant, so the official file's caches are never
clobbered -- see ntof_io.variant_cache). Without a path the staged file from
beam_july_paths is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def main() -> int:
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 224572
    # accepts a merged run<run>.root, a directory of per-job partials, or a
    # comma-separated list of partials
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if arg is None:
        path = None
    elif ',' in arg:
        path = [Path(p).resolve() for p in arg.split(',')]
    elif Path(arg).is_dir():
        path = sorted(Path(arg).resolve().glob(f'run{run}_[0-9]*.root'),
                      key=lambda p: int(p.stem.split('_')[-1]))
        if not path:
            print(f'no run{run}_NNNN.root partials in {arg}')
            return 1
    else:
        path = [Path(arg).resolve()]

    import ntof_dream_merge.ntof_io as ntof_io
    import ntof_dream_merge.tflash_repair as rep

    if path is not None:
        # sandbox: point the reader at the candidate file(s), and give it a cache
        # of its own -- persistent and fingerprinted on the file set, so the
        # bunch index survives between runs without ever being shared with
        # another processing (see ntof_io.variant_cache).
        ntof_io.ntof_paths = lambda r: path           # type: ignore
        ntof_io.ntof_path = lambda r: path[0]         # type: ignore
        cache = ntof_io.variant_cache(Path(arg).resolve(), path)
        rep.CACHE_DIR = ntof_io.CACHE_DIR = cache
        print(f'validating {len(path)} file(s), first = {path[0]}  '
              f'(caches in {cache})')
    else:
        print(f'validating staged file for run{run}')

    # ---- check 1: flash identification --------------------------------------
    c = rep.corrected_tflash(run)
    print('\n[1] gamma-flash identification (stored tflash vs per-run mode)')
    ok1 = True
    for t in rep.ALL_TREES:
        bad = c['_err_frac'][t]
        flag = 'PASS' if bad < 0.02 else 'FAIL'
        ok1 &= bad < 0.02
        print(f'    {t}: mode {c["_modes"][t]:9.1f} ns   bad bunches {bad:6.1%}   {flag}')

    # ---- check 2: cross-detector flash consistency --------------------------
    print('\n[2] flash-feature consistency vs same-arm wall (coincidence peak)')
    ok2 = True
    for t, off in c['_offsets'].items():
        if t.startswith('WAL') or t == 'PKUP':
            continue
        flag = 'PASS' if abs(off) < 25 else 'FAIL'
        ok2 &= abs(off) < 25
        print(f'    {t}: {off:+8.1f} ns   {flag}')

    # ---- check 3: prompt-coincidence capture + hit counts -------------------
    print('\n[3] prompt coincidences (late hits, big plastic amp>1000, +-40 ns)')
    edges = ntof_io.bunch_edges(run, 'WALA')
    nb = np.diff(edges)
    good = np.flatnonzero(nb > 0) + 1
    bl = good[len(good) // 3:len(good) // 3 + 60]
    for arm in 'ABCD':
        w = ntof_io.read_bunches(run, f'WAL{arm}', bl, branches=('BunchNumber',))
        p = ntof_io.read_bunches(run, f'PSS{arm}', bl,
                                 branches=('BunchNumber', 'amp'))
        lw = w['t_since_flash_ns'] > 20e6
        lp = (p['t_since_flash_ns'] > 20e6) & (p['amp'] > 1000)
        n_pair, amps, n_w = 0, [], 0
        for b in np.unique(w['BunchNumber']):
            tw = np.sort(w['t_since_flash_ns'][lw & (w['BunchNumber'] == b)])
            mh = lp & (p['BunchNumber'] == b)
            tp, pa = p['t_since_flash_ns'][mh], p['amp'][mh]
            n_w += tw.size
            if tw.size == 0 or tp.size == 0:
                continue
            j = np.searchsorted(tw, tp)
            j0 = np.clip(j - 1, 0, tw.size - 1)
            j1 = np.clip(j, 0, tw.size - 1)
            d = np.minimum(np.abs(tp - tw[j0]), np.abs(tp - tw[j1]))
            n_pair += int((d < 40).sum())
            amps.append(pa[d < 40])
        a = np.concatenate(amps) if amps else np.array([np.nan])
        print(f'    arm {arm}: {n_pair:5d} pairs / {n_w:6d} late wall hits '
              f'({n_pair / max(n_w, 1):6.1%})   partner amp p50 {np.median(a):7.0f}')

    print('\n    per-tree hits on sampled bunches (watch for silent losses):')
    for t in rep.ALL_TREES:
        e = ntof_io.bunch_edges(run, t)
        n = int(e[bl[-1]] - e[bl[0] - 1])
        print(f'    {t}: {n:>10,}')

    print(f'\nverdict: flash-id {"PASS" if ok1 else "FAIL"}, '
          f'consistency {"PASS" if ok2 else "FAIL"}')
    return 0 if (ok1 and ok2) else 1


if __name__ == '__main__':
    sys.exit(main())
