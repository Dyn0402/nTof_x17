#!/usr/bin/env python3
"""Run the DREAM<->n_TOF regression against a REPROCESSED n_TOF run.

This is the acceptance test that matters most, and the one the handoff calls
the definition of done: a correctly processed file must reach the numbers we
currently only get with the laptop-side tflash repair, WITH THAT REPAIR OFF.

Baselines to beat (official file + tflash_repair v2, run 224572 / run_79
stat090_0000):
    match_window          99.9 %
    wall AND plastic      93.7 % efficient / 0.5 % false  (89.9 / 1.3 at 1-3 ms)

Everything is sandboxed the way validate_reprocessing.py does it -- the reader
is pointed at the candidate files and the bunch-index / tflash caches go to a
throwaway directory -- so the official file's caches are never touched. That
matters more than usual here, because those caches are keyed by RUN NUMBER
only: a reprocessed run224572 read through the normal paths would silently
reuse the official file's index.

Usage:
    python dream_regression.py <parts-dir-or-file> [run_79] [stat090_0000] [nb]

Note on the time base: this uses the file's own stored `tflash`
(repair_tflash=False), which is what tests the reprocessing. For production
analysis prefer the PKUP-referenced calibration in
`ntof_processing/flash_timing/` -- t_flash = tof_PKUP + C, good to ~1 ns,
against the few-ns spread of any flash-finder output.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def candidate_files(arg: str, run: int):
    p = Path(arg).resolve()
    if p.is_dir():
        files = sorted(p.glob(f'run{run}_[0-9]*.root'),
                       key=lambda q: int(q.stem.split('_')[-1]))
        if not files:
            raise SystemExit(f'no run{run}_NNNN.root partials in {p}')
        return files
    return [p]


def main() -> int:
    arg = sys.argv[1]
    dream_run = sys.argv[2] if len(sys.argv) > 2 else 'run_79'
    sub = sys.argv[3] if len(sys.argv) > 3 else 'stat090_0000'
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    ntof_run = 224572

    import ntof_dream_merge.ntof_io as ntof_io
    import ntof_dream_merge.tflash_repair as rep

    # ORDER MATTERS. The DREAM<->bunch join runs off the PKUP tree and the
    # index tree, i.e. the beam record for the WHOLE run, and it is unaffected
    # by the PSA settings. A candidate may be only a few partials, so build the
    # join FIRST, against the official staged file, and only then point the
    # reader at the candidate for the hits.
    from ntof_dream_merge.bunch_join import dream_event_to_bunch
    ev = dream_event_to_bunch(dream_run, sub, ntof_run)

    files = candidate_files(arg, ntof_run)
    tmp = Path(tempfile.mkdtemp(prefix=f'dreamreg_{ntof_run}_'))
    ntof_io.ntof_paths = lambda r: files          # type: ignore
    ntof_io.ntof_path = lambda r: files[0]        # type: ignore
    rep.CACHE_DIR = ntof_io.CACHE_DIR = tmp
    ntof_io._TFLASH_FIX_CACHE.clear()
    print(f'candidate: {len(files)} file(s), first = {files[0].name}')
    print(f'caches sandboxed in {tmp}\n')
    have = set()
    for t in ('WALA', 'WALB', 'WALC', 'WALD'):
        e = ntof_io.bunch_edges(ntof_run, t)
        have |= set(np.flatnonzero(np.diff(e) > 0) + 1)
    all_b = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())
    bunches = np.array([b for b in all_b if b in have])[:nb]
    if bunches.size == 0:
        raise SystemExit('the candidate covers none of the bunches this DREAM '
                         'sub-run maps onto -- fetch the partials that do')
    print(f'{len(bunches)} of the first {nb} DREAM bunches are in the candidate '
          f'({bunches.min()}-{bunches.max()})\n')

    import ntof_dream_merge.match_window as mw
    # match_window calls read_bunches with the default repair_tflash=True. The
    # whole point here is to test the FILE, so force the repair off; if the
    # reprocessing is right, the stored tflash needs no help.
    import functools
    mw.read_bunches = functools.partial(ntof_io.read_bunches,
                                        repair_tflash=False)  # type: ignore
    K, T0 = 1.089e-4, -197.5          # as in match_window.__main__
    sel = ev[(ev['BunchNumber'].isin(bunches)) & (~ev['is_flash'])]
    eids, ets, best, dens = mw.nearest_residuals(ntof_run, sel, bunches, K, T0)

    print(f'[1] match_window  ({len(eids):,} events, {len(bunches)} bunches)')
    print(f'    accept bands {mw.BANDS} -> {mw.band_width():.0f} ns total')
    print('      t bin (ms)      n   n_TOF rate   efficiency   P(false match)')
    for r in mw.calibrate(ets, best, dens):
        print(f'      {r["t_lo"]:4d}-{r["t_hi"]:<4d} {r["n"]:7d} '
              f'{r["dens_per_us"]:8.2f}/us {r["eff"]:11.1%} {r["p_acc"]:14.1%}')
    hit = mw.in_bands(best).any(axis=1)
    eff = float(hit.mean())
    print(f'    OVERALL {eff:.1%} matched      '
          f'(baseline with the laptop repair ON: 99.9 %)')

    # Which tree family carried the match -- the plastic leg is the one that was
    # broken, so it is worth seeing separately.
    per = mw.in_bands(best)
    print('\n    matched by tree:')
    for ti, tree in enumerate(mw.TREES):
        print(f'      {tree}: {per[:, ti].mean():6.1%}')
    wal = per[:, [i for i, t in enumerate(mw.TREES) if t.startswith('WAL')]].any(axis=1)
    pss = per[:, [i for i, t in enumerate(mw.TREES) if t.startswith('PSS')]].any(axis=1)
    print(f'      any WAL: {wal.mean():6.1%}   any PSS: {pss.mean():6.1%}   '
          f'WAL and PSS: {(wal & pss).mean():6.1%}')
    print(f'      plastic partner | wall-matched: '
          f'{(wal & pss).sum() / max(wal.sum(), 1):6.1%}   '
          f'(official file 48.9 %, repaired 99.7 %)')

    # ---- [2] the purity test -------------------------------------------------
    # match_window measures EFFICIENCY only; at 1-3 ms the accidental
    # probability is ~100 % with the wide bands, so "matched" there says
    # nothing on its own. The thresholded wall AND plastic singles matcher is
    # the test that has any purity, and it is the number to compare.
    import ntof_dream_merge.dream_trigger as dt
    dt.read_bunches = functools.partial(ntof_io.read_bunches,
                                        repair_tflash=False)  # type: ignore

    thr, adc = dt.load_thresholds(dream_run, sub), dt.load_adc_mv()
    CB, CT = [], []
    offs = {}
    for arm in dt.ARMS:
        off = dt.measure_tb_offsets(ntof_run, bunches, arm)
        offs[arm] = off
        cb_, ct_ = dt.singles_candidates(ntof_run, bunches, arm, thr, adc,
                                         tb_off=off, require_plastic=True)
        CB.append(cb_)
        CT.append(ct_)
    cb = np.concatenate(CB)
    ct = np.concatenate(CT)
    o = np.lexsort((ct, cb))
    cb, ct = cb[o], ct[o]

    SHIFT = 100_000.0

    def match(shift):
        got = np.zeros(len(sel), bool)
        for b, g in sel.groupby('BunchNumber'):
            s, e = np.searchsorted(cb, [b, b + 1])
            tt = ct[s:e]
            if tt.size == 0:
                continue
            et = g['t_since_flash_ns'].to_numpy().astype(float)
            pred = et + K * et + T0 + shift
            lo = np.searchsorted(tt, pred - 1000)
            hi = np.searchsorted(tt, pred + 1000)
            for j, i in enumerate(g.index):
                if hi[j] <= lo[j]:
                    continue
                r = tt[lo[j]:hi[j]] - pred[j]
                for blo, bhi in mw.BANDS:
                    if ((r >= blo) & (r <= bhi)).any():
                        got[sel.index.get_loc(i)] = True
                        break
        return got

    sel = sel.reset_index(drop=True)
    got, ctl = match(0.0), match(SHIFT)
    ets2 = sel['t_since_flash_ns'].to_numpy().astype(float)
    print(f'\n[2] thresholded wall AND plastic SINGLES matcher')
    print(f'    {ct.size:,} candidates ({ct.size / len(bunches):.0f}/bunch)')
    print('      t bin (ms)      n    efficiency   control(false)')
    for lo, hi in ((1, 3), (3, 10), (10, 20), (20, 40), (40, 80)):
        m = (ets2 >= lo * 1e6) & (ets2 < hi * 1e6)
        if m.sum():
            print(f'      {lo:4d}-{hi:<4d} {m.sum():7d}   {got[m].mean():9.1%}   '
                  f'{ctl[m].mean():9.1%}')
    print(f'    OVERALL {got.mean():.1%} efficient / {ctl.mean():.1%} false   '
          f'(baseline with the laptop repair ON: 93.7 % / 0.5 %)')
    print(f'    measured time-base offsets (should be ~0 on a good file): '
          f'{ {a: round(float(np.median(list(v.values()))), 1) for a, v in offs.items()} }')

    print(f'\nverdict: match_window {"PASS" if eff >= 0.99 else "BELOW BASELINE"}, '
          f'singles {"PASS" if got.mean() >= 0.90 else "BELOW BASELINE"} '
          f'-- with repair_tflash OFF, i.e. on the file\'s own stored tflash')
    return 0


if __name__ == '__main__':
    sys.exit(main())
