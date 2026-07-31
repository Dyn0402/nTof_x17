#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm_activity_crosscheck.py -- do the Micromegas see tracks where n_TOF says
a wall AND plastic singles fired?

This is the first physics use of the DREAM<->n_TOF merge, run as a CROSS-CHECK
of the reprocessing: every non-flash DREAM event was hardware-triggered by the
N1081B sector singles (wall sum .AND. plastic, one of four arms), and each MM
chamber sits in its arm's line of flight. So if the merge and the reprocessed
hit content are right, then

  * a DREAM event matched to a reconstructed arm-a singles should show MM
    activity CONCENTRATED in chamber a (the other three chambers give the
    accidental floor, measured in the same events);
  * events the matcher misses should look like the matched ones (the matcher's
    ~4 % inefficiency should not select against MM content).

Activity here is CANDIDATE-LEVEL ONLY -- strip counts and amplitudes per
plane -- per the repo rule: no position/angle/depth from combined_hits times.

Usage:
    python mm_activity_crosscheck.py <parts-dir-or-file> [run] [subrun] [nb]

Writes a per-event npz next to nothing (scratch): --out <path> to keep it for
the liquid follow-up (eventId, bunch, matched arm, matched n_TOF time).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# *** STALE CONSTANTS (2026-07-30). The accept bands / clock map below were
# fitted on the OFFICIAL processing of run 224572 and do not describe the data
# we analyse. Current: K = 1.103724e-4, T0 = -253.64 ns, per-arm offsets, a
# per-bunch clock term, and a SINGLE +-25 ns band -- from
# ntof_dream_merge.calibration.load(). They are left inline here only so the
# numbers this script already published stay reproducible; anything re-run for
# physics must take the calibration from that module.
# See ntof_dream_merge/DREAM_NTOF_CALIBRATION.md. ***
K, T0 = 1.089e-4, -197.5        # DREAM->n_TOF clock map, as in match_window
ARMS = ('A', 'B', 'C', 'D')


def candidate_files(arg: str, run: int):
    p = Path(arg).resolve()
    if p.is_dir():
        files = sorted(p.glob(f'run{run}_[0-9]*.root'),
                       key=lambda q: int(q.stem.split('_')[-1]))
        if not files:
            raise SystemExit(f'no run{run}_NNNN.root partials in {p}')
        return files
    return [p]


def per_arm_match(sel: pd.DataFrame, cand: dict, bands) -> dict:
    """For each arm: (matched bool, matched n_TOF t_since_flash_ns) per event."""
    out = {}
    for arm, (cb, ct) in cand.items():
        got = np.zeros(len(sel), bool)
        tmatch = np.full(len(sel), np.nan)
        for b, g in sel.groupby('BunchNumber'):
            s, e = np.searchsorted(cb, [b, b + 1])
            tt = ct[s:e]
            if tt.size == 0:
                continue
            et = g['t_since_flash_ns'].to_numpy().astype(float)
            pred = et + K * et + T0
            lo = np.searchsorted(tt, pred - 1000)
            hi = np.searchsorted(tt, pred + 1000)
            for j, i in enumerate(g.index):
                if hi[j] <= lo[j]:
                    continue
                r = tt[lo[j]:hi[j]] - pred[j]
                inb = np.zeros(r.size, bool)
                for blo, bhi in bands:
                    inb |= (r >= blo) & (r <= bhi)
                if inb.any():
                    kk = np.flatnonzero(inb)[np.argmin(np.abs(r[np.flatnonzero(inb)]))]
                    got[i] = True
                    tmatch[i] = tt[lo[j]:hi[j]][kk]
        out[arm] = (got, tmatch)
    return out


def mm_event_table(run: str, subrun: str) -> pd.DataFrame:
    """Per (eventId, chamber): strip counts and max amplitude per plane."""
    from ntof_tracking.reco import io as tio
    cfg = tio.load_run_config(run)
    lut = tio.build_channel_lut(cfg)
    df = tio.load_subrun_hits(
        run, subrun, lut,
        columns=['eventId', 'feu', 'channel', 'amplitude', 'sample', 'time'])
    if df is None:
        raise SystemExit(f'no combined hits for {run}/{subrun}')
    g = (df.groupby(['eventId', 'det', 'plane'])
           .agg(n=('channel', 'nunique'), amax=('amplitude', 'max'))
           .reset_index())
    g['arm'] = g['det'].str[-1]
    wide = g.pivot_table(index='eventId', columns=['arm', 'plane'],
                         values=['n', 'amax'], fill_value=0.0)
    wide.columns = [f'{v}_{a}{p}' for v, a, p in wide.columns]
    return wide.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('target')
    ap.add_argument('run', nargs='?', default='run_79')
    ap.add_argument('subrun', nargs='?', default='stat090_0000')
    ap.add_argument('nb', nargs='?', type=int, default=300)
    ap.add_argument('--ntof-run', type=int, default=224572)
    ap.add_argument('--out', default=None,
                    help='npz of per-event match results, for follow-ups')
    ap.add_argument('--min-strips', type=int, default=2,
                    help='strips per plane for the "cluster" tier')
    args = ap.parse_args()

    import ntof_dream_merge.ntof_io as ntof_io
    import ntof_dream_merge.tflash_repair as rep

    # join FIRST (whole-run beam record), then point the reader at the candidate
    from ntof_dream_merge.bunch_join import dream_event_to_bunch
    ev = dream_event_to_bunch(args.run, args.subrun, args.ntof_run)

    files = candidate_files(args.target, args.ntof_run)
    ntof_io.ntof_paths = lambda r: files          # type: ignore
    ntof_io.ntof_path = lambda r: files[0]        # type: ignore
    # Persistent, per-variant and fingerprinted: same isolation as the mkdtemp
    # this replaces, but the bunch index survives between runs (~7 s/tree to
    # rebuild over 16 partials, so ~30 s for the four LIQ trees).
    rep.CACHE_DIR = ntof_io.CACHE_DIR = ntof_io.variant_cache(p, files)
    ntof_io._TFLASH_FIX_CACHE.clear()
    print(f'candidate: {len(files)} file(s), first = {files[0].name}')

    import functools
    import ntof_dream_merge.match_window as mw
    import ntof_dream_merge.dream_trigger as dt
    dt.read_bunches = functools.partial(ntof_io.read_bunches,
                                        repair_tflash=False)  # type: ignore

    have = set()
    for t in ('WALA', 'WALB', 'WALC', 'WALD'):
        e = ntof_io.bunch_edges(args.ntof_run, t)
        have |= set(np.flatnonzero(np.diff(e) > 0) + 1)
    all_b = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())
    bunches = np.array([b for b in all_b if b in have])[:args.nb]
    print(f'{len(bunches)} bunches ({bunches.min()}-{bunches.max()})')

    thr, adc = dt.load_thresholds(args.run, args.subrun), dt.load_adc_mv()
    cand = {}
    for arm in ARMS:
        offs = dt.measure_tb_offsets(args.ntof_run, bunches, arm)
        cb, ct = dt.singles_candidates(args.ntof_run, bunches, arm, thr, adc,
                                       tb_off=offs, require_plastic=True)
        o = np.lexsort((ct, cb))
        cand[arm] = (cb[o], ct[o])
        print(f'  arm {arm}: {ct.size:7,} wall+plastic singles')

    sel = (ev[(ev['BunchNumber'].isin(bunches)) & (~ev['is_flash'])]
           .reset_index(drop=True))
    arm_match = per_arm_match(sel, cand, mw.BANDS)
    M = np.stack([arm_match[a][0] for a in ARMS], axis=1)
    n_arm = M.sum(axis=1)
    print(f'\n{len(sel):,} non-flash DREAM events: '
          f'matched any arm {(n_arm > 0).mean():.1%}, '
          f'exactly one arm {(n_arm == 1).mean():.1%}, '
          f'>1 arm {(n_arm > 1).mean():.1%}')

    mm = mm_event_table(args.run, args.subrun)
    sel = sel.merge(mm, on='eventId', how='left').fillna(0.0)

    def active(df, a, tier):
        nx, ny = df[f'n_{a}x'], df[f'n_{a}y']
        if tier == 'any':
            return (nx >= 1) & (ny >= 1)
        return (nx >= args.min_strips) & (ny >= args.min_strips)

    for tier in ('any', 'cluster'):
        lab = ('>=1 strip' if tier == 'any'
               else f'>={args.min_strips} strips') + ' in BOTH planes'
        print(f'\nMM chamber activity, {lab}:')
        print('  DREAM events matched to      chA     chB     chC     chD      n')
        for a in ARMS:
            m = arm_match[a][0] & (n_arm == 1)
            if not m.sum():
                continue
            row = '  '.join(f'{active(sel[m], c, tier).mean():6.1%}' for c in ARMS)
            print(f'    arm {a} only               {row}  {m.sum():6d}')
        un = n_arm == 0
        row = '  '.join(f'{active(sel[un], c, tier).mean():6.1%}' for c in ARMS)
        print(f'    no arm (unmatched)        {row}  {un.sum():6d}')

    # does the matcher select against MM content? compare its misses to hits
    print('\nmatched-vs-unmatched MM content (all chambers pooled):')
    anyact = np.zeros(len(sel), bool)
    for c in ARMS:
        anyact |= active(sel, c, 'cluster').to_numpy()
    for lab, m in (('matched', n_arm > 0), ('unmatched', n_arm == 0)):
        print(f'  {lab:9s}: cluster in any chamber {anyact[m].mean():6.1%}   '
              f'(n={m.sum():,})')

    if args.out:
        best_arm = np.where(n_arm == 1, M.argmax(axis=1), -1)
        tm = np.full(len(sel), np.nan)
        for i, a in enumerate(ARMS):
            pick = best_arm == i
            tm[pick] = arm_match[a][1][pick]
        np.savez(args.out,
                 eventId=sel['eventId'].to_numpy(),
                 bunch=sel['BunchNumber'].to_numpy(),
                 t_dream_ns=sel['t_since_flash_ns'].to_numpy(),
                 arm=best_arm, t_ntof_ns=tm,
                 n_matched_arms=n_arm,
                 **{f'n_{a}{p}': sel[f'n_{a}{p}'].to_numpy()
                    for a in ARMS for p in 'xy'},
                 **{f'amax_{a}{p}': sel[f'amax_{a}{p}'].to_numpy()
                    for a in ARMS for p in 'xy'})
        print(f'\nper-event results -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
