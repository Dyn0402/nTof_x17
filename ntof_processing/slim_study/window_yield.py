#!/usr/bin/env python3
"""How much n_TOF survives a DREAM-trigger-window slim? Measured, not modelled.

For every non-flash DREAM trigger of run_79 we predict its place in the n_TOF
time base (calibration.py) and count the hits of every scintillator tree that
land within +-W of it, in the same bunch, for a scan of W.

USAGE
    python window_yield.py                       # the narrow scan
    python window_yield.py --out window_yield_wide.json \
                           --windows 250 1000 2000 5000 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ntof_dream_merge' / 'match_study' / 'scripts'))

import study_common as sc                                   # noqa: E402
from ntof_dream_merge.calibration import load as load_cal   # noqa: E402
import ntof_dream_merge.ntof_io as ntof_io                  # noqa: E402

TREES = ('WALA', 'WALB', 'WALC', 'WALD',
         'PSSA', 'PSSB', 'PSSC', 'PSSD',
         'LIQA', 'LIQB', 'LIQC', 'LIQD')
KEY = 1e9
CHUNK = 150          # bunches per read
HERE = Path(__file__).resolve().parent


def dream_keys(perbunch: bool = False, shift_ns: float = 0.0):
    """(sorted packed keys of every predicted trigger position, n_events).

    With `perbunch`, the bunch's own fitted (da_b, dk_b) is added -- i.e. the
    FINAL clock, the one a two-stage slim cuts on. Events in bunches with too
    few matches to fit fall back to the global map.
    """
    cal = load_cal()
    keys, nev, n_pb = [], 0, 0
    for sub in sc.SUBRUNS:
        d = np.load(sc.DATA / f'events_{sub}.npz')
        t, b = d['t'].astype(np.float64), d['bunch'].astype(np.float64)
        # arm-agnostic prediction: the slim must keep every arm, so the per-arm
        # offset (-16.8 .. +7.6 ns) is absorbed into the window, not applied.
        tp = cal.predict(t, arm=None)
        if perbunch:
            # `corr_in` is the bunch's own fit applied to its own events, which
            # is the production case. `corr_cv` exists only so that a quoted
            # efficiency is not the fit reading back its own input.
            c = np.load(sc.DATA / f'perbunch_corr_{sub}_wp.npz')['corr_in']
            ok = np.isfinite(c)
            tp = tp + np.where(ok, c, 0.0)
            n_pb += int(ok.sum())
        keys.append(b * KEY + tp + shift_ns)
        nev += t.size
    if perbunch:
        print(f'per-bunch correction applied to {n_pb:,} of {nev:,} triggers '
              f'({n_pb/nev:.2%}); the rest fall back to the global map')
    k = np.sort(np.concatenate(keys))
    return k, nev, cal


def nearest_dt(hit_key, pred_key):
    j = np.searchsorted(pred_key, hit_key)
    j0 = np.clip(j - 1, 0, pred_key.size - 1)
    j1 = np.clip(j, 0, pred_key.size - 1)
    return np.minimum(np.abs(pred_key[j0] - hit_key),
                      np.abs(pred_key[j1] - hit_key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=float, nargs='+',
                    default=[10, 25, 50, 100, 200, 500, 1000],
                    help='slim half-widths to scan, ns')
    ap.add_argument('--out', default='window_yield_narrow.json')
    ap.add_argument('--shift-ns', type=float, default=0.0,
                    help='offset every prediction by this much -- the +100 us '
                         'accidental control of the match study')
    ap.add_argument('--perbunch', action='store_true',
                    help='centre on the FINAL clock (global map + da_b, dk_b), '
                         'as a two-stage slim does')
    args = ap.parse_args()
    WINDOWS = np.asarray(sorted(args.windows), dtype=float)
    OUT = HERE / args.out
    report = f'{WINDOWS[-1]:g}'

    files = sc.use_variant()
    print(f'{len(files)} partials, cache {ntof_io.CACHE_DIR}')
    pred, nev, cal = dream_keys(args.perbunch, args.shift_ns)
    print(f'{nev:,} DREAM triggers   K={cal.K:.6e} T0={cal.T0_ns:+.2f} '
          f'window={cal.window_ns} ns')

    bunches = np.unique((pred // KEY).astype(np.int64))
    print(f'{bunches.size} bunches spanned ({bunches.min()}-{bunches.max()})')

    res = {}
    for tree in TREES:
        t0 = time.time()
        edges = ntof_io.bunch_edges(sc.NTOF_RUN, tree)
        n_run = int(edges[-1])
        kept = np.zeros(WINDOWS.size, dtype=np.int64)
        n_in = 0
        for i in range(0, bunches.size, CHUNK):
            blk = bunches[i:i + CHUNK]
            a = ntof_io.read_bunches(sc.NTOF_RUN, tree, blk,
                                     branches=('BunchNumber',),
                                     repair_tflash=False)
            if a['t_since_flash_ns'].size == 0:
                continue
            hk = a['BunchNumber'].astype(np.float64) * KEY + a['t_since_flash_ns']
            n_in += hk.size
            dt = nearest_dt(np.sort(hk), pred)
            kept += (dt[None, :] <= WINDOWS[:, None]).sum(axis=1)
        res[tree] = dict(n_run=n_run, n_in_dream_bunches=int(n_in),
                         kept={f'{w:g}': int(k) for w, k in zip(WINDOWS, kept)})
        nk = res[tree]['kept'][report]
        print(f'{tree}  run {n_run:>11,}  in-bunches {n_in:>11,}  '
              f'kept@{report}ns {nk:>10,} ({nk / max(n_in, 1):.4%})  '
              f'[{time.time()-t0:.0f}s]')

    res['_meta'] = dict(n_triggers=nev, n_bunches=int(bunches.size),
                        windows=WINDOWS.tolist(), perbunch=bool(args.perbunch),
                        shift_ns=float(args.shift_ns),
                        bunch_range=[int(bunches.min()), int(bunches.max())])
    OUT.write_text(json.dumps(res, indent=2))
    print('->', OUT)


if __name__ == '__main__':
    main()
