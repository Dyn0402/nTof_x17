#!/usr/bin/env python3
"""Flash-timing calibration from the divert-off runs.

Input : data/flash_run<N>.npz  (produced by extract_flash.py on lxplus)
Output: data/*.csv, data/flash_timing_calibration.json, figures/*.png

Definitions
-----------
For every (bunch, channel) the *flash hit* is the largest-amplitude hit inside
+-WIN ns of that bunch's anchor (the WALB tflash).  Its `tof` is the PSA's
pulse-time estimate (leading edge) and `peak_tof` the position of the maximum.
All times are quoted relative to the same bunch's PKUP pulse time, which is the
only channel that is never gated and whose cable never changed:

    dt = t_hit - t_PKUP        [ns]

`dt` therefore contains: flight time to the detector + cable + front-end delay
- the same for the pickup.  It is a constant per channel, and that constant is
the calibration.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA = BASE / 'data'
FIGS = BASE / 'figures'

WIN = 300.0          # ns half-window around the anchor for the flash hit
OFF_RUNS = [224356, 224357, 224358, 224359, 224360]
ON_RUNS = [224362, 224572]
WALLS = ['WALA', 'WALB', 'WALC', 'WALD']
PLASTICS = ['PSSA', 'PSSB', 'PSSC', 'PSSD']


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def load(run):
    f = np.load(DATA / f'flash_run{run}.npz', allow_pickle=False)
    return {k: f[k] for k in f.files}


def pkup_reference(d):
    """bunch -> pickup pulse time (largest-amp PKUP hit in the window)."""
    if 'PKUP' not in d:
        return {}
    r = d['PKUP']
    sel = np.abs(r['tof'] - r['anchor']) < 4000      # PKUP sits ~1.7 us after
    r = r[sel]
    order = np.lexsort((-r['amp'], r['BunchNumber']))
    r = r[order]
    first = np.ones(len(r), bool)
    first[1:] = r['BunchNumber'][1:] != r['BunchNumber'][:-1]
    r = r[first]
    return dict(zip(r['BunchNumber'].tolist(), r['tof'].tolist()))


def flash_hits(rec, win=WIN):
    """One row per (bunch, detn): the largest-amp hit within +-win of anchor."""
    sel = np.abs(rec['tof'] - rec['anchor']) < win
    r = rec[sel]
    key = r['BunchNumber'] * 100 + r['detn']
    order = np.lexsort((-r['amp'], key))
    r, key = r[order], key[order]
    first = np.ones(len(r), bool)
    first[1:] = key[1:] != key[:-1]
    return r[first]


def per_channel(run, d, pk, trees):
    """Return list of dict rows, one per (tree, channel)."""
    rows = []
    for t in trees:
        if t not in d:
            continue
        fh = flash_hits(d[t])
        tp = np.array([pk.get(int(b), np.nan) for b in fh['BunchNumber']])
        dt = fh['tof'] - tp
        dtp = fh['peak_tof'] - tp
        ok = np.isfinite(dt)
        for ch in range(1, 9):
            m = ok & (fh['detn'] == ch)
            if m.sum() < 20:
                continue
            x, xp = dt[m], dtp[m]
            med = np.median(x)
            core = np.abs(x - med) < 100                 # kill PSA mis-tags
            xc = x[core]
            pi = fh['PulseIntensity'][m][core]
            lo, hi = pi < 6e12, pi >= 6e12
            rows.append(dict(
                run=run, tree=t, ch=int(ch), n=int(m.sum()), n_core=int(core.sum()),
                frac_core=float(core.mean()),
                dt_med=float(np.median(xc)), dt_mean=float(xc.mean()),
                dt_sigma=float(robust_sigma(xc)), dt_std=float(xc.std()),
                dt_err=float(xc.std() / max(np.sqrt(len(xc)), 1)),
                peak_med=float(np.median(xp[core])),
                amp_med=float(np.median(fh['amp'][m][core])),
                area_med=float(np.median(fh['area'][m][core])),
                fwhm_med=float(np.median(fh['fwhm'][m][core])),
                rise_med=float(np.median(fh['risetime'][m][core])),
                sat_frac=float(np.mean(fh['satuflag'][m][core] > 0)),
                dt_lo=float(np.median(xc[lo])) if lo.sum() > 20 else np.nan,
                dt_hi=float(np.median(xc[hi])) if hi.sum() > 20 else np.nan,
                n_lo=int(lo.sum()), n_hi=int(hi.sum()),
                amp_lo=float(np.median(fh['amp'][m][core][lo])) if lo.sum() > 20 else np.nan,
                amp_hi=float(np.median(fh['amp'][m][core][hi])) if hi.sum() > 20 else np.nan,
            ))
    return rows


def write_csv(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(path, 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join('' if r.get(c) is None or (isinstance(r.get(c), float) and not np.isfinite(r[c]))
                              else (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]))
                              for c in cols) + '\n')


def main():
    DATA.mkdir(exist_ok=True), FIGS.mkdir(exist_ok=True)
    all_rows, per_bunch_store = [], {}
    for run in OFF_RUNS:
        p = DATA / f'flash_run{run}.npz'
        if not p.exists():
            print(f'missing {p}', file=sys.stderr); continue
        d = load(run)
        pk = pkup_reference(d)
        rows = per_channel(run, d, pk, WALLS + PLASTICS)
        all_rows += rows
        print(f'run{run}: {len(pk)} pkup bunches, {len(rows)} channel rows')

        # --- per-bunch series for the wall channels (jitter / drift studies)
        store = {}
        for t in WALLS:
            if t not in d:
                continue
            fh = flash_hits(d[t])
            tp = np.array([pk.get(int(b), np.nan) for b in fh['BunchNumber']])
            store[t] = dict(bunch=fh['BunchNumber'], ch=fh['detn'],
                            dt=fh['tof'] - tp, amp=fh['amp'],
                            pi=fh['PulseIntensity'], peak=fh['peak_tof'] - tp)
        per_bunch_store[run] = store
    if not all_rows:
        print('no data yet'); return

    write_csv(DATA / 'per_channel_flash_timing.csv', all_rows)
    np.savez_compressed(DATA / 'per_bunch_series.npz',
                        **{f'{r}_{t}_{k}': v
                           for r, s in per_bunch_store.items()
                           for t, dd in s.items() for k, v in dd.items()})
    print(f'wrote {DATA / "per_channel_flash_timing.csv"} ({len(all_rows)} rows)')


if __name__ == '__main__':
    main()
