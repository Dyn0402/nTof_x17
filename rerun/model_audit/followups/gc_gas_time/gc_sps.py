#!/usr/bin/env python3
"""gc_sps.py -- GC, SPS det4 at H4: column duration vs wall-clock time over the test beam (sps_colsum.py products).

Per sub-run, events in time order, blocks of BLOCK_N events with both views: the per-sample median of the peak-normalised
strip sums (the stack) per view, and its 50 % rise and fall crossings; T50 = fall - rise (the template's width cancels
to first order, ZS-suppressed tails do not matter).  Per-event box-fit T (trimmed mean) kept as a second estimator.
Each block is tagged with its registry plateau (drift field) from sps_beam_test_26/analysis/datasets.py; v is compared
only at a fixed field.  Writes results/sps_blocks.json.
"""
import glob
import json
import os
import sys

import numpy as np

from gc_common import REPO, RES, tmean

sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'analysis'))
import datasets  # noqa: E402

SPS_LOCAL = os.path.expanduser('~/x17/cosmic_bench/model_audit/followups/gc_sps')
BLOCK_N = 500
DTS = 60.0
DSET = {'run_55': 'run55_flatdrift', 'run_56': 'run56_m70V', 'run_57': 'run57_rot25_co2', 'run_59': 'run59_co2',
        'run_60': 'run60_flush', 'run_62': 'run62_rot25_ladder', 'run_71': 'run71_raw'}


def dataset_of(run, sub):
    if run == 'run_63':
        return 'run63_flat' if sub in ('operating_02', 'operating_03') else 'run63_rot25'
    return DSET[run]


def edges(st, frac_rise=0.5, frac_fall=0.3):
    """rise at frac_rise of the stack maximum, fall at frac_fall of it: the column loses ~half its charge from the
    mesh to the cathode end at H4 (ws_spsv), so the far end is taken lower than the near one."""
    m = np.nanmax(st)
    if not m > 0:
        return np.nan, np.nan
    ab = np.flatnonzero(st >= frac_rise * m); ab2 = np.flatnonzero(st >= frac_fall * m)
    if not len(ab) or not len(ab2):
        return np.nan, np.nan
    i, j = ab[0], ab2[-1]
    lvl, lvl2 = frac_rise * m, frac_fall * m
    if i == 0 or j >= len(st) - 1:
        return np.nan, np.nan
    r = DTS * (i - 1 + (lvl - st[i - 1]) / (st[i] - st[i - 1]))
    f = DTS * (j + (st[j] - lvl2) / (st[j] - st[j + 1]))
    return float(r), float(f)


def main():
    out = []
    for f in sorted(glob.glob(os.path.join(SPS_LOCAL, '*.npz'))):
        run, sub = os.path.basename(f)[:-4].split('__')
        ds = dataset_of(run, sub); D = datasets.get(ds)
        z = np.load(f)
        t = z['t_wall']; o = np.argsort(t)
        nb = len(t) // BLOCK_N
        for b in range(nb):
            sl = o[b * BLOCK_N:(b + 1) * BLOCK_N]
            tb = float(np.median(t[sl]))
            lab = next((p for p in D['plateaus'] if p[1] <= tb < p[2]), None)
            row = dict(run=run, subrun=sub, dataset=ds, t=tb, plateau=lab[0] if lab else '',
                       drift_V=float(lab[3]) if lab else np.nan, n=int(BLOCK_N), raw=bool(D.get('raw', False)),
                       mount=D['mount'], gas=D['gas'])
            for p in 'xy':
                S = z[f'{p}_S'][sl].astype(float)
                st = np.median(S / np.maximum(S.max(axis=1, keepdims=True), 1.0), axis=0)
                r, fa = edges(st)
                row[f'{p}_rise'] = r; row[f'{p}_fall'] = fa; row[f'{p}_T50'] = fa - r
                row[f'{p}_Tfit'] = tmean(z[f'{p}_T'][sl])
                row[f'{p}_q'] = float(np.median(z[f'{p}_q'][sl]))
            out.append(row)
        print(f'{run}/{sub}: {nb} blocks', flush=True)
    json.dump(out, open(os.path.join(RES, 'sps_blocks.json'), 'w'), indent=0)


if __name__ == '__main__':
    main()
