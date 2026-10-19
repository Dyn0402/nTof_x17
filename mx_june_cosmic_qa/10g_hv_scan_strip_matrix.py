#!/usr/bin/env python3
"""
10g_hv_scan_strip_matrix.py -- per-strip significance profile around the peak
strip, for every event of the det3 mesh ladder.

10f counted the strips over 5 sigma. That count is the thing we want to
explain, and to explain it you need the amplitudes of the strips that did *not*
make it: the question "does a low gain lose the faint strips" cannot be
answered from a threshold-limited quantity alone.

So this pass records, for each event and each view, the peak amplitude of every
strip in a +-``--half`` window around the peak strip, in units of that strip's
own noise sigma:

    s_k = max_t W[strip(k), t] / sigma[strip(k)],   k = -half .. +half

k is the offset in POSITION order (k = 0 is the peak strip). Sub-threshold and
negative values are kept -- they are the whole point. Two consequences, both
handled in the reduction (10h), not here:

  * ``s`` is a max over 32 samples, so an empty strip does not sit at 0 but at
    the expected maximum of 32 noise draws, ~2 sigma. Any statement of the form
    "scale this strip down by the gain ratio" must scale the signal and not
    that floor; 10h measures the floor from the outer columns of this matrix.
  * position order is not guaranteed to run the same way as the detector axis
    in both views, so only |k| and widths are comparable across views.

Selection (M3-golden, fiducial box) and the peak-strip definition are identical
to 10f, so the two tables merge on (subrun, view, event_id).

    ../.venv/bin/python 10g_hv_scan_strip_matrix.py
Output: <Analysis>/<run>/hv_scan/mx17_3/strip_matrix.parquet
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE, os.path.join(REPO, 'cosmic_bench_analysis')]

RUN = 'mx17_det3_saturday_scan_6-27-26'
DET = 'mx17_3'
BASE = '/home/dylan/x17/cosmic_bench/det3/'
ANALYSIS = '/home/dylan/x17/cosmic_bench/Analysis'
SUBRUN_RE = re.compile(r'^hv_scan(2?)_resist_(\d+)V_drift_1000V$')
DET_Z = 702.0
FID_X = (-190.0, 115.0)
FID_Y = (-190.0, 165.0)


def subruns():
    out = []
    for d in sorted(os.listdir(os.path.join(BASE, RUN))):
        m = SUBRUN_RE.match(d)
        if m:
            out.append((d, int(m.group(2)), 'scan2' if m.group(1) else 'scan1'))
    return sorted(out, key=lambda t: (t[2], t[1]))


def fiducial_ids(cfg):
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    from M3RefTracking import M3RefTracking
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    x, y, evn = rays.get_xy_positions(DET_Z)
    x, y, evn = np.asarray(x), np.asarray(y), np.asarray(evn)
    ok = (np.isfinite(x) & np.isfinite(y)
          & (x >= FID_X[0]) & (x <= FID_X[1])
          & (y >= FID_Y[0]) & (y <= FID_Y[1]))
    return set(int(e) for e in evn[ok])


def view_rows(decoded, feu, pos, want, sub, volt, scan, view, half):
    from wft.io import FeuReader
    files = sorted(glob.glob(os.path.join(decoded, f'*_{feu:02d}.root')))
    valid = ~np.isnan(pos)
    order = np.argsort(pos[valid])
    chs = np.flatnonzero(valid)[order]            # channels in position order
    rank = np.full(512, -1)
    rank[chs] = np.arange(len(chs))
    ncol = 2 * half + 1

    rows = []
    for path in files:
        rdr = FeuReader(path)
        noise = np.where(rdr.noise > 0, rdr.noise, np.inf)
        ids = set(int(e) for e in rdr.event_ids) & want
        if not ids:
            continue
        for eid, _ftst, wfm in rdr.iter_events(ids):
            amp = wfm.max(axis=1)
            pk = int(np.argmax(np.where(valid, amp, -np.inf)))
            r0 = rank[pk]
            lo, hi = max(0, r0 - half), min(len(chs), r0 + half + 1)
            win = chs[lo:hi]
            s = np.full(ncol, np.nan, dtype=np.float32)
            s[(rank[win] - r0) + half] = amp[win] / noise[win]
            row = dict(subrun=sub, hv=volt, scan=scan, view=view,
                       event_id=int(eid), peak_ch=pk,
                       peak_amp=float(amp[pk]), peak_noise=float(rdr.noise[pk]),
                       n_valid=int(len(win)))
            row.update({f's{k:+d}': float(s[k + half])
                        for k in range(-half, half + 1)})
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--half', type=int, default=10)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.io import strip_position_map

    out = a.out or os.path.join(ANALYSIS, RUN, 'hv_scan', DET,
                                'strip_matrix.parquet')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    base_cfg = get_config('sat_det3')
    base_cfg.BASE_PATH, base_cfg.RUN = BASE, RUN
    pos_maps = strip_position_map(base_cfg)

    allrows = []
    for sub, volt, scan in subruns():
        if a.only and sub not in a.only:
            continue
        decoded = os.path.join(BASE, RUN, sub, 'decoded_root')
        if not os.path.isdir(decoded):
            print(f'[10g] {sub}: no decoded_root, skip')
            continue
        cfg = get_config('sat_det3')
        cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = BASE, RUN, sub
        want = fiducial_ids(cfg)
        t0 = time.time()
        for view, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
            allrows += view_rows(decoded, feu, pos_maps[feu], want, sub, volt,
                                 scan, view, a.half)
        print(f'[10g] {sub} {volt} V: {len(want):,} rays, '
              f'{time.time() - t0:.0f} s', flush=True)

    df = pd.DataFrame(allrows)
    df.to_parquet(out)
    print(f'[10g] wrote {out} ({len(df):,} rows)')


if __name__ == '__main__':
    main()
