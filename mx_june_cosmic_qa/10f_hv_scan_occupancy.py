#!/usr/bin/env python3
"""
10f_hv_scan_occupancy.py — one threshold-free read pass over the det3 mesh
ladder, recording how much of the track the raw waveforms actually contain.

The forward fit (10d) reports total charge and column length after
deconvolution and after censoring the rail. That is the right number, but it is
a *model* number, and the two things asked here deserve an answer that does not
depend on the model:

  * ``q_win`` -- the plain sum of every pedestal- and CNS-corrected sample over
    a fixed strip window around the peak strip and the whole 32-sample record.
    No threshold anywhere: the noise is zero-mean after CNS, so summing it in
    is unbiased (it costs ~sqrt(n_cells) x sigma of resolution per event, a few
    hundred ADC against signals of thousands, and washes out in a median over
    ~650 events). This is the model-free total charge.
  * the same sum restricted to cells over 5 sigma (``q_5s``), the strips and
    the time samples that carry them -- the *threshold-limited* view, i.e. what
    the DAQ and the hits chain can ever see. The gap between the two is the
    part of the track that exists but is not reported.

``t_first``/``t_last`` bracket the 5 sigma cells in time, so ``span`` is the
drift-column length as the threshold sees it: at low gain the far end of the
column, the most diffused and most attenuated charge, is the first to go under,
and this is where that shows up.

Window: +-``--half`` strips in POSITION order around the peak strip (default 10
= +-7.8 mm), wide enough for an inclined cosmic's transverse spread and for the
+-2-strip sharing tail, narrow enough that the summed noise stays small.

Selection is identical to 10c/10d: M3-golden rays inside the same fiducial box.

    ../.venv/bin/python 10f_hv_scan_occupancy.py
Output: <Analysis>/<run>/hv_scan/mx17_3/occupancy_raw.parquet
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
SIGMA = 5.0
SAT_ADC = 3550.0
SAMPLE_NS = 60.0


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

    rows = []
    for path in files:
        rdr = FeuReader(path)
        noise = np.where(rdr.noise > 0, rdr.noise, np.inf)
        thr = SIGMA * noise
        med_noise = float(np.median(rdr.noise[valid]))
        ids = set(int(e) for e in rdr.event_ids) & want
        if not ids:
            continue
        for eid, _ftst, wfm in rdr.iter_events(ids):
            amp = wfm.max(axis=1)
            pk = int(np.argmax(np.where(valid, amp, -np.inf)))
            r0 = rank[pk]
            lo, hi = max(0, r0 - half), min(len(chs), r0 + half + 1)
            win = chs[lo:hi]                       # the strip window
            W = wfm[win]                           # [nwin, nsamp]
            T = thr[win][:, None]
            over = W >= T
            nsamp = W.shape[1]
            if over.any():
                cols = np.flatnonzero(over.any(axis=0))
                t_first, t_last = int(cols[0]), int(cols[-1])
                n_strip_5s = int(over.any(axis=1).sum())
                q_5s = float(W[over].sum())
                n_cell_5s = int(over.sum())
            else:
                t_first = t_last = -1
                n_strip_5s, q_5s, n_cell_5s = 0, 0.0, 0
            rows.append(dict(
                subrun=sub, hv=volt, scan=scan, view=view, event_id=int(eid),
                peak_ch=pk, peak_amp=float(amp[pk]),
                q_win=float(W.sum()), q_5s=q_5s,
                n_strip_5s=n_strip_5s, n_cell_5s=n_cell_5s,
                t_first=t_first, t_last=t_last,
                span_ns=(t_last - t_first + 1) * SAMPLE_NS if t_first >= 0 else 0.0,
                n_sat_cell=int((W >= SAT_ADC).sum()),
                n_win=int(W.shape[0]), n_samp=nsamp, noise=med_noise))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--half', type=int, default=10,
                    help='strip half-window in position order (default 10)')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.io import strip_position_map

    out = a.out or os.path.join(ANALYSIS, RUN, 'hv_scan', DET,
                                'occupancy_raw.parquet')
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
            print(f'[10f] {sub}: no decoded_root, skip')
            continue
        cfg = get_config('sat_det3')
        cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = BASE, RUN, sub
        want = fiducial_ids(cfg)
        t0 = time.time()
        for view, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
            r = view_rows(decoded, feu, pos_maps[feu], want, sub, volt, scan,
                          view, a.half)
            allrows += r
        print(f'[10f] {sub} {volt} V: {len(want):,} rays, '
              f'{time.time() - t0:.0f} s', flush=True)

    df = pd.DataFrame(allrows)
    df.to_parquet(out)
    print(f'[10f] wrote {out} ({len(df):,} rows)')


if __name__ == '__main__':
    main()
