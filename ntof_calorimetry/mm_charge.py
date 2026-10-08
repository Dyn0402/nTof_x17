#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm_charge.py -- PLAN.md M1: a model-independent ionisation charge per track
plane, from the raw waveforms.  NOT the reco's ``q_sum``/``q_total``/
``q_per_len`` (the NNLS fills unobservable depth bins without bound, O10).

THE ESTIMATOR.  For each gated track and each plane:

  road     the strips whose position lies on the track's own corridor,
           p0 + depth x tan over depth -3 .. 33 mm, padded 5 mm each side
           (`beam_cache`'s corridor; the pad covers the resistive kernel's
           +-2 strip reach).  The geometry comes from the waveform reco
           (x_p0, x_tan_theta in the strip-map frame), never from hits.
  baseline pedestal = the file's per-channel median (`wft.io.FeuReader`);
           COMMON MODE per sample from each 64-channel block's channels
           OUTSIDE the road.  The stock CNS takes the median of the whole
           block, which a steep track filling half a block pulls up -- it
           would subtract the track's own charge.
  charge   Q = sum over road strips and all 20 samples of the baseline-
           subtracted ADC.  The resistive layer moves charge between strips
           but conserves it, so the road sum is the induced charge.

Stored per track and plane: Q, the per-sample road profile (depth study and
truncated means), the per-strip sums, the road width, an off-road control
road of the same width (its Q must average 0), the largest raw sample
(saturation: 12-bit rail), and whether the track's drift fits the window.

    python -m ntof_calorimetry.mm_charge --arms A,C        # every local sub-run
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry.mip_sample import OUT  # noqa: E402
from wft import io as wio  # noqa: E402

RUN = 'run_149'
#: the in-situ calibration staged these waveforms (FEU 3/4 = A, 7/8 = C)
WAVE_BASE = Path('~/scratch/ntof_insitu/beam').expanduser()
TRACKS = REPO / 'ntof_cosmics' / 'results' / 'tracking' / 'k_run_147'
FEUS = {'A': (3, 4), 'B': (5, 6), 'C': (7, 8), 'D': (1, 2)}
Z_LO, Z_HI, PAD_MM = -3.0, 33.0, 5.0
GAP_MM = 30.0
SAMPLE_NS = 60.0
N_SAMPLE = 20
#: raw ADC above this is on the 12-bit rail (pedestal ~330)
SAT_RAW = 3700.0
CTRL_SHIFT_MM = 80.0
BLOCK = 64
M2 = OUT / 'm2'


def _pos_maps(arm: str) -> dict:
    from ntof_tracking.wft_beam import beam_config
    cfg = beam_config(arm, RUN, 'cosbounce_cos_0005')
    cfg.BASE_PATH = str(WAVE_BASE) + '/'
    return wio.strip_position_map(cfg)


def _road(pm: np.ndarray, p0: float, tn: float, shift: float = 0.0) -> np.ndarray:
    a, b = p0 + Z_LO * tn + shift, p0 + Z_HI * tn + shift
    lo, hi = min(a, b) - PAD_MM, max(a, b) + PAD_MM
    ch = np.where((pm >= lo) & (pm <= hi))[0]
    return ch[np.argsort(pm[ch])]


def _baseline(raw: np.ndarray, ped: np.ndarray, road: np.ndarray) -> np.ndarray:
    """(n_sample, 512) raw -> pedestal and road-excluded common mode off."""
    w = raw - ped[None, :]
    mask = np.ones(512, bool)
    mask[road] = False
    for b in range(512 // BLOCK):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        m = mask[sl]
        src = w[:, sl][:, m] if m.sum() >= 16 else w[:, sl]
        w[:, sl] -= np.median(src, axis=1)[:, None]
    return w


def process_subrun(sub: str, arm: str, pm: dict) -> pd.DataFrame:
    t = pd.read_parquet(TRACKS / f'tracks_{RUN}_{sub}.parquet')
    t = t[(t.arm == arm) & t.gated.astype('boolean').fillna(False)].copy()
    t['n_trk'] = t.groupby('event_id').event_id.transform('size')
    t = t[t.n_trk == 1].set_index('event_id')
    if not len(t):
        return pd.DataFrame()
    feu = dict(zip('xy', FEUS[arm]))
    rows = {int(e): dict(event_id=int(e)) for e in t.index}
    for plane in 'xy':
        files = sorted(glob.glob(str(WAVE_BASE / RUN / sub / 'decoded_root' / f'*_{feu[plane]:02d}.root')))
        for f in files:
            rdr = wio.FeuReader(f)
            want = np.intersect1d(rdr.event_ids, t.index.to_numpy())
            if not len(want):
                continue
            idx = np.where(np.isin(rdr.event_ids, want))[0]
            for lo in range(0, len(idx), 400):
                blk = idx[lo:lo + 400]
                arr = rdr.tree.arrays(['eventId', 'amplitude'], entry_start=int(blk[0]),
                                      entry_stop=int(blk[-1]) + 1, library='np')
                for i in blk:
                    j = i - int(blk[0])
                    eid = int(arr['eventId'][j])
                    if eid not in rows:
                        continue
                    raw = arr['amplitude'][j].reshape(-1, 512).astype(np.float32)
                    r = t.loc[eid]
                    p0, tn = float(r[f'{plane}_p0']), float(r[f'{plane}_tan_theta'])
                    road = _road(pm[feu[plane]], p0, tn)
                    ctrl = _road(pm[feu[plane]], p0, tn, CTRL_SHIFT_MM if p0 < 200 else -CTRL_SHIFT_MM)
                    if len(road) < 3:
                        continue
                    w = _baseline(raw, rdr.ped, np.union1d(road, ctrl))
                    prof = w[:, road].sum(1)
                    d = rows[eid]
                    d[f'{plane}_Q'] = float(prof.sum())
                    d[f'{plane}_prof'] = prof.astype(np.float32)
                    d[f'{plane}_strip_q'] = w[:, road].sum(0).astype(np.float32)
                    d[f'{plane}_n_road'] = int(len(road))
                    d[f'{plane}_Q_ctrl'] = float(w[:, ctrl].sum()) if len(ctrl) else np.nan
                    d[f'{plane}_n_ctrl'] = int(len(ctrl))
                    d[f'{plane}_noise'] = float(np.sqrt(np.sum(rdr.noise[road] ** 2) * raw.shape[0]))
                    d[f'{plane}_raw_max'] = float(raw[:, road].max())
                    d[f'{plane}_n_sample'] = int(raw.shape[0])
    D = pd.DataFrame(list(rows.values()))
    keep = ['x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'tan_raw_x', 'tan_raw_y', 'tanx', 'tany',
            'x_t0', 'y_t0', 'v_drift_um_ns', 'drift_t_end_ns', 'x_local', 'y_local', 'x_n_strips',
            'y_n_strips', 'chi2dof_x', 'chi2dof_y', 'x_slope_reliable', 'y_slope_reliable',
            'q_per_len', 'k_arm']
    D = D.merge(t[keep].reset_index(), on='event_id', how='left')
    D['subrun'], D['arm'] = sub, arm
    return D


def subruns() -> list:
    return sorted(Path(p).parent.name for p in glob.glob(str(WAVE_BASE / RUN / '*' / 'decoded_root')))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='A,C')
    ap.add_argument('--subs', default='')
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()
    subs = [s for s in a.subs.split(',') if s] or subruns()
    M2.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(s, arm) for arm in a.arms.split(',') for s in subs]
    pms = {arm: _pos_maps(arm) for arm in a.arms.split(',')}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(process_subrun, s, arm, pms[arm]): (s, arm) for s, arm in jobs}
        out = []
        for f in futs:
            s, arm = futs[f]
            D = f.result()
            print(f'{arm} {s}: {len(D)} tracks', flush=True)
            out.append(D)
    D = pd.concat([d for d in out if len(d)], ignore_index=True)
    D.to_pickle(M2 / 'charge_run149.pkl')
    print(f'wrote {M2 / "charge_run149.pkl"} ({len(D):,} track planes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
