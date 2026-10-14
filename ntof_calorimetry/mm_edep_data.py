#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm_edep_data.py -- the MM energy-deposit estimator on n_TOF BEAM data
(MM_EDEP.md, stage 3): run_145 (5 Aug, the post-23-July configuration, as are
the run_149 cosmics), chambers A and C.

Tracks: stage-3 fullpass, gated, the only gated track of its arm in the event.
Charge: `mm_charge`'s estimator (road from the waveform reco's strip-frame
p0 / tan, pedestal, common mode from the block's channels outside the road,
road sum over the 20 samples), per plane, on the run's decoded_root.
Joined per track to the scint-stack table (wall group and plastic bar
amplitudes, `ntof_scint_stack`) and to `wall_plastic`'s MeV calibration.

    python -m ntof_calorimetry.mm_edep_data          # -> OUT/mm_edep/beam_run145.pkl
"""
from __future__ import annotations

import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry import mm_charge as MC  # noqa: E402
from ntof_calorimetry.mip_sample import OUT  # noqa: E402

RUN = 'run_145'
WAVES = Path('/media/dylan/data/x17/beam_july/runs')
STAGE3 = Path('/media/dylan/data/x17/sept26_prelim/stage3_fullpass')
EDEP = OUT / 'mm_edep'
SUBS = ('stat090_0000', 'stat090_0001', 'stat090_0002')
KEEP = ['subrun', 'event_id', 'arm', 'track_id', 'x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'tanx',
        'tany', 'x_local', 'y_local', 'x_t0', 'y_t0', 'v_drift_um_ns', 't_since_flash_ns', 'k_arm',
        'chi2dof_x', 'chi2dof_y', 'x_n_strips', 'y_n_strips', 'dca_axis_mm', 'event_class',
        'coinc_this_arm', 'n_coinc_arms']


def beam_tracks(arm: str | None = None) -> pd.DataFrame:
    parts = []
    for s in SUBS:
        t = pd.read_parquet(STAGE3 / f'tracks_{RUN}_{s}.parquet')
        t = t[t.arm.isin(['A', 'C']) & t.gated.astype('boolean').fillna(False)]
        t = t[t.groupby(['event_id', 'arm']).event_id.transform('size') == 1]
        parts.append(t[[c for c in KEEP if c in t.columns]])
    T = pd.concat(parts, ignore_index=True)
    T['arm'] = T.arm.astype(str)
    return T[T.arm == arm] if arm else T


def _pos_map(arm: str, plane: str) -> np.ndarray:
    from ntof_tracking.wft_beam import beam_config
    from wft import io as wio
    cfg = beam_config(arm, RUN, SUBS[0])
    pm = wio.strip_position_map(cfg)
    return pm[MC.FEUS[arm]['xy'.index(plane)]]


def process(args) -> pd.DataFrame:
    sub, arm, T, pms = args
    from wft import io as wio
    t = T.set_index('event_id')
    rows = {int(e): dict(event_id=int(e)) for e in t.index}
    for k, plane in enumerate('xy'):
        feu = MC.FEUS[arm][k]
        for f in sorted(glob.glob(str(WAVES / RUN / sub / 'decoded_root' / f'*_{feu:02d}.root'))):
            rdr = wio.FeuReader(f)
            idx = np.where(np.isin(rdr.event_ids, t.index.to_numpy()))[0]
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
                    if raw.shape[0] != MC.N_SAMPLE:
                        continue
                    r = t.loc[eid]
                    p0, tn = float(r[f'{plane}_p0']), float(r[f'{plane}_tan_theta'])
                    if not (np.isfinite(p0) and np.isfinite(tn)):
                        continue
                    road = MC._road(pms[plane], p0, tn)
                    ctrl = MC._road(pms[plane], p0, tn, MC.CTRL_SHIFT_MM if p0 < 200 else -MC.CTRL_SHIFT_MM)
                    if len(road) < 3:
                        continue
                    w = MC._baseline(raw, rdr.ped, np.union1d(road, ctrl))
                    prof = w[:, road].sum(1)
                    d = rows[eid]
                    d[f'{plane}_Q'] = float(prof.sum())
                    d[f'{plane}_prof'] = prof.astype(np.float32)
                    d[f'{plane}_n_road'] = int(len(road))
                    d[f'{plane}_Q_ctrl'] = float(w[:, ctrl].sum()) if len(ctrl) else np.nan
                    d[f'{plane}_raw_max'] = float(raw[:, road].max())
    D = pd.DataFrame(list(rows.values()))
    D = D.merge(T, on='event_id', how='left')
    print(f'{arm} {sub}: {len(D)} tracks, {D.x_Q.notna().mean():.3f} with x charge', flush=True)
    return D


def scint(D: pd.DataFrame) -> pd.DataFrame:
    """Calibrated wall / plastic (wall_plastic's beam table, MeV) per track."""
    B = pd.read_parquet(OUT / 'wp' / 'beam_data.parquet')
    B = B[B.run == RUN][['event_id', 'arm', 't_since_flash_ns', 'e_plas', 'bar', 'p_sat', 'e_wall',
                         'wgrp', 'hw_A', 'hw_C']]
    B['tkey'] = B.t_since_flash_ns.round(-1)
    D['tkey'] = D.t_since_flash_ns.round(-1)
    out = D.merge(B.drop(columns='t_since_flash_ns'), on=['event_id', 'arm', 'tkey'], how='left')
    return out.drop(columns='tkey')


def main() -> int:
    EDEP.mkdir(parents=True, exist_ok=True)
    T = beam_tracks()
    pms = {arm: {p: _pos_map(arm, p) for p in 'xy'} for arm in 'AC'}
    jobs = [(s, arm, T[(T.subrun == s) & (T.arm == arm)], pms[arm]) for s in SUBS for arm in 'AC']
    with ProcessPoolExecutor(max_workers=6) as ex:
        D = pd.concat(list(ex.map(process, jobs)), ignore_index=True)
    D = scint(D)
    D.to_pickle(EDEP / f'beam_{RUN}.pkl')
    print(f'-> {EDEP / f"beam_{RUN}.pkl"}: {len(D)} tracks; scint matched {D.e_wall.notna().mean():.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
