#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scint_stack.py -- every gated track on all four arms, with the whole
scintillator stack behind it read out channel by channel.

`det_a_scint` asked one question of one arm: does an arm-A track point at a
wall group and a plastic bar that fired?  This module builds the table that
lets the same question be asked of every arm and every layer -- SiPM wall,
plastic, AND the liquid, which no track has ever been compared with -- and
keeps enough of each hit (amplitude, time, saturation, both wall ends
separately) to turn the yes/no into maps of efficiency and gain and a position
along the wall bars.

WHAT IS DIFFERENT FROM `det_a_scint`, AND WHY.

**The extrapolation uses the RAW tangent, and the scale is left open.**  The
calibrated direction is ``tan_raw / k_arm``, but `k_arm` is the open blocker
(STATUS 2026-09-10: the wall measures arm A's tangents 33 % too large; C and
D's `k` moves 24-29 % run to run) and arm B has no `k` at all.  Every track
here carries ``u, v`` at the strip plane and ``tan_raw_x, tan_raw_y``; the
crossing at a layer a lever ``L`` past the strips is

    u_layer = u + L * s_u * tan_raw_x,      v_layer = v + L * s_v * tan_raw_y

with the scale ``s`` fitted downstream FROM THE SCINTILLATORS
(`scint_stack_ana.fit_scale`).  The frame was verified on run_145 against the
3D projection of `det_a_scint.project`: identical to 0.000 mm once the plane
centre ``u_mm`` is added, on A, C and D.  So this table needs no `k`, includes
arm B, and includes run_126/154/156, whose stage-3 tables carry no direction.

**Every channel, not "did the group fire".**  Wall ``detn`` 1..8 are the two
ends of four groups (``detn = 2g + {1, 2}``); each end is kept separately
(amplitude, time, saturation), because whether the two ends agree is half of
what this pass is for.  Plastic bars 1 and 2, and the liquid's one channel
(with its area for the slow component), likewise.  Where a channel has more
than one hit in a window the LARGEST is kept, as `scintillators.wall_pairs`
does.

**Two windows of identical width.**  ``on`` is the production accept window
(-100, +60) ns; ``off`` is the same 160 ns displaced to (-560, -400), before
the trigger, which `det_a_scint` showed is flat for the wall and the plastic
and equal to the slim's own ``is_control`` level.  The plastic decays for
~1 us AFTER the trigger, so a post-trigger control would be wrong.  The
liquid's pre-trigger side is checked rather than assumed (`scint_stack_ana`).

    python -m ntof_scint_stack.extract --runs run_145
    python -m ntof_scint_stack.extract --jobs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.campaign_imaging import (  # noqa: E402
    PRE_ACCESS_RUNS, run_number)
from sept26_prelim_analysis.det_a_scint import layer_geometry  # noqa: E402

SCHEMA = 'ntof_scint_stack/4'
ARMS = ('A', 'B', 'C', 'D')
#: ``det`` codes: wall 0-3, plastic 4-7, liquid 8-11, in A B C D order.
WAL, PSS, LIQ = 0, 4, 8

#: name -> (lo, hi) on ``dt_ns``, signed.  SAME WIDTH, see the docstring.
WINDOWS = {'on': (-100.0, 60.0), 'off': (-560.0, -400.0)}

#: Every amplitude in the slim is ADC counts at 0.0306 mV/count on every
#: channel (``mx_july_beam_qa/calib/adc_to_mv_run224524.json``: 0.03039 to
#: 0.03079 across all 44 channels, so one factor is good to 0.7 %).
MV_PER_ADC = 0.0306

#: The hardware trigger, emulated: per arm, (the largest top+bottom SUM of a
#: wall group) >= WALL_THR AND (the larger plastic bar) >= PLAS_THR, in mV, in
#: the prompt window; DREAM fires on the OR over arms.  The values are the
#: discriminator thresholds READ BACK from the N1081B boards on run_79
#: (`ntof_dream_merge.dream_trigger`, 2026-07-26).  Every production run states
#: it ran 'at the run_67 optimum', but the per-sub-run n1081b_config.json files
#: are not on this machine, so this is the run_79 setting assumed campaign-wide
#: -- labelled as an emulation wherever it is used.
WALL_THR = {'A': 25.0, 'B': 35.0, 'C': 34.0, 'D': 36.0}
PLAS_THR = {'A': 118.0, 'B': 139.0, 'C': 157.0, 'D': 134.0}

TRACK_COLS = [
    'subrun', 'event_id', 'arm', 'track_id', 'gated',
    'x_local', 'y_local', 'tan_raw_x', 'tan_raw_y', 'k_arm',
    'x_slope_reliable', 'y_slope_reliable',
    'x_p0_err', 'y_p0_err', 'x_tan_err', 'y_tan_err',
    'chi2dof_x', 'chi2dof_y', 'x_n_strips', 'y_n_strips',
    'q_total', 'dca_axis_mm', 'drift_railed', 't_since_flash_ns',
    'is_flash', 'coinc_A', 'coinc_B', 'coinc_C', 'coinc_D', 'n_coinc_arms',
    'x_t0', 'y_t0', 'q_per_len',
]


def channels() -> list:
    """[(prefix, family offset, detn)] in the order the columns are written."""
    out = [(f'w{n}', WAL, n) for n in range(1, 9)]
    out += [('p1', PSS, 1), ('p2', PSS, 2), ('l', LIQ, 1)]
    return out


def hit_columns(win: str) -> list:
    cols = []
    for pre, fam, _n in channels():
        cols += [f'{pre}_amp_{win}', f'{pre}_dt_{win}', f'{pre}_sat_{win}']
        if fam == LIQ:
            cols += [f'{pre}_area_{win}', f'{pre}_pu_{win}']
    return cols


def read_tracks(src: Path, run: str, sub: str) -> pd.DataFrame:
    p = src / f'tracks_{run}_{sub}.parquet'
    t = pd.read_parquet(p, columns=TRACK_COLS)
    t = t[t.gated.astype('boolean').fillna(False)].drop(columns='gated')
    t['subrun'] = sub
    for c in ('x_slope_reliable', 'y_slope_reliable', 'drift_railed',
              'is_flash', 'coinc_A', 'coinc_B', 'coinc_C', 'coinc_D'):
        t[c] = t[c].astype('boolean').fillna(False).astype(bool)
    # THE TRIGGER.  DREAM fires on (wall segment SUM) AND (plastic bar) in ANY
    # arm (`ntof_dream_merge.dream_trigger`), so on an event where this arm's
    # own coincidence is what triggered, its wall and plastic fired BY
    # CONSTRUCTION.  ``other_trig`` marks the events some OTHER arm could have
    # triggered: there, this arm's layers are probed without that bias.
    oth = np.zeros(len(t), bool)
    for a in ARMS:
        oth |= (t.arm.to_numpy() != a) & t[f'coinc_{a}'].to_numpy()
    t['other_trig'] = oth
    t['n_trk'] = (t.groupby(['arm', 'event_id']).event_id
                  .transform('size').astype('int16'))
    return t.reset_index(drop=True)


def read_hits(slim_dir: Path, run: str, sub: str) -> pd.DataFrame:
    p = slim_dir / f'ntof_hits_{run}_{sub}.parquet'
    h = pd.read_parquet(p, columns=['eventId', 'det', 'detn', 'dt_ns', 'amp',
                                    'area_0', 'satuflag', 'pileup1',
                                    'is_control'])
    lo = min(w[0] for w in WINDOWS.values())
    hi = max(w[1] for w in WINDOWS.values())
    return h[(h.is_control == 0) & (h.dt_ns >= lo) & (h.dt_ns <= hi)]


def wide_hits(h: pd.DataFrame, win: str) -> pd.DataFrame:
    """One row per (event, arm): every channel's largest hit in ``win``."""
    lo, hi = WINDOWS[win]
    x = h[(h.dt_ns >= lo) & (h.dt_ns <= hi)].copy()
    x['arm'] = np.array(ARMS)[x.det.to_numpy() % 4]
    fam = (x.det.to_numpy() // 4) * 4
    pre = np.where(fam == WAL, 'w' + x.detn.astype(str),
                   np.where(fam == PSS, 'p' + x.detn.astype(str), 'l'))
    x['ch'] = pre
    x = (x.sort_values('amp', ascending=False)
          .drop_duplicates(['eventId', 'arm', 'ch']))
    x['amp'] = (x.amp * MV_PER_ADC).astype('float32')
    x['area_0'] = (x.area_0 * MV_PER_ADC).astype('float32')
    val = {'amp': 'amp', 'dt': 'dt_ns', 'sat': 'satuflag',
           'area': 'area_0', 'pu': 'pileup1'}
    parts = []
    for short, col in val.items():
        p = x.pivot_table(index=['eventId', 'arm'], columns='ch', values=col,
                          aggfunc='first')
        p.columns = [f'{c}_{short}_{win}' for c in p.columns]
        parts.append(p)
    W = pd.concat(parts, axis=1)
    keep = [c for c in hit_columns(win) if c in W.columns]
    W = W[keep]
    for c in hit_columns(win):
        if c not in W.columns:
            W[c] = np.nan
    return W[hit_columns(win)].astype('float32').reset_index()


def hw_trigger(W: pd.DataFrame) -> pd.DataFrame:
    """Per event: did each arm satisfy the (emulated) hardware trigger?"""
    w = W.copy()
    sums = np.stack([np.nan_to_num(w[f'w{2 * g + 1}_amp_on'].to_numpy())
                     + np.nan_to_num(w[f'w{2 * g + 2}_amp_on'].to_numpy())
                     for g in range(4)], 1).max(1)
    pl = np.fmax(np.nan_to_num(w.p1_amp_on.to_numpy()),
                 np.nan_to_num(w.p2_amp_on.to_numpy()))
    thw = w.arm.map(WALL_THR).to_numpy(float)
    thp = w.arm.map(PLAS_THR).to_numpy(float)
    w['hw'] = (sums >= thw) & (pl >= thp)
    T = (w.pivot_table(index='event_id', columns='arm', values='hw',
                       aggfunc='max').reindex(columns=list(ARMS))
         .astype('boolean').fillna(False).astype(bool))
    T.columns = [f'hw_{a}' for a in T.columns]
    return T.reset_index()


def extract_subrun(run: str, sub: str, src: Path, slim_dir: Path,
                   geos: dict) -> pd.DataFrame:
    t = read_tracks(src, run, sub)
    if not len(t):
        return t
    h = read_hits(slim_dir, run, sub)
    for win in WINDOWS:
        W = wide_hits(h, win).rename(columns={'eventId': 'event_id'})
        if win == 'on':
            T = hw_trigger(W)
        t = t.merge(W, on=['event_id', 'arm'], how='left')
    t = t.merge(T, on='event_id', how='left')
    for a in ARMS:
        t[f'hw_{a}'] = t[f'hw_{a}'].astype('boolean').fillna(False).astype(bool)
    oth = np.zeros(len(t), bool)
    for a in ARMS:
        oth |= (t.arm.to_numpy() != a) & t[f'hw_{a}'].to_numpy()
    t['other_hw'] = oth
    # global u of the track at the strip plane: x_local is measured from the
    # plane centre, the layers are placed in the structure frame.
    u0 = t.arm.map({a: geos[a]['u_mm'] for a in ARMS}).astype(float)
    t['u_mm'] = (t.x_local + u0).astype('float32')
    t['v_mm'] = t.y_local.astype('float32')
    t = t.drop(columns=['x_local', 'y_local'])
    t['run'] = run
    return t


def extract_run(run: str, src: str, slim: str, out_dir: str) -> tuple:
    """(run, n_tracks, n_subruns, missing, error)."""
    try:
        src, slim, od = Path(src), Path(slim), Path(out_dir)
        subs = sorted(p.name[len(f'tracks_{run}_'):-len('.parquet')]
                      for p in src.glob(f'tracks_{run}_*.parquet'))
        geos = {a: layer_geometry(run, a) for a in ARMS}
        have, missing = [], []
        for s in subs:
            (have if (slim / f'ntof_hits_{run}_{s}.parquet').exists()
             else missing).append(s)
        if missing:
            # A sub-run with tracks and no slim would read as one in which no
            # scintillator ever fired.  Skip it by NAME and say so.
            print(f'  {run}: no slim for {missing}', flush=True)
        T = [extract_subrun(run, s, src, slim, geos) for s in have]
        T = pd.concat([x for x in T if len(x)], ignore_index=True)
        for c in ('arm', 'subrun', 'run'):
            T[c] = T[c].astype('category')
        od.mkdir(parents=True, exist_ok=True)
        T.to_parquet(od / f'stack_{run}.parquet', index=False,
                     compression='snappy')
        return run, len(T), len(have), missing, ''
    except Exception:
        return run, 0, 0, [], traceback.format_exc(limit=5)


def geometry_table(runs) -> pd.DataFrame:
    """Every arm's layer positions per run, and a check that they never move."""
    rows = []
    for r in runs:
        for a in ARMS:
            g = layer_geometry(r, a)
            rows.append(dict(
                run=r, arm=a, u_mm=g['u_mm'], w_strip=g['w_strip'],
                L_wall=g['w_wall'] - g['w_strip'],
                L_plas=g['w_plas'] - g['w_strip'],
                L_ls=g['w_ls'] - g['w_strip'],
                u_ls=g['u_ls'], v_ls=g['v_ls'],
                plas_u_1=g['plas_u'][1], plas_u_2=g['plas_u'][2],
                bar_u=json.dumps([round(g['bar_u'][b], 3)
                                  for b in sorted(g['bar_u'])])))
    G = pd.DataFrame(rows)
    num = [c for c in G.columns if c not in ('run', 'arm', 'bar_u')]
    spread = G.groupby('arm')[num].agg(lambda s: s.max() - s.min()).max(1)
    nbar = G.groupby('arm').bar_u.nunique()
    if (spread > 1e-3).any() or (nbar > 1).any():
        raise ValueError(f'scintillator geometry moves between runs:\n'
                         f'{spread}\n{nbar}\n  split the pass by configuration.')
    return G


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.out('stage3_fullpass')))
    ap.add_argument('--slim', default=str(paths.out('slim')))
    ap.add_argument('--runs', default='')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--include-pre-access', action='store_true')
    a = ap.parse_args()

    src = Path(paths.require(a.src, 'the stage-3 track tables'))
    runs = ([r for r in a.runs.split(',') if r] or
            sorted({p.name.split('_stat')[0].replace('tracks_', '')
                    for p in src.glob('tracks_run_*_*.parquet')},
                   key=run_number))
    if not a.include_pre_access:
        runs = [r for r in runs if r not in PRE_ACCESS_RUNS]
    od = paths.spell('scint')
    (od / 'tracks').mkdir(parents=True, exist_ok=True)
    G = geometry_table(runs)
    G.to_csv(od / 'geometry.csv', index=False)
    print(f'scintillator stack, {len(runs)} runs, geometry identical in all\n')

    t0 = time.time()
    args = (str(src), a.slim, str(od / 'tracks'))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        fut = {ex.submit(extract_run, r, *args): r for r in runs}
        res = []
        for f in as_completed(fut):
            run, n, ns, miss, err = f.result()
            res.append(dict(run=run, n_tracks=n, n_subruns=ns,
                            missing_slim=','.join(miss), error=err))
            msg = (f'FAILED -- {err.strip().splitlines()[-1]}' if err
                   else f'{n:>9,} tracks, {ns} sub-runs')
            print(f'  {run:8s} {msg}   [{time.time() - t0:5.0f} s]', flush=True)
    R = pd.DataFrame(res)
    R['rn'] = R.run.map(run_number)
    R = R.sort_values('rn').drop(columns='rn')
    R.to_csv(od / 'extract_runs.csv', index=False)
    (od / 'extract.meta.json').write_text(json.dumps(dict(
        schema=SCHEMA, windows=WINDOWS, mv_per_adc=MV_PER_ADC,
        n_runs=len(R), n_tracks=int(R.n_tracks.sum()),
        failed=R[R.error != ''].run.tolist()), indent=1))
    return int((R.error != '').any())


if __name__ == '__main__':
    raise SystemExit(main())
