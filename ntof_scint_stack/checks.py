#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checks.py -- the two checks behind the sample choices in `ana`.

1. **The trigger emulation** (`extract.WALL_THR`, `PLAS_THR`), from the slim
   hits of every trigger in two sub-runs per run (the first and the middle).
   An arm's thresholds are read off the events no OTHER arm could have
   triggered (other arms below 0.8 x their thresholds): there this arm fired
   the trigger, so its wall-group sum and its plastic must sit above the real
   discriminator, and the low edge of their distributions IS the threshold.
   Also: how many triggers the emulation gives to 0, 1, 2+ arms.  Measured
   2026-10-06: 97.7 % exactly one arm, 2.1 % none, 0.2 % two -- so "another
   arm triggered" is rare because n_TOF triggers are single-arm, not because
   the emulation misses them.

2. **The time cut** (`ana.LATE_MS`).  Tag-and-probe efficiency of the wall,
   the plastic and the liquid in bins of time since the flash, net of
   accidentals with the measured accidental-tag correction, on the good
   single tracks (`ana`'s in-time window and chamber mask).  Below 10 ms the
   corrected numbers have not converged (the wall reads 0.4-0.9 against
   0.92-0.98 past 20 ms) and the bins hold < 15 % of the late statistics, so
   the cut stays where it is.

    python -m ntof_scint_stack.checks            # both; after `ana`
    python -m ntof_scint_stack.checks --only trigger
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from ntof_scint_stack import ana  # noqa: E402
from ntof_scint_stack.extract import (  # noqa: E402
    ARMS, MV_PER_ADC, PLAS_THR, WALL_THR, WINDOWS)

#: other arms below this fraction of their thresholds = this arm triggered
VETO_FRAC = 0.8
#: bins of time since the flash, ms
LATE_EDGES = [0.5, 1, 1.5, 2, 3, 5, 7, 10, 20, 1e9]


def out_dir() -> Path:
    d = paths.spell('scint') / 'checks'
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# 1. trigger
# --------------------------------------------------------------------------- #
def _event_maxima(f: str) -> pd.DataFrame:
    """Per trigger: each arm's largest wall-group SUM and larger plastic, mV,
    in the prompt window (the emulation's own definition)."""
    h = pd.read_parquet(f, columns=['eventId', 'det', 'detn', 'dt_ns', 'amp',
                                    'tof', 'is_control'])
    ev = h.groupby('eventId').tof.min()
    lo, hi = WINDOWS['on']
    h = h[(h.is_control == 0) & (h.dt_ns >= lo) & (h.dt_ns <= hi)]
    h = h.assign(arm=h.det % 4, fam=h.det // 4, a=h.amp * MV_PER_ADC)
    w = (h[h.fam == 0].groupby(['eventId', 'arm', 'detn']).a.max()
         .unstack('detn').reindex(columns=range(1, 9)).fillna(0))
    ws = np.stack([w[2 * g + 1] + w[2 * g + 2] for g in range(4)], 1).max(1)
    W = (pd.Series(ws, index=w.index).unstack('arm')
         .reindex(index=ev.index, columns=range(4)).fillna(0))
    p = (h[h.fam == 1].groupby(['eventId', 'arm']).a.max().unstack('arm')
         .reindex(index=ev.index, columns=range(4)).fillna(0))
    out = pd.DataFrame({'t_ms': ev.to_numpy() / 1e6}, index=ev.index)
    for i, a in enumerate(ARMS):
        out[f'w{a}'] = W[i].to_numpy()
        out[f'p{a}'] = p[i].to_numpy()
    out['run'] = Path(f).name.split('_')[2] + '_' + Path(f).name.split('_')[3]
    return out.reset_index(drop=True)


def trigger(jobs: int = 8) -> tuple:
    runs = pd.read_csv(paths.spell('scint') / 'extract_runs.csv').run
    slim = paths.out('slim')
    fs = []
    for r in runs:
        g = sorted(glob.glob(str(slim / f'ntof_hits_{r}_stat*_[0-9][0-9][0-9][0-9].parquet')))
        fs += g[:1] + ([g[len(g) // 2]] if len(g) > 2 else [])
    with ProcessPoolExecutor(jobs) as ex:
        D = pd.concat(list(ex.map(_event_maxima, fs)), ignore_index=True)
    hw = {a: (D[f'w{a}'] >= WALL_THR[a]) & (D[f'p{a}'] >= PLAS_THR[a])
          for a in ARMS}
    loose = {a: (D[f'w{a}'] >= VETO_FRAC * WALL_THR[a])
             & (D[f'p{a}'] >= VETO_FRAC * PLAS_THR[a]) for a in ARMS}
    n = sum(hw[a].astype(int) for a in ARMS)
    late = D.t_ms > ana.LATE_MS
    S = []
    for lab, m in (('all', np.ones(len(D), bool)), ('late', late),
                   ('early', ~late)):
        vc = n[m].value_counts(normalize=True)
        S.append(dict(sample=lab, n_triggers=int(m.sum()),
                      **{f'frac_{k}_arms': float(vc.get(k, 0.0))
                         for k in range(3)},
                      frac_3plus_arms=float(vc[vc.index >= 3].sum())))
    E = []
    for a in ARMS:
        alone = sum(loose[b].astype(int) for b in ARMS if b != a) == 0
        for run, x in [('campaign', D[alone])] + list(D[alone].groupby('run')):
            w = x[f'w{a}'][x[f'p{a}'] >= PLAS_THR[a]]
            p = x[f'p{a}'][x[f'w{a}'] >= WALL_THR[a]]
            E.append(dict(run=run, arm=a, n=len(x),
                          pass_frac=float(hw[a][x.index].mean()),
                          wall_thr=WALL_THR[a], plas_thr=PLAS_THR[a],
                          **{f'wall_q{q}': float(np.percentile(w, q))
                             for q in (0.5, 1, 2)},
                          **{f'plas_q{q}': float(np.percentile(p, q))
                             for q in (0.5, 1, 2)}))
    S, E = pd.DataFrame(S), pd.DataFrame(E)
    S.to_csv(out_dir() / 'trigger_arms.csv', index=False)
    E.to_csv(out_dir() / 'trigger_edges.csv', index=False)
    return S, E, len(fs)


# --------------------------------------------------------------------------- #
# 2. time cut
# --------------------------------------------------------------------------- #
def late_scan(arm: str) -> pd.DataFrame:
    meta = json.loads((ana.out_dir() / 'ana' / 'meta.json').read_text())
    t = ana.load_arm(arm)
    g = ana.geometry(arm)
    cal = meta['cal'][arm]
    sig = {'wall_u': cal['sig_w'], 'plas_u': cal['sig_p']}
    P = ana.predict(t, g, cal, sig, arm)
    _, (lo, hi) = ana.t0_window(t, P, sig)
    M = pd.read_parquet(ana.out_dir() / 'ana' / f'mask_{arm}.parquet')
    good = (((t.t0 >= lo) & (t.t0 < hi)).to_numpy() & ana.apply_mask(t, M)
            & (t.n_trk == 1).to_numpy())
    A = lambda c: P[c].to_numpy()  # noqa: E731
    in_w = A('on_w') & (A('d_edge_w') > 2 * sig['wall_u'])
    in_p = A('on_p') & (A('d_edge_p') > ana.PLAS_MARGIN)
    in_l = A('on_l') & (A('d_edge_l') > 40)
    tms = t.t_since_flash_ns.to_numpy() / 1e6
    rows = []
    for lo_, hi_ in zip(LATE_EDGES[:-1], LATE_EDGES[1:]):
        b = good & (tms >= lo_) & (tms < hi_)
        for samp, sm in (('all', b), ('unbiased', b & t.other_hw.to_numpy())):
            for lay, par, ton, toff, pon, poff in (
                    ('wall', sm & in_w & in_p, A('pm_on'), A('pm_off'),
                     A('wany_on'), A('wany_off')),
                    ('plas', sm & in_w & in_p, A('wboth_on'), A('wboth_off'),
                     A('pm_on'), A('pm_off')),
                    ('liq', sm & in_w & in_l, A('wboth_on'), A('wboth_off'),
                     A('lf_on'), A('lf_off'))):
                if not par.any():
                    continue
                r, *_ = ana.tag_probe(par, ton, toff, pon, poff)
                rows.append(dict(arm=arm, lo_ms=lo_, hi_ms=hi_, sample=samp,
                                 layer=lay, **{k: r[k] for k in (
                                     'n', 'off', 'c', 'eff_raw', 'eff',
                                     'err')}))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', choices=('trigger', 'late'))
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    if a.only in (None, 'trigger'):
        S, E, nf = trigger(2 * a.jobs)
        print(f'trigger emulation, {nf} sub-runs, {S.n_triggers.iloc[0]:,} triggers')
        print(S.round(4).to_string(index=False))
        print(E[E.run == 'campaign'].round(1).to_string(index=False))
    if a.only in (None, 'late'):
        with ProcessPoolExecutor(min(a.jobs, 4)) as ex:
            L = pd.concat(list(ex.map(late_scan, ARMS)), ignore_index=True)
        L.to_csv(out_dir() / 'late_scan.csv', index=False)
        print(L[(L['sample'] == 'all') & (L.layer == 'wall')]
              .pivot(index='lo_ms', columns='arm', values='eff').round(3))
    return 0


if __name__ == '__main__':
    sys.exit(main())
