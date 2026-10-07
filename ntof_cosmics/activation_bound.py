#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
activation_bound.py -- is there activation (T1/2 of minutes to hours) in the
beam-off trigger rate after the beam stops?

Per beam-off sub-run at the production point (no n_TOF pulse inside it): the
DREAM trigger rate, the minutes since the last n_TOF proton pulse (per-minute
slow-control `beam_class` logs), and a decay-weighted proton history for each
candidate isotope, sum_i P_i exp(-(t - t_i)/tau).  A real activation component
makes the rate rise with that index.  HANDOFF_TRACKING_2026-10-06.md §10f.

    python ntof_cosmics/activation_bound.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOGS = '/media/dylan/data/x17/beam_july/slow_control/beam_intensity/beam_class_*.csv'
OUT = HERE / 'results' / 'activation'
#: candidate isotopes, half-life in minutes (Ar/iso 90/10 gas, Al/Cu/steel structure)
ISO = {'28Al': 2.24, '66Cu': 5.1, '41Ar': 109.6, '56Mn': 154.6, '24Na': 897.0}


def main() -> int:
    B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(LOGS))])
    B['p'] = B.ntof_beam.fillna(0) * B.ntof_mean_e10.fillna(0)      # 1e10 protons per minute
    B = B[B.p > 0].sort_values('unix_ts')
    tb, pb = B.unix_ts.to_numpy() + 30, B.p.to_numpy()
    d = pd.read_csv(HERE / 'results' / 'cosmic_subruns.csv')
    d = d[(d.at_prod_hv == True) & (d.beam_state == 'none') & d.rate_hz.notna()].copy()  # noqa: E712
    tm = d.t_start.to_numpy() + d.seconds.to_numpy() / 2
    i = np.searchsorted(tb, d.t_start.to_numpy()) - 1
    d['min_since_beam'] = (d.t_start.to_numpy() - tb[i]) / 60
    summ = {}
    for name, th in ISO.items():
        tau = th * 60 / np.log(2)
        d[name] = [np.sum(pb[:k + 1] * np.exp(-(t - tb[:k + 1]) / tau)) for t, k in zip(tm, i)]
        X = np.column_stack([np.ones(len(d)), d[name]])
        c, *_ = np.linalg.lstsq(X, d.rate_hz.to_numpy(), rcond=None)
        summ[name] = dict(half_life_min=th, corr=float(np.corrcoef(d[name], d.rate_hz)[0, 1]),
                          delta_rate_hz=float(c[1] * (d[name].max() - d[name].min())))
        print(f'{name}: corr {summ[name]["corr"]:+.2f}, rate change over the index range {summ[name]["delta_rate_hz"]:+.2f} Hz')
    OUT.mkdir(parents=True, exist_ok=True)
    d[['run', 'subrun', 'min_since_beam', 'rate_hz'] + list(ISO)].to_csv(OUT / 'subruns.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
