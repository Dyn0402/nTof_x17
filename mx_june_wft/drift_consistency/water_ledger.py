#!/usr/bin/env python3
"""
water_ledger.py — drift velocity and implied water for every June cosmic
sub-run, on one time axis, with one calibration.

Inputs (both written by sibling scripts / pulled from EOS):
  column_ledger.csv        per sub-run column-time quantiles (column_ledger.py)
  hv/<run>/<sub>/hv_monitor.csv, hv/<run>/run_config.json
                           every bench HV log, pulled from EOS june_tests

Method
------
T_col = C * T90, where T90 = median(u90)/0.9 from the waveform fit. C is fixed
once, on det3 at 6-27 (top slot, 1000 V): there the M3-anchored geometric
velocity is 36.6 um/ns and the charge-visible column is 27.9 mm
(GAP_STUDY_2026-07-30.md as amended), i.e. T_col = 762 ns. The same C is
applied to every detector -- the reconstruction and the column shape are the
same everywhere, so C is a property of the u90 estimator, not of a chamber.

v = gap / T_col, with each chamber's own measured column (ANALYSIS_STATE
2026-07-31 sec. 10.4: det2 30.6, det3 27.9, det6 27.85, det7 27.5; det4's
column is unusable, so 30 is used with that caveat). E = V_drift / gap from
the logged vmon of the chamber's own drift channel (run_config hv_channels).

Water is read off the Ar/iso 95/5 + H2O Magboltz tables at 745.8 Torr (the
bench) by log-v interpolation in water fraction at that E.

Sub-runs whose profiles rail on the 1080 ns basis (rail > 0.3) give only a
lower bound on T_col, hence a lower bound on water; they are flagged.

    ../../.venv/bin/python mx_june_wft/drift_consistency/water_ledger.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = '/media/dylan/data/x17/cosmic_bench/Analysis/drift_consistency'
GARF = os.path.join(REPO, 'garfield_sim', 'results')

GAP_MM = {'mx17_2': 30.6, 'mx17_3': 27.9, 'mx17_6': 27.85, 'mx17_7': 27.5,
          'mx17_4': 30.0, 'mx17_1': 30.0}
ANCHOR = dict(det='mx17_3', run='mx17_det3_saturday_scan_6-27-26',
              v=36.6, gap=27.9)
RAIL_MAX = 0.3


def water_tables():
    """{water %: (E, v)} for Ar/iso 95/5 + H2O, bench pressure, no N2/O2."""
    tabs = {}
    def add(w, rows):
        E = np.array([r['E_Vcm'] for r in rows])
        v = np.array([r['v_um_per_ns'] for r in rows])
        tabs.setdefault(round(w, 3), (E, v))
    for f in ('water_grid.json', 'water_grid2.json',
              'drift_velocity_candidates.json',
              'drift_velocity_candidates2.json'):
        d = json.load(open(os.path.join(GARF, f)))
        for k, rows in d['mixtures'].items():
            m = re.fullmatch(r'Ar(?:\d+(?:\.\d+)?)?_iso5_H2O([\d.]+)', k)
            if m:
                add(float(m.group(1)), rows)
    d = json.load(open(os.path.join(GARF, 'water2d.json')))
    for k, rows in d['mixtures'].items():
        m = re.fullmatch(r'iso5\.0_h2o([\d.]+)_n20\.0_o20\.0', k)
        if m:
            add(float(m.group(1)), rows)
    d = json.load(open(os.path.join(GARF, 'drift_velocity_Ar_iC4H10_95_5_Saclay.json')))
    add(0.0, d['points'])
    return dict(sorted(tabs.items()))


def invert_water(v, E, tabs):
    ws = np.array(list(tabs))
    vs = np.array([np.interp(E, *tabs[w]) for w in ws])
    o = np.argsort(vs)
    if v > vs.max():
        return 0.0
    if v < vs.min():
        return np.inf
    return float(np.interp(np.log(v), np.log(vs[o]), ws[o]))


def drift_channel(run_dir, det):
    cfg = json.load(open(os.path.join(run_dir, 'run_config.json')))
    for d in cfg.get('detectors', []):
        if d.get('name') == det:
            c = d['hv_channels']['drift']
            return f'{c[0]}:{c[1]}'
    return None


def hv_row(run, sub, det):
    run_dir = os.path.join(OUT, 'hv', run)
    f = os.path.join(run_dir, sub, 'hv_monitor.csv')
    if not os.path.exists(f):
        return {}
    ch = drift_channel(run_dir, det)
    d = pd.read_csv(f)
    s = d.iloc[len(d) // 5:]
    ts = pd.to_datetime(d['timestamp'])
    out = dict(t_start=ts.iloc[0], t_mid=ts.iloc[0] + (ts.iloc[-1] - ts.iloc[0]) / 2,
               drift_ch=ch)
    if ch and f'{ch} vmon' in d:
        out.update(vmon=float(s[f'{ch} vmon'].median()),
                   imon=float(s[f'{ch} imon'].median()))
    return out


def main():
    led = pd.read_csv(os.path.join(OUT, 'column_ledger.csv'))
    led = led[(led.plane == 'x') & led.u90.notna()].copy()
    hv = [hv_row(r.run, r.subrun, r.det) for r in led.itertuples()]
    led = pd.concat([led.reset_index(drop=True), pd.DataFrame(hv)], axis=1)

    a = led[(led.det == ANCHOR['det']) & (led.run == ANCHOR['run'])
            & (led.rail < RAIL_MAX)]
    T_anchor = ANCHOR['gap'] / ANCHOR['v'] * 1e3
    C = T_anchor / np.average(a.T90, weights=a.n)
    print(f'C = {C:.4f}  (anchor T_col {T_anchor:.0f} ns over {len(a)} sub-runs)')

    tabs = water_tables()
    print('water tables (%):', list(tabs))
    led['T_col'] = C * led.T90
    led['gap'] = led.det.map(GAP_MM)
    led['v'] = led.gap / led.T_col * 1e3
    led['E'] = led.vmon / (led.gap / 10.0)
    led['h2o'] = [invert_water(v, E, tabs) if np.isfinite(E) else np.nan
                  for v, E in zip(led.v, led.E)]
    led['lower_bound'] = led.rail >= RAIL_MAX
    led['ohm_GOhm'] = led.vmon / led.imon / 1e3
    led.to_csv(os.path.join(OUT, 'water_ledger.csv'), index=False)
    print(f'wrote {os.path.join(OUT, "water_ledger.csv")}')


if __name__ == '__main__':
    main()
