#!/usr/bin/env python3
"""
column_ledger.py — drift-column time and charge-arrival shape for every June
cosmic sub-run, from the r06 waveform reconstruction.

Input: the r06 condor campaign outputs (one events.parquet per run/sub-run/
detector), pulled from lxplus to
    /media/dylan/data/x17/cosmic_bench/condor_campaign_r06/out/
and its campaign_manifest.csv (HV, slot, FEUs per sub-run).

Per (sub-run, detector, plane) it records, for well-reconstructed inclined
tracks (ok, quality_ok, slope_reliable, not isochronous, 0.08 <= |tan| <= 0.40):

  u50, u90, uend   medians of the per-track charge-arrival quantiles [ns after
                   the fitted t0]. These are times on the fit's 60 ns depth
                   basis; they never use v, so they are a pure timing measure.
  T90              = u90 / 0.9, the column time if charge is uniform along it
  r5090            = median(u50 / u90): 0.556 for a uniform column; attachment
                   (charge falling with depth) pulls it down
  rail             fraction of tracks whose profile reaches the end of the basis
                   (column longer than the basis, T90 then biased low)

Everything here is a timing quantity from the waveform fit; no combined_hits
time is used (RECONSTRUCTION_BASIS.md).

    ../../.venv/bin/python mx_june_wft/drift_consistency/column_ledger.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

R06 = '/media/dylan/data/x17/cosmic_bench/condor_campaign_r06'
OUT = '/media/dylan/data/x17/cosmic_bench/Analysis/drift_consistency'
TAN_LO, TAN_HI = 0.08, 0.40
MIN_TRACKS = 40


def parse_key(name):
    parts = name.split('__')
    return dict(run=parts[1], subrun=parts[2], det=parts[3],
                arm=parts[4] if len(parts) > 4 else 'prod')


def date_of(run):
    tok = [t for t in run.split('_') if t.count('-') == 2]
    return tok[0] if tok else ''


def summarise(df, plane, basis_end):
    p = plane + '_'
    sel = (df[p + 'ok'] & df[p + 'quality_ok'] & df[p + 'slope_reliable']
           & ~df[p + 'isochronous'])
    t = df[p + 'tan_theta'].abs()
    s = df[sel & (t >= TAN_LO) & (t <= TAN_HI)]
    n = len(s)
    if n < MIN_TRACKS:
        return dict(n=n)
    u50, u90, uend = (s[p + k].to_numpy() for k in ('q_u50', 'q_u90', 'q_uend'))
    boot = np.random.default_rng(1).choice(u90, (200, n))
    return dict(n=n, u50=np.median(u50), u90=np.median(u90),
                u90_err=np.std(np.median(boot, axis=1)),
                uend=np.median(uend), T90=np.median(u90) / 0.9,
                r5090=np.median(u50 / u90),
                rail=float(np.mean(uend >= basis_end - 1)),
                chi2dof=float(np.median(s[p + 'chi2'] / s[p + 'dof'].clip(1))))


def main():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(R06, 'campaign_manifest.csv'))
    man_idx = man.set_index(['run', 'subrun', 'det'])
    rows = []
    for d in sorted(glob.glob(os.path.join(R06, 'out', 'c26__*'))):
        ev = os.path.join(d, 'events.parquet')
        if not os.path.exists(ev):
            continue
        k = parse_key(os.path.basename(d))
        meta = json.load(open(os.path.join(d, 'events.meta.json')))
        b = meta.get('bundle', {})
        df = pd.read_parquet(ev)
        basis_end = float(np.nanmax(df[['x_q_uend', 'y_q_uend']].to_numpy()))
        try:
            m = man_idx.loc[(k['run'], k['subrun'], k['det'])]
            m = m.iloc[0] if isinstance(m, pd.DataFrame) else m
            hv = dict(drift_V=m.drift_V, resist_V=m.resist_V, det_z=m.det_z,
                      feu_x=m.feu_x, gas=m.gas)
        except KeyError:
            hv = {}
        for plane in ('x', 'y'):
            rows.append(dict(**k, date=date_of(k['run']), plane=plane, **hv,
                             bundle_v=b.get('v_drift'), basis_end=basis_end,
                             **summarise(df, plane, basis_end)))
    led = pd.DataFrame(rows)
    led['slot'] = led['det_z'].map({232.0: 'bottom', 702.0: 'top'})
    out = os.path.join(OUT, 'column_ledger.csv')
    led.to_csv(out, index=False)
    print(f'wrote {out} ({len(led)} rows)')


if __name__ == '__main__':
    main()
