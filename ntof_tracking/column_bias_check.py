#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
column_bias_check.py -- is the fitted charge column longer than the data?

The preliminary run_79 pass left three mutually inconsistent handles on
v_drift for arm A:

    hits-level column span      ~43 um/ns   (fast, ~ clean-Magboltz 90/10)
    fitted profile end q_uend   ~36-39 um/ns
    2 x fitted median arrival   ~28-29 um/ns
    target-pointing slope       ~0.67 of nominal, i.e. v ~ 28.5 if nothing
                                else is biased

They cannot all be right. The question this script answers is whether the
FITTED charge profile is a trustworthy ruler at all, and it answers it the
only way that is not circular: by running the same comparison on the BENCH,
where v (36.6 um/ns) and the drift gap (27.9 mm, measured) are known, and
where the window is 32 samples so nothing is truncated.

Per (event, plane) it compares two numbers that should track each other:

    fit    :  t0 + q_uend + t_peak(template)   -- when the deepest charge's
              pulse should PEAK, in ns from the start of the window
    hits   :  max_sample of the deepest strip in the seed cluster, x sample_ns

If the fit's column ends where the hits say it does, the difference is ~0 and
q_uend is a ruler. A large positive bias on the bench means the estimator
over-runs by construction (NNLS parking un-modelled charge in late depth bins)
and the run_79 numbers derived from it have to be corrected by that amount --
which is exactly what decides whether run_79's gas is fast or slow.

Usage:
    python -m ntof_tracking.column_bias_check bench
    python -m ntof_tracking.column_bias_check beam
    python -m ntof_tracking.column_bias_check both
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BENCH = dict(
    tag='bench det3 (sat_det3, 32 samples, v=36.6, gap 27.9 mm)',
    events='/media/dylan/data/x17/cosmic_bench/Analysis/'
           'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/'
           'mx17_3/wft/events.parquet',
    bundle='/media/dylan/data/x17/cosmic_bench/Analysis/'
           'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/'
           'mx17_3/wft/calib_bundle_lp2',
    hits_dir='/media/dylan/data/x17/cosmic_bench/det3/'
             'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/'
             'combined_hits_root/',
    feu_x=7, feu_y=8, v_true=36.6, gap_mm=27.9, n_sample=32)

BEAM = dict(
    tag='run_79 mx17_A (20 samples, v assumed 42.6, gap 27.9 mm assumed)',
    events='/media/dylan/data/x17/beam_july/analysis/wft/run_79/stat090_0000/'
           'mx17_A/events_prelim.parquet',
    bundle='/media/dylan/data/x17/beam_july/analysis/wft/run_79/stat090_0000/'
           'mx17_A/calib_bundle_prelim',
    hits_dir='/media/dylan/data/x17/beam_july/runs/run_79/stat090_0000/'
             'combined_hits_root/',
    feu_x=3, feu_y=4, v_true=None, gap_mm=27.9, n_sample=20)


def template_peak_ns(bundle_path: str) -> dict:
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(bundle_path)
    return {p: float(cal.grid[int(np.argmax(cal.tmpl[p]))]) for p in ('x', 'y')}, cal


def hits_column(hits_dir: str, feus, n_files=None) -> pd.DataFrame:
    """Per (event, plane): first and last peak sample of the largest cluster."""
    import uproot
    from wft import seed as wseed
    files = sorted(f for f in os.listdir(hits_dir)
                   if f.endswith('.root') and '_datrun_' in f)
    if n_files:
        files = files[:n_files]
    rows = []
    for fn in files:
        df = uproot.open(os.path.join(hits_dir, fn))['hits'].arrays(
            ['eventId', 'feu', 'channel', 'amplitude', 'significance',
             'max_sample'], library='pd')
        df = df[df['feu'].isin(list(feus))]
        # largest pulse per channel: at beam a channel can carry several hits
        df = df.sort_values('amplitude').drop_duplicates(
            ['eventId', 'feu', 'channel'], keep='last')
        df = wseed.apply_significance_floor(df, wseed.SIG_REL_FLOOR)
        for (eid, feu), g in df.groupby(['eventId', 'feu'], sort=False):
            if len(g) < 5:
                continue
            m = g['max_sample'].to_numpy()
            rows.append(dict(event_id=int(eid), feu=int(feu), n=len(g),
                             onset=float(np.nanmin(m)), edge=float(np.nanmax(m))))
    return pd.DataFrame(rows)


def run(cfg: dict, n_files: int = 1) -> dict:
    peaks, cal = template_peak_ns(cfg['bundle'])
    ev = pd.read_parquet(cfg['events'])
    hc = hits_column(cfg['hits_dir'], (cfg['feu_x'], cfg['feu_y']), n_files)
    if not len(hc):
        raise SystemExit(f'no hit columns found in {cfg["hits_dir"]} for FEUs '
                         f'{cfg["feu_x"]}/{cfg["feu_y"]} -- wrong FEU pair?')
    sns = float(cal.sample_ns)
    k_end = cal.n_depth_bins * sns          # the last basis bin: q_uend rails here
    print(f"\n=== {cfg['tag']} ===")
    print(f'    {len(ev):,} reconstructed events, {len(hc):,} hit columns, '
          f'basis {cal.n_depth_bins} x {sns:.0f} = {k_end:.0f} ns')
    out = {}
    for plane, feu in (('x', cfg['feu_x']), ('y', cfg['feu_y'])):
        h = hc[hc['feu'] == feu].set_index('event_id')
        d = ev[ev[f'{plane}_ok'] & ev[f'{plane}_quality_ok']].set_index('event_id')
        j = d.join(h, how='inner', rsuffix='_h')
        if len(j) < 50:
            print(f'    {plane}: only {len(j)} joined events, skipped')
            continue
        railed = j[f'{plane}_q_uend'] >= k_end - 30
        # where the fit says the deepest charge's pulse peaks, in ns
        t_fit = j[f'{plane}_t0'] + j[f'{plane}_q_uend'] + peaks[plane]
        t_hit = j['edge'] * sns
        # and where it says the FIRST charge peaks (a t0 cross-check that does
        # not depend on the column length at all)
        t_fit0 = j[f'{plane}_t0'] + peaks[plane]
        t_hit0 = j['onset'] * sns
        d_end = (t_fit - t_hit)[~railed]
        d_on = t_fit0 - t_hit0
        col_fit = j.loc[~railed, f'{plane}_q_uend']
        col_hit = (j.loc[~railed, 'edge'] - j.loc[~railed, 'onset']) * sns
        res = dict(
            n=int(len(j)), railed_frac=float(railed.mean()),
            onset_bias_ns=float(np.median(d_on)),
            end_bias_ns=float(np.median(d_end)),
            col_fit_ns=float(np.median(col_fit)),
            col_hit_ns=float(np.median(col_hit)),
            v_from_col_fit=float(cfg['gap_mm'] * 1e3 / np.median(col_fit)),
            v_from_col_hit=float(cfg['gap_mm'] * 1e3 / np.median(col_hit)))
        out[plane] = res
        print(f'    {plane}: n={res["n"]:,}  railed {res["railed_frac"]:.0%}   '
              f'onset bias {res["onset_bias_ns"]:+.0f} ns   '
              f'column end bias {res["end_bias_ns"]:+.0f} ns')
        print(f'        column length: fit {res["col_fit_ns"]:.0f} ns  '
              f'-> v {res["v_from_col_fit"]:.1f}    '
              f'hits {res["col_hit_ns"]:.0f} ns -> v {res["v_from_col_hit"]:.1f} '
              f'um/ns   (hit span is a LOWER bound: thresholds cut both ends)')
        if cfg['v_true']:
            print(f'        truth v = {cfg["v_true"]:.1f} um/ns -> the fit ruler '
                  f'reads {res["v_from_col_fit"] / cfg["v_true"]:.2f}x, '
                  f'the hits ruler {res["v_from_col_hit"] / cfg["v_true"]:.2f}x')
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    res = {}
    if which in ('bench', 'both'):
        res['bench'] = run(BENCH, nf)
    if which in ('beam', 'both'):
        res['beam'] = run(BEAM, nf)
    if 'bench' in res and 'beam' in res:
        print('\n=== transfer: correct the beam ruler by the bench bias ===')
        for plane in ('x', 'y'):
            b, m = res['bench'].get(plane), res['beam'].get(plane)
            if not b or not m:
                continue
            # the bench tells us the fit's column is (fit/true) too long; apply
            # the same factor to the beam column before turning it into v
            f = b['v_from_col_fit'] / BENCH['v_true']
            print(f'    {plane}: beam v_raw {m["v_from_col_fit"]:.1f} / bench '
                  f'ruler factor {f:.2f} -> corrected v '
                  f'{m["v_from_col_fit"] / f:.1f} um/ns')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
