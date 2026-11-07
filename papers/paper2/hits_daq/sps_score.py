#!/usr/bin/env python3
"""sps_score.py --tables DIR [--rows k1,k2] [--stage 25.4] [--json OUT] -- the SPS det4 known-angle score (R7) of the
hits families against the waveform fit, on the same planes, with no reference track.

Inputs: the analysis-DAG tables of one SPS row, tiers_sps_<key>.parquet (u, ur, ue, uq per family; w_W) and
h2c_sps_<key>_*of*.parquet (w_h2c per family).  All slopes are dx/dt in mm/ns.

The rot25 rows' Y view is the inclined one (det4 on the 25.4 deg stage; |tan| only, COORDINATES.md 4b); X is ~normal.
The CO2 rows have no column-duration v (rerun/paper1_spsfigs: the column runs out of the 64-sample window), so v is set
per row from the stage angle through the waveform fit: v_stage = median|w_W| / tan(stage) on the inclined view.  That
makes W the calibration on these rows (its own known-angle check, 0.981 +- 0.03 of tan 25.4 on the 13 CF4 rows with a
column v, is Paper I's); what this scores is the hits families against W on the same planes:
  ratio      median of u / w_W per plane (inclined view): > 1 = the hits ladder reads steeper than W (the compression,
             no v and no reference involved)
  tan error  tan = slope / v_stage against sign(w_W) * tan(stage): s68 / steep / in5 with tiers.score
On the X view (normal incidence) the reference is tan 0, with the same v_stage.  The 0 deg row (run55 flatdrift) has no
inclined view: there v = the window meta's v, quoted only for the head-on s68.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiers import score  # noqa: E402

FAMS = ('dream', 'peak', 'thr', 'tot', 'fd', 'binary', 'ideal')
CODES = dict(dream='V0', peak='V1', thr='V2', tot='V3', fd='V4', binary='V5', ideal='V6')


def load(d, key):
    T = pd.read_parquet(os.path.join(d, f'tiers_sps_{key}.parquet'))
    H = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(d, f'h2c_sps_{key}_*of*.parquet')))],
                  ignore_index=True)
    keep = ['event_id', 'view'] + [f'w_h2c_{f}' for f in FAMS if f'w_h2c_{f}' in H]
    return T.merge(H[keep], on=['event_id', 'view'], how='left')


def row_score(T, stage_deg):
    ts = np.tan(np.radians(stage_deg))
    inc = T[T.view == 'y']
    rot = stage_deg > 0
    v = float(np.nanmedian(np.abs(inc.w_W)) / ts) if rot else float(T.v_drift.iloc[0]) / 1e3
    out = dict(inclined=bool(rot), v_mm_ns=v, views={})
    for view, g in T.groupby('view'):
        w = g.w_W.to_numpy(float)
        tr = np.sign(w) * ts if (rot and view == 'y') else np.zeros(len(g))
        V = dict(n=len(g), W=score(w / v, tr), tiers={})
        for f in FAMS:
            for suf, col in (('', f'u_{f}'), ('r', f'ur_{f}'), ('e', f'ue_{f}'), ('q', f'uq_{f}'), ('/H2c', f'w_h2c_{f}')):
                if col not in g:
                    continue
                s = g[col].to_numpy(float)
                o = dict(score=score(s / v, tr), coverage=float(np.isfinite(s).mean()))
                if rot and view == 'y':
                    k = np.isfinite(s) & np.isfinite(w) & (np.abs(w) > 0)
                    o['ratio'] = float(np.median(s[k] / w[k])) if k.sum() > 30 else None
                V['tiers'][f'{CODES[f]}{suf}_{f}'] = o
        out['views'][view] = V
    return out


def fmt(x, p='.2f'):
    return format(x, p) if x is not None else '  - '


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', required=True)
    ap.add_argument('--rows', help='comma list of row keys (default: every tiers_sps_* in --tables)')
    ap.add_argument('--stage', type=float, default=25.4, help='stage angle of the rot25 rows [deg] (measured yaw)')
    ap.add_argument('--json')
    a = ap.parse_args()
    keys = a.rows.split(',') if a.rows else sorted(
        os.path.basename(f)[len('tiers_sps_'):-len('.parquet')] for f in glob.glob(os.path.join(a.tables, 'tiers_sps_*.parquet')))
    res = dict(stage_deg=a.stage, rows={})
    for key in keys:
        if not glob.glob(os.path.join(a.tables, f'h2c_sps_{key}_*of*.parquet')):
            print(f'[sps] {key}: no H2c tables yet, skipped')
            continue
        r = row_score(load(a.tables, key), a.stage if 'rot25' in key or 'run59' in key else 0.0)
        res['rows'][key] = r
        print(f'== {key}  inclined {r["inclined"]}  v {r["v_mm_ns"] * 1e3:.1f} um/ns')
        for view, V in r['views'].items():
            W = V['W']['all']
            print(f'  [{view}] n {V["n"]}  W s68 {fmt(W["s68"])} in5 {fmt(W["frac_in5"], ".3f")}')
            for name, t in V['tiers'].items():
                if name[2:3] in ('e', 'q'):
                    continue
                s = t['score']['all']
                print(f'     {name:14s} ratio {fmt(t.get("ratio"), ".3f")}  s68 {fmt(s["s68"])}  steep {fmt(s["steep"], "+.2f")}'
                      f'  in5 {fmt(s["frac_in5"], ".3f")}  cov {t["coverage"]:.2f}')
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
