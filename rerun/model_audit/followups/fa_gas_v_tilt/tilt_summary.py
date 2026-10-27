#!/usr/bin/env python3
"""tilt_summary.py -- FA item 3: group tilt_rows.json by chamber and run (mounting candidates), test constancy.

A "run" (the directory date) is the finest mounting unit the registry records: chambers were moved between runs, never
inside one.  Per chamber: the weighted tilt per run (angle and position routes, X and Y), chi2 of the rows of one run
about their mean (constant within a mounting?), chi2 of the run means about the chamber mean (does a change of run move
it?), and the curvature (bow) per row and its chi2 about one chamber value (static per chamber?).
Writes results/tilt_summary.json and results/tilt_table.md.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_common import RES  # noqa: E402

SLOT = {232.0: 'bottom', 702.0: 'top'}


def wmean(v, e):
    v, e = np.asarray(v, float), np.asarray(e, float)
    m = np.isfinite(v) & np.isfinite(e) & (e > 0)
    if not m.any():
        return np.nan, np.nan, np.nan, 0
    w = 1 / e[m] ** 2; mu = np.sum(w * v[m]) / w.sum()
    return float(mu), float(1 / np.sqrt(w.sum())), float(np.sum(w * (v[m] - mu) ** 2)), int(m.sum() - 1)


def main():
    rows = [r for r in json.load(open(os.path.join(RES, 'tilt_rows.json'))) if 'failed' not in r and r['n_events'] >= 150]
    for r in rows:                       # an angle route without a working ladder is not a measurement (tilt.py)
        if not r.get('angle_valid', True):
            for p in 'xy':
                for k in ('angle_deg', 'angle_err', 'curv_angle_per_km', 'curv_angle_err'):
                    r[f'tilt_{p}'][k] = float('nan')
    out = {}
    md = ['| chamber | run | subrun | slot | drift V | resist V | n | X tilt angle [deg] | X tilt position | Y tilt angle | '
          'Y tilt position | curv X [/km] angle / pos | curv Y [/km] angle / pos |', '|' + '---|' * 13]
    for det in sorted({r['det'] for r in rows}):
        R = sorted([r for r in rows if r['det'] == det], key=lambda r: (tuple(int(x) for x in r['date'].split('-')[:2]), r['subrun']))
        runs = {}
        for r in R:
            runs.setdefault(r['run'], []).append(r)
            md.append(f"| {det} | {r['run'].replace('mx17_', '')} | {r['subrun'][:38]} | {SLOT.get(r['slot_z'], '?')} | "
                      f"{r['drift_V']:.0f} | {r['resist_V']:.0f} | {r['n_events']} | "
                      + ' | '.join(f"{r[f'tilt_{p}'][k + '_deg']:+.2f} ± {r[f'tilt_{p}'][k + '_err']:.2f}" for p in 'xy' for k in ('angle', 'pos'))
                      + ' | ' + ' | '.join(f"{r[f'tilt_{p}']['curv_angle_per_km']:+.0f} / {r[f'tilt_{p}']['curv_pos_per_km']:+.0f}" for p in 'xy') + ' |')
        D = dict(runs={})
        for key in ('angle', 'pos'):
            for p in 'xy':
                per_run = []
                for run, rr in runs.items():
                    mu, e, chi, ndf = wmean([r[f'tilt_{p}'][f'{key}_deg'] for r in rr], [r[f'tilt_{p}'][f'{key}_err'] for r in rr])
                    D['runs'].setdefault(run, dict(slot=SLOT.get(rr[0]['slot_z'], '?'), n_rows=len(rr)))[f'{key}_{p}'] = \
                        dict(mean=mu, err=e, chi2=chi, ndf=ndf)
                    per_run.append((mu, e))
                mu, e, chi, ndf = wmean(*zip(*per_run))
                D[f'between_runs_{key}_{p}'] = dict(mean=mu, err=e, chi2=chi, ndf=ndf)
                within = [D['runs'][run][f'{key}_{p}'] for run in runs]
                D[f'within_runs_{key}_{p}'] = dict(chi2=float(np.nansum([w['chi2'] for w in within])), ndf=int(sum(w['ndf'] for w in within if np.isfinite(w['chi2']))))
            for p in 'xy':
                mu, e, chi, ndf = wmean([r[f'tilt_{p}'][f'curv_{key}_per_km'] for r in R], [r[f'tilt_{p}'][f'curv_{key}_err'] for r in R])
                D[f'curv_{key}_{p}'] = dict(mean=mu, err=e, chi2=chi, ndf=ndf)
        out[det] = D
        print(det, ' '.join(f"{k}: chi2 {D[k]['chi2']:.0f}/{D[k]['ndf']}" for k in D if k != 'runs'))
    json.dump(out, open(os.path.join(RES, 'tilt_summary.json'), 'w'), indent=1, default=float)
    open(os.path.join(RES, 'tilt_table.md'), 'w').write('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
