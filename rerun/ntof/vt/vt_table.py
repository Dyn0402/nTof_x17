#!/usr/bin/env python3
"""vt_table.py -- P7a: drift velocity v(t) per (acquisition, arm) from the vt_acq.py outputs; vt_<arm>.parquet.

    vt_table.py --vt DIR [--out DIR] [--boot 60]

Per (run, sub-run, tag, arm, view): the events vt_acq kept (the pass's clean single tracks with one colsum cluster),
minus saturated ones and those with < 5 pulse strips; each strip sum normalised to its maximum; the median stack fitted
with colsum.GridFit (the frozen bundle's X template, T grid 5 ns), T = box duration; error = bootstrap rms (--boot
resamples of the events); v = gap / T with the bundle's timing-free gap (A 28.08 ± 0.19, C 28.0 (no error), D
29.44 ± 1.12 mm).  The two views are combined inverse-variance.  This is the recipe of the 11-02 in-situ v (plan §6)
and of rerun/ntof/tests/d_b/bd_cosmic.py, applied per acquisition.

Columns: run, subrun, tag, arm, trigger (beam / cosmic), t_start, t_end (UTC from the registry's acquisition start plus
the DREAM timestamps, 10 ns ticks), n_x, n_y, T_x, T_y (+ _err), v_x, v_y (+ _err), v, v_err (statistical; the gap
error is a common scale, `gap_err_rel`), v_bundle, hv (resist/drift of that arm), hv_change (the HV differs from the
arm's previous acquisition), and t0 of the stack.

Caveats carried into the sidecar: T is gain-confounded (+18 % across a det3 mesh scan), so v(t) is quoted at fixed HV
and flagged across an HV change; beam and cosmic samples differ by a selection systematic (~3 % on A), so the two
trigger classes are kept apart; the 1.2 us window can clip slow columns (C closest).
"""
import argparse, csv, glob, json, os, sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fa_gas_v_tilt')]
import colsum   # noqa: E402

WD = os.environ.get('WORK_DATA', '/media/ucla/work')
BUNDLES = f'{WD}/ntof_pass1/bundles/pf'
GAP = {'A': (28.08, 0.19), 'C': (28.0, None), 'D': (29.44, 1.12)}
MIN_N = 40


def bundle_facts(arm):
    b = json.load(open(f'{BUNDLES}/{arm}_E2_beam/bundle.json'))
    z = np.load(f'{BUNDLES}/{arm}_E2_beam/arrays.npz')
    gap = b['provenance']['physics_inputs']['det']['gap_mm']
    assert abs(gap - GAP[arm][0]) < 1e-6, (arm, gap)
    return b['v_drift'], z['grid'], z['tmpl_x'] / z['tmpl_x'].max()


def stack_fit(g, S, rng, nboot):
    Sn = S / np.maximum(S.max(axis=1, keepdims=True), 1.0)
    r = g.fit(np.median(Sn, axis=0))
    bT = [g.fit(np.median(Sn[rng.integers(0, len(Sn), len(Sn))], axis=0))[1] for _ in range(nboot)]
    return r[0], r[1], float(np.std(bT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vt', required=True, help='local copy of Analysis_pass1/vt (run_<r>/<sub>/vt_*.npz)')
    ap.add_argument('--out', default=f'{WD}/ntof_pass1/vt')
    ap.add_argument('--boot', type=int, default=60)
    a = ap.parse_args()
    colsum.T0_GRID = np.arange(-300.0, 900.0 + 1e-9, 5.0)
    colsum.T_GRID = np.arange(300.0, 1500.0 + 1e-9, 5.0)
    reg = {(int(r['run']), r['subrun'], r['tag']): r for r in csv.DictReader(open(os.path.join(HERE, '..', 'ntof_registry.csv')))}
    rng = np.random.default_rng(7)
    facts, fits = {}, {}
    rows = []
    files = sorted(glob.glob(f'{a.vt}/run_*/*/vt_*.npz'))
    for i, f in enumerate(files):
        stem = os.path.basename(f)[3:-4]                       # run_<r>_<sub>_<tag>_<arm>
        arm = stem[-1]
        run = int(stem.split('_')[1])
        sub = os.path.basename(os.path.dirname(f))
        tag = stem[len(f'run_{run}_{sub}_'):-2]
        r = reg[(run, sub, tag)]
        if arm not in facts:
            facts[arm] = bundle_facts(arm)
            fits[arm] = colsum.GridFit(colsum.Kern(facts[arm][1], facts[arm][2]), 20)
        z = np.load(f)
        row = dict(run=run, subrun=sub, tag=tag, arm=arm, trigger='beam' if r['rclass'] == 'production' else 'cosmic',
                   v_bundle=facts[arm][0], hv=f"{r[f'resist_{arm}']}/{r[f'drift_{arm}']}")
        t0 = datetime.strptime(r['t_local'], '%Y-%m-%d %H:%M')
        ts = np.concatenate([z['x_ts'], z['y_ts']]).astype(float)
        ts = ts[ts >= 0]
        row['t_start'] = t0 + timedelta(seconds=float(ts.min()) * 1e-8) if len(ts) else t0
        row['t_end'] = t0 + timedelta(seconds=float(ts.max()) * 1e-8) if len(ts) else t0
        gap = GAP[arm][0]
        for p in 'xy':
            ok = ~z[f'{p}_sat'].astype(bool) & (z[f'{p}_nsig'] >= 5) & (z[f'{p}_ncl'] == 1) if len(z['eid']) else np.zeros(0, bool)
            S = z[f'{p}_S'][ok].astype(float) if len(z['eid']) else np.zeros((0, 20))
            row[f'n_{p}'] = int(len(S))
            if len(S) >= MIN_N:
                t0f, T, eT = stack_fit(fits[arm], S, rng, a.boot)
                row.update({f't0_{p}': t0f, f'T_{p}': T, f'T_{p}_err': eT, f'v_{p}': gap * 1e3 / T,
                            f'v_{p}_err': gap * 1e3 / T ** 2 * eT})
            else:
                row.update({f't0_{p}': np.nan, f'T_{p}': np.nan, f'T_{p}_err': np.nan, f'v_{p}': np.nan,
                            f'v_{p}_err': np.nan})
        vs = [(row[f'v_{p}'], row[f'v_{p}_err']) for p in 'xy' if np.isfinite(row[f'v_{p}']) and row[f'v_{p}_err'] > 0]
        if vs:
            w = np.array([1 / e ** 2 for _, e in vs])
            row['v'] = float(np.sum(w * np.array([v for v, _ in vs])) / w.sum())
            row['v_err'] = float(1 / np.sqrt(w.sum()))
        else:
            row['v'] = row['v_err'] = np.nan
        row['gap_err_rel'] = (GAP[arm][1] / gap) if GAP[arm][1] else np.nan
        rows.append(row)
        if i % 500 == 0:
            print(f'{i:,}/{len(files):,}', flush=True)
    T = pd.DataFrame(rows).sort_values(['arm', 't_start']).reset_index(drop=True)
    T['hv_change'] = T.groupby('arm').hv.transform(lambda s: s.ne(s.shift()) & s.shift().notna())
    os.makedirs(a.out, exist_ok=True)
    for arm, g in T.groupby('arm'):
        g.to_parquet(f'{a.out}/vt_{arm}.parquet', index=False)
    side = dict(made_by='rerun/ntof/vt/vt_table.py', plan='NTOF_PASS_PLAN P7a', n_files=len(files), min_n=MIN_N,
                boot=a.boot, gap_mm={k: v[0] for k, v in GAP.items()}, gap_err_mm={k: v[1] for k, v in GAP.items()},
                caveats=['T is gain-confounded (+18 % across a det3 mesh scan): quote v(t) at fixed HV; hv_change '
                         'marks where it is not comparable', 'beam and cosmic samples differ by a selection '
                         'systematic (~3 % on A): trigger classes kept apart', '1.2 us window can clip slow columns (C)',
                         'B: no box (no v); D: cosmic rows only (beam column not a box)'])
    json.dump(side, open(f'{a.out}/vt.meta.json', 'w'), indent=1)
    print(T.groupby(['arm', 'trigger'])[['v', 'v_err', 'n_x']].median())


if __name__ == '__main__':
    main()
