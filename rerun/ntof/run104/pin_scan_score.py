#!/usr/bin/env python3
"""pin_scan_score.py -- score the D beam t0-pin scan (condor 4674646, make_pin_scan.py) per pin f.

    pin_scan_score.py [--scan DIR] [--stage2 DIR] [--control DIR]

pin(f) = L2 + f (L1 - L2).  Sources per (acquisition, f): scan DIR/f<NNN>/ for the jobs; f = 0 on runs 106/108 and f = 1
on runs 100/102 from the main pass (--stage2, a local copy of Analysis_pass1/stage2); f = 1 on runs 106/108 from the
control (--control, tests/d104_control_L1).  Per acquisition and f, the same readings as d104_latency.py:

  point  capsule pointing: gated tracks with DCA to the capsule axis < 15 mm (capsule_point.score)
  chi2   per plane fitted ok under f and under the reference f = 1, with each pin's prior term ((t0 - pin)/15)^2
         removed: median of (chi2_f - chi2_ref)/dof; < 0 means the waveforms prefer f over L1

Summed over acquisitions per (era, f): pointing total (and relative to f = 1), median chi2 difference, fraction of planes
better than f = 1.  A parabola through the pointing totals gives the best f per era.  Writes DIR/results/pin_scan.csv
(per acquisition x f) and pin_scan_summary.csv.
"""
import argparse, glob, json, os, sys, warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [NT, os.path.abspath(os.path.join(NT, '..', '..'))]
warnings.filterwarnings('ignore')
L1 = {'x': -17.8, 'y': -11.2}
L2 = {'x': 18.3, 'y': 20.0}
SIG = 15.0
RUNS = '/media/dylan/data/x17/beam_july/runs'
WD = os.environ.get('WORK_DATA', '/media/ucla/work')


def pin(f):
    return {p: L2[p] + f * (L1[p] - L2[p]) for p in 'xy'}


def cd(df, f, p):
    """chi2 of plane p with the pin's prior term removed."""
    P = pin(f)[p]
    return df[f'{p}_chi2'] - ((df[f'{p}_t0'] - (P - 10 * df[f'{p}_ftst'])) / SIG) ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', default=f'{WD}/ntof_pass1/pinscan')
    ap.add_argument('--stage2', default=f'{WD}/ntof_pass1/stage2')
    ap.add_argument('--control', default=f'{WD}/ntof_pass1/run104/control_L1')
    a = ap.parse_args()
    import capsule_point
    scan = json.load(open(f'{a.scan}/scan.json'))
    acqs = [s.split() for s in scan['acquisitions']]
    fs = sorted({0.0, 1.0} | {int(t[1:]) / 100 for t in scan['pins']})
    rows = []
    for run, sub, tag, arm in acqs:
        stem = f'run_{run}_{sub}_{tag}_D.parquet'
        src = {}
        for f in fs:
            t = f'f{int(round(f * 100)):03d}'
            cand = [f'{a.scan}/out/{t}/run_{run}/{sub}/D/{stem}']
            if f == 0.0 and run in ('106', '108'):
                cand = [f'{a.stage2}/run_{run}/{sub}/D/{stem}']
            if f == 1.0:
                cand = [f'{a.control}/run_{run}/{sub}/D/{stem}'] if run in ('106', '108') else \
                       [f'{a.stage2}/run_{run}/{sub}/D/{stem}']
            src[f] = next((c for c in cand if os.path.exists(c)), None)
        if src[1.0] is None:
            print('no reference for', stem); continue
        ref = pd.read_parquet(src[1.0])
        rc = f'{RUNS}/run_{run}/run_config.json'
        for f, path in src.items():
            if path is None:
                continue
            d = pd.read_parquet(path)
            try:
                s, _ = capsule_point.score(path, 'D', rc, str(f))
            except Exception:
                s = dict(n_point_15=0, n_tracks=0, frac_point_15=np.nan)
            r = dict(run=int(run), subrun=sub, tag=tag, era='L2' if run in ('106', '108') else 'L1', f=f,
                     n_point15=s['n_point_15'], n_tracks=s['n_tracks'], n_events=len(d))
            m = d.merge(ref, on='event_id', suffixes=('', '_r'))
            for p in 'xy':
                ok = (m[f'{p}_ok'] == True) & (m[f'{p}_ok_r'] == True) & (m[f'{p}_ftst'] >= 0)
                mm = m[ok]
                ref_cd = mm[f'{p}_chi2_r'] - ((mm[f'{p}_t0_r'] - (pin(1.0)[p] - 10 * mm[f'{p}_ftst_r'])) / SIG) ** 2
                dd = (cd(mm, f, p) - ref_cd) / mm[f'{p}_dof'].clip(lower=1)
                r[f'dchi2_{p}'] = float(np.median(dd)) if len(dd) else np.nan
                r[f'frac_better_{p}'] = float((dd < 0).mean()) if len(dd) else np.nan
                g = d[(d[f'{p}_ok'] == True) & (d[f'{p}_quality_ok'] == True) & (d.n_tracks == 1) & (d[f'{p}_ftst'] >= 0)]
                v = (g[f'{p}_t0'] + 10 * g[f'{p}_ftst']).to_numpy(float)
                r[f'pull_{p}'] = float(np.median(v) - pin(f)[p]) if len(v) > 10 else np.nan
            rows.append(r)
        print(f'{stem}: {len(src)} pins', flush=True)
    T = pd.DataFrame(rows)
    os.makedirs(f'{a.scan}/results', exist_ok=True)
    T.to_csv(f'{a.scan}/results/pin_scan.csv', index=False)
    # only acquisitions with every f present are summed
    full = T.groupby(['run', 'subrun', 'tag']).f.transform('nunique') == T.groupby('era').f.transform('nunique')
    S = (T[full].groupby(['era', 'f']).agg(n_acq=('tag', 'size'), n_point15=('n_point15', 'sum'),
                                             n_tracks=('n_tracks', 'sum'), dchi2_x=('dchi2_x', 'median'),
                                             dchi2_y=('dchi2_y', 'median'), frac_better_x=('frac_better_x', 'median'),
                                             frac_better_y=('frac_better_y', 'median'), pull_x=('pull_x', 'median'),
                                             pull_y=('pull_y', 'median')).reset_index())
    S['point_rel_L1'] = S.n_point15 / S.groupby('era').apply(lambda g: g.set_index('f').n_point15.get(1.0, np.nan)) \
        .reindex(S.era).to_numpy()
    for p in 'xy':
        S[f'pin_{p}'] = [pin(f)[p] for f in S.f]
    S.to_csv(f'{a.scan}/results/pin_scan_summary.csv', index=False)
    pd.set_option('display.width', 220)
    print(S.round(4).to_string(index=False))
    for era, g in S.groupby('era'):
        if len(g) >= 3:
            c = np.polyfit(g.f, g.n_point15, 2)
            fbest = -c[1] / (2 * c[0]) if c[0] < 0 else np.nan
            print(f'{era}: best f by pointing (parabola) {fbest:.2f}; pin x {pin(fbest)["x"]:.1f} y {pin(fbest)["y"]:.1f} ns'
                  if np.isfinite(fbest) else f'{era}: pointing not peaked in the scanned range')


if __name__ == '__main__':
    main()
