#!/usr/bin/env python3
"""vt_line_campaign.py -- does the angle scale follow the timing v(t)?  The A-C cosmic line on every beam-off sub-run.

The P7a v(t) (`vt_smooth.parquet`, v_fit) is a TIMING velocity: 30 mm / column duration.  The table correction
tan_vt = tan * v_bundle / v_t assumes it is also the ANGLE velocity (the A/B 4677861 showed tan scales exactly as 1/v
at a fixed fit, so the question is only which v).  The A-C through-going line (ac_line.score; positions only, so
independent of v) measures the angle velocity directly: fitted/true tan = v_angle / v_bundle on each chamber.

Per cosmic sub-run (registry rclass cosmic, the pass's stage-2 copy): the line ratio per view of A and C on
|true tan| in [0.15, 0.45], against the sub-run's mean v_t / v_bundle.  Slope 1 = the timing v is the angle v.

    vt_line_campaign.py [--stage2 $WORK_DATA/ntof_pass1/stage2] [--jobs 10] [--out rerun/ntof/stage3/vt_score]
"""
import argparse
import csv
import glob
import json
import os
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(NT, '..', '..'))
sys.path[:0] = [HERE, NT, REPO]
warnings.filterwarnings('ignore')

WD = os.environ.get('WORK_DATA') or '/media/ucla/work'
RUNS = '/media/dylan/data/x17/beam_july/runs'
WIN = (0.15, 0.45)


def one(args):
    run, sub, tags, stage2 = args
    import ac_line
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for arm in 'AC':
                parts = []
                for i, tag in enumerate(sorted(tags)):
                    f = f'{stage2}/run_{run}/{sub}/{arm}/run_{run}_{sub}_{tag}_{arm}.candidates.parquet'
                    if not os.path.exists(f):
                        return run, sub, None, f'missing {f}'
                    c = pd.read_parquet(f)
                    c['event_id'] = c['event_id'] + 100_000_000 * i     # same offset for A and C of one acquisition
                    parts.append(c)
                pd.concat(parts, ignore_index=True).to_parquet(f'{tmp}/{arm}.candidates.parquet')
            o, M = ac_line.score(f'{tmp}/A.parquet', f'{tmp}/C.parquet', f'{RUNS}/run_{run}/run_config.json')
        rows = []
        for s, arm in (('a', 'A'), ('c', 'C')):
            for ax in 'xy':
                fit, tt = M[f'tan{ax}_{s}'].to_numpy(float), M[f'true_tan{ax}_{s}'].to_numpy(float)
                w = (np.abs(tt) > WIN[0]) & (np.abs(tt) < WIN[1]) & np.isfinite(fit)
                rows.append(pd.DataFrame(dict(run=int(run), subrun=sub, arm=arm, view=ax, r=fit[w] / tt[w])))
        return run, sub, pd.concat(rows, ignore_index=True), None
    except Exception as e:                                           # noqa: BLE001
        return run, sub, None, repr(e)


def fit_slope(x, y, e):
    w = 1 / e ** 2
    X = np.c_[np.ones_like(x), x - 1]
    C = np.linalg.inv(X.T @ (X * w[:, None]))
    b = C @ (X.T @ (w * y))
    chi2 = float(np.sum(w * (y - X @ b) ** 2))
    return dict(intercept_at_1=float(b[0]), slope=float(b[1]), slope_err=float(np.sqrt(C[1, 1])),
                intercept_err=float(np.sqrt(C[0, 0])), chi2=chi2, ndof=int(len(x) - 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage2', default=f'{WD}/ntof_pass1/stage2')
    ap.add_argument('--vt', default=f'{WD}/ntof_pass1/vt/vt_smooth.parquet')
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--out', default=os.path.join(HERE, 'vt_score'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = [r for r in csv.DictReader(open(os.path.join(NT, 'ntof_registry.csv')))
           if r['excluded'] != 'True' and r['rclass'] == 'cosmic']
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r['tag'])
    ev, bad = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for run, sub, d, err in ex.map(one, [(r, s, t, a.stage2) for (r, s), t in sorted(groups.items())]):
            if err:
                bad.append((run, sub, err))
            elif d is not None:
                ev.append(d)
    ev = pd.concat(ev, ignore_index=True)
    ev.to_parquet(f'{a.out}/line_campaign_events.parquet', index=False)

    vt = pd.read_parquet(a.vt)
    vt = vt[vt.trigger == 'cosmic']
    vs = (vt.assign(q=vt.v_fit / vt.v_bundle, m=vt.measured.astype(float))
          .groupby(['run', 'subrun', 'arm']).agg(vt_ratio=('q', 'mean'), measured=('m', 'mean'),
                                                  tm=('tm', 'median')).reset_index())
    g = (ev.groupby(['run', 'subrun', 'arm', 'view'])['r']
         .agg(n='size', ratio='median', mad=lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))).reset_index())
    g['ratio_err'] = 1.2533 * g['mad'] / np.sqrt(g['n'])
    g = g.merge(vs, on=['run', 'subrun', 'arm'], how='left')
    g.to_csv(f'{a.out}/line_campaign_subruns.csv', index=False)
    out = dict(n_subruns=len(groups), n_failed=len(bad), failed=bad[:20], window=WIN, fits={})
    for (arm, view), s in g[(g.n >= 20) & g.vt_ratio.notna()].groupby(['arm', 'view']):
        out['fits'][f'{arm}{view}'] = dict(n_subruns=int(len(s)), n_events=int(s.n.sum()),
                                           vt_ratio_range=[float(s.vt_ratio.min()), float(s.vt_ratio.max())],
                                           **fit_slope(s.vt_ratio.to_numpy(), s.ratio.to_numpy(),
                                                       s.ratio_err.to_numpy()))
    json.dump(out, open(f'{a.out}/line_campaign.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

    # kappa = v_angle / v_t per (arm, view) with the slope fixed at 1: the constant the table correction needs
    s = g[(g.n >= 20) & g.vt_ratio.notna()].copy()
    s['k'], s['ke'] = s.ratio / s.vt_ratio, s.ratio_err / s.vt_ratio
    s['late'] = pd.to_datetime(s.tm) >= pd.to_datetime(s.tm).median()

    def wm(d):
        w = 1 / d.ke ** 2
        k = float((w * d.k).sum() / w.sum())
        return k, float(w.sum() ** -0.5), float((w * (d.k - k) ** 2).sum() / max(len(d) - 1, 1))
    kap = dict(doc='kappa = v_angle / v_t: the A-C cosmic line (fitted/true tan) over v_t/v_bundle per cosmic sub-run, '
                   'slope fixed at 1, weighted mean.  tan_vt = tan * v_bundle / (kappa * v_t).  D has no A-C line: '
                   'kappa_D = 1 (timing ratio only, absolute scale unanchored).',
               source='vt_line_campaign.py', window=WIN, kappa={}, syst_early_late={})
    for (arm, view), d in s.groupby(['arm', 'view']):
        k, e, c = wm(d)
        ke, _, _ = wm(d[~d.late])
        kl, _, _ = wm(d[d.late])
        kap['kappa'][f'{arm}{view}'] = dict(kappa=k, err=e, chi2_dof=c, n_subruns=int(len(d)))
        kap['syst_early_late'][f'{arm}{view}'] = dict(early=ke, late=kl, half_diff=abs(kl - ke) / 2)
    json.dump(kap, open(f'{a.out}/vt_kappa.json', 'w'), indent=1)
    print(json.dumps(kap['kappa'], indent=1))


if __name__ == '__main__':
    main()
