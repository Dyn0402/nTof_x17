"""pilot_compare.py -- is2_v1 pilot stage 3 against production, per run
(HANDOFF_TRACKING §16 next step 1, §18 next step 2).

The pilot reconstructed the first sub-run of each of the 36 runs with the
is2 in-situ bundles, min 3 strips, stage 2 at TAN_MAX 1.0 raw (one-sided
search) and a TRUE-tan acceptance of 0.6 at stage 3.  Production is
stage3_fullpass: TAN_MAX 0.6 raw at stage 2, its own per-run k where certified.

Per (run, arm):
  gated_reco / gated      what the reco accepts, and after the 0.6-true cut
  confirmed               SiPM-wall matches minus the off-time control, on
                          gated tracks inside the wall (det_a_scint.match_run)
  confirmed_acc           the same restricted to |true tan| < 0.6 in BOTH
                          chains -- the like-for-like yield comparison
  late_frac               max(x_t0, y_t0) > 300 ns among gated
  mirror_zone             gated pilot tracks with 0.3 <= |true tan| < 0.6,
                          where the one-sided mirror basin lives (§17)

Usage:  pilot_compare.py match   (wall matching, cached per run/arm/chain)
        pilot_compare.py summary
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import det_a_scint as DS

ROOT = Path('/media/dylan/data/x17/sept26_prelim')
PILOT = ROOT / 'stage3_is2_v1_pilot'
PROD = ROOT / 'stage3_fullpass'   # stage3_campaign is the early low-statistics campaign (~1/15 of the events)
SUBSET = json.loads((ROOT / 'pilot_subset_is2_v1.json').read_text())
OUT = Path(__file__).resolve().parent / 'results' / 'pilot_is2_v1'
CACHE = ROOT / 'pilot_is2_v1_compare'
ARMS = ('A', 'C')
ACC_TRUE = 0.6
LATE_NS = 300
CHAINS = {'prod': PROD, 'pilot': PILOT}


def _subs(run):
    return SUBSET[run]


def match():
    for run in sorted(SUBSET, key=lambda r: int(r.split('_')[1])):
        for chain, src in CHAINS.items():
            for arm in ARMS:
                f = CACHE / chain / f'scint_{run}_{arm}.parquet'
                if f.exists():
                    continue
                f.parent.mkdir(parents=True, exist_ok=True)
                try:
                    M, _ = DS.match_run(run, _subs(run), src, None, DS.layer_geometry(run, arm))
                except Exception as e:  # noqa: BLE001 -- a run without k is reported, not fatal
                    print(f'  !! {run} {chain} {arm}: {str(e).splitlines()[0][:120]}')
                    continue
                M.to_parquet(f, index=False)
                print(f'  {run} {chain} {arm}: {len(M):,} matched rows', flush=True)


COLS = ['arm', 'gated', 'x_t0', 'y_t0', 'tanx', 'tany', 'angle_calibrated']


def summary():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(SUBSET, key=lambda r: int(r.split('_')[1])):
        for chain, src in CHAINS.items():
            T = pd.concat([pd.read_parquet(src / f'tracks_{run}_{s}.parquet',
                                           columns=COLS + (['gated_reco', 'in_acceptance'] if chain == 'pilot' else []))
                           for s in _subs(run)], ignore_index=True)
            for arm in ARMS:
                g = T[T.arm == arm]
                G = g[g.gated]
                tt = np.maximum(G.tanx.abs(), G.tany.abs())
                r = dict(run=run, arm=arm, chain=chain, gated=len(G),
                         gated_reco=int(g.gated_reco.sum()) if chain == 'pilot' else len(G),
                         gated_acc=int((tt < ACC_TRUE).sum()),
                         mirror_zone=int(((tt >= 0.3) & (tt < ACC_TRUE)).sum()),
                         late_frac=float((np.maximum(G.x_t0, G.y_t0) > LATE_NS).mean()) if len(G) else np.nan,
                         calibrated=bool(g.angle_calibrated.astype('boolean').fillna(False).any()))
                f = CACHE / chain / f'scint_{run}_{arm}.parquet'
                if f.exists():
                    S = pd.read_parquet(f)
                    w = S[S.on_wall.astype(bool)]   # an empty object column would select columns, not rows
                    wa = w[np.maximum(w.tanx.abs(), w.tany.abs()) < ACC_TRUE]
                    r.update(on_wall=len(w),
                             confirmed=int(w.match_wall.sum() - w.match_wall_ctrl.sum()),
                             confirmed_acc=int(wa.match_wall.sum() - wa.match_wall_ctrl.sum()),
                             wall_rate=float(w.match_wall.mean()) if len(w) else np.nan,
                             wall_ctrl=float(w.match_wall_ctrl.mean()) if len(w) else np.nan)
                rows.append(r)
    R = pd.DataFrame(rows)
    P = R[R.chain == 'prod'].set_index(['run', 'arm'])
    idx = R.set_index(['run', 'arm']).index
    for c in ('gated', 'gated_acc', 'confirmed', 'confirmed_acc'):
        if c in R:
            R[f'{c}_vs_prod'] = R[c].values / idx.map(P[c]).values.astype(float)
    R.to_csv(OUT / 'pilot_compare.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500):
        print(R[R.chain == 'pilot'].round(3).to_string(index=False))
        tot = R.groupby(['arm', 'chain'])[['gated', 'gated_reco', 'gated_acc', 'mirror_zone',
                                           'confirmed', 'confirmed_acc']].sum()
        print('\n', tot)
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('match', 'summary'))
    a = ap.parse_args()
    {'match': match, 'summary': summary}[a.step]()
