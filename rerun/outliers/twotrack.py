#!/usr/bin/env python3
"""twotrack.py <det> -- what the multi-track reconstruction does with the outliers (C22 follow-up, Dylan 10-25: try
refitting the multi-particle events with the multi-track fit).

Arms (reco_subset.py on the outliers + 300 controls, finalize environment): `off` (must reproduce the finalize table)
and `on` (WFT_TWO_TRACK_FIT=1: a joint two-track fit offered to every candidate window; accepted splits replace the
parent). Both write the full candidate list (events.candidates.parquet: every spatial cluster's fit, plus split
children). Per class (classify.py) and view, against M3 (|dθ| < 5° and |res| < 3 mm = "agrees"):
  winner    the arm's selected track
  any       some candidate (or split child) agrees -- the ceiling a reference-free choice could reach
  rules     reference-free choices among the plausible candidates: rank (production), largest charge, longest column,
            lowest chi2/dof, closest t0 to the other view's winner
The controls measure the cost (a rule that moves good events away from M3).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from olib import BASE, load  # noqa: E402


def agree(p0, w, v, row, lad):
    s, c = lad[v]
    tl = (w * 1e3 - c) / s
    d = np.degrees(np.arctan(tl) - np.arctan(row[f'ref_tan_{v}']))
    return (np.abs(d) < 5) & (np.abs(p0 - row[f'ref_mesh_{v}']) < 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('det'); ap.add_argument('--arms', nargs='+', default=['off', 'on'])
    a = ap.parse_args(); det = a.det
    J = load(det, quality=False); lad = J.attrs['ladder']; J = J.set_index('event_id')
    X = pd.read_parquet(os.path.join(BASE, f'classified_{det}.parquet'))[['event_id', 'cls', 'outlier']].set_index('event_id')
    out = {}
    for arm in a.arms:
        p = os.path.join(BASE, 'reco2', f'{det}_{arm}', 'events.parquet')
        if not os.path.exists(p):
            continue
        E = pd.read_parquet(p).drop_duplicates('event_id', keep=False).set_index('event_id')
        C = pd.read_parquet(p.replace('.parquet', '.candidates.parquet'))
        rows = []
        for eid, g in C.groupby('event_id'):
            if eid not in J.index or eid not in X.index:
                continue
            r = J.loc[eid]; cls = X.loc[eid, 'cls'] if X.loc[eid, 'outlier'] else 'CONTROL'
            for v in 'xy':
                if not np.isfinite(r[f'ref_tan_{v}']):
                    continue
                gv = g[(g.plane == v) & ~g.split_replaced]
                if not len(gv):
                    continue
                ag = agree(gv.p0.to_numpy(), gv.w.to_numpy(), v, r, lad)
                pl = gv.plausible.to_numpy()
                gp = gv[pl] if pl.any() else gv
                agp = ag[pl] if pl.any() else ag
                o = 'y' if v == 'x' else 'x'
                t0o = E.loc[eid, f'{o}_t0'] if eid in E.index else np.nan
                pick = {
                    'rank': int(np.argmin(gp['rank'].to_numpy())),
                    'charge': int(np.argmax(gp.q_sum.to_numpy())),
                    'column': int(np.argmax(np.nan_to_num(gp.q_uend.to_numpy()))),
                    'chi2dof': int(np.argmin((gp.chi2 / gp.dof.clip(lower=1)).to_numpy())),
                    't0match': int(np.argmin(np.abs(gp.t0.to_numpy() - t0o))) if np.isfinite(t0o) else int(np.argmin(gp['rank'].to_numpy())),
                }
                win_ok = False
                if eid in E.index and E.loc[eid, f'{v}_ok']:
                    win_ok = bool(agree(np.array([E.loc[eid, f'{v}_p0']]), np.array([E.loc[eid, f'{v}_w']]), v, r, lad)[0])
                base_ok = bool(r[f'{v}_ok'] and agree(np.array([r[f'{v}_p0']]), np.array([r[f'{v}_w']]), v, r, lad)[0])
                rows.append(dict(event_id=eid, view=v, cls=cls, n_cand=len(gv), n_split=int(gv.split_child.sum()),
                                 finalize=base_ok, winner=win_ok, any=bool(ag.any()),
                                 **{f'rule_{k}': bool(agp[i]) for k, i in pick.items()}))
        R = pd.DataFrame(rows); R.to_parquet(os.path.join(BASE, 'reco2', f'twotrack_{det}_{arm}.parquet'))
        cols = ['finalize', 'winner', 'any', 'rule_rank', 'rule_charge', 'rule_column', 'rule_chi2dof', 'rule_t0match']
        T = (R.groupby('cls')[cols].mean() * 100).round(1); T['n'] = R.groupby('cls').size(); T['split%'] = (R.groupby('cls').n_split.apply(lambda s: (s > 0).mean()) * 100).round(1)
        print(f'== {det} arm {arm}: % of event-views agreeing with M3'); print(T.to_string())
        out[arm] = T.reset_index().to_dict('records')
    json.dump(out, open(os.path.join(BASE, 'report', f'twotrack_{det}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
