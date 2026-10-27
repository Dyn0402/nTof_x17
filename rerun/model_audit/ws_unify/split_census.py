#!/usr/bin/env python3
"""split_census.py <gate_full_dir> [--json out] -- what the joint two-track fit splits on clean bench single muons.

Input: per chamber <dir>/<det>/{prod,twotrack}/ with events.parquet, events.splits.parquet (twotrack) and the
effbudget rays_<det>.parquet (copied from the fleet gate, Analysis_audit/<row>/wft/<arm>/).
Clean single muon: interior M3 ray, production track within 3 mm of it in BOTH views (cat_xy_3 == 'detected').
On the cosmic bench almost every M3-matched trigger is one muon, so the accepted-split rate on these events is the
false-split rate up to the (small) delta-ray / shower fraction.  Per chamber and view: split rate, and the parent's
chi2/dof, brightness (q_sum), separation and statistic of the accepted splits against the population.
"""
import argparse, json, os
import numpy as np
import pandas as pd

DETS = ('det2', 'det3', 'det4', 'det6', 'det7')


def census(root, det):
    d = os.path.join(root, det)
    rays = pd.read_parquet(os.path.join(d, 'prod', f'rays_{det}.parquet'))
    ev = pd.read_parquet(os.path.join(d, 'prod', 'events.parquet')).set_index('event_id')
    sp = pd.read_parquet(os.path.join(d, 'twotrack', 'events.splits.parquet'))
    clean = rays[(rays['region'] == 'interior') & (rays['cat_xy_3'] == 'detected')]['event_id'].unique()
    clean = np.intersect1d(clean, ev.index.values)
    out = dict(n_clean=int(len(clean)))
    acc = sp[sp['accepted'] & (sp['rank'] == 0)]
    for v in 'xy':
        e = ev.loc[clean]
        cd = (e[f'{v}_chi2'] / e[f'{v}_dof'].clip(lower=1)).to_numpy()
        a = acc[(acc['plane'] == v) & acc['event_id'].isin(clean)]
        att = sp[(sp['plane'] == v) & (sp['rank'] == 0) & sp['event_id'].isin(clean)]
        ids = a['event_id'].to_numpy()
        cda = (ev.loc[ids, f'{v}_chi2'] / ev.loc[ids, f'{v}_dof'].clip(lower=1)).to_numpy()
        q = e[f'{v}_q_sum'].to_numpy()
        qa = ev.loc[ids, f'{v}_q_sum'].to_numpy()
        out[v] = dict(
            n_attempt=int(len(att)), n_split=int(len(a)),
            split_rate_pct=100.0 * len(a) / max(len(clean), 1),
            chi2dof_p50_all=float(np.nanmedian(cd)), chi2dof_p90_all=float(np.nanpercentile(cd, 90)),
            chi2dof_p50_split=float(np.nanmedian(cda)) if len(cda) else None,
            # where the split parents sit in the chamber's own chi2/dof distribution
            chi2dof_pctile_split=float(np.nanmedian([100.0 * np.mean(cd < c) for c in cda])) if len(cda) else None,
            q_pctile_split=float(np.nanmedian([100.0 * np.mean(q < c) for c in qa])) if len(qa) else None,
            sep_p50=float(a['sep'].median()) if len(a) else None,
            dist_p50=float(a['dist'].median()) if len(a) else None,
            fstat_p50=float(a['fstat'].median()) if len(a) else None,
            qfrac_p50=float(a['qfrac'].median()) if len(a) else None,
            frac_sep_lt3=float((a['sep'] < 3.0).mean()) if len(a) else None)
        both = np.intersect1d(acc[(acc['plane'] == 'x')]['event_id'], acc[(acc['plane'] == 'y')]['event_id'])
        out['n_both_views'] = int(len(np.intersect1d(both, clean)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--json')
    a = ap.parse_args()
    res = {det: census(a.root, det) for det in DETS if os.path.isdir(os.path.join(a.root, det, 'twotrack'))}
    L = ['| det view | clean singles | attempts | split | rate % | χ²/dof p50 all / split | split parent χ²/dof pctile | q pctile | sep p50 mm | sep<3 mm | qfrac p50 | both views |',
         '|' + '---|' * 12]
    for det, r in res.items():
        for v in 'xy':
            s = r[v]
            f = lambda x, fm='.1f': '—' if x is None else format(x, fm)
            L.append(f"| {det} {v} | {r['n_clean']} | {s['n_attempt']} | {s['n_split']} | {s['split_rate_pct']:.2f} | "
                     f"{f(s['chi2dof_p50_all'])} / {f(s['chi2dof_p50_split'])} | {f(s['chi2dof_pctile_split'], '.0f')} | "
                     f"{f(s['q_pctile_split'], '.0f')} | {f(s['sep_p50'], '.2f')} | {f(s['frac_sep_lt3'], '.2f')} | {f(s['qfrac_p50'], '.2f')} | {r['n_both_views']} |")
    md = '\n'.join(L) + '\n'; print(md)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1); open(a.json.replace('.json', '.md'), 'w').write(md)


if __name__ == '__main__':
    main()
