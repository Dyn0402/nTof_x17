#!/usr/bin/env python3
"""collect.py [--root OUTROOT] [--out results/] -- one table of every bench row's Paper I efficiency numbers from the
p1eff DAG outputs (budget.py, 03_angles --fiducial interior_good, band_census.py), plus the per-chamber golden-row
headline. Run on lxplus (the outputs are on EOS) with the lock venv; writes small CSV / JSON into the repo.

Columns per row (all in %, R = 5 mm unless the name says 3; 'ig' = interior_good = interior AND det7 X ch 256-511):
  n_<g>, eff_<g>                    detection (budget 'detected': both views fitted, |r| <= R raw frame,
                                    n_dropped <= 2) for g in active / interior / ig / interior_live / edge
  eff_<g>_r3, eff_x_<g>, eff_y_<g>  R = 3 mm; per view (1D residual <= R, the view's own ladder)
  rule_<g>, any_<g>, posonly_<g>    trigger-track rule pick / multi-track credit / position-only (no fit-sanity
                                    cut: the skeptical rows' broken angle scale) (budget.cand_arms)
  loss_<cat>_ig                     the ladder categories on interior_good
  band_any, band_quiet, band_x/y    noise-band census: fraction of triggers with a band (quiet = no spark-like)
  s68_x/y_ig, n_x/y_ig              03_angles on interior_good (DEPENDS ON v(t) for the drifting rows, VT_PLAN)
"""
import argparse, csv, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTROOT = '/eos/user/d/dneff/x17rerun/paper1_eff'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
LADDER = ['detected', 'rejected_cut', 'spark', 'no_hit', 'hw_known', 'one_view_silent', 'charge_elsewhere',
          'not_seeded', 'fit_failed', 'far']
GROUPS = {'active': 'active', 'interior': 'interior', 'ig': 'interior_good', 'live': 'interior_live', 'edge': 'edge'}


def binom_err(p, n):
    return float(100 * np.sqrt(max(p / 100 * (1 - p / 100), 1e-12) / n)) if n else np.nan


def row_numbers(r, root):
    rel = f"{r['run']}/{r['subrun']}/{r['det']}"; d = SHORT[r['det']]
    o = dict(key=r['key'], det=d, run=r['run'], subrun=r['subrun'], resist_V=r['resist_V'], drift_V=r['drift_V'],
             cls=r['cls'], group=r['group'])
    p = os.path.join(root, 'budget', rel, f'budget_{d}.json')
    if os.path.exists(p):
        S = json.load(open(p)); B = S['budget']
        o['n_rays_recipe'] = S['n_rays_recipe']; o['n_not_measured'] = S['n_not_measured']
        o['n_beyond_table'] = S['n_beyond_table']; o['n_outside'] = S['n_outside']
        for k, g in GROUPS.items():
            e = B.get(f'{g}|xy|5')
            if not e:
                continue
            o[f'n_{k}'] = e['n']; o[f'eff_{k}'] = e['detected']; o[f'err_{k}'] = binom_err(e['detected'], e['n'])
            o[f'eff_{k}_r3'] = B[f'{g}|xy|3']['detected']
            o[f'eff_x_{k}'] = B[f'{g}|x|5']['detected']; o[f'eff_y_{k}'] = B[f'{g}|y|5']['detected']
            o[f'rule_{k}'] = e.get('detected_rule'); o[f'any_{k}'] = e.get('detected_any')
            o[f'posonly_{k}'] = e.get('detected_posonly')
            if k == 'ig':
                for c in LADDER[1:]:
                    o[f'loss_{c}_ig'] = e[c]
        c = S.get('cand', {}).get('interior_good', {}).get('5')
        if c:
            o['closure_mismatch_ig'] = c['prod_closure_mismatch']
            for v in 'xy':
                o[f'multi_views_{v}_ig'] = c[f'multi_views_{v}']; o[f'prod_right_{v}_ig'] = c[f'prod_right_{v}']
                o[f'rule_right_{v}_ig'] = c[f'rule_right_{v}']
    p = os.path.join(root, 'bands', rel, 'census.json')
    if os.path.exists(p):
        C = json.load(open(p))
        o['band_n'] = C['n_triggers']; o['band_any'] = 100 * (C['p_band_any'] or 0)
        o['band_quiet'] = 100 * (C['p_band_quiet'] or 0) if C['p_band_quiet'] is not None else np.nan
        o['band_no_discharge'] = 100 * (C['p_band_no_discharge'] or 0) if C.get('p_band_no_discharge') is not None else np.nan
        o['band_x'] = 100 * (C['p_band_x'] or 0); o['band_y'] = 100 * (C['p_band_y'] or 0)
        o['band_both'] = 100 * (C['p_band_both'] or 0)
        o['spark_trig'] = 100 * C['n_spark'] / max(C['n_triggers'], 1)
        o['discharge_trig'] = 100 * C.get('n_discharge', 0) / max(C['n_triggers'], 1)
        sc = sum(f['scanned'] for f in C['files'].values()); tot = sum(f['n_entries'] for f in C['files'].values())
        o['band_scan_frac'] = sc / max(tot, 1)
        for v in 'xy':
            o[f'band_chips_{v}'] = json.dumps(C.get(f'chip_rate_{v}_quiet'))
    p = os.path.join(root, 'angles', rel, 'angular_resolution.json')
    if os.path.exists(p):
        A = json.load(open(p))
        for v in 'xy':
            q = A['planes'].get(v, {})
            o[f's68_{v}_ig'] = q.get('s68_deg'); o[f'n_{v}_ig'] = q.get('n'); o[f'bias_{v}_ig'] = q.get('bias_deg')
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=OUTROOT); ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    ap.add_argument('--registry', default=os.path.join(HERE, '..', '..', 'pass_registry.csv'))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.registry)) if r['cls'] != 'excluded']
    T = pd.DataFrame([row_numbers(r, a.root) for r in rows])
    os.makedirs(a.out, exist_ok=True)
    T.to_csv(os.path.join(a.out, 'rows.csv'), index=False)
    missing = {k: int(T[c].isna().sum()) for k, c in (('budget', 'eff_active'), ('bands', 'band_any'), ('angles', 's68_x_ig'))
               if c in T}
    gold = T[T.cls == 'golden_calib'].set_index('det')
    keep = [c for c in T.columns if c.startswith(('n_', 'eff_', 'err_', 'rule_', 'any_', 'posonly_', 'loss_', 'band_', 's68_', 'closure',
                                                     'multi_views', 'prod_right', 'rule_right', 'spark_trig',
                                                     'discharge_trig'))]
    S = dict(n_rows=len(T), missing=missing, golden={d: {c: (None if pd.isna(v) else v) for c, v in gold.loc[d, keep + ['key', 'resist_V', 'drift_V']].items()}
                                                    for d in gold.index})
    json.dump(S, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, default=str)
    print(json.dumps(dict(n_rows=len(T), missing=missing), indent=1))
    cols = ['det', 'key', 'n_ig', 'eff_interior', 'eff_ig', 'err_ig', 'eff_ig_r3', 'rule_ig', 'any_ig', 'band_any', 's68_x_ig', 's68_y_ig']
    print(gold.reset_index()[[c for c in cols if c in gold.reset_index()]].to_string(index=False))


if __name__ == '__main__':
    main()
