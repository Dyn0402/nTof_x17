#!/usr/bin/env python3
"""a1_recat.py [--root OUTROOT] [--out results/] -- Paper I appendix A1 on the PASS tables: every interior M3 ray of
each golden row in exactly one class, first match wins, with the satstudy/summarize.recat order (10-26):

  detected        the budget's production detection
  band            the event has a noise band (band_census.py, either view) and was lost to the n_dropped cut or the
                  spark veto, and is not a discharge (sparks rule)
  multi:<sub>     >= 2 particle candidates (time extent <= 24 samples, a1_job cand_feat) in a view:
                  m3_picked (lost to the n_dropped cut) / other_picked (an x AND a y candidate within R exist) /
                  m3_not_found
  spark:<cls>     the rest of the spark veto: discharge / muon+other / shower (a1_job, satstudy/sparks.py rules)
  far:<C22 cls>   fitted in both views, |r| > R: the C22 class (a1_job: classify.py on this table)
  fit_failed      a view's fit not ok (pre-pass: the 2-strip head-on class); |tan_x| < 0.03 fraction reported
  cut:other       rejected by the n_dropped cut, no band, no second particle
  other:<cat>     no_hit, one_view_silent, charge_elsewhere, not_seeded, hw_known (outside interior_good only)
Populations: interior (as the pre-pass table), interior_good (det7: X ch 256-511) and, for det4, interior_live (dead /
low-gain bands out). Writes results/a1_<det>.json and results/a1.json; per-ray classes to OUTROOT/a1/<rel>/recat.parquet.
"""
import argparse, csv, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTROOT = '/eos/user/d/dneff/x17rerun/paper1_eff'
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
EXT_MAX, R_MM = 24, 5.0


def band_events(root, rel):
    p = os.path.join(root, 'bands', rel, 'bands.npz')
    if not os.path.exists(p):
        return None, None
    Z = np.load(p)
    ev = set(Z['eid_x'][Z['mask_x'] > 0].tolist()) | set(Z['eid_y'][Z['mask_y'] > 0].tolist())
    scanned = set(Z['eid_x'].tolist()) & set(Z['eid_y'].tolist())
    return ev, scanned


def recat_row(r, root):
    rel = f"{r['run']}/{r['subrun']}/{r['det']}"; d = SHORT[r['det']]
    R = pd.read_parquet(os.path.join(root, 'budget', rel, f'rays_{d}.parquet'))
    R = R[~R.not_measured & (R.region == 'interior')].set_index('event_id')
    a1 = os.path.join(root, 'a1', rel)
    K = pd.read_parquet(os.path.join(a1, f'classified_{d}.parquet')).set_index('event_id')
    Sp = pd.read_parquet(os.path.join(a1, f'sparks_{d}.parquet'))
    Sp = Sp.set_index('event_id').cls if len(Sp) else pd.Series(dtype=object)
    F = pd.read_parquet(os.path.join(a1, f'cand_feat_{d}.parquet'))
    C = pd.read_parquet(os.path.join(A, rel, 'wft', 'events.candidates.parquet'))
    C = C[(C['rank'] >= 0) & C.event_id.isin(R.index)]
    C = C.merge(F, on=['event_id', 'plane', 'rank'], how='left')
    # candidates of views with a single candidate carry no time extent (not extracted): one candidate is never 'M'
    C['particle'] = C.ext.fillna(0) <= EXT_MAX
    C['mesh'] = np.where(C.plane == 'x', R.mesh_x.reindex(C.event_id).to_numpy(), R.mesh_y.reindex(C.event_id).to_numpy())
    C['near'] = (C.p0 - C.mesh).abs() <= R_MM
    npart = C[C.particle].groupby(['event_id', 'plane']).size().unstack(fill_value=0).reindex(R.index, fill_value=0)
    multi = set(R.index[(npart.max(axis=1) >= 2) & (R.cat_xy_5 != 'spark')])
    nx = set(C[C.near & (C.plane == 'x')].event_id); ny = set(C[C.near & (C.plane == 'y')].event_id)
    band, scanned = band_events(root, rel)
    cat = []
    for e, x in R.iterrows():
        c = x.cat_xy_5
        if c == 'detected':
            k = 'detected'
        elif band is not None and e in band and c in ('rejected_cut', 'spark') and Sp.get(e, '') != 'discharge':
            k = 'band'
        elif e in multi:
            k = 'multi:' + ('m3_picked' if c == 'rejected_cut' else ('other_picked' if (e in nx and e in ny) else 'm3_not_found'))
        elif c == 'spark':
            k = 'spark:' + str(Sp.get(e, 'unclassified'))
        elif c == 'far':
            k = 'far:' + (str(K.cls.get(e)) if e in K.index and K.outlier.get(e, False) else 'unclassified')
        elif c == 'fit_failed':
            k = 'fit_failed'
        elif c == 'rejected_cut':
            k = 'cut:other'
        else:
            k = 'other:' + c
        cat.append(k)
    R['recat'] = cat
    R['good_live'] = R.good_x & ~R.hw
    R[['recat', 'cat_xy_5', 'good_x', 'hw', 'good_live']].to_parquet(os.path.join(a1, 'recat.parquet'))
    out = dict(key=r['key'], det=d, rel=rel, resist_V=r['resist_V'], drift_V=r['drift_V'],
               band_coverage=(float(np.isin(R.index, list(scanned)).mean()) if scanned is not None else None))
    pops = {'interior': np.ones(len(R), bool), 'interior_good': R.good_x.to_numpy()}
    if d == 'det4':
        pops['interior_live'] = (~R.hw).to_numpy()
    for nm, m in pops.items():
        U = R[m]; n = len(U); vc = U.recat.value_counts()
        ff = U[U.recat == 'fit_failed']
        out[nm] = dict(n=int(n), pct={k: float(100 * v / n) for k, v in vc.items()}, counts={k: int(v) for k, v in vc.items()},
                       fit_failed_headon_frac=float((ff.tan_x.abs() < 0.03).mean()) if len(ff) else None,
                       multi_frac=float(100 * U.index.isin(list(multi)).mean()),
                       detected_any_credit=float(100 * (U.cat_xy_5.eq('detected') | (U.index.isin(list(multi)) & U.index.isin(list(nx & ny)))).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=OUTROOT); ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    ap.add_argument('--registry', default=os.path.join(HERE, '..', '..', 'pass_registry.csv'))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.registry)) if r['cls'] in ('golden_calib', 'golden_quoted')]
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for r in rows:
        try:
            o = recat_row(r, a.root)
        except FileNotFoundError as e:
            print('missing', r['key'], e); continue
        res[r['key']] = o
        print(o['det'], r['key'], o['interior']['n'], {k: round(v, 2) for k, v in sorted(o['interior']['pct'].items(), key=lambda t: -t[1])})
    json.dump(res, open(os.path.join(a.out, 'a1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
