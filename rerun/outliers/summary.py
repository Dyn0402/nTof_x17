#!/usr/bin/env python3
"""summary.py -- every number of the outlier study into one results.json (make_report.py reads only this).

Inputs: classified_<det>.parquet (classify.py), an/edge_response.json, an/kinkvtx.json, an/bent_fraction.json,
sixpt.parquet, sps/*.json, ../pogo_check/results/sps_run71_*.json.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE]
from olib import BASE, DETS, HEADON, FAR, load  # noqa: E402
from classify import ORDER  # noqa: E402

GROUPS = {'not a failure': ['REJECTED_OK'], 'geometry (acceptance, edge field)': ['OUTSIDE', 'EDGE'],
          'known hardware': ['HW_KNOWN'], 'reference (M3)': ['M3_ELSEWHERE', 'POS_TAIL_BENT'],
          'second particle / delta ray': ['MULTI', 'DELTA'],
          'MX17 response / reconstruction': ['MX17_DEAD', 'ONE_VIEW_ELSEWHERE', 'POS_TAIL_MX17', 'FIT_SAT', 'FIT']}


def main():
    R = {'dets': list(DETS), 'order': ORDER, 'classes': {}, 'headon': [], 'groups': {}}
    bent = json.load(open(os.path.join(BASE, 'an', 'bent_fraction.json')))
    for det in DETS:
        d = load(det); n = len(d)
        x = pd.read_parquet(os.path.join(BASE, f'classified_{det}.parquet'))
        o = x[x.outlier]; vc = o.cls.value_counts()
        cl = {k: dict(n=int(vc.get(k, 0)), pct=100 * vc.get(k, 0) / n) for k in ORDER}
        f = bent[det]['f']
        cl['POS_TAIL_BENT'] = dict(n=f * cl['POS_TAIL']['n'], pct=f * cl['POS_TAIL']['pct'])
        cl['POS_TAIL_MX17'] = dict(n=(1 - f) * cl['POS_TAIL']['n'], pct=(1 - f) * cl['POS_TAIL']['pct'])
        R['classes'][det] = dict(n_events=n, n_outliers=int(len(o)), pct_outliers=100 * len(o) / n, cls=cl,
                                 bent_f=f, bent_err=bent[det]['err'])
        R['groups'][det] = {g: sum(cl[k]['pct'] for k in ks) for g, ks in GROUPS.items()}
        d = d.merge(x[['event_id', 'cls']], on='event_id', how='left')
        for v in 'xy':
            u = d[d[f'use_{v}']]; far = u[f'dl_{v}'].abs() > FAR
            for lab, m in (('head-on', u[f'ref_tan_{v}'].abs() < HEADON), ('inclined', u[f'ref_tan_{v}'].abs() >= HEADON)):
                c = u[m & far].cls.value_counts(); nn = int(m.sum())
                known = sum(c.get(k, 0) for k in ('OUTSIDE', 'EDGE', 'HW_KNOWN'))
                R['headon'].append(dict(det=det, view=v, cls=lab, n=nn, far=100 * far[m].mean(),
                                        far_after_known=100 * (far[m].sum() - known) / nn,
                                        by=dict((k, 100 * c.get(k, 0) / nn) for k in ORDER)))
    R['groups_def'] = GROUPS
    R['edge'] = json.load(open(os.path.join(BASE, 'an', 'edge_response.json')))
    R['kinkvtx'] = json.load(open(os.path.join(BASE, 'an', 'kinkvtx.json')))
    S = pd.read_parquet(os.path.join(BASE, 'sixpt.parquet'))
    R['sixpt'] = dict(n_views=len(S), n_events=int(S.event_id.nunique()),
                      worst={str(t): S[(S.chi2_6 > t) & (S.chi2_5 < 10)].worst.value_counts().to_dict()
                             for t in (10, 20, 50)})
    sps = {}
    for f in sorted(glob.glob(os.path.join(BASE, 'sps', '*.json'))):
        try:
            sps[os.path.basename(f)[:-5]] = json.load(open(f))['summary']
        except Exception:
            pass
    R['sps'] = sps
    pogo = {}
    for f in sorted(glob.glob(os.path.join(REPO, 'rerun', 'pogo_check', 'results', '*.json'))):
        z = json.load(open(f))
        pogo[os.path.basename(f)[:-5]] = [c['verdict'] for c in z.get('cards', [])]
    R['pogo'] = pogo
    json.dump(R, open(os.path.join(BASE, 'report', 'results.json'), 'w'), indent=1, default=float)
    print('wrote', os.path.join(BASE, 'report', 'results.json'))
    for det in DETS:
        print(det, {g: round(v, 2) for g, v in R['groups'][det].items()})


if __name__ == '__main__':
    os.makedirs(os.path.join(BASE, 'report', 'figures'), exist_ok=True)
    main()
