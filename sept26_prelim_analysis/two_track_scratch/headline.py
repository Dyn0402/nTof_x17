"""Handoff §2 table: coincident overlays, both donors found and correctly paired."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sept26_prelim_analysis import intra_bench as ib

BASE = os.path.realpath(os.path.expanduser('~/x17/sept26_prelim/intra_bench'))
BANDS = [(0, 12, '<12'), (12, 24, '12-24'), (24, 1e9, '>=24')]


def scored(v):
    d = os.path.join(BASE, v) if v else BASE
    M = pd.read_parquet(d + '/overlays.parquet')
    C = pd.read_parquet(d + '/candidates.parquet')
    dp = d + '/donors.parquet'
    D = pd.read_parquet(dp if os.path.exists(dp) else BASE + '/donors.parquet')
    return ib.score(M, C, D)


def table(S, cls='coincident'):
    o = S[(S['mode'] == 'overlay') & (S.cls == cls)]
    ev = o.groupby('oid').agg(arm=('arm', 'first'), sep=('sep_x', 'first'),
                              sepy=('sep_y', 'first'), both=('track_found', 'all'))
    ev['sep'] = np.minimum(ev.sep, ev.sepy)
    rows = {}
    for lo, hi, lab in BANDS:
        g = ev[(ev.sep >= lo) & (ev.sep < hi)]
        rows[lab] = g.groupby('arm').both.mean()
    return pd.DataFrame(rows).T


def donor_split(S):
    o = S[(S['mode'] == 'overlay') & (S.sbin >= 4)]
    return o.groupby(['arm', 'donor']).track_found.mean().unstack()


if __name__ == '__main__':
    for v in sys.argv[1:]:
        S = scored(v)
        print(f'== {v or "production"}')
        for cls in ('coincident', 'offset'):
            print(cls); print((100 * table(S, cls)).round(1).to_string())
        print('>=24 mm, per donor'); print((100 * donor_split(S)).round(1).to_string())
        if 'noise' in set(S['mode']):
            n = S[S['mode'] == 'noise']
            print('noise control, a found:', (100 * n.groupby('arm').track_found.mean()).round(1).to_dict())
