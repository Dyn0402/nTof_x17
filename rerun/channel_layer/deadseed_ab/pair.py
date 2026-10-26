"""pair.py <row wft dir> <det>: per-ray category transitions s0 -> s1 and per-event fit changes."""
import sys, numpy as np, pandas as pd
W, det = sys.argv[1], sys.argv[2]
R = {s: pd.read_parquet(f'{W}/ds{s}/budget/rays_{det}.parquet') for s in (0, 1)}
cols = [c for c in R[0].columns if c.startswith('cat')]
print('rays', len(R[0]), 'category columns', cols)
for c in cols:
    m = R[0][[c]].join(R[1][[c]], rsuffix='_1')
    ch = m[m[c] != m[c + '_1']]
    print(f'{c}: {len(ch)} rays change category')
    if len(ch):
        print(ch.groupby([c, c + '_1']).size().sort_values(ascending=False).head(12).to_string())
E = {s: pd.read_parquet(f'{W}/ds{s}/events.parquet') for s in (0, 1)}
j = E[0].merge(E[1], on='event_id', suffixes=('_0', '_1'))
print('events', len(E[0]), len(E[1]), 'common', len(j))
for v in 'xy':
    a, b = f'{v}_p0_0', f'{v}_p0_1'
    if a in j:
        d = (j[a] - j[b]).abs()
        print(v, 'p0 changed > 0.01 mm:', int((d > 0.01).sum()), ' > 1 mm:', int((d > 1).sum()),
              ' n_strips changed:', int((j[f'{v}_n_strips_0'] != j[f'{v}_n_strips_1']).sum()) if f'{v}_n_strips_0' in j else '?')
