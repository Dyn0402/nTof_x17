#!/usr/bin/env python3
"""census_table.py <census dir> <out.csv>: the census jsons (census.py) as one table; prints the rows that change."""
import glob, json, os, sys
import pandas as pd

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    j = json.load(open(f))
    rows.append(dict(key=j['key'], changes=j['changes'], n_new=j['n_new'], n_old=j['n_old'], only_new=j['only_new'],
                     only_old=j['only_old'], ref_moved=j['ref_moved'], censored_events=j['censored_events'],
                     hot_old=' '.join(map(str, j['hot_old'])), hot_new=' '.join(map(str, j['hot_new'])),
                     coupled=json.dumps(j['coupled']), z372_old=j['junk_old']['z372'], z372_new=j['junk_new']['z372'],
                     z510_old=j['junk_old']['z510'], z510_new=j['junk_new']['z510']))
df = pd.DataFrame(rows); df.to_csv(sys.argv[2], index=False, float_format='%.4f')
with pd.option_context('display.width', 250, 'display.max_colwidth', 60, 'display.max_rows', 200):
    print(df.drop(columns=['n_old']).to_string(index=False))
print(f'{int(df.changes.sum())} of {len(df)} rows change')
