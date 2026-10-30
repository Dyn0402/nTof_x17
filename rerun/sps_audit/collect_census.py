#!/usr/bin/env python3
"""collect_census.py DIR [--out results/seed_census.csv]: one row per pass-1 SPS row from seed_census.py's json files,
with the re-run verdict (any event whose pre-fix seed the committed fix excludes)."""
import argparse, glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'seed_census.csv'))
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, '*.json'))):
        d = json.load(open(f))
        rows.append(dict(key=d['key'], n_events=d['n_events'], n_changed=d['n_changed'], frac=d['n_changed'] / max(d['n_events'], 1),
                         hot=' '.join(map(str, d['hot'])),
                         seeds_on_excluded=' '.join(f'{k}:{v}' for k, v in d['seeds_on_excluded'].items())))
    df = pd.DataFrame(rows).sort_values('frac', ascending=False)
    df['rerun'] = df.n_changed > 0
    df.to_csv(a.out, index=False, float_format='%.4f')
    print(df.to_string(index=False))
    print(f'{len(df)} rows; re-run (any changed event) {int(df.rerun.sum())}; above 1 %: {int((df.frac > 0.01).sum())}')


if __name__ == '__main__':
    main()
