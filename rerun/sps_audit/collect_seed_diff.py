#!/usr/bin/env python3
"""collect_seed_diff.py DIR [--out results/seed_diff.csv]: one row per pass-1 SPS row from seed_diff.py's json files."""
import argparse, glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'seed_diff.csv'))
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, '*.json'))):
        d = json.load(open(f))
        o = d['old_seed_y_of_changed']
        # the jsons were made with a wider seed mask (census-dead channels too); the committed fix excludes only the
        # oscillating ch 372/510, so the events it changes are exactly those whose pre-fix Y seed was one of them
        rows.append(dict(key=d['key'], n_events=d['n_events'], n_seed_372=int(o.get('372', 0)), n_seed_510=int(o.get('510', 0)),
                         n_changed_wide_mask=len(d['changed_keys'])))
    df = pd.DataFrame(rows)
    df['n_changed'] = df.n_seed_372 + df.n_seed_510
    df['frac'] = df.n_changed / df.n_events.clip(lower=1)
    df.sort_values('frac', ascending=False).to_csv(a.out, index=False, float_format='%.4f')
    print(df.sort_values('frac', ascending=False).to_string(index=False))
    print(f'{len(df)} rows; {int((df.n_changed > 0).sum())} with any changed event; {int((df.frac > 0.001).sum())} above 0.1 %')


if __name__ == '__main__':
    main()
