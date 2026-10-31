#!/usr/bin/env python3
"""compare.py --stage S --out O.csv [--pass-dir P]: per staged SPS row against its pass-1 product, the oscillation
junk before/after and what else moved.

Junk = the excess of Y fits (y_p0, y_ok) in a 4.5 mm bin around ch 372/373 (158.0 mm), around ch 375/376 (155.6 mm) and
around ch 510 (250.38 mm) over the mean of the two neighbouring bins (the beam's own density): rerun/sps_audit/pile_rows.py
with the 155.6 mm bin added.  Also: events in common, y_ok / x_ok, and the median |shift| of y_p0 and x_p0 on the events
whose Y window did not hold a coupled strip (those must be unchanged)."""
import argparse, json, os
import numpy as np
import pandas as pd

PASS = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps'
BINS = {'b158': 158.0, 'b155': 155.6, 'b250': 250.38}
W = 4.5


def excess(y):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    f = lambda lo: float(((y >= lo) & (y < lo + W)).mean()) if len(y) else np.nan
    return {k: f(c - W / 2) - 0.5 * (f(c - 1.5 * W) + f(c + 0.5 * W)) for k, c in BINS.items()}


def zone(y, lo=153.5, hi=160.5):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    return float(((y >= lo) & (y < hi)).mean()) if len(y) else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--pass-dir', default=PASS)
    a = ap.parse_args()
    rows = []
    for k in sorted(os.listdir(a.stage)):
        ps, pp = os.path.join(a.stage, k, 'events.parquet'), os.path.join(a.pass_dir, k, 'events.parquet')
        if not (os.path.exists(ps) and os.path.exists(pp)):
            continue
        new, old = pd.read_parquet(ps), pd.read_parquet(pp)
        r = dict(key=k, n_old=len(old), n_new=len(new))
        for tag, t in (('old', old), ('new', new)):
            y = t.y_p0.where(t.y_ok.astype(bool))
            for b, v in excess(y).items():
                r[f'{b}_{tag}'] = v
            r[f'zone_{tag}'] = zone(y)
            r[f'yok_{tag}'], r[f'xok_{tag}'] = float(t.y_ok.mean()), float(t.x_ok.mean())
        m = old.merge(new, on='event_id', suffixes=('_o', '_n'))
        r['n_common'] = len(m)
        clean = m['y_n_coupled'].fillna(0).eq(0) & m['x_n_coupled'].fillna(0).eq(0) if 'y_n_coupled' in m else np.ones(len(m), bool)
        for p in 'xy':
            d = (m[f'{p}_p0_n'] - m[f'{p}_p0_o']).abs()
            r[f'{p}_moved_clean'] = int((d[clean] > 1e-3).sum())
            r[f'{p}_moved_all'] = int((d > 1e-3).sum())
            r[f'{p}_n_coupled_events'] = int((m[f'{p}_n_coupled'].fillna(0) > 0).sum()) if f'{p}_n_coupled' in m else 0
        meta = json.load(open(os.path.join(a.stage, k, 'events.meta.json')))
        r['hot'] = ' '.join(map(str, meta.get('hot_seed_channels') or []))
        r['coupled'] = json.dumps(meta.get('coupled_channels') or {})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format='%.4f')
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(df[['key', 'n_old', 'n_new', 'n_common', 'zone_old', 'zone_new', 'b155_old', 'b155_new', 'b158_old',
                  'b158_new', 'b250_old', 'b250_new', 'y_moved_clean', 'x_moved_clean', 'y_n_coupled_events', 'hot']].to_string(index=False))


if __name__ == '__main__':
    main()
