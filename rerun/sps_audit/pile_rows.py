#!/usr/bin/env python3
"""pile_rows.py --pass-dir /eos/.../Analysis_pass1/sps --out results/pile_rows.csv

Per pass-1 SPS row (read-only), the Y fits that sit on the two oscillating channels' strips: ch 372 (158.34 mm) and
ch 510 (250.38 mm).  Excess = the fraction in a 4.5 mm bin around the strip minus the mean of the two neighbouring
4.5 mm bins (the beam's own density there).  A row whose excess is above ~0.5 % had windows seeded on the oscillation
(sps_build_cache.SEED_BAD, rerun/sps_audit/README.md) and needs its sps node re-run.
"""
import argparse, os
import numpy as np
import pandas as pd

BINS = {'ch372': 158.34, 'ch510': 250.38}
W = 4.5


def excess(y, c):
    f = lambda lo: float(((y >= lo) & (y < lo + W)).mean())
    lo = c - W / 2
    return f(lo) - 0.5 * (f(lo - W) + f(lo + W)), f(lo)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pass-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = []
    for k in sorted(os.listdir(a.pass_dir)):
        p = os.path.join(a.pass_dir, k, 'events.parquet')
        if not os.path.exists(p):
            continue
        t = pd.read_parquet(p, columns=['y_ok', 'y_p0', 'ref_y'])
        y = t.y_p0[t.y_ok.astype(bool)].values
        r = dict(key=k, n_y_ok=len(y))
        for name, c in BINS.items():
            r[f'excess_{name}'], r[f'frac_{name}'] = excess(y, c)
            r[f'ref_excess_{name}'], _ = excess(t.ref_y.values, c)
        rows.append(r)
    df = pd.DataFrame(rows)
    df['excess_total'] = df.excess_ch372.clip(lower=0) + df.excess_ch510.clip(lower=0)
    df['rerun'] = df.excess_total > 0.005
    df.to_csv(a.out, index=False, float_format='%.4f')
    print(df.sort_values('excess_total', ascending=False).head(25).to_string(index=False))
    print(f'{int(df.rerun.sum())} of {len(df)} rows above 0.5 % excess')


if __name__ == '__main__':
    main()
