#!/usr/bin/env python3
"""pass_hists.py --pass-dir /eos/.../Analysis_pass1/sps --out results/pass_hists.json

Position histograms of a few pass-1 SPS rows (read-only), for the report: the run71 RAW rows, their ZS twins, and the
ZS rows the oscillating channels also hit.
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROWS = ['run71_raw__raw700', 'run70_flat_drift__d700', 'run66_flat_resist__r780', 'run55_flatdrift__d700',
        'run57_rot25_co2__d700', 'run60_flush__ov07']
EDGES = np.arange(90.0, 310.01, 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pass-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = dict(edges=EDGES.tolist())
    for k in ROWS:
        t = pd.read_parquet(os.path.join(a.pass_dir, k, 'events.parquet'), columns=['x_ok', 'x_p0', 'y_ok', 'y_p0'])
        out[k] = {v: np.histogram(t[f'{v}_p0'][t[f'{v}_ok'].astype(bool)], bins=EDGES)[0].tolist() for v in 'xy'}
    json.dump(out, open(a.out, 'w'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
