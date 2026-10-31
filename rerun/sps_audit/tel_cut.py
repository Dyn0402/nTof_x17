#!/usr/bin/env python3
"""tel_cut.py --hits hits_run53.npz [...] [--out results/tel_cut.json]

Justifies the telescope cut (one cluster in each of the four uRWELL plane-views) from the telescope alone: for every
four-view event of a ZS run (hits_baseline.py), the back plane against the front plane extrapolated (MAPPING_AND_ALIGNMENT.md
§2: back = slope x front + offset), per coordinate.  If the multi-cluster events were as clean, their back-front residual
would have the same core fraction.
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SLOPE = {'x': (0.99960, -0.958), 'y': (1.04177, -4.107)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--hits', nargs='+', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'tel_cut.json'))
    a = ap.parse_args()
    out = {}
    for f in a.hits:
        H = np.load(f); name = os.path.basename(f)[:-4]
        single = np.all([H[f'{p}_n'] == 1 for p in ('fx', 'fy', 'bx', 'by')], axis=0)
        R = dict(n_four_view=int(len(single)), frac_single=float(single.mean()))
        for c in 'xy':
            r = H[f'b{c}_p'] - (SLOPE[c][0] * H[f'f{c}_p'] + SLOPE[c][1]); r = r - np.nanmedian(r[single])
            for tag, m in (('single', single), ('multi', ~single)):
                R[f'{c}_{tag}_within_3mm'] = float(np.mean(np.abs(r[m]) < 3.0))
                R[f'{c}_{tag}_s68_mm'] = float(np.nanpercentile(np.abs(r[m]), 68.27))
        out[name] = R
        print(name, {k: round(v, 3) for k, v in R.items()})
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
