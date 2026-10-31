#!/usr/bin/env python3
"""chi2_scaling.py [--out results/chi2_scaling.json] -- how the data's chi2/dof grows with amplitude, against the toy.

With noise-only weights, an under-modelled component of the waveform shows in chi2/dof as
  flat in Q       extra additive noise (not in the pedestal sigma)
  ~ Q             a fluctuation that scales with the signal's own statistics (gain / cluster-size type)
  ~ Q^2           a fixed shape mismatch (template, footprint, kernel) -- a model error proportional to the signal
Data: unsaturated planes of the calibration rows (interior_good), chi2/dof in q_sum deciles, slope of log(chi2/dof - 1)
against log q_sum.  The toy has one amplitude scale per point (Polya and cluster fluctuations only), so its chi2/dof
is read at the data's median charge (toy_inputs q50).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import floorlib as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(C.HERE, 'results', 'chi2_scaling.json'))
    a = ap.parse_args()
    out = {}
    for det, key in C.CALIB.items():
        F = C.load(key)
        for p in ('x', 'y'):
            g = C.sample(F, p)
            g = g[(g.n_sat_strips == 0) & (g.q_sum > 0)]
            g = g.assign(cd=g.chi2 / g.dof, qd=pd.qcut(g.q_sum, 10, labels=False, duplicates='drop'))
            t = g.groupby('qd').agg(q=('q_sum', 'median'), cd=('cd', 'median'), n=('cd', 'size'))
            ok = t.cd > 1.5
            sl = float(np.polyfit(np.log(t.q[ok]), np.log(t.cd[ok] - 1), 1)[0]) if ok.sum() >= 4 else np.nan
            out[f'{det}|{p}'] = dict(slope_log_excess=sl, deciles=[(float(q), float(c)) for q, c in zip(t.q, t.cd)])
            print(det, p, f'd log(chi2/dof - 1) / d log Q = {sl:.2f}', [(int(q), round(c, 1)) for q, c in zip(t.q, t.cd)][::3])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
