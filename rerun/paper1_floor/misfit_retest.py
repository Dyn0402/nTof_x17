#!/usr/bin/env python3
"""misfit_retest.py [--out results/misfit_retest.json] -- the 10-27 finding 'the unexplained part is not a waveform misfit'
(FD REPORT: resolution flat across the bottom four chi2/dof quintiles, flat in charge) re-tested on pass 1.

Per calibration row and view (interior_good, v(t) arm, after the 2D tilt map): s68 in chi2/dof quintiles, and in chi2/dof
rank WITHIN amplitude deciles (the misfit at fixed brightness: chi2 grows with amplitude under noise-only weights, K1),
on unsaturated planes; and the data's median chi2/dof against the X toy's (L8 rung, ladder_pass1_ig_n1500).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import floorlib as C
from analyze import derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(C.HERE, 'results', 'misfit_retest.json'))
    a = ap.parse_args()
    out = {}
    for det, key in C.CALIB.items():
        lf = os.path.join(C.REPO, 'rerun', 'model_audit', 'followups', 'fd_floor_realism', 'results',
                          f'ladder_pass1_ig_n1500_{det}.json')
        T = json.load(open(lf))['rungs']['L8']
        toy_chi = float(np.median([T[k]['chi2dof'] for k in ('3', '9', '16')]))
        F = C.load(key)
        for p in ('x', 'y'):
            g = derived(C.sample(F, p))
            q = np.nanpercentile(g.cd, [0, 20, 40, 60, 80, 100])
            q[-1] += 1e-9
            quint = [C.s68(g.dt[(g.cd >= lo) & (g.cd < hi)]) for lo, hi in zip(q[:-1], q[1:])]
            u = g.n_sat_strips == 0
            fixed = [C.s68(g.dt[u & (g.cdr >= lo) & (g.cdr < hi)]) for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, .9),
                                                                                (.9, 1.01))]
            # the same rank within amplitude-decile x angle-tercile x strip-count-tercile cells: low chi2 must not just
            # mean an easier geometry
            ath = np.abs(g.tan_ref)
            cell = (g.qdec.astype(str) + '|' + pd.qcut(ath, 3, labels=False, duplicates='drop').astype(str) + '|'
                    + pd.qcut(g.n_strips.rank(method='first'), 3, labels=False).astype(str))
            g['cdr3'] = g.groupby(cell).cd.rank(pct=True)
            fixed3 = [C.s68(g.dt[u & (g.cdr3 >= lo) & (g.cdr3 < hi)]) for lo, hi in ((0, .25), (.25, .5), (.5, .75),
                                                                                    (.75, .9), (.9, 1.01))]
            allu = C.s68(g.dt[u])
            share = (allu ** 2 - fixed3[0] ** 2) if np.isfinite(fixed3[0]) else np.nan
            out[f'{det}|{p}'] = dict(chi2dof_median=float(np.nanmedian(g.cd)), toy_chi2dof_median_x=toy_chi,
                                     s68_chi2_quintiles=quint, s68_chi2_rank_at_amplitude=fixed,
                                     s68_chi2_rank_at_amp_angle_nstrips=fixed3, s68_unsat=allu,
                                     misfit_delta2_vs_best_quartile=float(share))
            print(det, p, f'chi2/dof data {np.nanmedian(g.cd):.0f} toy(X) {toy_chi:.0f}', 'quintiles',
                  np.round(quint, 3), 'rank at amplitude (unsat)', np.round(fixed, 3), '| at amp x angle x nstrips',
                  np.round(fixed3, 3), f'| unsat all {allu:.3f} -> best quartile: Delta^2 {share:.3f}')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
