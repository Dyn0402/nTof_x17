#!/usr/bin/env python3
"""spstune_score.py '<glob of spstune_tr*_st*.parquet>' [--stage 25.4] [--json OUT] -- score the SPS H2c tuning grid
(t_rise x S_T, h2c_run.py on one row) on the inclined Y view with sps_score.row_score: the same v_stage and the same
score as the hits families.

All grid points are the same planes, so differences between them are measured with a paired bootstrap over planes.
The cost column is reported too: a fit that ends at a very large cost or a slope several times W's is a failed fit, and
it stays in the score (the tails enter s68 and in5 as they would in a real analysis).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sps_score import row_score  # noqa: E402
from tiers import s68  # noqa: E402

NBOOT = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tables')
    ap.add_argument('--stage', type=float, default=25.4)
    ap.add_argument('--json')
    a = ap.parse_args()
    res, err = {}, {}
    ts = np.tan(np.radians(a.stage))
    for f in sorted(glob.glob(a.tables)):
        tr, st = (int(x) for x in re.search(r'tr(\d+)_st(\d+)', os.path.basename(f)).groups())
        T = pd.read_parquet(f)
        r = row_score(T, a.stage)
        V = r['views']['y']
        t = V['tiers']['V0/H2c_dream']
        s = t['score']['all']
        g = T[T.view == 'y']
        w = g.w_W.to_numpy(float)
        tr_tan = np.sign(w) * ts
        d = lambda sl: np.degrees(np.arctan(sl / r['v_mm_ns']) - np.arctan(tr_tan))          # as tiers.score
        dh, dw, dv = d(g.w_h2c_dream.to_numpy(float)), d(w), d(g.u_dream.to_numpy(float))
        err[(tr, st)] = dh
        wgood = np.abs(dw) <= 5                 # planes where W is on the stage angle
        sub = {}
        for nm, m in (('W_on', wgood), ('W_off', ~wgood)):
            sub[nm] = dict(n=int(m.sum()), h2c_s68=s68(dh[m])[0], h2c_in5=float(np.mean(np.abs(dh[m]) <= 5)),
                           V0_s68=s68(dv[m])[0], V0_in5=float(np.mean(np.abs(dv[m]) <= 5)),
                           W_s68=s68(dw[m])[0], W_in5=float(np.mean(np.abs(dw[m]) <= 5)))
        wild = float(np.mean(np.abs(g.w_h2c_dream / g.w_W) > 2))
        res[f'tr{tr}_st{st}'] = dict(t_rise=tr, s_t=st, n=int(V['n']), v_um_ns=r['v_mm_ns'] * 1e3, ratio=t['ratio'], s68=s['s68'],
                                     bias=s['bias'], in5=s['frac_in5'], cov=t['coverage'], wild_frac=wild,
                                     W_s68=V['W']['all']['s68'], W_in5=V['W']['all']['frac_in5'],
                                     V0_ratio=V['tiers']['V0_dream']['ratio'], V0_s68=V['tiers']['V0_dream']['score']['all']['s68'],
                                     V0_in5=V['tiers']['V0_dream']['score']['all']['frac_in5'],
                                     sub=sub, cost_median=float(g.cost_dream.median()), cost_p90=float(g.cost_dream.quantile(.9)))
    keys = sorted(err)
    n = len(err[keys[0]])
    rng = np.random.default_rng(1)
    B = np.array([[s68(err[k][i])[0] for k in keys]
                  for i in (rng.integers(0, n, n) for _ in range(NBOOT))])
    best = int(np.nanargmin(np.nanmedian(B, 0)))
    for j, k in enumerate(keys):
        d = B[:, j] - B[:, best]
        o = res[f'tr{k[0]}_st{k[1]}']
        o.update(boot_s68_err=float(np.nanstd(B[:, j])), delta_to_best=float(np.nanmedian(d)), delta_err=float(np.nanstd(d)))
    print(f'inclined Y, stage {a.stage} deg, v {res[next(iter(res))]["v_um_ns"]:.2f} um/ns, n {n}; '
          f'W s68 {res[next(iter(res))]["W_s68"]:.2f} in5 {res[next(iter(res))]["W_in5"]:.3f}; '
          f'V0 ratio {res[next(iter(res))]["V0_ratio"]:.3f} s68 {res[next(iter(res))]["V0_s68"]:.2f} in5 {res[next(iter(res))]["V0_in5"]:.3f}')
    print(f'{"t_rise":>6s} {"S_T":>4s} {"ratio":>6s} {"s68":>5s} {"+-":>5s} {"bias":>6s} {"in5":>6s} {"wild":>6s} {"cost50":>7s} {"cost90":>7s} {"d(best)":>12s}')
    for k in keys:
        o = res[f'tr{k[0]}_st{k[1]}']
        print(f'{k[0]:6d} {k[1]:4d} {o["ratio"]:6.3f} {o["s68"]:5.2f} {o["boot_s68_err"]:5.2f} {o["bias"]:+6.2f} {o["in5"]:6.3f} {o["wild_frac"]:6.3f}'
              f' {o["cost_median"]:7.1f} {o["cost_p90"]:7.1f} {o["delta_to_best"]:+6.2f}+-{o["delta_err"]:.2f}')
    print('split by whether W is within 5 deg of the stage angle (selection on W: favours W)')
    print(f'{"t_rise":>6s} {"S_T":>4s} | W-on n, H2c s68 in5, V0 s68 in5, W s68 | W-off n, H2c in5, V0 in5')
    for k in keys:
        q = res[f'tr{k[0]}_st{k[1]}']['sub']
        on, off = q['W_on'], q['W_off']
        print(f'{k[0]:6d} {k[1]:4d} | {on["n"]:3d} {on["h2c_s68"]:.2f} {on["h2c_in5"]:.3f}  {on["V0_s68"]:.2f} {on["V0_in5"]:.3f}  {on["W_s68"]:.2f}'
              f' | {off["n"]:3d} {off["h2c_in5"]:.3f} {off["V0_in5"]:.3f}')
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
