"""g_poly.py -- does the FINE structure of the bench map add anything beyond a smooth (plane / quadratic) tilt and scale surface?
(audit test G, step 4 follow-up).  Same held-out protocol as g_direct.py: train one run parity, test the other, both ways.

Models  a(x,y) = sum_k c_k phi_k + alpha * bench_a_shape(x,y) ;  s(x,y) = sum_k d_k phi_k + beta * bench_s_shape(x,y)
phi = (1, X, Y) 'plane' or (1, X, Y, X^2, XY, Y^2) 'quad', X = x_local/100, Y = y_local/100.  LS with 3-sigma clipping on the clean training lines.
"""
import os
import sys

os.environ.setdefault('X17_TQ_OUT', '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/g/tq')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import g_common as G
import g_direct as GD
from rerun.ntof.improve import angle_map as AM


def feats(x, y, order):
    X, Y = x / 100.0, y / 100.0
    cols = [np.ones_like(X)]
    if order >= 1:
        cols += [X, Y]
    if order >= 2:
        cols += [X * X, X * Y, Y * Y]
    return np.c_[tuple(cols)] if len(cols) > 1 else cols[0][:, None]


def design(x, y, j, order, bench):
    P = feats(x, y, order)
    cols = [P, P * j[:, None]]
    if bench is not None:
        a_sh, s_sh = bench
        ash = AM.interp(dict(a=a_sh, s=0 * a_sh, centres=G.CENTRES, a0=0.0, s0=0.0), x, y)[0]
        ssh = AM.interp(dict(a=s_sh, s=0 * s_sh, centres=G.CENTRES, a0=0.0, s0=0.0), x, y)[0]
        cols += [ash[:, None], (ssh * j)[:, None]]
    return np.hstack(cols), P.shape[1]


def fit(Gt, order, bench):
    j, r = Gt.j.to_numpy(), Gt.r.to_numpy()
    Xd, _ = design(Gt.x.to_numpy(), Gt.y.to_numpy(), j, order, bench)
    keep = np.ones(len(j), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(Xd[keep], r[keep], rcond=None)[0]
        res = r - Xd @ coef
        sig = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < 3 * sig
    return coef


def apply(coef, slope, x, y, order, bench):
    # a(x,y) and s(x,y) from the coefficient vector (the design with j = 0 gives a, the coefficient on j gives s)
    n = feats(x, y, order).shape[1]
    P = feats(x, y, order)
    a = P @ coef[:n]
    s = P @ coef[n:2 * n]
    if bench is not None:
        a_sh, s_sh = bench
        a = a + coef[2 * n] * AM.interp(dict(a=a_sh, s=0 * a_sh, centres=G.CENTRES, a0=0.0, s0=0.0), x, y)[0]
        s = s + coef[2 * n + 1] * AM.interp(dict(a=s_sh, s=0 * s_sh, centres=G.CENTRES, a0=0.0, s0=0.0), x, y)[0]
    return (slope - a) / (1.0 + s)


def main(margin=0.0):
    M = pd.read_parquet(GD.LINES)
    rn = M.run.str.replace('run_', '').astype(int)
    rng = np.random.default_rng(4)
    rows = []
    for s, arm in (('a', 'A'), ('c', 'C')):
        for p in 'xy':
            Gc = AM.clean_lines(M, s, p)
            parc = Gc.run.str.replace('run_', '').astype(int) % 2
            _, _, a_sh, s_sh, _, _ = GD.shape(arm, p, margin)
            for tr in (0, 1):
                Gt = Gc[parc == tr]
                te = (rn % 2) != tr
                j = M.loc[te, f'j{p}'].to_numpy()
                slope = (M.loc[te, f'd_{p}_{s}'] / M.loc[te, f'd_z_{s}']).to_numpy()
                x, y = M.loc[te, f'x_local_{s}'].to_numpy(), M.loc[te, f'y_local_{s}'].to_numpy()
                g = np.abs(j) < 0.45
                for lab, order, bench in [('const', 0, None), ('plane', 1, None), ('quad', 2, None), ('bench', 0, (a_sh, s_sh)),
                                          ('plane+bench', 1, (a_sh, s_sh)), ('quad+bench', 2, (a_sh, s_sh))]:
                    coef = fit(Gt, order, bench)
                    sc = apply(coef, slope, x, y, order, bench)
                    v = (sc - j)[g]
                    b = [G.mad(v[rng.integers(0, len(v), len(v))]) for _ in range(100)]
                    n = feats(x[:1], y[:1], order).shape[1]
                    rows.append(dict(arm=arm, view=p, train=tr, model=lab, sig=G.mad(v), err=float(np.std(b)),
                                     alpha=coef[2 * n] if bench else np.nan, beta=coef[2 * n + 1] if bench else np.nan))
                insitu = AM.fit_map(Gt) | dict(centres=G.CENTRES)
                v = (AM.correct(slope, x, y, insitu) - j)[g]
                rows.append(dict(arm=arm, view=p, train=tr, model='insitu', sig=G.mad(v), err=0.0, alpha=np.nan, beta=np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(f'{G.OUT}/poly_m{margin:.0f}.csv', index=False)
    T = R.groupby(['arm', 'view', 'model'], sort=False).agg(sig=('sig', 'mean'), err=('err', 'mean'), alpha=('alpha', 'mean'), beta=('beta', 'mean')).reset_index()
    out = ['| chamber view | const | plane | quad | bench shape | plane + bench | quad + bench | in-situ 8x8 | alpha, beta (plane + bench) |', '|---|---|---|---|---|---|---|---|---|']
    for arm, p in (('A', 'x'), ('A', 'y'), ('C', 'x'), ('C', 'y')):
        t = T[(T.arm == arm) & (T.view == p)].set_index('model')
        f = lambda m: f"{t.loc[m, 'sig']*1e3:.1f} +- {t.loc[m, 'err']*1e3:.1f}" if m != 'insitu' else f"{t.loc[m, 'sig']*1e3:.1f}"  # noqa: E731
        out.append(f"| {arm} {p} | " + ' | '.join(f(m) for m in ('const', 'plane', 'quad', 'bench', 'plane+bench', 'quad+bench', 'insitu'))
                   + f" | {t.loc['plane+bench', 'alpha']:.2f}, {t.loc['plane+bench', 'beta']:.2f} |")
    open(f'{G.OUT}/poly_m{margin:.0f}.md', 'w').write('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
