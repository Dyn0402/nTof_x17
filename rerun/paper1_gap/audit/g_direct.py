"""g_direct.py -- apply the bench map to the n_TOF A-C cosmic lines and compute the held-out single-view sigma (audit test G, step 4).

Read only on the n_TOF side.  Lines: ana_r2/track_quality/lines.parquet (31,387 A-C lines of the true beam-off runs, built on the
unmapped v(t) slopes, 2026-11-23).  Models, per chamber and view, trained on one run parity and tested on the other (both ways):
  none      no correction
  global    one tilt and one scale per chamber and view (the in-situ model with a single cell): c + gamma j
  insitu    the 8x8 in-situ map (angle_map.fit_map on the training parity)
  bench_raw the bench map as it is (bench a, s per cell; sign-mapped to the n_TOF frame), cells without bench data take the bench chamber-wide value
  bench_1   global (trained) + the bench SHAPE (bench a, s minus their median over cells) with unit amplitude
  bench_fit global (trained) + the bench shape with a free amplitude each for a and s (trained)
  insitu_x_bench: the in-situ map with its shape replaced by the in-situ shape of the other parity (reference for the noise ceiling)
Held-out sigma = MAD of (corrected slope - joined slope) on the test parity: 'all' lines |j| < 0.45, and 'clean' lines (angle_map.clean_lines).
"""
import json
import os
import pickle
import sys

os.environ.setdefault('X17_TQ_OUT', '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/g/tq')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import g_common as G
from rerun.ntof.improve import angle_map as AM

LINES = '/media/ucla/work/ntof_pass1/ana_r2/track_quality/lines.parquet'
BENCH = pickle.load(open(f'{G.OUT}/bench_maps.pkl', 'rb'))
NBOOT = 200


def mp_from(A, S, a0=0.0, s0=0.0):
    A = np.where(np.isfinite(A), A, a0)
    S = np.where(np.isfinite(S), S, s0)
    return dict(a=A, s=S, centres=G.CENTRES, a0=a0, s0=s0)


def shape(arm, p, margin):
    B = BENCH[margin][arm]['all'][p]
    sg = G.SGN[(arm, p)]
    a, s = sg * B['a'], B['s']
    a_sh = a - np.nanmedian(a)
    s_sh = s - np.nanmedian(s)
    return a, s, np.where(np.isfinite(a_sh), a_sh, 0.0), np.where(np.isfinite(s_sh), s_sh, 0.0), sg * B['a0'], B['s0']


def sig_boot(v, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = [G.mad(v[rng.integers(0, len(v), len(v))]) for _ in range(NBOOT // 2)]
    return G.mad(v), float(np.std(b))


def fit_amp(Gt, a_sh_mp, s_sh_mp, c0, g0):
    """r - c - gamma j = alpha*ash(x,y) + beta*ssh(x,y) j; LS with 3-sigma clipping on the training lines.  Returns (c, gamma, alpha, beta)."""
    ash, _ = AM.interp(dict(a=a_sh_mp['a'], s=a_sh_mp['s'], centres=G.CENTRES, a0=0.0, s0=0.0), Gt.x.to_numpy(), Gt.y.to_numpy())
    ssh = AM.interp(dict(a=s_sh_mp['a'], s=s_sh_mp['s'], centres=G.CENTRES, a0=0.0, s0=0.0), Gt.x.to_numpy(), Gt.y.to_numpy())[0]
    j, r = Gt.j.to_numpy(), Gt.r.to_numpy()
    X = np.c_[np.ones_like(j), j, ash, ssh * j]
    keep = np.ones(len(j), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(X[keep], r[keep], rcond=None)[0]
        res = r - X @ coef
        sig = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < 3 * sig
    cov = np.linalg.pinv(X[keep].T @ X[keep]) * (sig ** 2)
    return coef, np.sqrt(np.diag(cov))


def main(margin=0.0):
    M = pd.read_parquet(LINES)
    rn = M.run.str.replace('run_', '').astype(int)
    rng = np.random.default_rng(11)
    rows, amps = [], []
    for s, arm in (('a', 'A'), ('c', 'C')):
        for p in 'xy':
            Gc = AM.clean_lines(M, s, p)
            parc = Gc.run.str.replace('run_', '').astype(int) % 2
            a_b, s_b, a_sh, s_sh, a0b, s0b = shape(arm, p, margin)
            for tr in (0, 1):
                Gt = Gc[parc == tr]
                te = (rn % 2) != tr
                j = M.loc[te, f'j{p}'].to_numpy()
                slope = (M.loc[te, f'd_{p}_{s}'] / M.loc[te, f'd_z_{s}']).to_numpy()
                x, y = M.loc[te, f'x_local_{s}'].to_numpy(), M.loc[te, f'y_local_{s}'].to_numpy()
                g = np.abs(j) < 0.45
                Gs = Gc[parc != tr]
                slope_c = Gs.r.to_numpy() + Gs.j.to_numpy()
                a_g, s_g, _ = AM.robust_line(Gt.j.to_numpy(), Gt.r.to_numpy())

                def evals(mp, tag):
                    sc = AM.correct(slope, x, y, mp)
                    cc = AM.correct(slope_c, Gs.x.to_numpy(), Gs.y.to_numpy(), mp)
                    sa, ea = sig_boot((sc - j)[g], rng)
                    sc_, ec = sig_boot(cc - Gs.j.to_numpy(), rng)
                    rows.append(dict(arm=arm, view=p, train=tr, model=tag, sig_all=sa, err_all=ea, sig_clean=sc_, err_clean=ec,
                                     tail_all=float(np.mean(np.abs(sc - j)[g] > 0.15)), n_all=int(g.sum()), n_clean=len(Gs)))

                const = lambda c, gm: dict(a=np.full((G.NC, G.NC), c), s=np.full((G.NC, G.NC), gm), centres=G.CENTRES, a0=c, s0=gm)  # noqa: E731
                evals(const(0.0, 0.0), 'none')
                evals(const(a_g, s_g), 'global')
                evals(AM.fit_map(Gt) | dict(centres=G.CENTRES), 'insitu')
                evals(mp_from(a_b, s_b, a0b, s0b), 'bench_raw')
                A1 = a_g + a_sh
                S1 = s_g + s_sh
                evals(dict(a=A1, s=S1, centres=G.CENTRES, a0=a_g, s0=s_g), 'bench_1')
                coef, err = fit_amp(Gt, dict(a=a_sh, s=np.zeros_like(a_sh)), dict(a=s_sh, s=np.zeros_like(s_sh)), a_g, s_g)
                Af = coef[0] + coef[2] * a_sh
                Sf = coef[1] + coef[3] * s_sh
                evals(dict(a=Af, s=Sf, centres=G.CENTRES, a0=coef[0], s0=coef[1]), 'bench_fit')
                amps.append(dict(arm=arm, view=p, train=tr, c=coef[0], gamma=coef[1], alpha=coef[2], alpha_err=err[2], beta=coef[3], beta_err=err[3]))
                # a-only free amplitude (scale shape ignored): the scale map is the weaker one
                X_ = fit_amp(Gt, dict(a=a_sh, s=np.zeros_like(a_sh)), dict(a=np.zeros_like(s_sh), s=np.zeros_like(s_sh)), a_g, s_g)
                evals(dict(a=X_[0][0] + X_[0][2] * a_sh, s=np.full((G.NC, G.NC), X_[0][1]), centres=G.CENTRES, a0=X_[0][0], s0=X_[0][1]), 'bench_fit_a')
                # in-situ shape of the OTHER parity as a map (the noise ceiling of "a map of this chamber")
                other = Gc[parc != tr]
                evals(AM.fit_map(other) | dict(centres=G.CENTRES), 'insitu_other_parity_shape')
                print(arm, p, tr, 'done', flush=True)
    R = pd.DataFrame(rows)
    P = pd.DataFrame(amps)
    tag = f'm{margin:.0f}'
    R.to_csv(f'{G.OUT}/direct_{tag}.csv', index=False)
    P.to_csv(f'{G.OUT}/direct_amps_{tag}.csv', index=False)
    pd.set_option('display.width', 250)
    # parity-averaged table
    T = R.groupby(['arm', 'view', 'model'], sort=False)[['sig_all', 'err_all', 'sig_clean', 'err_clean', 'tail_all']].mean()
    print(T.round(4).to_string())
    print(P.round(3).to_string())


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
