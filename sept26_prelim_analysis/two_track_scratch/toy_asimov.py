"""Ideal-world two-track detectability: the Asimov Delta-chi2.

Two tracks, same w, same t0, separation d, generated noise-free by the real
forward model (run_145 bundle). The best ONE-track fit to that window leaves
chi2 = lambda(d): the expected Delta-chi2 of a perfect analysis on real noisy
data (the noncentrality of the one-vs-two test). No seeder, no trigger, no
optimiser luck, no noise fluctuation.
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sept26_prelim_analysis import two_track_synth as ts

SEPS = np.array([0, .25, .5, .75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0])
NOISE = ts.NOISE_MED          # ADC per strip per sample, run_145 median
QTOT = ts.Q_MED               # per track
UEND = 840.0                  # ns: mid of the synthetic 600-1080 range
T0 = 0.0


def window(plane, tracks):
    from wft import model as wm
    pos = np.arange(-40, 41) * wm.PITCH + 100.0 + 0.3   # 0.3: not on a strip centre
    W = np.zeros((len(pos), wm.NSAMP))
    k = int(round(UEND / wm.DT))
    for p0, w, t0 in tracks:
        q = np.zeros(wm.K); q[:k] = 1.0; q *= QTOT / q.sum()
        W += (wm.build_matrix(plane, pos, p0, w, t0, wm.HYPER) @ q).reshape(len(pos), wm.NSAMP)
    return W, pos


def best_one(plane, W, pos, p0s, w_true, t0_true):
    """Global best single track: grid over (p0, t0) at the true w, then NM on all three."""
    from wft import model as wm
    noise = np.full(len(pos), NOISE); sat = np.zeros_like(W, bool)
    f = lambda p0, w, t0: wm.chi2_plane(plane, W, noise, pos, sat, p0, w, t0, wm.HYPER,
                                        snap_t0=False)[0]
    lo, hi = min(p0s) - 1.0, max(p0s) + 1.0
    grid = [(f(p, w_true, t), p, t) for p in np.arange(lo, hi + 1e-9, 0.05)
            for t in np.arange(t0_true - 120, t0_true + 121, 20)]
    grid.sort()
    best = np.inf
    for c, p, t in grid[:4]:
        r = minimize(lambda v: f(*v), [p, w_true, t], method='Nelder-Mead',
                     options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000, maxfev=4000))
        best = min(best, r.fun, c)
    return best


def job(args):
    arm, plane, tan, d = args
    ts._init(arm)
    from wft import model as wm
    w = tan * ts._CAL.v_drift * 1e-3
    W, pos = window(plane, [(100.0, w, T0), (100.0 + d, w, T0)])
    lam = best_one(plane, W, pos, [100.0, 100.0 + d], w, T0)
    # reference: chi2 of the ONE-track truth itself must be 0 at d = 0
    return dict(arm=arm, plane=plane, tan=tan, d=d, lam=lam,
                peak_snr=float(W.max() / NOISE),
                sigma_p0=ts._CAL.hyper['sigma_p0'], Dp=ts._CAL.hyper['Dp'])


if __name__ == '__main__':
    jobs = [(a, p, t, d) for a in ('A', 'C') for p in ('x', 'y') for t in (0.0, 0.3)
            for d in SEPS]
    with ProcessPoolExecutor(14) as ex:
        R = pd.DataFrame(list(ex.map(job, jobs)))
    R.to_csv(__file__.replace('.py', '.csv'), index=False)
    pd.set_option('display.width', 200)
    print(R.pivot_table(index='d', columns=['arm', 'plane', 'tan'], values='lam').round(1))
    print(R.groupby(['arm', 'plane']).peak_snr.first().round(1))
