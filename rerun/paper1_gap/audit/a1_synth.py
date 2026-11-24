"""a1_synth.py -- control: a PURE gradient field (the n = 12 Phi spline, no noise) pushed through the same cell averaging, plane
subtraction, finite-difference curl test and Helmholtz fit.  Gives the floor of 'solenoidal' fraction and slope bias that
the method itself produces.  -> synth.json"""
import json, pickle
import numpy as np
from scipy.interpolate import RectBivariateSpline
from a1_lib import *  # noqa
import a1_part2 as P2

maps = pickle.load(open(f'{OUT}/maps.pkl', 'rb'))['maps']
PH = pickle.load(open(f'{OUT}/phi.pkl', 'rb'))
eu, ew, q, e, q1, q2 = PH[12]           # non-quadratic part of Phi (mm); add the quadratic part from phi.json
pj = json.load(open(f'{OUT}/phi.json'))['12']['quad_coef_mm']
UU, WW = np.meshgrid(eu, ew, indexing='ij')
quad = pj['c'] + pj['U'] * UU + pj['W'] * WW + pj['UU'] * UU ** 2 + pj['UW'] * UU * WW + pj['WW'] * WW ** 2
Q = np.where(np.isfinite(q), q, 0.0) + quad                      # unused nodes: quadratic only
spl = RectBivariateSpline(eu, ew, Q, kx=3, ky=3, s=0)
out = {}
for n in (4, 6, 8, 10, 12):
    mp = {}
    for run in ('run1', 'run2'):
        mp[run] = {}
        m0 = maps[n][run]['x']
        eu_, ew_ = np.linspace(*UL, n + 1) * 100, np.linspace(*WL, n + 1) * 100
        sub = 6
        tx = np.full((n, n), np.nan); ty = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                xs = eu_[i] + (np.arange(sub) + .5) / sub * (eu_[i + 1] - eu_[i]); ys = ew_[j] + (np.arange(sub) + .5) / sub * (ew_[j + 1] - ew_[j])
                tx[i, j] = spl(xs, ys, dx=1).mean(); ty[i, j] = spl(xs, ys, dy=1).mean()
        for p, t in (('x', tx), ('y', ty)):
            d = dict(maps[n][run][p]); valid = np.isfinite(d['tau'])
            d['tau'] = np.where(valid, t, np.nan)
            mp[run][p] = d
    Rx = {r: residual(mp[r]['x'], 'tau', 'etau', 'plane')[0] for r in mp}
    Ry = {r: residual(mp[r]['y'], 'tau', 'etau', 'plane')[0] for r in mp}
    ex = {r: np.where(np.isfinite(Rx[r]), mp[r]['x']['etau'], np.nan) for r in mp}
    ey = {r: np.where(np.isfinite(Ry[r]), mp[r]['y']['etau'], np.nan) for r in mp}
    cs = curl_stats(Rx, ex, Ry, ey, n)
    h = {k: P2.helm_stats(n, k, mp)[0] for k in ('plane', 'none')}
    out[n] = dict(curl={k: cs[k] for k in ('rxy', 'frac', 'slope', 'slope_cov', 'rms_cx', 'rms_cy', 'ncorner')},
                  helm={k: dict(sol_sig=v['sol_sig'], tot_sig=v['tot_sig'], frac=v['sol_frac_var'], chi2dof=v['both']['chi2_dof']) for k, v in h.items()},
                  tau_plane_grad=[float(residual(mp['run1']['x'], 'tau', 'etau', 'plane')[1][1]), float(residual(mp['run1']['y'], 'tau', 'etau', 'plane')[1][2])])
    print(n, out[n]['curl'], out[n]['helm'], flush=True)
json.dump(out, open(f'{OUT}/synth.json', 'w'), indent=1, default=float)
