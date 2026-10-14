#!/usr/bin/env python3
"""footprint_test.py -- is X's lateral footprint Gaussian, or does it have induction tails?

Head-on tracks (|tan| < 0.05): every depth lands at the same place, so the time-integrated
charge per strip, against the strip's offset from the charge centroid, is the lateral
footprint (prompt + drift diffusion integrated over depth).  Stacked over events by the
centroid's sub-strip position (so the profile is sampled finely), compared with
  gauss   : a Gaussian of width s integrated over each 0.78 mm strip
  induct  : the charge induced on a plane at depth d below a point charge,
            dQ/dx ~ d / (x^2 + d^2)^(3/2) (2-D), convolved with a Gaussian of width s
            and integrated over the strip
  mix     : the model's own shape -- uniform ionisation over the gap, each depth z a
            Gaussian of width sqrt(s0^2 + a^2 z / gap) (prompt footprint + drift diffusion),
            i.e. a scale-mixture of Gaussians (heavier tails than one Gaussian)
  mixind  : the same mixture with each landing convolved with the induction kernel (depth d)
  mixlor  : the mixture plus a fraction eta of a Lorentzian of half-width gamma (pseudo-Voigt);
            the form wft's strip fractions can carry (keys lor_frac_x, lor_gamma_x)
All fitted to the stacked strip fractions; the tails (|offset| >= 2 strips) decide.

    footprint_test.py det2=<big_cache.pkl> ...
Output: results/footprint_test.json
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
PITCH = 0.78
OFFS = np.arange(-4, 5)


def profile(ev, view):
    rows = []
    for e in ev.values():
        if view not in e or abs(e[f'tan_{view}']) > 0.05:
            continue
        P = e[view]; pos = np.asarray(P['pos'], float); W = np.asarray(P['W'], float)
        q = W.sum(1); k = int(np.argmax(q))
        if k < 4 or k > len(q) - 5 or q[k] <= 0:
            continue
        sel = slice(k - 1, k + 2)
        c = (pos[sel] * np.clip(q[sel], 0, None)).sum() / np.clip(q[sel], 0, None).sum()
        tot = q[k - 4:k + 5].sum()
        if tot <= 0:
            continue
        frac = q[k - 4:k + 5] / tot
        rows.append((c - pos[k], frac))                 # sub-strip offset of the centroid, fractions
    return rows


def strip_int(kernel_cdf, x_c):
    """fractions on strips at OFFS*PITCH for a footprint centred at x_c (cdf given)."""
    hi = kernel_cdf(OFFS * PITCH + PITCH / 2 - x_c); lo = kernel_cdf(OFFS * PITCH - PITCH / 2 - x_c)
    f = hi - lo
    return f / f.sum()


def gauss_cdf(s):
    return lambda x: 0.5 * (1 + erf(x / (np.sqrt(2) * s)))


def induct_cdf(d, s):
    # induced-charge cdf for a line charge (2-D): (1/pi) atan(x/d) ... point-charge-at-height in 2-D
    # gives dQ/dx ~ d/(x^2+d^2); convolve numerically with a Gaussian of width s
    xs = np.linspace(-15, 15, 6001); dx = xs[1] - xs[0]
    k = d / (xs ** 2 + d ** 2); k /= k.sum() * dx
    g = np.exp(-0.5 * (xs / max(s, 1e-3)) ** 2); g /= g.sum() * dx
    pdf = np.convolve(k, g, mode='same') * dx
    cdf = np.cumsum(pdf) * dx
    return lambda x: np.interp(x, xs, cdf)


def mix_cdf(s0, a, d=None):
    xs = np.linspace(-15, 15, 6001); dx = xs[1] - xs[0]
    pdf = np.zeros_like(xs)
    for u in np.linspace(0.0, 1.0, 21):
        sg = np.sqrt(s0 ** 2 + a ** 2 * u)
        pdf += np.exp(-0.5 * (xs / sg) ** 2) / sg
    pdf /= pdf.sum() * dx
    if d is not None:
        k = d / (xs ** 2 + d ** 2); k /= k.sum() * dx
        pdf = np.convolve(pdf, k, mode='same') * dx
    cdf = np.cumsum(pdf) * dx
    return lambda x: np.interp(x, xs, cdf)


def mixlor_cdf(s0, a, eta, gam):
    base = mix_cdf(s0, a)
    return lambda x: (1 - eta) * base(x) + eta * (0.5 + np.arctan(x / gam) / np.pi)


def fit(rows, kind):
    xc = np.array([r[0] for r in rows]); F = np.array([r[1] for r in rows])
    bins = np.linspace(-0.39, 0.39, 7)
    idx = np.digitize(xc, bins) - 1
    keep = [b for b in range(len(bins) - 1) if (idx == b).sum() >= 10]
    M = np.array([F[idx == b].mean(0) for b in keep])
    E = np.array([F[idx == b].std(0) / np.sqrt((idx == b).sum()) for b in keep])
    mid = (0.5 * (bins[:-1] + bins[1:]))[keep]

    mk = {'gauss': lambda p: gauss_cdf(p[0]), 'induct': lambda p: induct_cdf(p[0], p[1]),
          'mix': lambda p: mix_cdf(p[0], p[1]), 'mixind': lambda p: mix_cdf(p[0], p[1], p[2]),
          'mixlor': lambda p: mixlor_cdf(*p)}[kind]
    p0 = {'gauss': [0.45], 'induct': [0.3, 0.3], 'mix': [0.4, 0.8], 'mixind': [0.35, 0.8, 0.2],
          'mixlor': [0.4, 0.7, 0.05, 0.8]}[kind]

    def resid(p):
        cdf = mk(p)
        return np.concatenate([(strip_int(cdf, m) - M[b]) / np.maximum(E[b], 1e-4) for b, m in enumerate(mid)])
    lo = [0.02] * len(p0); hi = [3.0] * len(p0)
    if kind == 'mixlor':
        lo[2], hi[2] = 0.0, 0.5
    sol = least_squares(resid, p0, bounds=(lo, hi))
    cdf = mk(sol.x)
    pred = np.array([strip_int(cdf, m) for m in mid])
    return dict(par=sol.x.tolist(), chi2=float(np.sum(sol.fun ** 2)), n=int(sol.fun.size),
                data=M.mean(0).tolist(), model=pred.mean(0).tolist())


def main():
    res = {'offsets': OFFS.tolist()}
    for arg in sys.argv[1:]:
        det, path = arg.split('=', 1)
        ev = pickle.load(open(path, 'rb'))
        res[det] = {}
        for view in ('x', 'y'):
            rows = profile(ev, view)
            g = fit(rows, 'gauss'); i = fit(rows, 'mix'); mi = fit(rows, 'mixind'); ml = fit(rows, 'mixlor')
            res[det][view] = dict(n=len(rows), gauss=g, mix=i, mixind=mi, mixlor=ml)
            print(f'{det} {view} n={len(rows):4d}  data |o|=0..4: ' +
                  ' '.join(f'{x:.4f}' for x in np.array(g['data'])[4:]) +
                  f'\n      gauss  s={g["par"][0]:.3f}  chi2 {g["chi2"]:7.0f}/{g["n"]}  model ' +
                  ' '.join(f'{x:.4f}' for x in np.array(g['model'])[4:]) +
                  f'\n      mix    s0={i["par"][0]:.3f} a={i["par"][1]:.3f}  chi2 {i["chi2"]:7.0f}/{i["n"]}  model ' +
                  ' '.join(f'{x:.4f}' for x in np.array(i['model'])[4:]) +
                  f'\n      mixind s0={mi["par"][0]:.3f} a={mi["par"][1]:.3f} d={mi["par"][2]:.3f}  chi2 {mi["chi2"]:7.0f}/{mi["n"]}  model ' +
                  ' '.join(f'{x:.4f}' for x in np.array(mi['model'])[4:]) +
                  f'\n      mixlor s0={ml["par"][0]:.3f} a={ml["par"][1]:.3f} eta={ml["par"][2]:.3f} gam={ml["par"][3]:.3f}'
                  f'  chi2 {ml["chi2"]:7.0f}/{ml["n"]}  model ' + ' '.join(f'{x:.4f}' for x in np.array(ml['model'])[4:]))
    json.dump(res, open(os.path.join(HERE, 'results', 'footprint_test.json'), 'w'))


if __name__ == '__main__':
    main()
