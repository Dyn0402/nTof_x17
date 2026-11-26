#!/usr/bin/env python3
"""decompose.py -- split the M3 chi2 of every track into an ordinary part and an anomalous part.

Model, per M3 coordinate c (X or Y), for the single-track NClus = 4 rays of the relaxed tables:

    chi2 = g + k

  g  the ordinary part: Gaussian plane errors of a 4-point line fit give chi2 = s^2 chi2_2, an exponential with mean
     mu = 2 s^2.  The plane resolution s depends on the track angle (the MGv2 planes resolve an inclined track worse),
     so mu = mu(t), t = |slope| of the coordinate.  mu(t) is a free value per angle bin, interpolated in between.
  k  the anomalous part (multiple-scattering kinks, delta rays, a wrong cluster): the same distribution at every angle.
     k = 0 with probability 1 - sum f_m, else exponential with mean lam_m (m = 1..M, M = 3 by default).

The density of chi2 in an angle bin is then (1 - F) Exp(mu) + sum_m f_m Hypo(mu, lam_m), Hypo the convolution of two
exponentials.  The fit is a binned Poisson likelihood over every (angle bin, chi2 bin), chi2 < CHI2_MAX.

Per track, the posterior mean of the anomalous part given its chi2 and angle,
    E[k | chi2, t] = int_0^chi2 k p_g(chi2 - k) dF(k) / int_0^chi2 p_g(chi2 - k) dF(k),
is what carries the information about the pointing: for Gaussian errors the pointing is independent of the chi2, so
only the anomalous part can make the pointing worse.  posterior() evaluates it, and mu(t) for the ordinary part.

Writes $WORK/decompose.json (per coordinate: angle bin edges, mu per bin, f, lam).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m3common import TABLES, WORK  # noqa: E402
from rows import ROWS  # noqa: E402

T_EDGES = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4])
CHI2_MAX = 60.0
X_EDGES = np.concatenate([np.linspace(0, 1, 21), np.linspace(1.1, 3, 20), np.geomspace(3.2, CHI2_MAX, 25)])
M = 3
OUT = os.path.join(WORK, 'decompose.json')
SLOPE = {'X': 'm3_sx', 'Y': 'm3_sy'}


def comp_cdf(x, mu, lam):
    """CDF of Exp(mu) (lam None) or of Exp(mu) * Exp(lam) (hypoexponential)."""
    if lam is None:
        return 1 - np.exp(-x / mu)
    if abs(lam - mu) < 1e-9 * mu:
        lam = mu * (1 + 1e-6)
    return 1 - (lam * np.exp(-x / lam) - mu * np.exp(-x / mu)) / (lam - mu)


def comp_pdf(x, mu, lam):
    if lam is None:
        return np.exp(-x / mu) / mu
    if abs(lam - mu) < 1e-9 * mu:
        lam = mu * (1 + 1e-6)
    return (np.exp(-x / lam) - np.exp(-x / mu)) / (lam - mu)


def unpack(p, nt):
    mu = np.exp(p[:nt])
    z = np.exp(p[nt:nt + M]); f = z / (1 + z.sum())            # f_m > 0, sum < 1
    lam = np.cumsum(np.exp(p[nt + M:nt + 2 * M]))             # ordered means
    return mu, f, lam


def bin_probs(mu_j, f, lam, edges=X_EDGES):
    P = (1 - f.sum()) * np.diff(comp_cdf(edges, mu_j, None))
    for fm, lm in zip(f, lam):
        P = P + fm * np.diff(comp_cdf(edges, mu_j, lm))
    return P


def fit(chi2, t, edges_t=T_EDGES):
    jb = np.digitize(t, edges_t) - 1
    nt = len(edges_t) - 1
    keep = (jb >= 0) & (jb < nt) & (chi2 >= 0)
    H = np.array([np.histogram(chi2[keep & (jb == j)], X_EDGES)[0] for j in range(nt)], float)
    N = H.sum(1) + np.array([np.sum(keep & (jb == j) & (chi2 >= CHI2_MAX)) for j in range(nt)])

    def nll(p):
        mu, f, lam = unpack(p, nt)
        L = 0.0
        for j in range(nt):
            P = np.clip(bin_probs(mu[j], f, lam), 1e-300, None)
            L -= np.sum(H[j] * np.log(P)) + (N[j] - H[j].sum()) * np.log(max(1 - P.sum(), 1e-300))
        return L

    med = np.array([np.median(chi2[keep & (jb == j)]) for j in range(nt)])
    p0 = np.concatenate([np.log(0.6 * med), np.log([0.3, 0.15, 0.05]), np.log([0.3, 1.5, 6.0])])
    r = minimize(nll, p0, method='L-BFGS-B', options=dict(maxiter=5000))
    r = minimize(nll, r.x, method='Powell', options=dict(maxiter=20000, xtol=1e-4, ftol=1e-7))
    r = minimize(nll, r.x, method='L-BFGS-B', options=dict(maxiter=5000))
    mu, f, lam = unpack(r.x, nt)
    # goodness of fit: Pearson chi2 over bins with expectation >= 5
    pear, nb = 0.0, 0
    for j in range(nt):
        e = N[j] * bin_probs(mu[j], f, lam); k = e >= 5
        pear += np.sum((H[j][k] - e[k]) ** 2 / e[k]); nb += k.sum()
    return dict(t_edges=list(edges_t), t_centres=[float(np.median(t[keep & (jb == j)])) for j in range(nt)],
                mu=list(mu), f=list(f), lam=list(lam), n=[int(x) for x in N], nll=float(r.fun),
                pearson=float(pear), nbins=int(nb), npar=int(len(r.x)), H=H.tolist())


def mu_of_t(par, t):
    """ordinary chi2 mean at slope t: linear interpolation of the bin values in the bin medians (flat outside)."""
    return np.interp(np.abs(t), par['t_centres'], par['mu'])


def posterior(par, chi2, t, ngrid=400):
    """E[k | chi2, t] (anomalous part) and mu(t), per track.  Exact for the exponential-mixture prior: with
    beta_m = 1/lam_m - 1/mu, int_0^x k e^{-beta k} dk and int_0^x e^{-beta k} dk in closed form."""
    chi2 = np.asarray(chi2, float); mu = mu_of_t(par, t)
    f = np.array(par['f']); lam = np.array(par['lam'])
    x = np.maximum(chi2, 0.0)
    # weights relative to the 'no anomaly' term (1-F) p_g(x)/... ; common factor exp(-x/mu)/mu dropped
    num = np.zeros_like(x); den = np.full_like(x, 1 - f.sum())
    for fm, lm in zip(f, lam):
        b = 1.0 / lm - 1.0 / mu                       # rate of the tilted exponential on [0, x]
        bx = b * x
        small = np.abs(bx) < 1e-6
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            I0 = np.where(small, x, -np.expm1(-bx) / np.where(small, 1, b))
            I1 = np.where(small, x ** 2 / 2, (1 - np.exp(-bx) * (1 + bx)) / np.where(small, 1, b) ** 2)
        num += fm / lm * I1
        den += fm / lm * I0
    return num / den, mu


def load_relaxed(tags=None):
    """the relaxed M3 rays, one table per acquisition (the two chambers of a pair share their M3 rays)"""
    seen, out = set(), []
    for r in ROWS:
        if (tags is not None and r['tag'] not in tags) or (r['run'], r['subrun']) in seen:
            continue
        seen.add((r['run'], r['subrun']))
        out.append(pd.read_parquet(os.path.join(TABLES, f"{r['tag']}.relaxed.parquet")).assign(tag=r['tag']))
    return pd.concat(out, ignore_index=True)


def main():
    R = load_relaxed()
    out = {}
    for c in ('X', 'Y'):
        p = fit(R[f'chi2_{c}'].to_numpy(), R[SLOPE[c]].abs().to_numpy())
        out[c] = p
        print(f"{c}: n {sum(p['n'])}  f {np.round(p['f'], 4)}  lam {np.round(p['lam'], 3)}  "
              f"Pearson {p['pearson']:.0f}/{p['nbins'] - p['npar']}")
        print('   t  ', ' '.join(f'{x:.3f}' for x in p['t_centres']))
        print('   mu ', ' '.join(f'{x:.3f}' for x in p['mu']))
    json.dump(out, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()


def posterior_components(par, chi2, t):
    """E[k 1(component m) | chi2, t] per anomaly component m (rows), and mu(t): the posterior mean of the anomalous
    chi2 split by which component produced it.  Their sum is posterior()'s E[k | chi2, t]."""
    chi2 = np.asarray(chi2, float); mu = mu_of_t(par, t)
    f = np.array(par['f']); lam = np.array(par['lam'])
    x = np.maximum(chi2, 0.0)
    num = []; den = np.full_like(x, 1 - f.sum())
    for fm, lm in zip(f, lam):
        b = 1.0 / lm - 1.0 / mu
        bx = b * x
        small = np.abs(bx) < 1e-6
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            I0 = np.where(small, x, -np.expm1(-bx) / np.where(small, 1, b))
            I1 = np.where(small, x ** 2 / 2, (1 - np.exp(-bx) * (1 + bx)) / np.where(small, 1, b) ** 2)
        num.append(fm / lm * I1)
        den += fm / lm * I0
    return np.array(num) / den, mu
