#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
landau.py -- the expected minimum-ionising deposit in a thin scintillator,
and a Landau (x) Gauss fit to a measured spectrum.

EXPECTED.  The most probable energy loss of the Landau-Vavilov-Bichsel
distribution (PDG 2024, eq. 34.12),

    Delta_p = xi [ ln(2 m c^2 b^2 g^2 / I) + ln(xi / I) + j - b^2 - delta(bg) ],
    xi = (K/2) (Z/A) x / b^2,   j = 0.200,

with the Sternheimer density correction.  It is the right quantity for a
20 mm plastic: the MEAN (Bethe) is ~15 % higher and is pulled by delta rays
that leave the bar.  It is a step-free calculation, not a simulation -- no
delta-ray escape, no Birks.  Birks quenching of a MIP in PVT is ~2.5 %
(kB ~ 0.126 mm/MeV at 2 MeV/cm), and the keVee scale was set with Compton
electrons of 0.5-1.6 MeV whose own quenching is of the same size, so
MeVee = MeV for a MIP to ~2 %.

FIT.  Landau(loc, scale) convolved with a Gaussian of width sigma (photo-
statistics, readout), binned Poisson likelihood.  Reported: ``mpv_landau``,
the Landau's own peak -- the quantity to set against Delta_p -- and
``peak``, the peak of the convolution, which is what a histogram shows.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import landau, norm

K = 0.307075            # MeV cm^2 / mol
ME = 0.51099895         # MeV
#: (rho g/cm3, Z/A, I eV, Sternheimer C, x0, x1, a, k)
MATERIALS = {
    # PDG: polyvinyltoluene, the BC-408/EJ-200 base
    'PVT': (1.032, 0.54141, 64.7, 3.1997, 0.1464, 2.4855, 0.16101, 3.2393),
    # LAB (C18H30) is not in the PDG tables; polyethylene-like values with
    # LAB's density and Z/A.  (est.) -- a few % on Delta_p at most.
    'LAB': (0.86, 0.5574, 57.4, 3.0016, 0.1370, 2.5177, 0.12108, 3.4292),
}
#: the standard Landau's peak, in scipy's parametrisation
LANDAU_PEAK = float(minimize(lambda z: -landau.pdf(z[0]), [-0.2]).x[0])


def _delta(bg: float, C: float, x0: float, x1: float, a: float, k: float) -> float:
    x = np.log10(bg)
    if x >= x1:
        return 2 * np.log(10) * x - C
    if x >= x0:
        return 2 * np.log(10) * x - C + a * (x1 - x) ** k
    return 0.0


def mpv(thick_mm: float, bg: float = 30.0, material: str = 'PVT') -> float:
    """Most probable loss, MeV, of a singly-charged particle at beta*gamma bg."""
    rho, za, I, C, x0, x1, a, k = MATERIALS[material]
    b2 = bg ** 2 / (1 + bg ** 2)
    xi = 0.5 * K * za * rho * thick_mm / 10 / b2
    I = I * 1e-6
    return xi * (np.log(2 * ME * bg ** 2 / I) + np.log(xi / I) + 0.200 - b2
                 - _delta(bg, C, x0, x1, a, k))


def mean_loss(thick_mm: float, bg: float = 30.0, material: str = 'PVT') -> float:
    """Bethe mean loss, MeV (no restriction), for comparison."""
    rho, za, I, C, x0, x1, a, k = MATERIALS[material]
    b2 = bg ** 2 / (1 + bg ** 2)
    g = np.sqrt(1 + bg ** 2)
    tmax = 2 * ME * bg ** 2 / (1 + 2 * g * ME / 105.658 + (ME / 105.658) ** 2)
    I = I * 1e-6
    dedx = K * za / b2 * (0.5 * np.log(2 * ME * bg ** 2 * tmax / I ** 2) - b2
                          - _delta(bg, C, x0, x1, a, k) / 2)
    return dedx * rho * thick_mm / 10


def langaus_pdf(x: np.ndarray, loc: float, scale: float, sigma: float) -> np.ndarray:
    """Landau(loc, scale) (x) N(0, sigma), on an arbitrary grid x (normalised
    numerically over a wide support)."""
    lo, hi = loc - 8 * scale - 5 * sigma, loc + 60 * scale + 5 * sigma
    z = np.linspace(lo, hi, 1500)
    dz = z[1] - z[0]
    L = landau.pdf(z, loc, scale)
    x = np.atleast_1d(x)
    return norm.pdf(x[:, None] - z[None, :], 0, sigma) @ L * dz


def fit(vals: np.ndarray, lo_q: float = 0.35, hi_mult: float = 2.2, nbins: int = 40,
        floor: float | None = None) -> dict:
    """Langaus fit to ``vals`` over [lo, hi]: lo at quantile ``lo_q`` of the
    sample's own core (or ``floor``, whichever is higher), hi = hi_mult x the
    median.  The low side is cut because it holds the trigger turn-on, edge
    clips and partial crossings; the MPV is set by the peak and the rising
    tail.  Returns the fit and a bootstrap error on the MPV."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 40:
        return dict(n=len(v), mpv_landau=np.nan, peak=np.nan, scale=np.nan, sigma=np.nan,
                    mpv_err=np.nan, median=float(np.median(v)) if len(v) else np.nan)
    med = float(np.median(v))
    lo = max(float(np.quantile(v, lo_q)) * 0.75, floor or 0.0)
    hi = hi_mult * med
    edges = np.linspace(lo, hi, nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    def nll_for(sample):
        n, _ = np.histogram(sample, edges)
        N = len(sample)

        def nll(p):
            loc, ls, lsg = p
            s, sg = np.exp(ls), np.exp(lsg)
            pdf = langaus_pdf(ctr, loc, s, sg)
            mu = N * pdf * (edges[1] - edges[0])
            # the window holds a fraction f of the sample; normalise inside
            mu = mu / max(mu.sum(), 1e-12) * n.sum()
            mu = np.clip(mu, 1e-9, None)
            return float(np.sum(mu - n * np.log(mu)))
        return nll

    p0 = [med * 0.85, np.log(med * 0.08), np.log(med * 0.12)]
    r = minimize(nll_for(v), p0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=1e-3, maxiter=2000))
    loc, s, sg = r.x[0], np.exp(r.x[1]), np.exp(r.x[2])
    m_l = loc + LANDAU_PEAK * s
    grid = np.linspace(lo, hi, 300)
    pk = float(grid[np.argmax(langaus_pdf(grid, loc, s, sg))])
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(15):
        b = rng.choice(v, len(v))
        rb = minimize(nll_for(b), r.x, method='Nelder-Mead',
                      options=dict(xatol=1e-3, fatol=1e-3, maxiter=600))
        bs.append(rb.x[0] + LANDAU_PEAK * np.exp(rb.x[1]))
    return dict(n=int(len(v)), n_fit=int(((v >= lo) & (v <= hi)).sum()), lo=lo, hi=hi,
                mpv_landau=float(m_l), peak=pk, scale=float(s), sigma=float(sg),
                mpv_err=float(np.std(bs)), median=med, loc=float(loc))


if __name__ == '__main__':
    for bg in (5, 10, 30, 100):
        print(f'bg {bg:4d}: PVT 20 mm MPV {mpv(20, bg):.3f} MeV, mean {mean_loss(20, bg):.3f}; '
              f'LAB 18 mm MPV {mpv(18, bg, "LAB"):.3f}; PVT 3 mm {mpv(3, bg):.3f}')


def _langaus_grid(grid: np.ndarray, loc: float, scale: float, sigma: float) -> np.ndarray:
    """`langaus_pdf` on a UNIFORM grid, by discrete convolution (fast).
    The Landau is evaluated on the grid extended by 6 sigma each side, so
    mass that the Gaussian smears into the window from outside it is kept."""
    dz = grid[1] - grid[0]
    nk = int(np.ceil(6 * sigma / dz))
    z = grid[0] + dz * np.arange(-nk, len(grid) + nk)
    L = landau.pdf(z, loc, scale)
    k = norm.pdf(dz * np.arange(-nk, nk + 1), 0, sigma) * dz
    return np.convolve(L, k, mode='valid')


#: Landau scale / MPV for 20 mm PVT: xi / Delta_p (xi = 0.174 MeV at b = 1).
#: The fit holds the Landau's own width at this physics value and lets the
#: Gaussian (photostatistics, readout, path spread) carry the rest.
XI_OVER_MPV_PVT20 = 0.0512


def fit_trunc(e: np.ndarray, t: np.ndarray, hi: float, xi_ratio: float = XI_OVER_MPV_PVT20,
              p0: float | None = None, n_boot: int = 30, seed: int = 1,
              res_prior: tuple | None = None) -> dict:
    """Unbinned Landau (x) Gauss fit with a PER-EVENT lower truncation.

    ``e``  the measured deposits; ``t`` each event's own lower bound (its
    trigger threshold where the trigger needed this channel, else a common
    floor); ``hi`` a common upper bound.  Likelihood per event
    f(e_i) / [F(hi) - F(t_i)], so events sculpted by the threshold carry
    exactly the information they hold.  Free: MPV (the Landau's own peak) and
    sigma; the Landau scale is ``xi_ratio`` x MPV.

    ``res_prior`` = (mean, width) of a Gaussian prior on sigma / MPV.  When
    the threshold cuts near the peak, a low MPV with a wide sigma fits the
    surviving tail almost as well as the truth (the closure test shows it);
    the prior, taken from bars whose threshold is far below their peak, is
    what breaks that degeneracy."""
    e, t = np.asarray(e, float), np.asarray(t, float)
    ok = np.isfinite(e) & (e >= t) & (e <= hi)
    e, t = e[ok], t[ok]
    if len(e) < 40:
        return dict(n=int(len(e)), mpv=np.nan, sigma=np.nan, mpv_err=np.nan)
    grid = np.linspace(0.0, hi, 700)

    def nll_of(ee, tt):
        def nll(p):
            m, ls = p
            sg = np.exp(ls)
            sc = xi_ratio * m
            loc = m - LANDAU_PEAK * sc
            f = _langaus_grid(grid, loc, sc, sg)
            F = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(grid))])
            fe = np.interp(ee, grid, f)
            Ft = np.interp(tt, grid, F)
            den = np.clip(F[-1] - Ft, 1e-12, None)
            pen = 0.0 if res_prior is None else 0.5 * ((sg / m - res_prior[0]) / res_prior[1]) ** 2
            return float(-np.sum(np.log(np.clip(fe, 1e-300, None)) - np.log(den)) + pen)
        return nll

    m0 = p0 if p0 else float(np.median(e))
    best = None
    for start in (0.8 * m0, m0, 1.15 * m0):
        r = minimize(nll_of(e, t), [start, np.log(0.15 * m0)], method='Nelder-Mead',
                     options=dict(xatol=0.05, fatol=1e-3, maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(e), len(e))
        rb = minimize(nll_of(e[i], t[i]), best.x, method='Nelder-Mead',
                      options=dict(xatol=0.05, fatol=1e-3, maxiter=400))
        bs.append(rb.x[0])
    return dict(n=int(len(e)), mpv=float(best.x[0]), sigma=float(np.exp(best.x[1])),
                mpv_err=float(np.std(bs)), frac_truncated=float(np.mean(t > 0.6 * best.x[0])),
                nll=float(best.fun))
