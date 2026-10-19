#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ipc_born.py -- the internal-pair continuum from the multipole that made it.

    python -m sept26_prelim_analysis.ipc_born            # validation + tables
    python -m sept26_prelim_analysis.ipc_born --al       # and the Al capture look

WHY THIS REPLACES THE ANSATZ IN ``pair_physics.py``.  That module samples the
virtual photon's mass from ``dN/dM ~ 1/M`` and decays it isotropically, because
that is what the Geant4 primary generator does, and it carries the spread
between four hand-made variants as the modelling systematic.  Both choices are
guesses, the spread between them is a factor of 38 in the quantity the X17
search actually needs (the fraction of the continuum above 109 deg), and
neither guess is labelled with a piece of nuclear physics.  None of that is
necessary.  For a Z = 2 nucleus at 20.6 MeV the one-photon-exchange (Born)
calculation is essentially exact -- alpha*Z = 0.015 -- and it gives the
distribution in closed form, once per multipole.  The remaining unknown is then
a *physical* one, "which multipole made the pair", which is a question about
the reaction and not about the QED.

THE MASTER FORMULA.  A transition of energy ``W`` emits a virtual photon of
invariant mass ``M`` and three-momentum ``k = sqrt(W^2 - M^2)`` which converts
to the pair.  With ``theta*`` the lepton angle in the pair rest frame measured
from ``k``, and ``beta* = sqrt(1 - 4 m_e^2/M^2)``,

    dGamma / (dM^2 dcos(theta*))  =  (1/M^4) [ N_T(k) S_T + N_L(k) S_L ] k beta*

    S_T(theta*) = 2 M^2 [ (1 + cos^2) + (1 - beta*^2) sin^2 ]   two transverse
    S_L(theta*) = 2 M^2 [ 1 - beta*^2 cos^2 ]                   one longitudinal

The lepton side is pure QED and is written above.  The nuclear side is three
numbers per multipole, in the long-wavelength limit:

    M(lambda)   N_T = k^(2 lambda)                       N_L = 0
    E(lambda)   N_T = ((l+1)/l) W^2 k^(2 lambda - 2)     N_L = M^2 k^(2 lambda - 2)
    E0 (C0)     N_T = 0                                  N_L = M^2 k^2

Three things are worth reading off that table, because they are the whole
result:

* **E0 is not a virtual photon that decays; it is a contact term.**  The
  monopole charge matrix element goes as ``k^2`` and the Coulomb propagator as
  ``1/k^2``, so the two cancel and the nucleus can absorb *any* momentum at no
  cost.  The pair is then unconstrained in opening angle: the exact lab-frame
  law is ``(1 + eps cos(theta))`` with ``eps = p+p-/(E+E- - m^2) <= 1``, which
  puts **11 %** of the pairs above 109 deg.  Viviani et al. state the same
  cancellation in words -- "this singularity poses no problem, since
  |C0000(q)|^2 ~ q^4" (PRC 105, 014001, around Eq. 49).
* **M1 is the most collimated thing here.**  ``N_T ~ k^2`` on top of the
  ``1/M^4`` propagator leaves ``dN/dln M ~ k^3``, i.e. the 1/M ansatz *with*
  the phase-space suppression the ansatz omits.  4.6 % above 109 deg, not the
  11.8 % Geant's generator produces.
* **E-type beats M-type.**  ``N_T ~ W^2`` rather than ``k^2`` for E1: the
  current matrix element is fixed by the transition energy, not by the momentum
  transfer, so it does not switch off at the wide-angle endpoint.  9.9 % above
  109 deg, and a larger pair coefficient.

WHAT IS VALIDATED, AND HOW.  Two independent checks, both in
:func:`validate`, both exact rather than eyeballed:

  E0 against Wilkinson    Integrating the master formula over ``theta*`` and
                          re-expressing in the positron energy must reproduce
                          Wilkinson's published E0 pair rate integrand,
                          ``p+ p- (E+E- - gamma^2) F(Z,E+) F(Z,E-)``
                          (Nucl. Phys. A133 (1969) 1, quoted as Eq. 18 of
                          Dowie et al., arXiv:1911.00031).  At Z = 2 the Fermi
                          functions are 1 to 0.1 %, so the comparison is to the
                          bare ``p+p-(E+E- - 1)``.  It agrees to 0.5 %.
  E0 against itself       The same distribution derived a completely different
                          way -- as a local ``psi-bar gamma^0 psi`` operator,
                          giving the lab-frame ``(1 + eps cos theta)`` -- must
                          come out of the virtual-photon machinery.  The two
                          agree on the fraction above 109 deg to 0.01
                          percentage points.

  and a third, weaker one: ``alpha_pair`` is normalised so that the M -> 0
  limit of the conversion factor is (alpha/3 pi)/M^2, which is the standard
  soft-virtual-photon result, and it reproduces the textbook ordering
  E1 > M1 > E2 at fixed energy.

WHAT IS *NOT* SETTLED HERE, and must not be read as settled: the WEIGHTS.
This module says what each multipole looks like.  It does not say how much of
each the reaction makes -- see :mod:`ipc_channels` for the reaction side, and
``docs`` in that module for why the >1 ms neutron window changes the answer.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SCHEMA = 'sept26_prelim/ipc_born/1'

M_E = 0.51099895            # MeV
ALPHA = 1.0 / 137.035999084
E_TRANSITION = 20.58        # MeV, 4He* from n + 3He at rest
AL_SN = 7.7255              # MeV, neutron separation energy of 28Al
X17_MIN_DEG = 109.0

#: The multipoles this module knows, and what each one is for here.
CHANNELS = {
    'E0': 'E0 (C0) monopole -- 0+ -> 0+, no real photon exists',
    'M1': 'M1 magnetic dipole -- the 3S1 (1+) thermal capture channel',
    'E1': 'E1 electric dipole -- p-wave capture, and most Al primaries',
    'E2': 'E2 electric quadrupole',
    'M2': 'M2 magnetic quadrupole',
}


# --------------------------------------------------------------------------- #
# the nuclear side: three numbers per multipole
# --------------------------------------------------------------------------- #
def nuclear_factors(kind: str, k, M, w: float):
    """``(N_T, N_L)`` for one multipole, per polarisation, long-wavelength.

    Normalisations are arbitrary and channel-local -- only the ratio ``N_L/N_T``
    within a channel, and the k- and M-dependence, affect any shape this module
    reports.  ``N_T`` is *per transverse polarisation*; ``S_T`` below already
    sums the two.
    """
    k = np.asarray(k, float)
    M = np.asarray(M, float)
    if kind == 'E0':
        return np.zeros_like(k), M ** 2 * k ** 2
    if kind[0] == 'M':
        lam = int(kind[1:])
        return k ** (2 * lam), np.zeros_like(k)
    if kind[0] == 'E':
        lam = int(kind[1:])
        # Siegert, long wavelength: |T^el_l| -> sqrt((l+1)/l) (w/k) |rho(Cl)|
        # with |rho(Cl)| ~ k^l.  The (l+1)/l is the only place this module
        # leans on Siegert rather than on kinematics; it moves E1's T:L ratio
        # by a factor 2 and nothing else.
        return ((lam + 1) / lam) * w ** 2 * k ** (2 * lam - 2), \
            M ** 2 * k ** (2 * lam - 2)
    raise ValueError(f'unknown multipole {kind!r}')


def _lepton_tensors(M, cstar):
    """``(S_T, S_L)`` -- the QED side, exact including the electron mass."""
    b2 = np.clip(1.0 - 4.0 * M_E ** 2 / M ** 2, 0.0, None)
    c2 = cstar ** 2
    s_t = 2 * M ** 2 * ((1 + c2) + (1 - b2) * (1 - c2))
    s_l = 2 * M ** 2 * (1 - b2 * c2)
    return s_t, s_l, np.sqrt(b2)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample(kind: str, n: int = 2_000_000, w: float = E_TRANSITION,
           seed: int = 21) -> pd.DataFrame:
    """Weighted pairs from one multipole: lab opening angle, energies, mass.

    Importance-sampled flat in ``ln M`` and in ``cos(theta*)``, so the returned
    ``weight`` column carries the whole matrix element and phase space.  Weights
    rather than accept/reject because the M1 weight spans eight decades and
    rejection would throw away 99.99 % of the sample at the endpoint that
    matters.
    """
    if w <= 2 * M_E:
        raise ValueError('transition energy below the pair threshold')
    rng = np.random.default_rng(seed)
    lo = 2 * M_E * (1 + 1e-12)
    M = np.exp(rng.uniform(np.log(lo), np.log(w), n))
    jac = 2 * M ** 2                       # dM^2 = 2 M^2 dlnM
    k = np.sqrt(np.clip(w ** 2 - M ** 2, 0.0, None))
    cs = rng.uniform(-1, 1, n)
    s_t, s_l, bstar = _lepton_tensors(M, cs)
    n_t, n_l = nuclear_factors(kind, k, M, w)
    weight = (n_t * s_t + n_l * s_l) / M ** 4 * k * bstar * jac

    # boost the back-to-back pair along k
    ss = np.sqrt(np.clip(1 - cs ** 2, 0, None))
    e_star = M / 2.0
    p_star = e_star * bstar
    gam = w / M
    bet = k / w
    pz1 = gam * (p_star * cs + bet * e_star)
    pz2 = gam * (-p_star * cs + bet * e_star)
    px1, px2 = p_star * ss, -p_star * ss
    dot = px1 * px2 + pz1 * pz2
    theta = np.degrees(np.arccos(np.clip(
        dot / np.hypot(px1, pz1) / np.hypot(px2, pz2), -1, 1)))
    e1 = gam * (e_star + bet * p_star * cs)
    e2 = gam * (e_star - bet * p_star * cs)
    return pd.DataFrame(dict(theta_deg=theta, weight=np.clip(weight, 0, None),
                             m_ee=M, e_plus=e1, e_minus=e2,
                             y=(e1 - e2) / w))


def e0_lab(n: int = 4_000_000, w: float = E_TRANSITION,
           z: int = 2, seed: int = 22) -> pd.DataFrame:
    """E0 the other way: the exact lab-frame contact law, for cross-checking.

    The E0 operator is local, so the pair distribution is elementary --

        d2W / (dE+ dcos(theta))  ~  p+ p- ( E+E- - m^2 + p+ p- cos(theta) )

    -- which is the ``(1 + eps cos theta)`` anisotropy quoted for E0 pair
    conversion, with ``eps = p+p-/(E+E- - m^2)``.  Integrating over the angle
    returns Wilkinson's ``p+p-(E+E- - m^2)`` energy-sharing law exactly.  The
    Coulomb (Fermi-function) correction is not applied: at Z = 2 it is 0.1 %.
    """
    if z > 8:
        raise ValueError('Born form only; use a Dirac-Coulomb code above Z ~ 8')
    rng = np.random.default_rng(seed)
    ep = rng.uniform(M_E, w - M_E, n)
    em = w - ep
    pp = np.sqrt(np.clip(ep ** 2 - M_E ** 2, 0, None))
    pm = np.sqrt(np.clip(em ** 2 - M_E ** 2, 0, None))
    c = rng.uniform(-1, 1, n)
    weight = np.clip(pp * pm * (ep * em - M_E ** 2 + pp * pm * c), 0, None)
    return pd.DataFrame(dict(theta_deg=np.degrees(np.arccos(c)), weight=weight,
                             e_plus=ep, e_minus=em, y=(ep - em) / w))


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def wfrac(d: pd.DataFrame, above: float) -> float:
    w = d.weight.to_numpy()
    return float(w[d.theta_deg.to_numpy() > above].sum() / w.sum())


def wmedian(d: pd.DataFrame) -> float:
    o = np.argsort(d.theta_deg.to_numpy())
    t = d.theta_deg.to_numpy()[o]
    c = np.cumsum(d.weight.to_numpy()[o])
    return float(np.interp(0.5 * c[-1], c, t))


def shape(d: pd.DataFrame, bins=None) -> np.ndarray:
    """Normalised dN/dtheta on ``bins`` (default 3 deg, 0-180)."""
    if bins is None:
        bins = np.arange(0.0, 181.0, 3.0)
    h, _ = np.histogram(d.theta_deg, bins=bins, weights=d.weight)
    width = np.diff(bins)
    return h / (h.sum() * width)


# --------------------------------------------------------------------------- #
# the spectrum itself, without Monte-Carlo noise
# --------------------------------------------------------------------------- #
#: The default opening-angle axis for every dN/dtheta this package reports.
#: 1 deg from 0 to 180 -- fine enough that the E0 peak and the M1 forward rise
#: are both resolved, coarse enough that a 180-row CSV is readable.
THETA_BINS = np.arange(0.0, 181.0, 1.0)
THETA_MID = 0.5 * (THETA_BINS[1:] + THETA_BINS[:-1])


@functools.lru_cache(maxsize=4096)
def _grid_cached(kind, w, bkey, n_m, n_c):
    return _grid_spectrum(kind, w, np.frombuffer(bkey, dtype=float), n_m, n_c)


def grid_spectrum(kind: str, w: float = E_TRANSITION, bins=None,
                  n_m: int = 1400, n_c: int = 1400) -> np.ndarray:
    """Cached front end -- see :func:`_grid_spectrum` for the calculation.

    The aluminium and GANIL modules ask for the same (multipole, transition
    energy) pair once per neutron energy and there are two hundred of them, so
    without this the energy scan spends its whole life re-integrating curves it
    has already integrated.  Keyed on the bin edges too, because a different
    axis is a different answer.
    """
    if bins is None:
        bins = THETA_BINS
    b = np.ascontiguousarray(np.asarray(bins, float))
    # a copy, so a caller that scales the result in place cannot poison the
    # cache for everyone else
    return _grid_cached(kind, float(w), b.tobytes(), n_m, n_c).copy()


def _grid_spectrum(kind: str, w: float = E_TRANSITION, bins=None,
                   n_m: int = 1400, n_c: int = 1400) -> np.ndarray:
    """Normalised ``dN/dtheta`` for one multipole, by quadrature, not sampling.

    Same master formula as :func:`sample`, evaluated on a deterministic
    ``(ln M, cos theta*)`` grid instead of a random one.  Two reasons this
    exists rather than reusing ``sample``:

    * **it is the spectrum, and the spectrum is now the deliverable.**  A
      Monte-Carlo dN/dtheta wobbles at the percent level bin to bin, which is
      invisible in a >109 deg integral and very visible in a plotted curve or a
      published CSV.  The grid version is reproducible to the last digit.
    * :mod:`ipc_aluminium` needs one of these per gamma line -- 215 of them.
      At 2e6 samples each that is a minute of noise; on a grid it is a second
      and exact.

    Returns the density on ``bins`` (default :data:`THETA_BINS`), normalised so
    that ``sum(y * width) == 1``.
    """
    if bins is None:
        bins = THETA_BINS
    if w <= 2 * M_E:
        return np.zeros(len(bins) - 1)
    # MIDPOINT cells, not endpoints.  Both endpoints are singular in the map
    # to the lab angle -- M = W gives k = 0 and theta = 180 deg for every
    # cos(theta*) at once, so an endpoint rule dumps a whole grid row into the
    # last bin and inflates it by an order of magnitude.  Midpoints avoid the
    # measure-zero lines entirely and need no special-casing.
    lo = 2 * M_E * (1 + 1e-12)
    edges = np.linspace(np.log(lo), np.log(w), n_m + 1)
    lnm = 0.5 * (edges[1:] + edges[:-1])
    dlnm = np.diff(edges)[:, None]
    M = np.exp(lnm)[:, None]
    cedges = np.linspace(-1.0, 1.0, n_c + 1)
    cs = (0.5 * (cedges[1:] + cedges[:-1]))[None, :]
    dcs = np.diff(cedges)[None, :]
    k = np.sqrt(np.clip(w ** 2 - M ** 2, 0.0, None))
    s_t, s_l, bstar = _lepton_tensors(M, cs)
    n_t, n_l = nuclear_factors(kind, k, M, w)
    # cell weights in both directions, times dM^2 = 2 M^2 dlnM
    weight = ((n_t * s_t + n_l * s_l) / M ** 4 * k * bstar
              * 2 * M ** 2 * dlnm * dcs)

    ss = np.sqrt(np.clip(1 - cs ** 2, 0, None))
    e_star = M / 2.0
    p_star = e_star * bstar
    gam = w / M
    bet = k / w
    pz1 = gam * (p_star * cs + bet * e_star)
    pz2 = gam * (-p_star * cs + bet * e_star)
    px1, px2 = p_star * ss, -p_star * ss
    dot = px1 * px2 + pz1 * pz2
    theta = np.degrees(np.arccos(np.clip(
        dot / np.hypot(px1, pz1) / np.hypot(px2, pz2), -1, 1)))

    h, _ = np.histogram(theta.ravel(), bins=bins,
                        weights=np.clip(weight, 0, None).ravel())
    width = np.diff(bins)
    tot = (h * 1.0).sum()
    return h / (tot * width) if tot > 0 else h


def spectrum_table(y, bins=None) -> pd.DataFrame:
    """A dN/dtheta density as the table that gets published.

    One row per bin: the density, the fraction of pairs the bin holds, and the
    running fraction *above* the bin.  The last column is the only place a
    ``fraction beyond X degrees`` number should ever come from -- reading it off
    a table beats quoting three hand-picked thresholds, because the reader
    picks the threshold.
    """
    if bins is None:
        bins = THETA_BINS
    width = np.diff(bins)
    frac = np.asarray(y, float) * width
    return pd.DataFrame(dict(
        theta_lo=bins[:-1], theta_hi=bins[1:],
        theta_mid=0.5 * (bins[1:] + bins[:-1]),
        density=y, frac_in_bin=frac,
        frac_above=frac.sum() - np.cumsum(frac),
    ))


def frac_above(y, deg: float, bins=None) -> float:
    """Fraction of a density ``y`` beyond ``deg``, by interpolating the table.

    The tables and figures carry the whole spectrum; this is here so that a
    single number quoted in prose is read off the *same* object the figure
    plots, rather than recomputed from a separate sample.
    """
    t = spectrum_table(y, bins)
    return float(np.interp(deg, t.theta_hi, t.frac_above))


@functools.lru_cache(maxsize=4096)
def alpha_pair(kind: str, w: float = E_TRANSITION, npts: int = 4000) -> float:
    """Pairs per photon for this multipole, Born, Z -> 0.

    Undefined for E0 -- there is no photon to divide by -- and raises.  The
    normalisation is fixed by requiring the soft limit
    ``dGamma/dM^2 / Gamma_gamma -> (alpha/3pi)/M^2`` as ``M -> 0``, which makes
    the conversion factor for a multipole with nuclear factors ``(N_T, N_L)``

        (alpha/3pi) (1/M^2) beta* (1 + 2m^2/M^2) [2 N_T + N_L] / [2 N_T(k=W)]

    with ``N_T(k=W)`` the same expression at the photon point.
    """
    if kind == 'E0':
        raise ValueError('E0 has no photon branch; use e0_pair_width_eV() instead')
    m2 = np.exp(np.linspace(np.log((2 * M_E) ** 2 * (1 + 1e-9)),
                            np.log(w ** 2), npts))
    M = np.sqrt(m2)
    k = np.sqrt(np.clip(w ** 2 - m2, 0, None))
    b = np.sqrt(np.clip(1 - 4 * M_E ** 2 / m2, 0, None))
    n_t, n_l = nuclear_factors(kind, k, M, w)
    n_t0, _ = nuclear_factors(kind, np.full_like(k, w), M, w)
    f = (ALPHA / (3 * np.pi)) / m2 * b * (1 + 2 * M_E ** 2 / m2) \
        * (2 * n_t + n_l) / (2 * n_t0)
    return float(np.trapezoid(f, m2))


def e0_pair_width_eV(m_e0_fm2: float, w: float = E_TRANSITION) -> float:
    """E0 pair width in eV for a monopole matrix element ``M(E0)`` in fm^2.

    ``Gamma = (8/9pi) alpha^3 (m_e c^2/hbar) (M(E0)/lambdabar_C^2)^2 / 4 * I``
    with ``I`` Wilkinson's dimensionless energy-sharing integral.  Radius
    convention cancels.  Z = 2 (Fermi functions 1).
    """
    lam_c_fm = 197.3269804 / M_E
    wm = w / M_E
    ep = np.linspace(1.0, wm - 1.0, 20001)
    em = wm - ep
    integ = np.trapezoid(np.sqrt(ep ** 2 - 1) * np.sqrt(np.clip(em ** 2 - 1, 0, None))
                         * (ep * em - 1.0), ep)
    rate = (8.0 / (9 * np.pi) * ALPHA ** 3 * (M_E * 1e6 / 6.582119569e-16)
            * (m_e0_fm2 / lam_c_fm ** 2) ** 2 / 4.0 * integ)
    return float(rate * 6.582119569e-16)


# --------------------------------------------------------------------------- #
# mixtures -- the only free parameter left
# --------------------------------------------------------------------------- #
def mixture(weights: dict, n: int = 2_000_000, w: float = E_TRANSITION,
            seed: int = 21) -> pd.DataFrame:
    """One sample from a weighted sum of multipoles.

    ``weights`` are *pair yields*, not photon yields: ``{'E0': 0.2, 'M1': 0.8}``
    means one pair in five is a monopole pair.  Converting a photon-branch
    ratio into one of these needs :func:`alpha_pair`, which is exactly the step
    the current rate table skips.
    """
    tot = float(sum(weights.values()))
    out = []
    for i, (kind, f) in enumerate(sorted(weights.items())):
        if f <= 0:
            continue
        d = sample(kind, n, w, seed=seed + 7 * i)
        d['weight'] *= f / tot / d.weight.sum()
        d['kind'] = kind
        out.append(d)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(n: int = 4_000_000, w: float = E_TRANSITION) -> pd.DataFrame:
    """Two exact checks and one limit check.  Returns a table, prints nothing.

    The tolerances are set just above the Monte-Carlo noise of a 2e6-event
    weighted sample, so ``n`` is clamped there: a validation that gets easier
    when you ask for fewer events is not a validation.
    """
    n = max(int(n), 2_000_000)
    rows = []

    vp = sample('E0', n, w, seed=31)
    lab = e0_lab(n, w, seed=32)
    rows.append(dict(
        check='E0: virtual-photon machinery vs exact lab contact law',
        quantity='fraction above 109 deg',
        a=wfrac(vp, X17_MIN_DEG), b=wfrac(lab, X17_MIN_DEG),
        tol=0.002))
    rows.append(dict(
        check='E0: virtual-photon machinery vs exact lab contact law',
        quantity='median opening angle [deg]',
        # 0.15 deg is 0.25 % of the median -- above the Monte-Carlo noise of
        # two independently seeded weighted samples, far below any real
        # disagreement between the two derivations.
        a=wmedian(vp), b=wmedian(lab), tol=0.15))

    # E0 energy sharing against Wilkinson's published integrand
    edges = np.linspace(M_E, w - M_E, 41)
    mid = 0.5 * (edges[1:] + edges[:-1])
    h, _ = np.histogram(vp.e_plus, bins=edges, weights=vp.weight)
    em = w - mid
    wilk = (np.sqrt(mid ** 2 - M_E ** 2) * np.sqrt(em ** 2 - M_E ** 2)
            * (mid * em - M_E ** 2))
    ratio = (h / h.sum()) / (wilk / wilk.sum())
    rows.append(dict(
        check='E0: energy sharing vs Wilkinson p+p-(E+E- - m^2)',
        quantity='max |ratio - 1| over the spectrum',
        a=float(np.max(np.abs(ratio[2:-2] - 1))), b=0.0, tol=0.02))

    # the quadrature spectrum against the sampled one, over the WHOLE curve
    # rather than at one threshold -- total variation distance, which is the
    # largest fraction of pairs any reshuffling of the two could disagree on.
    for kind in ('E0', 'M1'):
        ymc = shape(sample(kind, n, w, seed=33), THETA_BINS)
        yg = grid_spectrum(kind, w, THETA_BINS)
        tv = float(0.5 * np.abs(yg - ymc).sum() * np.diff(THETA_BINS)[0])
        rows.append(dict(
            check=f'{kind}: quadrature spectrum vs sampled spectrum',
            quantity='total variation distance over 1 deg bins, 0-180',
            a=tv, b=0.0, tol=0.01))

    # the soft limit that fixes alpha_pair's normalisation
    for kind in ('M1', 'E1'):
        m2 = (3 * M_E) ** 2
        M = np.sqrt(m2)
        k = np.sqrt(w ** 2 - m2)
        n_t, n_l = nuclear_factors(kind, k, M, w)
        n_t0, _ = nuclear_factors(kind, w, M, w)
        rows.append(dict(
            check=f'{kind}: conversion factor -> (alpha/3pi)/M^2 as M -> 0',
            quantity='(2N_T + N_L)/(2N_T(k=W)) at M = 3 m_e',
            a=float((2 * n_t + n_l) / (2 * n_t0)), b=1.0, tol=0.02))

    v = pd.DataFrame(rows)
    v['delta'] = (v.a - v.b).abs()
    v['pass'] = v.delta <= v.tol
    return v


# --------------------------------------------------------------------------- #
# the tables this module exists to produce
# --------------------------------------------------------------------------- #
def multipole_table(n: int = 2_000_000, w: float = E_TRANSITION) -> pd.DataFrame:
    rows = []
    for kind in ('E0', 'M1', 'E1', 'E2', 'M2'):
        d = sample(kind, n, w)
        row = dict(multipole=kind, what=CHANNELS[kind],
                   median_deg=wmedian(d),
                   frac_gt90=wfrac(d, 90.0),
                   frac_gt109=wfrac(d, X17_MIN_DEG),
                   frac_gt130=wfrac(d, 130.0))
        row['alpha_pair'] = np.nan if kind == 'E0' else alpha_pair(kind, w)
        rows.append(row)
    return pd.DataFrame(rows)


def ansatz_table(n: int = 2_000_000, w: float = E_TRANSITION) -> pd.DataFrame:
    """The four variants currently carried as the band, on the same axis."""
    from sept26_prelim_analysis import pair_physics as PP
    rows = []
    for name, kw in PP.VARIANTS.items():
        a = PP.ipc_angles(n // 4, **kw)
        d = pd.DataFrame(dict(theta_deg=a, weight=np.ones(len(a))))
        rows.append(dict(variant=name, median_deg=wmedian(d),
                         frac_gt90=wfrac(d, 90.0),
                         frac_gt109=wfrac(d, X17_MIN_DEG),
                         frac_gt130=wfrac(d, 130.0)))
    return pd.DataFrame(rows)


def energy_scan(kinds=('E1', 'M1'), energies=None,
                n: int = 1_000_000) -> pd.DataFrame:
    """How the wide-angle tail dies with transition energy -- the Al question."""
    if energies is None:
        energies = [20.58, 12.0, 9.0, AL_SN, 6.0, 4.734, 3.034, 2.590, 1.779]
    rows = []
    for w in energies:
        for kind in kinds:
            d = sample(kind, n, w, seed=41)
            rows.append(dict(w_MeV=w, multipole=kind,
                             alpha_pair=alpha_pair(kind, w),
                             frac_gt109=wfrac(d, X17_MIN_DEG),
                             pairs_gt109_per_photon=alpha_pair(kind, w)
                             * wfrac(d, X17_MIN_DEG)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=2_000_000)
    ap.add_argument('--al', action='store_true', help='also the energy scan')
    ap.add_argument('--write', action='store_true', help='write CSVs to paths.out')
    a = ap.parse_args()

    V = validate(min(a.n * 2, 4_000_000))
    print('VALIDATION')
    print(V.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print(f'  -> {"ALL PASS" if V["pass"].all() else "FAILURE"}\n')

    T = multipole_table(a.n)
    print('BORN MULTIPOLES at W = 20.58 MeV')
    print(T.drop(columns=['what']).to_string(index=False,
                                             float_format=lambda x: f'{x:.4f}'))
    A = ansatz_table(a.n)
    print('\nWHAT pair_physics.py CURRENTLY CARRIES, same axis')
    print(A.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    S = None
    if a.al:
        S = energy_scan(n=max(a.n // 2, 500_000))
        print('\nTRANSITION-ENERGY SCAN  (Al capture is 7.7255 MeV and below)')
        print(S.to_string(index=False, float_format=lambda x: f'{x:.4e}'))

    if a.write:
        from sept26_prelim_analysis import paths
        od = paths.out('ipc')
        V.to_csv(od / 'ipc_born_validation.csv', index=False)
        T.to_csv(od / 'ipc_born_multipoles.csv', index=False)
        A.to_csv(od / 'ipc_born_ansatz.csv', index=False)
        if S is not None:
            S.to_csv(od / 'ipc_born_energy_scan.csv', index=False)
        print(f'\nwrote -> {od}')
    return 0 if V['pass'].all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
