#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ganil_background.py -- the same experiment at NFS/GANIL, where E_n is MeV.

    python -m sept26_prelim_analysis.ganil_background
    python -m sept26_prelim_analysis.ganil_background --write

THE ONE SENTENCE.  At n_TOF every neutron this analysis sees is below 2 eV, so
the 4He excitation is 20.578 MeV to eight digits and one template covers the
whole run.  At NFS the neutron carries 1-40 MeV, the excitation becomes
``E_x = S_n + 0.749 E_n`` and runs from 21 to 51 MeV, and **everything moves
with it -- except the capsule background, which does not move at all.**  That
asymmetry is the whole content of this page.

FOUR THINGS CHANGE, AND THEY DO NOT ALL POINT THE SAME WAY.

1.  **The X17 signature stops being "a bump near 110 deg".**  A 16.8 MeV boson
    from a transition of energy ``E_x`` has a hard minimum opening angle
    ``cos(theta_min) = 1 - 2 m^2 / E_x^2``, which is 109 deg only because
    E_x happens to be 20.58 MeV at n_TOF.  At 40 MeV neutrons it is 39 deg --
    the signal has walked down into the middle of the internal-pair continuum.
    :func:`x17_min_angle`.

2.  **But the neutron energy is measured, event by event, and the signal angle
    is a known function of it.**  So the signature becomes a *correlation*:
    theta_peak tracking theta_min(E_n) across three decades of rate.  The
    internal-pair continuum from the gas partly tracks it too (same E_x); the
    capsule background does not track it at all, because a 2.2 MeV inelastic
    gamma from 27Al is 2.2 MeV whatever the neutron did.  This is a handle that
    does not exist at n_TOF and it is worth more than the loss in 1.

3.  **The signal per neutron gets dramatically better.**  At thermal the 3He
    cell is optically thick to (n,p) at 5333 b, so essentially every neutron is
    eaten by a channel that makes nothing, and only 1e-8 of them go radiative.
    At MeV the competition has collapsed to barns while (n,gamma) has only
    fallen by a factor of a few -- the radiative branch per neutron *entering
    the cell* goes up by more than two orders of magnitude.
    :func:`he3_rates`.

4.  **The capsule background changes character completely.**  Capture is over;
    at MeV the capsule radiates by INELASTIC SCATTERING, at cross sections
    ~1 b rather than 0.2 b, and the photons are the 27Al level scheme rather
    than the 28Al one.  Two consequences that pull in opposite directions: the
    two strongest lines (844 and 1014 keV) are BELOW the pair threshold and
    make no pairs at all, but 12C's 4.44 MeV level turns on hard above 6 MeV
    and is a 460 mb E2 source of exactly the photons that make wide pairs.
    :func:`capsule_lines`.

WHERE THE NUMBERS COME FROM.  Cross sections are ENDF/B-VIII.0, read straight
off the evaluations staged in ``data/nuclear/`` by :mod:`endf` -- including the
inelastic level energies, which are minus the QI of each MT.  The pair physics
is :mod:`ipc_born`, unchanged: it was never specific to thermal neutrons, only
the *weights* were.  What is assumed rather than read is listed in
:func:`missing` and it is a short list.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import endf as EN  # noqa: E402
from sept26_prelim_analysis import ipc_born as IB  # noqa: E402
from sept26_prelim_analysis import ipc_aluminium as AL  # noqa: E402

SCHEMA = 'sept26_prelim/ganil/1'

# --------------------------------------------------------------------------- #
# the reaction
# --------------------------------------------------------------------------- #
S_N_HE4 = 20.5776           # MeV, neutron separation energy of 4He
M_X17 = 16.8                # MeV, the boson mass the anomaly is quoted at
#: m(3He)/(m(3He) + m(n)) -- the fraction of the lab neutron energy that is
#: available in the centre of mass, and therefore goes into the excitation.
CM_FRAC = 2808.391 / (2808.391 + 939.565)

#: The NFS neutron range, and the two production modes.  These set the axis of
#: every figure here; they are not fitted to anything.
EN_MIN_MEV, EN_MAX_MEV = 1.0, 40.0

#: The n_TOF signal window, kept so the two pages can be compared: 36 degrees
#: wide, starting at the kinematic minimum.  At n_TOF that is 109-145 deg.
WINDOW_WIDTH_DEG = 36.0

# --------------------------------------------------------------------------- #
# what radiates, and how much of it there is
# --------------------------------------------------------------------------- #
#: Areal densities of the existing capsule, atoms per barn, path-averaged over
#: the 4 cm sphere.  Reused from the n_TOF geometry so the two pages are the
#: same apparatus, which is the comparison worth making.
N_AL, N_C, N_HE3 = AL.N_AL_ATB, AL.N_CF_ATB, AL.N_HE3_ATB

#: Below this a photon cannot make a pair at all; quoted because it removes the
#: two strongest aluminium lines outright.
PAIR_THRESHOLD_MEV = 2 * IB.M_E

#: How a de-excitation gamma's multipole is taken, per nuclide.  The low-lying
#: levels of 27Al are all positive parity (it is an sd-shell nucleus and the
#: first negative-parity state is high), so the strong inelastic gammas are
#: M1/E2 rather than E1 -- which converts less, and more collimated.  E2 and M1
#: agree to 4 % in everything computed here, so one stands for both.  Levels
#: above ``MIXED_ABOVE_MEV`` are carried as an M1-to-E1 bracket instead.
MIXED_ABOVE_MEV = 5.0
DEFAULT_MULTIPOLE = {'Al27': 'M1', 'C12': 'E2'}


def quiet_band() -> tuple:
    """``(lo, hi)`` in MeV: the neutron energies at which the wall is silent.

    The upper edge is not chosen, it is a threshold: the lowest 27Al or 12C
    level that is ABOVE the pair-creation threshold, converted to the lab
    neutron energy that opens it.  Below it the capsule's only pair source is
    its own millibarn radiative capture, because its two strongest inelastic
    lines cannot convert.  Computed from the evaluations rather than typed in,
    so a different capsule material moves it automatically.
    """
    best = np.inf
    for nuc, a in (('Al27', 27), ('C12', 12)):
        lv = EN.levels(nuc)
        lv = lv[lv.e_level_MeV > PAIR_THRESHOLD_MEV]
        if len(lv):
            best = min(best, float(lv.e_level_MeV.min()) * (a + 1) / a)
    return EN_MIN_MEV, float(best)


def excitation(en_mev) -> np.ndarray:
    """4He excitation energy for a lab neutron energy, MeV."""
    return S_N_HE4 + CM_FRAC * np.asarray(en_mev, float)


def x17_min_angle(en_mev) -> np.ndarray:
    """Smallest opening angle a 16.8 MeV boson can give, degrees.

    ``cos(theta_min) = 1 - 2 m^2 / E_x^2`` for the symmetric decay, and no
    sharing of the energy between the two tracks does better.  This is the
    single most consequential number on the page: it is 109 deg at n_TOF only
    because the neutron brings nothing.
    """
    ex = excitation(en_mev)
    return np.degrees(np.arccos(np.clip(1 - 2 * M_X17 ** 2 / ex ** 2, -1, 1)))


def x17_spectrum(en_mev: float, bins=None, n: int = 400_000,
                 seed: int = 71) -> np.ndarray:
    """Opening-angle spectrum of X17 -> e+e- at one neutron energy.

    Two-body kinematics, exactly: the boson is emitted with
    ``E = E_x``, ``p = sqrt(E_x^2 - m^2)`` in the 4He frame (the recoil is
    taken as negligible, which costs under a degree at 40 MeV), and decays
    isotropically because a spin-0 or spin-1 boson at rest has no preferred
    axis relative to nothing.
    """
    if bins is None:
        bins = IB.THETA_BINS
    ex = float(excitation(en_mev))
    if ex <= M_X17:
        return np.zeros(len(bins) - 1)
    rng = np.random.default_rng(seed)
    gam = ex / M_X17
    bet = np.sqrt(max(1 - 1 / gam ** 2, 0.0))
    c = rng.uniform(-1, 1, n)
    s = np.sqrt(np.clip(1 - c ** 2, 0, None))
    e_star = M_X17 / 2.0
    p_star = np.sqrt(max(e_star ** 2 - IB.M_E ** 2, 0.0))
    pz1 = gam * (p_star * c + bet * e_star)
    pz2 = gam * (-p_star * c + bet * e_star)
    px1, px2 = p_star * s, -p_star * s
    dot = px1 * px2 + pz1 * pz2
    th = np.degrees(np.arccos(np.clip(
        dot / np.hypot(px1, pz1) / np.hypot(px2, pz2), -1, 1)))
    h, _ = np.histogram(th, bins=bins)
    return h / (h.sum() * np.diff(bins))


def window(en_mev: float) -> tuple:
    """The signal window at this neutron energy: ``(lo, hi)`` in degrees."""
    lo = float(x17_min_angle(en_mev))
    return lo, min(lo + WINDOW_WIDTH_DEG, 180.0)


def _frac_in(y, lo, hi, bins=None) -> float:
    """Fraction of a density between two angles."""
    if bins is None:
        bins = IB.THETA_BINS
    t = IB.spectrum_table(y, bins)
    above = np.interp([lo, hi], t.theta_hi, t.frac_above)
    return float(above[0] - above[1])


# --------------------------------------------------------------------------- #
# the gas
# --------------------------------------------------------------------------- #
#: Above this, ENDF/B-VIII.0 sets every DISCRETE inelastic level cross section
#: to zero and moves the strength into MT = 91, the continuum -- for aluminium
#: and for carbon alike.  So above 20 MeV neither the signal nor the background
#: has the data this page needs, from opposite directions.  Every table stops
#: there and says so.
DISCRETE_EVAL_MAX_MEV = 20.0

#: The top of every evaluated 3He(n,gamma) cross section there is.  Both
#: ENDF/B-VIII.0 and TENDL-2021 stop at 20 MeV, so the upper half of the NFS
#: range has no evaluation behind it -- see :func:`missing`.
HE3_EVAL_MAX_MEV = 20.0


def he3_rates(en_mev, extrapolate: bool = True) -> pd.DataFrame:
    """What the 3He gas does per neutron entering the cell, against energy.

    The interesting column is ``radiative_per_neutron``.  At thermal it is
    1.0e-8, because the cell is black to (n,p) and only one absorption in 1e8
    is radiative.  At MeV the (n,p) has fallen by three and a half decades
    while (n,gamma) has fallen by less than one, so the *same cell* converts
    far more of its neutrons into the reaction this experiment is looking for.
    """
    e = np.atleast_1d(np.asarray(en_mev, float)) * 1e6
    s_g = EN.sigma_at('He3', 'capture', e)
    s_p = EN.sigma_at('He3', 'np', e)
    s_t = EN.sigma_at('He3', 'total', e)
    beyond = e > HE3_EVAL_MAX_MEV * 1e6
    if extrapolate and beyond.any():
        # continue each cross section as the power law of the evaluation's own
        # last decade.  Labelled everywhere it is used; it is an extension of a
        # trend, not a calculation, and the alternative is a blank half-page.
        for arr, mt in ((s_g, 'capture'), (s_p, 'np'), (s_t, 'total')):
            d = EN.xs('He3', mt)
            m = d.E_eV >= 2e6
            x, y = np.log(d.E_eV[m]), np.log(np.clip(d.sigma_b[m], 1e-12, None))
            k = np.polyfit(x, y, 1)
            arr[beyond] = np.exp(np.polyval(k, np.log(e[beyond])))
    # the radiative branch of what is ABSORBED, not the thin-target product:
    # at thermal the cell is optically thick and the two differ by the optical
    # depth, which is a factor of 170.
    tau_abs = N_HE3 * (s_p + s_g)
    p_abs = 1.0 - np.exp(-tau_abs)
    frac_rad = np.where(s_p + s_g > 0, s_g / np.clip(s_p + s_g, 1e-30, None), 0)
    return pd.DataFrame(dict(
        En_MeV=e * 1e-6, Ex_MeV=excitation(e * 1e-6),
        sigma_ngamma_b=s_g, sigma_np_b=s_p, sigma_total_b=s_t,
        tau_absorption=tau_abs, p_absorbed=p_abs,
        radiative_per_neutron=p_abs * frac_rad,
        np_per_neutron=p_abs * (1 - frac_rad),
        # (n,p) makes a proton and a triton from one vertex, which is a
        # two-prong topology a TPC has to tell from a pair.  How many of them
        # there are per useful event is a number worth carrying next to the
        # signal rather than in a footnote.
        np_per_radiative=np.where(frac_rad > 0, (1 - frac_rad) / np.clip(
            frac_rad, 1e-30, None), np.inf),
        beyond_evaluation=beyond))


def he3_pair_spectrum(en_mev: float, multipole: str = 'E1',
                      bins=None) -> np.ndarray:
    """The gas's own internal-pair continuum at this neutron energy.

    ``multipole`` is the assumption, not a measurement: above 1 MeV the entrance
    channel is no longer two s-waves and the capture is direct/semi-direct,
    which is E1-dominated in every light nucleus where it has been measured.
    M1 is carried as the alternative everywhere this matters.
    """
    return IB.grid_spectrum(multipole, float(excitation(en_mev)), bins)


# --------------------------------------------------------------------------- #
# the capsule
# --------------------------------------------------------------------------- #
def capsule_lines(en_mev: float) -> pd.DataFrame:
    """Every photon the capsule wall makes at one neutron energy.

    Each discrete inelastic level (MF = 3, MT = 51..90) is taken to de-excite
    by a single photon of the level energy straight to the ground state.  That
    is exact for 12C, whose 4.44 MeV level has nowhere else to go, and it is
    the HARD end of the bracket for 27Al, where the upper levels feed the 844
    and 1014 keV states instead -- a cascade makes softer photons, which convert
    less but more widely, so the error does not all go one way.
    :func:`missing` carries it.

    Radiative capture is included as one line at ``S_n``, which at MeV energies
    is a rounding correction and is kept only so the two pages use one formula.
    """
    rows = []
    for nuc, n_atb, sn in (('Al27', N_AL, 7.7255), ('C12', N_C, 4.9463)):
        lv = EN.levels(nuc)
        for _, r in lv.iterrows():
            s = float(EN.sigma_at(nuc, int(r.MT), en_mev * 1e6))
            if s <= 0 or r.e_level_MeV <= PAIR_THRESHOLD_MEV:
                # below 1.022 MeV a photon cannot make a pair at all; the two
                # strongest aluminium lines live here and are simply absent
                # from this background, which is worth seeing in the table.
                rows.append(dict(nuclide=nuc, source=f'(n,n\'){r.level}',
                                 e_gamma_MeV=r.e_level_MeV, sigma_b=s,
                                 per_neutron=n_atb * s, multipole='',
                                 below_threshold=True))
                continue
            mult = ('M1' if r.e_level_MeV > MIXED_ABOVE_MEV
                    else DEFAULT_MULTIPOLE[nuc])
            rows.append(dict(nuclide=nuc, source=f'(n,n\') level {int(r.level)}',
                             e_gamma_MeV=r.e_level_MeV, sigma_b=s,
                             per_neutron=n_atb * s, multipole=mult,
                             below_threshold=False))
        s_cap = float(EN.sigma_at(nuc, 'capture', en_mev * 1e6))
        rows.append(dict(nuclide=nuc, source='(n,g) to g.s.', e_gamma_MeV=sn,
                         sigma_b=s_cap, per_neutron=n_atb * s_cap,
                         multipole='E1', below_threshold=False))
    d = pd.DataFrame(rows)
    d['En_MeV'] = en_mev
    return d.sort_values('per_neutron', ascending=False).reset_index(drop=True)


def inelastic_completeness(en_mev) -> pd.DataFrame:
    """How much of the inelastic strength the discrete levels account for.

    The same kind of check the n_TOF page runs on the capture scheme, and it
    matters more here: above a few MeV the evaluation moves strength out of the
    named levels and into MT = 91, the continuum, whose photons this module
    cannot see.  Wherever ``discrete_fraction`` is well below 1 the capsule
    numbers are an UNDER-estimate, and by roughly that factor.
    """
    e = np.atleast_1d(np.asarray(en_mev, float)) * 1e6
    rows = []
    for nuc in ('Al27', 'C12'):
        lv = EN.levels(nuc)
        disc = sum(EN.sigma_at(nuc, int(mt), e) for mt in lv.MT)
        tot = EN.sigma_at(nuc, 'inelastic', e)
        rows.append(pd.DataFrame(dict(
            nuclide=nuc, En_MeV=e * 1e-6, sigma_discrete_b=disc,
            sigma_inelastic_b=tot,
            discrete_fraction=np.where(tot > 0, disc / np.clip(tot, 1e-30,
                                                              None), np.nan))))
    return pd.concat(rows, ignore_index=True)


def capsule_pairs(en_mev: float, bins=None) -> tuple:
    """``(spectrum, pairs_per_neutron)`` for the whole capsule wall.

    Sums the Born curve of every line above the pair threshold, weighted by
    that line's production rate and its conversion coefficient.
    """
    if bins is None:
        bins = IB.THETA_BINS
    d = capsule_lines(en_mev)
    d = d[(~d.below_threshold) & (d.per_neutron > 0)
          & (d.e_gamma_MeV > 1.05 * PAIR_THRESHOLD_MEV)]
    acc = np.zeros(len(bins) - 1)
    tot = 0.0
    for _, r in d.iterrows():
        a = IB.alpha_pair(r.multipole, float(r.e_gamma_MeV))
        wgt = r.per_neutron * a
        acc += wgt * IB.grid_spectrum(r.multipole, float(r.e_gamma_MeV), bins)
        tot += wgt
    if tot > 0:
        acc = acc / (acc * np.diff(bins)).sum()
    return acc, float(tot)


# --------------------------------------------------------------------------- #
# the comparison this page exists to make
# --------------------------------------------------------------------------- #
def energy_scan(energies=None, he_multipole: str = 'E1') -> pd.DataFrame:
    """One row per neutron energy: signal window, and what is in it.

    ``capsule_in_window`` versus ``he3_in_window`` is the answer; the reason it
    is worth tabulating rather than plotting alone is that the two move for
    different reasons -- the gas's continuum follows the window because both
    follow E_x, and the capsule's does not follow it at all.
    """
    if energies is None:
        # stops at 20 MeV, which is where BOTH evaluations stop carrying what
        # this page needs -- 3He(n,gamma) simply ends, and the aluminium and
        # carbon discrete levels are zeroed in favour of the continuum.
        energies = np.array([1., 1.5, 2., 2.5, 3., 4., 5., 7., 10., 14., 20.])
    rows = []
    for en in np.atleast_1d(energies):
        lo, hi = window(en)
        y_he = he3_pair_spectrum(en, he_multipole)
        y_cap, cap_tot = capsule_pairs(en)
        he = he3_rates(en).iloc[0]
        he_pairs = float(he.radiative_per_neutron
                         * IB.alpha_pair(he_multipole, float(he.Ex_MeV)))
        f_he = _frac_in(y_he, lo, hi)
        f_cap = _frac_in(y_cap, lo, hi) if cap_tot > 0 else 0.0
        f_x17 = _frac_in(x17_spectrum(en), lo, hi)
        rows.append(dict(
            En_MeV=float(en), Ex_MeV=float(excitation(en)),
            theta_min_deg=lo, window_hi_deg=hi,
            x17_in_window=f_x17,
            he3_pairs_per_neutron=he_pairs,
            he3_in_window=he_pairs * f_he,
            capsule_pairs_per_neutron=cap_tot,
            capsule_in_window=cap_tot * f_cap,
            capsule_frac_in_window=f_cap,
            he3_frac_in_window=f_he,
            discrete_fraction_Al=float(inelastic_completeness(en)
                                       .query('nuclide == "Al27"')
                                       .discrete_fraction.iloc[0]),
            he3_beyond_evaluation=bool(he.beyond_evaluation)))
    d = pd.DataFrame(rows)
    d['capsule_over_he3'] = d.capsule_in_window / d.he3_in_window
    return d


def ntof_comparison(he_multipole: str = 'E1') -> pd.DataFrame:
    """NFS against n_TOF, on the quantities that decide whether it is worth it.

    The thermal row is computed the same way as the MeV ones -- same cell, same
    formula -- so the two columns differ only by the neutron energy, which is
    the point.
    """
    from sept26_prelim_analysis import ipc_channels as IC
    e_th = 0.0253e-6                       # MeV
    s_g = float(EN.sigma_at('He3', 'capture', e_th * 1e6))
    s_p = float(EN.sigma_at('He3', 'np', e_th * 1e6))
    p_abs = 1.0 - np.exp(-N_HE3 * (s_p + s_g))
    th_rad = p_abs * s_g / (s_p + s_g)
    q_hi = quiet_band()[1]
    rows = [dict(quantity='neutron energy', ntof='0.001&ndash;2 eV',
                 nfs='1&ndash;40 MeV, and 1&ndash;20 MeV is where the data is'),
            dict(quantity='&#8308;He excitation E<sub>x</sub>',
                 ntof=f'{S_N_HE4:.3f} MeV, fixed to eight digits',
                 nfs=f'{excitation(1.0):.1f}&ndash;{excitation(40.0):.1f} MeV '
                     f'&mdash; a variable'),
            dict(quantity='X17 minimum opening angle',
                 ntof='109&deg;, fixed',
                 nfs=f'{x17_min_angle(1.0):.0f}&deg; down to '
                     f'{x17_min_angle(40.0):.0f}&deg;, and E<sub>n</sub> tells '
                     f'you which'),
            dict(quantity='&sup3;He cell to (n,p)',
                 ntof=f'optically thick, &tau; = {N_HE3 * s_p:.0f} &mdash; the '
                      f'beam is consumed making nothing',
                 nfs=f'thin, &tau; = '
                     f'{float(EN.sigma_at("He3", "np", 5e6)) * N_HE3:.3f} '
                     f'at 5 MeV'),
            dict(quantity='the quiet window',
                 ntof='none &mdash; the capsule captures at every energy',
                 nfs=f'below {q_hi:.2f} MeV the wall has no line above the '
                     f'pair threshold'),
            dict(quantity='radiative captures per neutron entering',
                 ntof=f'{th_rad:.2e}',
                 nfs=f'{float(he3_rates(1.0).radiative_per_neutron.iloc[0]):.2e}'
                     f' at 1 MeV, '
                     f'{float(he3_rates(14.0).radiative_per_neutron.iloc[0]):.2e}'
                     f' at 14 MeV'),
            dict(quantity='what the capsule radiates by',
                 ntof='capture, 0.231 b on &sup2;&#8311;Al',
                 nfs='inelastic scattering, ~1 b, but its two strongest lines '
                     'are below the pair threshold'),
            dict(quantity='does the expected spectrum move with E<sub>n</sub>',
                 ntof='no &mdash; 2&times;10&#8315;&#8313; across the whole '
                      'window, so one template covers it',
                 nfs='the gas does, the capsule does not -- which is the new '
                     'handle'),
            ]
    return pd.DataFrame(rows)


def missing() -> pd.DataFrame:
    """What this page assumes rather than reads, worst first."""
    rows = [
        ('the capsule cascade is taken as one photon per level', 'x0.5-1',
         'each discrete inelastic level is de-excited by a single photon of '
         'the level energy straight to the ground state. Exact for 12C, whose '
         '4.44 MeV level has nowhere else to go. For 27Al the upper levels '
         'mostly feed the 844 and 1014 keV states instead, so the real photons '
         'are softer than assumed -- fewer pairs, but wider ones. ENDF/B-VIII.0 '
         'carries this in MF=6 with ZAP=0 and this module does not read MF=6 '
         'yet; that is the single biggest improvement available here.'),
        ('the multipole of the gas capture above 1 MeV', 'x1.5',
         'taken as E1 because direct/semi-direct capture is E1-dominated in '
         'light nuclei. Above the 20.58 MeV threshold the 4He compound states '
         'are broad and overlapping and no partial-wave decomposition exists '
         'at these energies. M1 is carried as the alternative and moves the '
         'wide-angle yield by about 1.5.'),
        ('X17 production at E_x = 21-51 MeV', 'unknown',
         'the anomaly is reported for the 20.21 and 21.01 MeV states of 4He. '
         'At 1-40 MeV neutrons the compound sits above them, in a region with '
         'no resonance to enhance anything, and the X17-to-photon ratio there '
         'is a model statement rather than a measurement. This page is about '
         'the BACKGROUND and deliberately quotes no signal rate.'),
        ('everything above the 4He breakup thresholds', 'unknown',
         'E_x above ~23.8 MeV opens d+d, and the compound is unbound to n+3He '
         'and p+3H throughout. Radiative capture is a small branch of a system '
         'that mostly falls apart, and ENDF MT=102 for 3He above ~10 MeV is an '
         'evaluation with little data behind it. Treat the high-energy end as '
         'indicative.'),
        ('neutron transport and the beam profile', 'x1-6',
         'as at n_TOF: single-pass optical depths, no scattering in the wall, '
         'no beam profile, no self-shielding beyond the analytic sphere. The '
         'same Geant4 run fixes both pages.'),
        ('external conversion and charged-particle backgrounds', 'unknown',
         'at MeV energies the cell also makes recoil protons, (n,p) and (n,d) '
         'charged products, and far more high-energy photons that can convert '
         'in material. None of that is a PAIR from a transition, so none of it '
         'is on this page, and all of it is a trigger load.'),
        ('the facility numbers', 'n/a',
         'flight path, pulse rate and flux are not used anywhere here -- every '
         'quantity is per neutron entering the cell. A rate projection needs '
         'the NFS flux, which is the one input that has to come from GANIL.'),
    ]
    return pd.DataFrame(rows, columns=['what is assumed', 'how much it moves',
                                       'why it matters'])


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--multipole', default='E1', choices=('E1', 'M1'))
    a = ap.parse_args()

    print('THE KINEMATICS MOVE')
    k = pd.DataFrame(dict(En_MeV=[0.0, 1, 2, 5, 10, 20, 40]))
    k['Ex_MeV'] = excitation(k.En_MeV)
    k['x17_theta_min_deg'] = x17_min_angle(k.En_MeV)
    print(k.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    print('\nTHE GAS')
    H = he3_rates([0.0253e-6, 1, 2, 5, 10, 14, 20, 40])
    print(H.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    print('\nWHAT THE CAPSULE RADIATES AT 5 MeV (top 12 by rate)')
    C = capsule_lines(5.0)
    print(C.head(12).to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    S = energy_scan(he_multipole=a.multipole)
    print('\nTHE SIGNAL WINDOW AND WHAT IS IN IT')
    print(S.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    print('\nNFS AGAINST n_TOF')
    N = ntof_comparison(a.multipole)
    for _, r in N.iterrows():
        print(f'  {r.quantity:<44s} n_TOF: {r.ntof}')
        print(f'  {"":<44s} NFS  : {r.nfs}')

    print('\nWHAT IS ASSUMED')
    for _, r in missing().iterrows():
        print(f'  [{r["how much it moves"]:>9s}]  {r["what is assumed"]}')

    if a.write:
        from sept26_prelim_analysis import paths
        od = paths.out('ganil')
        k.to_csv(od / 'ganil_kinematics.csv', index=False)
        H.to_csv(od / 'ganil_he3_rates.csv', index=False)
        C.to_csv(od / 'ganil_capsule_lines_5MeV.csv', index=False)
        S.to_csv(od / 'ganil_energy_scan.csv', index=False)
        N.to_csv(od / 'ganil_ntof_comparison.csv', index=False)
        missing().to_csv(od / 'ganil_missing.csv', index=False)
        print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
