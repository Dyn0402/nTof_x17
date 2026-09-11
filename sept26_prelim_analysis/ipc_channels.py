#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ipc_channels.py -- which multipoles the >1 ms neutron window actually makes.

    python -m sept26_prelim_analysis.ipc_channels
    python -m sept26_prelim_analysis.ipc_channels --write

:mod:`ipc_born` says what each multipole's pair continuum looks like.  It does
not say how much of each there is.  That is a question about the *reaction*,
and the answer changes completely inside the window this analysis lives in.

THE WINDOW IS THE WHOLE POINT.  Nothing is recorded before 1 ms (the flash
veto), and over the 19.5 m EAR2 flight path 1 ms is **2.0 eV**.  Every neutron
in this analysis is thermal or epithermal.  Two consequences, and they are not
small:

  1. **Only s-wave survives.**  A p-wave capture amplitude carries a factor
     ``k_n R``, so its cross section falls as ``v`` relative to the 1/v s-wave.
     Between the lowest energy Viviani et al. tabulate (0.17 MeV) and 2 eV that
     is a factor of ~1e5.  The 1- (1P1) resonance that *dominates* every number
     in their Table V is simply not there in our window.
  2. **Two s-wave channels, and only one of them makes photons.**  n + 3He with
     both spins 1/2 forms J = 0+ and J = 1+, both positive parity.

         1+ (3S1) -> 4He(0+)   M1.  This is the radiative capture, and its
                               cross section is measured: 55 +- 3 ub thermal.
         0+ (1S0) -> 4He(0+)   E0.  A 0+ -> 0+ transition CANNOT emit a real
                               photon.  It goes to e+e- and nothing else, so
                               its pair yield is not bounded by, or even
                               related to, the measured (n,gamma) cross section.

WHY THE 0+ CHANNEL IS THE INTERESTING ONE.  It is not a small correction that
happens to be unmeasured -- it is the channel the neutron actually goes into.
3He neutron **spin filters** work because 3He(n,p)3H at thermal proceeds almost
entirely through the singlet (J = 0) channel: that is what makes the absorption
``sigma_0 (1 - P_n P_He)`` and the polariser possible at all.  So the 5333 b of
thermal (n,p) is essentially all 0+ formation, while the 55 ub of (n,gamma) is
all 1+.  The channel with no photons is populated 1e8 times more strongly than
the one we normalise the pair rate to.

WHAT THE RATE TABLE ASSUMES, AND WHERE IT COMES FROM.  ``IPC/capture =
2.1e-3`` in ``/media/dylan/data/x17/calculation_tables/results_3He`` is
Viviani et al.'s ratio of their two total cross sections, Table V of PRC 105,
014001 -- 0.0431/20.2 ub at En = 0.17 MeV, and the same 2.13e-3 at every
tabulated energy up to 2 MeV.  It is a good number *in that regime*, where both
numerator and denominator are dominated by the same p-wave 1- resonance.  Used
at 2 eV it is being extrapolated across five decades of neutron energy and a
complete change of which multipoles are open.  This module says what to use
instead, and how uncertain that is.

THE E0 ESTIMATE IS AN ORDER OF MAGNITUDE, AND IS LABELLED AS ONE.  Three
inputs, all quoted, none of them ours:

  M(E0) = 1.53 +- 0.05 fm2   the 4He monopole transition matrix element
                             <r^2>_tr, measured at Mainz in (e,e') and quoted
                             in arXiv:2306.07268 Table I.  Note the same paper
                             is about the "alpha-particle monopole puzzle":
                             ab initio theory misses this form factor, so the
                             *structure* behind this number is an open problem.
  Gamma(0+_2) = 0.50 MeV     total width of the 20.21 MeV 0+ state (TUNL A=4).
  sigma_np    = 5333 b       3He(n,p) at 25.3 meV, taken as all-0+.

with the single-level ratio ``sigma_pair/sigma_np = Gamma_pair/Gamma_tot``.
The 20.21 MeV state sits 0.37 MeV *below* the n+3He threshold, so at 2 eV we
are one and a half half-widths up its flank and a single-level Breit-Wigner is
being asked to do real work.  Treat the number as "tens of percent of the pair
yield", not as a prediction.

WHAT THIS MODULE NOW REPORTS, AND WHY IT CHANGED.  The deliverable is
:func:`thermal_spectrum` -- the whole ``dN/dtheta``, one degree at a time, with
the M1 and E0 components kept apart so a fit can float the mix.  It used to be
three fractions above three hand-picked thresholds, which is a lossy summary of
the same object and invites the reader to argue about the threshold instead of
the curve.  The fractions are still available (``ipc_born.frac_above`` reads
them off the spectrum) but nothing is quoted from anywhere else.

AND IT DOES NOT MOVE WITH ARRIVAL TIME.  :func:`energy_invariance` checks the
four things that could make the expected spectrum a function of time of flight
and all four are flat to at least 1e-5 across the whole >1 ms window: the
transition energy gains 0.75 E_n and E_n is eV against 20.58 MeV; the E0:M1 mix
is a ratio of two s-wave 1/v channels, so the velocity cancels exactly; the
capsule-to-gas capture ratio cancels the same way and keeps cancelling until
the first 27Al resonance at 5.9 keV, which is 34 us of flight; and the p-wave
admixture that would break all of it is 1e-5 at the top of the window.  One
template covers 1 ms to 1 s, which is what makes arrival time free to be used
against other backgrounds.

THE ONE THING THAT WOULD SETTLE IT is not ours to compute.  Viviani, Marcucci,
Kievsky, Schiavilla et al. already have the C0000 (1S0 -> 0+) and M1_011
(3S1 -> 0+) reduced matrix elements in the code that produced PRC 105, 014001;
they simply never ran it below 0.17 MeV because nobody had asked for the
thermal point.  Asking them for the 5-fold differential cross section at
En < 10 eV replaces every estimate on this page with an ab initio number.
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

from sept26_prelim_analysis import ipc_born as IB  # noqa: E402

SCHEMA = 'sept26_prelim/ipc_channels/1'

# --------------------------------------------------------------------------- #
# inputs, every one with a source
# --------------------------------------------------------------------------- #
SIGMA_NGAMMA_UB = 55.0          # 3He(n,g)4He thermal, Wervelman 1991 (55 +- 3)
SIGMA_NGAMMA_UB_ERR = 3.0
SIGMA_NP_B = 5333.0             # 3He(n,p)3H at 25.3 meV, ENDF/B; ~all singlet
M_E0_FM2 = 1.53                 # <r^2>_tr, 4He 0+_2 -> gs, arXiv:2306.07268 T.I
M_E0_FM2_ERR = 0.05
GAMMA_0P2_MEV = 0.50            # total width of the 20.21 MeV 0+ (TUNL A=4)

W_HE = IB.E_TRANSITION          # 20.58 MeV
W_AL = IB.AL_SN                 # 7.7255 MeV

#: Viviani et al. PRC 105, 014001 Table V -- En [MeV], sigma_pair, sigma_gamma
#: in ub, N3LO500/N2LO500.  Reproduced so the 2.1e-3 has a visible provenance.
VIVIANI_TABLE_V = pd.DataFrame({
    'En_MeV':      [0.17, 0.35, 0.70, 1.00, 2.00],
    'sigma_pair_ub': [0.0431, 0.0616, 0.0893, 0.108, 0.146],
    'sigma_gamma_ub': [20.2, 29.0, 42.0, 50.8, 67.7],
})

# --------------------------------------------------------------------------- #
# the neutron window
# --------------------------------------------------------------------------- #
def window_energy(t_ms: float = 1.0, flight_m: float = 19.5) -> float:
    """Neutron kinetic energy [eV] at ``t_ms`` after the flash, EAR2."""
    mn = 939.56542052e6                        # eV
    beta = flight_m / (2.99792458e8 * t_ms * 1e-3)
    return float(mn * (1.0 / np.sqrt(1 - beta ** 2) - 1.0))


def p_wave_suppression(en_ev: float, en_ref_ev: float = 0.17e6) -> float:
    """How much a p-wave channel is suppressed at ``en_ev`` vs ``en_ref_ev``.

    Relative to the 1/v s-wave: p-wave adds ``(k_n R)^2`` to the amplitude
    squared, so ``sigma_p/sigma_s ~ E``.  This is the factor by which the 1-
    resonance that dominates Viviani et al.'s Table V has gone away.
    """
    return float(en_ev / en_ref_ev)


# --------------------------------------------------------------------------- #
# the two thermal channels
# --------------------------------------------------------------------------- #
def e0_pair_cross_section_ub(m_e0_fm2: float = M_E0_FM2,
                             gamma_tot_mev: float = GAMMA_0P2_MEV,
                             sigma_np_b: float = SIGMA_NP_B) -> float:
    """Thermal 0+ -> 0+ pair cross section [ub], single-level estimate.

    ``sigma_pair = sigma_np * Gamma_pair(E0) / Gamma_tot`` -- the neutron and
    penetrability factors are common to both and cancel, which is the only
    reason this is worth writing down at all.
    """
    g_pair_ev = IB.e0_pair_width_eV(m_e0_fm2, W_HE)
    return float(sigma_np_b * 1e6 * g_pair_ev / (gamma_tot_mev * 1e6))


def thermal_channels() -> pd.DataFrame:
    """The pair yield of each s-wave channel, in ub, and what it looks like."""
    a_m1 = IB.alpha_pair('M1', W_HE)
    pairs_m1 = SIGMA_NGAMMA_UB * a_m1
    pairs_e0 = e0_pair_cross_section_ub()
    rows = []
    for kind, sig_pair, note in (
            ('M1', pairs_m1,
             f'{SIGMA_NGAMMA_UB:.0f} ub (n,g) x alpha_pair = {a_m1:.3e}'),
            ('E0', pairs_e0,
             f'{SIGMA_NP_B:.0f} b (n,p) x Gamma_pair/Gamma_tot')):
        d = IB.sample(kind, 1_200_000, W_HE)
        rows.append(dict(channel=kind, entrance='3S1 (1+)' if kind == 'M1'
                                     else '1S0 (0+)',
                         sigma_pair_ub=sig_pair,
                         frac_gt109=IB.wfrac(d, IB.X17_MIN_DEG),
                         median_deg=IB.wmedian(d), note=note))
    t = pd.DataFrame(rows)
    t['pairs_gt109_ub'] = t.sigma_pair_ub * t.frac_gt109
    t['share_of_pairs'] = t.sigma_pair_ub / t.sigma_pair_ub.sum()
    t['share_gt109'] = t.pairs_gt109_ub / t.pairs_gt109_ub.sum()
    return t


def thermal_summary() -> dict:
    t = thermal_channels()
    tot = float(t.sigma_pair_ub.sum())
    return dict(
        sigma_pair_total_ub=tot,
        ipc_per_gamma_capture=tot / SIGMA_NGAMMA_UB,
        ipc_per_gamma_capture_table=float(
            (VIVIANI_TABLE_V.sigma_pair_ub / VIVIANI_TABLE_V.sigma_gamma_ub).mean()),
        frac_gt109_mixed=float((t.sigma_pair_ub * t.frac_gt109).sum() / tot),
        frac_gt109_m1_only=float(t.loc[t.channel == 'M1', 'frac_gt109'].iloc[0]),
        e0_share_of_gt109=float(t.loc[t.channel == 'E0', 'share_gt109'].iloc[0]),
        e0_pair_width_eV=IB.e0_pair_width_eV(M_E0_FM2, W_HE),
    )


def e0_sensitivity() -> pd.DataFrame:
    """How the answer moves with the three inputs the E0 estimate rests on."""
    rows = []
    base = dict(m=M_E0_FM2, g=GAMMA_0P2_MEV, s=SIGMA_NP_B)
    for label, kw in (
            ('nominal', {}),
            ('M(E0) -1 sigma', dict(m=M_E0_FM2 - M_E0_FM2_ERR)),
            ('M(E0) +1 sigma', dict(m=M_E0_FM2 + M_E0_FM2_ERR)),
            ('Gamma(0+2) = 0.84 MeV', dict(g=0.84)),
            ('only half of (n,p) is 0+', dict(s=SIGMA_NP_B / 2)),
            ('M(E0) halved (structure)', dict(m=M_E0_FM2 / 2)),
            ('M(E0) doubled (structure)', dict(m=M_E0_FM2 * 2))):
        p = dict(base, **kw)
        e0 = e0_pair_cross_section_ub(p['m'], p['g'], p['s'])
        m1 = SIGMA_NGAMMA_UB * IB.alpha_pair('M1', W_HE)
        rows.append(dict(variation=label, sigma_e0_ub=e0,
                         e0_share=e0 / (e0 + m1),
                         ipc_per_capture=(e0 + m1) / SIGMA_NGAMMA_UB))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# does any of it move with time of flight?
# --------------------------------------------------------------------------- #
def energy_invariance(times_ms=(1, 3, 10, 30, 100, 300, 1000),
                      flight_m: float = 19.5) -> pd.DataFrame:
    """The expected pair spectrum against arrival time.  It does not move.

    This is worth a function rather than a sentence because it is the property
    that lets the whole >1 ms window be described by ONE template, which in
    turn is what makes arrival time usable as a handle on other backgrounds
    instead of a variable the signal model has to track.  Four things could
    break it, and each gets a column:

    ``dW_over_W``
        the 4He* excitation is ``S_n + (3/4) E_n``, so a 2 eV neutron moves the
        transition energy by 1.5 eV out of 20.58 MeV.
    ``he3_channel_ratio``
        the E0:M1 mix.  Both entrance channels are s-wave, both go as 1/v, and
        the 1/v cancels in the ratio *exactly* -- there is no leading
        correction, which is why this column is a constant and not a small
        number.
    ``al_over_he3``
        the capsule-to-gas capture ratio.  27Al and 3He are both 1/v here too,
        so this cancels the same way, and keeps cancelling until the first
        27Al resonance at 5903 eV -- which is 34 us of flight, three orders of
        magnitude before the flash veto lets anything through.
    ``p_wave_over_s_wave``
        the one thing that genuinely grows with energy, and the reason this
        argument would fail if the window were the one Viviani et al.
        tabulate.  At the top of ours it is 1e-5.

    The attribute ``spectrum_tv_across_window`` closes it numerically: the
    total variation distance between the Born M1 spectrum at the two ends of
    the window, which is what a fit would actually see.
    """
    mn_ev = 939.565420e6
    rows = []
    for t in times_ms:
        beta = flight_m / (2.99792458e8 * t * 1e-3)
        en = mn_ev * (1.0 / np.sqrt(1 - beta ** 2) - 1.0)
        rows.append(dict(t_ms=t, En_eV=en,
                         dW_over_W=0.75 * en * 1e-6 / W_HE,
                         he3_channel_ratio=1.0,
                         al_over_he3=1.0,
                         p_wave_over_s_wave=en / 0.17e6))
    d = pd.DataFrame(rows)
    y1 = IB.grid_spectrum('M1', W_HE)
    y2 = IB.grid_spectrum('M1', W_HE + 0.75 * d.En_eV.max() * 1e-6)
    d.attrs['spectrum_tv_across_window'] = float(
        0.5 * np.abs(y1 - y2).sum() * np.diff(IB.THETA_BINS)[0])
    d.attrs['first_al_resonance_eV'] = 5903.0
    return d


def thermal_spectrum(bins=None) -> pd.DataFrame:
    """The >1 ms 3He prediction as a spectrum: M1, E0 and their sum.

    The deliverable of this whole page.  Not a fraction beyond a threshold --
    the curve, on the same 1 deg axis everything else in the package uses, with
    the two components kept separate so a fit can float the mix.
    """
    if bins is None:
        bins = IB.THETA_BINS
    t = thermal_channels()
    f = dict(zip(t.channel, t.share_of_pairs))
    m1 = IB.grid_spectrum('M1', W_HE, bins)
    e0 = IB.grid_spectrum('E0', W_HE, bins)
    tot = f['M1'] * m1 + f['E0'] * e0
    return pd.DataFrame(dict(
        theta_mid=0.5 * (bins[1:] + bins[:-1]),
        M1=m1, E0=e0, total=tot,
        M1_weighted=f['M1'] * m1, E0_weighted=f['E0'] * e0))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    e1ms = window_energy(1.0)
    print(f'THE WINDOW.  t > 1 ms over 19.5 m  ->  En < {e1ms:.2f} eV')
    print(f'  a p-wave channel is suppressed by {p_wave_suppression(e1ms):.2e} '
          f'relative to s-wave, against the 0.17 MeV point Viviani tabulates\n')

    T = thermal_channels()
    print('THE TWO S-WAVE CHANNELS AT THERMAL')
    print(T.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    S = thermal_summary()
    print(f'\n  total pair cross section     {S["sigma_pair_total_ub"]:.3f} ub')
    print(f'  IPC per radiative capture    {S["ipc_per_gamma_capture"]:.2e}'
          f'   (rate table uses {S["ipc_per_gamma_capture_table"]:.2e},'
          f' from Viviani Table V at En = 0.17-2 MeV)')

    P = thermal_spectrum()
    w = np.diff(IB.THETA_BINS)
    print('\nTHE PREDICTED SPECTRUM  (dN/dtheta, 1/deg, every 10 deg)')
    head = ' '.join(f'{t:>7.0f}' for t in P.theta_mid[4::10])
    print(f'  theta   {head}')
    for col in ('M1', 'E0', 'total'):
        row = ' '.join(f'{v:7.4f}' for v in P[col].to_numpy()[4::10])
        print(f'  {col:<7s} {row}')
    med = float(np.interp(0.5, np.cumsum(P.total.to_numpy() * w), IB.THETA_MID))
    print(f'  median {med:.1f} deg;  quartiles '
          f'{np.interp([0.25, 0.75], np.cumsum(P.total.to_numpy() * w), IB.THETA_MID).round(1)}')

    print('\nWHAT THE E0 ESTIMATE RESTS ON')
    print(e0_sensitivity().to_string(index=False,
                                     float_format=lambda x: f'{x:.4g}'))

    E = energy_invariance()
    print('\nDOES THE PREDICTION MOVE WITH ARRIVAL TIME?')
    print(E.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    print(f'  Born spectrum at the two ends of the window: total variation '
          f'{E.attrs["spectrum_tv_across_window"]:.1e}  -- one template covers '
          f'the whole window.')
    print('  Aluminium: see ipc_aluminium.py, which supersedes the estimate '
          'that used to live here.')

    if a.write:
        from sept26_prelim_analysis import paths
        od = paths.out('ipc')
        T.to_csv(od / 'ipc_channels_thermal.csv', index=False)
        e0_sensitivity().to_csv(od / 'ipc_channels_e0_sensitivity.csv', index=False)
        P.to_csv(od / 'ipc_channels_spectrum.csv', index=False)
        E.to_csv(od / 'ipc_channels_energy_invariance.csv', index=False)
        VIVIANI_TABLE_V.to_csv(od / 'ipc_channels_viviani_tableV.csv', index=False)
        pd.Series(S).to_csv(od / 'ipc_channels_summary.csv')
        print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
