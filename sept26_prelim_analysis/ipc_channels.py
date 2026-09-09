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

#: Capsule vs gas, thermal bin, from the December-2025 rate calculation
#: (``results_3He``, the 0.01-0.1 eV row): captures per pulse.
GC_CAPTURES_PER_PULSE = 5.38e4      # Al capsule + CF, "GC-captures"
HE3_CAPTURES_PER_PULSE = 4.37       # 3He radiative captures, "He3-captures"

#: Fraction of 27Al thermal captures emitting a primary above ~7.5 MeV.
#: The 7724.0 keV (to the ground state) and 7693.4 keV lines.  The IAEA PGAA
#: database returns two normalisations for these and this module does not
#: pretend to have resolved which is which, so the answer is carried as a
#: bracket and the conclusion is quoted as a range.
AL_HARD_PRIMARY_FRAC = (0.024, 0.25)


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
# the Al question
# --------------------------------------------------------------------------- #
def aluminium_comparison(n: int = 1_000_000) -> pd.DataFrame:
    """Wide-angle pairs from the capsule against wide-angle pairs from the gas.

    The two things that make this a real question rather than a footnote:

      * ``pairs above 109 deg per photon`` is nearly FLAT in transition energy.
        A 7.7 MeV E1 primary makes 4.0e-4 of them, a 20.6 MeV one makes 4.3e-4.
        Losing two thirds of the energy costs 8 %.
      * there are ~1.2e4 capsule captures for every 3He radiative capture.

    and the one thing that would have removed it is missing: this setup has no
    magnet and no calorimetry, so a pair's TOTAL ENERGY -- the 20.6 vs 7.7 MeV
    that separates the two outright -- is not measured.  The n_TOF proposal's
    detector has a 50 mT coil for exactly this.
    """
    rows = []
    for label, w, kind in (('3He (n,g), M1', W_HE, 'M1'),
                           ('3He (n,g), E0', W_HE, 'E0'),
                           ('27Al (n,g) primary, E1', W_AL, 'E1'),
                           ('27Al (n,g) primary, M1', W_AL, 'M1')):
        d = IB.sample(kind, n, w)
        f = IB.wfrac(d, IB.X17_MIN_DEG)
        a = np.nan if kind == 'E0' else IB.alpha_pair(kind, w)
        rows.append(dict(source=label, w_MeV=w, multipole=kind,
                         alpha_pair=a, frac_gt109=f,
                         pairs_gt109_per_photon=a * f,
                         median_deg=IB.wmedian(d)))
    return pd.DataFrame(rows)


def aluminium_ratio() -> pd.DataFrame:
    """The bracket: wide-angle Al pairs per wide-angle 3He pair, per pulse."""
    a_al = IB.alpha_pair('E1', W_AL)
    f_al = IB.wfrac(IB.sample('E1', 1_000_000, W_AL), IB.X17_MIN_DEG)
    t = thermal_channels()
    he_gt109 = float((t.sigma_pair_ub * t.frac_gt109).sum()
                     / t.sigma_pair_ub.sum())
    he_pairs = HE3_CAPTURES_PER_PULSE * float(t.sigma_pair_ub.sum()) \
        / SIGMA_NGAMMA_UB * he_gt109
    rows = []
    for f_hard in AL_HARD_PRIMARY_FRAC:
        al_pairs = GC_CAPTURES_PER_PULSE * f_hard * a_al * f_al
        rows.append(dict(al_hard_primary_frac=f_hard,
                         al_gt109_per_pulse=al_pairs,
                         he3_gt109_per_pulse=he_pairs,
                         ratio=al_pairs / he_pairs))
    return pd.DataFrame(rows)


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
    print(f'  fraction above 109 deg       {S["frac_gt109_mixed"]:.3f}'
          f'   (M1 alone would be {S["frac_gt109_m1_only"]:.3f};'
          f' Geant ansatz gives 0.118)')
    print(f'  E0 share of the >109 deg yield {S["e0_share_of_gt109"]:.2f}')

    print('\nWHAT THE E0 ESTIMATE RESTS ON')
    print(e0_sensitivity().to_string(index=False,
                                     float_format=lambda x: f'{x:.4g}'))

    print('\nALUMINIUM -- first look')
    print(aluminium_comparison().to_string(index=False,
                                           float_format=lambda x: f'{x:.4g}'))
    R = aluminium_ratio()
    print()
    print(R.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    print(f'  => the capsule makes {R.ratio.min():.0f}-{R.ratio.max():.0f} '
          f'times as many >109 deg pairs as the gas does, and this setup '
          f'cannot tell them apart by energy.')

    if a.write:
        from sept26_prelim_analysis import paths
        od = paths.out('ipc')
        T.to_csv(od / 'ipc_channels_thermal.csv', index=False)
        e0_sensitivity().to_csv(od / 'ipc_channels_e0_sensitivity.csv', index=False)
        aluminium_comparison().to_csv(od / 'ipc_channels_al.csv', index=False)
        R.to_csv(od / 'ipc_channels_al_ratio.csv', index=False)
        VIVIANI_TABLE_V.to_csv(od / 'ipc_channels_viviani_tableV.csv', index=False)
        pd.Series(S).to_csv(od / 'ipc_channels_summary.csv')
        print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
