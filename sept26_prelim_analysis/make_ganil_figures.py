#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ganil_figures.py -- the figures for the NFS/GANIL background page.

  ganil_kinematics  where the X17 signature goes as the neutron energy rises,
                    and where the n_TOF point sits on that curve.
  ganil_spectra     the three opening-angle spectra at three neutron energies.
                    The capsule curve is the same curve three times, which is
                    the whole argument.
  ganil_rates       per neutron entering the cell: what the gas does, what the
                    wall does, and the two-prong (n,p) load, against energy.
  ganil_lines       which capsule photons exist at which neutron energy, and
                    the two thresholds that switch them on.
  ganil_window      the decision plot: capsule pairs per gas pair inside the
                    moving signal window, against the n_TOF value.

    python -m sept26_prelim_analysis.make_ganil_figures
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

from sept26_prelim_analysis import paths          # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import endf as EN      # noqa: E402
from sept26_prelim_analysis import ipc_born as IB  # noqa: E402
from sept26_prelim_analysis import ganil_background as G  # noqa: E402

BINS = IB.THETA_BINS
MID = IB.THETA_MID
W = np.diff(BINS)

#: The three neutron energies the spectra figure uses: the clean end, the
#: middle, and the top of what an evaluation covers.
SHOW = (1.5, 5.0, 20.0)

#: The X17-to-internal-pair ratio the December-2025 rate table assumes.  Used
#: ONLY to put the signal on the same axis as the background; this page makes
#: no claim about it, and at E_x = 21-51 MeV there is no measurement behind it.
X17_PER_IPC = 2.5e-2


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs.use()
    return plt


def _scaled(plt, f=0.72):
    return plt.rc_context({'font.size': fs.BASE_PT * f,
                           'axes.labelsize': fs.BASE_PT * f,
                           'axes.titlesize': fs.BASE_PT * f * 1.05,
                           'legend.fontsize': fs.BASE_PT * f * 0.9,
                           'xtick.labelsize': fs.BASE_PT * f * 0.86,
                           'ytick.labelsize': fs.BASE_PT * f * 0.86})


def _median(y):
    return float(np.interp(0.5, np.cumsum(y * W), MID))


# --------------------------------------------------------------------------- #
def fig_kinematics(out):
    plt = _plt()
    en = np.linspace(0.0, 40.0, 400)
    tmin = G.x17_min_angle(en)
    hi = np.minimum(tmin + G.WINDOW_WIDTH_DEG, 180)
    med = np.array([_median(IB.grid_spectrum('E1', float(G.excitation(e)),
                                             BINS)) for e in
                    np.linspace(0.0, 40.0, 41)])
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        ax.fill_between(en, tmin, hi, color=fs.BAND_SIGNAL, alpha=0.16, lw=0,
                        label='the signal window, 36° wide')
        ax.plot(en, tmin, color=fs.BAND_SIGNAL, lw=3.0,
                label='X17 minimum opening angle')
        ax.plot(np.linspace(0, 40, 41), med, color=fs.DET_COLOR['C'], lw=2.2,
                ls='--', label='median of the ³He internal-pair continuum')
        cap = _median(G.capsule_pairs(6.0)[0])
        ax.axhline(cap, color=fs.DET_COLOR['B'], lw=2.2, ls=':',
                   label='median of the capsule continuum — it does not move')
        ax.plot([0], [G.x17_min_angle(0.0)], 'o', ms=11, color=fs.INK, zorder=6)
        ax.annotate('n_TOF sits here:\n109°, and nothing moves',
                    (0, 109.5), xytext=(16, -4), textcoords='offset points',
                    color=fs.INK, fontsize=fs.BASE_PT * 0.77, va='top')
        qlo, qhi = G.quiet_band()
        ax.axvspan(qlo, qhi, color=fs.DET_COLOR['A'], alpha=0.10, lw=0)
        ax.annotate(f'{qlo:g}–{qhi:.1f} MeV: the capsule\ncannot make pairs yet',
                    (0.5 * (qlo + qhi), 26), ha='center',
                    color=fs.DET_COLOR['A'], fontsize=fs.BASE_PT * 0.77)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 180)
        ax.set_yticks(np.arange(0, 181, 30))
        ax.set_xlabel('neutron energy  [MeV]')
        ax.set_ylabel('opening angle  [deg]')
        ax.set_title('At NFS the signal angle is a function of the neutron '
                     'energy, and the neutron energy is measured')
        ax.legend(loc='upper right')
        fs.save(fig, out / 'ganil_kinematics',
                data=pd.DataFrame(dict(En_MeV=en, Ex_MeV=G.excitation(en),
                                       theta_min_deg=tmin, window_hi_deg=hi)))


def fig_spectra(out):
    """Absolute, not normalised.  The shapes barely move; the rates move by 10^4."""
    plt = _plt()
    cols = {'theta_mid': MID}
    with _scaled(plt, 0.66):
        fig, axes = plt.subplots(1, 3, figsize=fs.WIDE, sharey=True)
        for ax, en in zip(axes, SHOW):
            lo, hi = G.window(en)
            ex = float(G.excitation(en))
            he = G.he3_rates(en).iloc[0]
            r_he = float(he.radiative_per_neutron) * IB.alpha_pair('E1', ex)
            y_he = r_he * G.he3_pair_spectrum(en, 'E1', BINS)
            y_cap_shape, r_cap = G.capsule_pairs(en, BINS)
            y_cap = r_cap * y_cap_shape
            y_x17 = r_he * X17_PER_IPC * G.x17_spectrum(en, BINS)
            cols[f'x17_{en:g}MeV'] = y_x17
            cols[f'he3_{en:g}MeV'] = y_he
            cols[f'capsule_{en:g}MeV'] = y_cap
            ax.axvspan(lo, hi, color=fs.BAND_SIGNAL, alpha=0.13, lw=0, zorder=0)
            ax.plot(MID, np.maximum(y_he, 1e-16), color=fs.ACCENT, lw=2.4,
                    label='³He internal pairs')
            if y_cap.sum() > 0:
                ax.plot(MID, np.maximum(y_cap, 1e-16), color=fs.DET_COLOR['B'],
                        lw=2.4, label='capsule wall')
            ax.plot(MID, np.maximum(y_x17, 1e-16), color=fs.INK, lw=2.0,
                    ls='--', label=f'X17 at X17/IPC = {X17_PER_IPC:g}')
            ax.set_yscale('log')
            ax.set_xlim(0, 180)
            ax.set_ylim(1e-14, 1e-6)
            ax.set_xticks(np.arange(0, 181, 45))
            ax.set_xlabel('opening angle  [deg]')
            ax.set_title(f'Eₙ = {en:g} MeV   →   Eₓ = {ex:.1f} MeV')
        axes[0].set_ylabel('pairs per neutron per degree')
        axes[0].legend(loc='lower left', fontsize=fs.BASE_PT * 0.64)
        fig.suptitle('Absolute rates. The shapes hardly change; what changes '
                     'is that the wall rises past the gas by four decades.',
                     fontsize=fs.BASE_PT * 0.86, y=0.99)
        fig.tight_layout()
        fs.save(fig, out / 'ganil_spectra', data=pd.DataFrame(cols))


def fig_rates(out):
    """Per neutron entering the cell, against energy.  The whole rate story."""
    plt = _plt()
    en = np.geomspace(0.5, G.DISCRETE_EVAL_MAX_MEV, 70)
    H = G.he3_rates(en)
    cap = np.array([G.capsule_pairs(e)[1] for e in en])
    he_pairs = H.radiative_per_neutron.to_numpy() * np.array(
        [IB.alpha_pair('E1', float(x)) for x in H.Ex_MeV])
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        ax.loglog(en, H.radiative_per_neutron, color=fs.ACCENT, lw=3.0,
                  label='³He radiative captures, per neutron entering the cell')
        ax.loglog(en, he_pairs, color=fs.ACCENT, lw=1.8, ls='--',
                  label='and the internal pairs they make')
        ax.loglog(en, cap, color=fs.DET_COLOR['B'], lw=3.0,
                  label='pairs from the capsule wall')
        ax.axhline(1.03e-8, color=fs.INK, lw=1.5, ls=':',
                   label='n_TOF, thermal: 1.0×10⁻⁸ radiative captures '
                         'per neutron')
        for x, lab in ((2.29, '²⁷Al 2.21 MeV level opens'),
                       (4.81, '¹²C 4.44 MeV level opens')):
            ax.axvline(x, color=fs.MUTED, lw=1.2, ls='--')
            ax.text(x * 1.05, 2.4e-4, lab, rotation=90, va='top', ha='left',
                    color=fs.MUTED, fontsize=fs.BASE_PT * 0.67)
        ax.axvspan(*G.quiet_band(), color=fs.DET_COLOR['A'], alpha=0.10, lw=0)
        ax.set_xlim(0.5, 20)
        ax.set_ylim(1e-9, 4e-4)
        ax.set_xticks([0.5, 1, 2, 3, 5, 10, 20])
        ax.set_xticklabels(['0.5', '1', '2', '3', '5', '10', '20'])
        ax.set_xlabel('neutron energy  [MeV]')
        ax.set_ylabel('per neutron entering the cell')
        ax.set_title('The gas gets 200× better than thermal, and the wall '
                     'stays quiet until 2.3 MeV')
        ax.legend(loc='upper left', framealpha=0.92)
        fs.save(fig, out / 'ganil_rates',
                data=pd.DataFrame(dict(En_MeV=en,
                                       he3_radiative=H.radiative_per_neutron,
                                       he3_pairs=he_pairs,
                                       capsule_pairs=cap,
                                       np_per_radiative=H.np_per_radiative)))


def fig_lines(out):
    plt = _plt()
    en = np.geomspace(0.8, G.DISCRETE_EVAL_MAX_MEV, 120)
    keep = [('C12', 51, '¹²C 4.44 MeV  (E2)', fs.DET_COLOR['C']),
            ('Al27', 53, '²⁷Al 2.21 MeV', fs.DET_COLOR['B']),
            ('Al27', 56, '²⁷Al 3.00 MeV', fs.COPPER),
            ('Al27', 54, '²⁷Al 2.73 MeV', fs.DET_COLOR['D']),
            ('Al27', 51, '²⁷Al 0.84 MeV — below the pair threshold', fs.MUTED),
            ('Al27', 52, '²⁷Al 1.01 MeV — below the pair threshold', fs.LINE)]
    n_atb = {'Al27': G.N_AL, 'C12': G.N_C}
    cols = {'En_MeV': en}
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        for nuc, mt, lab, col in keep:
            y = n_atb[nuc] * EN.sigma_at(nuc, mt, en * 1e6)
            cols[lab] = y
            ax.semilogx(en, 1e3 * y, color=col, lw=2.6,
                        ls=':' if 'below' in lab else '-', label=lab)
        ax.axvline(2 * IB.M_E * 28 / 27, color=fs.INK, lw=1.4, ls='--')
        ax.set_xlim(0.8, 20)
        ax.set_xticks([1, 2, 3, 5, 10, 20])
        ax.set_xticklabels(['1', '2', '3', '5', '10', '20'])
        ax.set_xlabel('neutron energy  [MeV]')
        ax.set_ylabel('photons per 1000 neutrons entering the cell')
        ax.set_title('What the capsule radiates, and the two strongest lines '
                     'cannot make a pair at all')
        ax.legend(loc='upper left')
        fs.save(fig, out / 'ganil_lines', data=pd.DataFrame(cols))


def fig_window(out, S):
    plt = _plt()
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        ax.axhspan(1e4, 1.3e6, color=fs.BAND_DEAD, alpha=0.12, lw=0)
        ax.text(1.05, 1.3e5, 'where n_TOF sits: 10⁴–10⁶ capsule pairs per gas '
                             'pair', color=fs.BAND_DEAD,
                fontsize=fs.BASE_PT * 0.77)
        # only where both terms exist: above 20 MeV there is no evaluated
        # 3He(n,gamma) to divide by, and a ratio to an extrapolation is not a
        # measurement of anything
        ok = np.isfinite(S.capsule_over_he3) & (S.capsule_over_he3 > 0) \
            & (~S.he3_beyond_evaluation)
        ax.loglog(S.En_MeV[ok], S.capsule_over_he3[ok], 'o-',
                  color=fs.DET_COLOR['B'], lw=3.0, ms=8)
        qlo, qhi = G.quiet_band()
        ax.axvspan(qlo, qhi, color=fs.DET_COLOR['A'], alpha=0.12, lw=0)
        ax.annotate(f'below {qhi:.1f} MeV the capsule background is\nsmaller '
                    f'than the gas signal it sits on',
                    (1.05, 12), ha='left', color=fs.DET_COLOR['A'],
                    fontsize=fs.BASE_PT * 0.79)
        ax.set_xlim(0.9, 25)
        ax.set_ylim(1, 3e6)
        ax.set_xticks([1, 2, 3, 5, 10, 20])
        ax.set_xticklabels(['1', '2', '3', '5', '10', '20'])
        ax.set_xlabel('neutron energy  [MeV]')
        ax.set_ylabel('capsule pairs per ³He pair, inside the signal window')
        ax.set_title('The decision plot: run below 2.5 MeV and the capsule '
                     'stops being the problem')
        fs.save(fig, out / 'ganil_window', data=S)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--multipole', default='E1', choices=('E1', 'M1'))
    a = ap.parse_args()
    out = paths.out('ganil') / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    S = G.energy_scan(he_multipole=a.multipole)
    fig_kinematics(out)
    fig_spectra(out)
    fig_rates(out)
    fig_lines(out)
    fig_window(out, S)
    print(f'\nfigures -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
