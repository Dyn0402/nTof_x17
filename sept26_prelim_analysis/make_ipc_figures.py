#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ipc_figures.py -- the four figures for the IPC deep-dive page.

  ipc_shapes     the opening-angle law per multipole, against the band that
                 pair_physics.py currently carries.  Log y, because the thing
                 the X17 search needs is a tail.
  ipc_mass       where the difference comes from: the virtual photon's mass
                 spectrum.  The 1/M ansatz is the M1 answer with the
                 phase-space suppression left out.
  ipc_thermal    the >1 ms prediction -- M1 + E0 in the proportion the reaction
                 makes them -- against the four-variant band it replaces.
  ipc_energy     wide-angle pairs per photon against transition energy.  Flat.
                 Which is why aluminium is a problem.

    python -m sept26_prelim_analysis.make_ipc_figures
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
from sept26_prelim_analysis import ipc_born as IB  # noqa: E402
from sept26_prelim_analysis import ipc_channels as IC  # noqa: E402

BINS = np.arange(0.0, 181.0, 3.0)
MID = 0.5 * (BINS[1:] + BINS[:-1])
KIND_COLOR = {'E0': fs.ACCENT, 'M1': fs.DET_COLOR['A'], 'E1': fs.DET_COLOR['C'],
              'E2': fs.MUTED, 'M2': fs.LINE}
KIND_LABEL = {'E0': 'E0  0⁺→0⁺ monopole', 'M1': 'M1  magnetic dipole',
              'E1': 'E1  electric dipole', 'E2': 'E2', 'M2': 'M2'}


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


def _band_from_ansatz(n=600_000):
    from sept26_prelim_analysis import pair_physics as PP
    rows = {}
    for name, kw in PP.VARIANTS.items():
        a = PP.ipc_angles(n, **kw)
        h, _ = np.histogram(a, bins=BINS, density=True)
        rows[name] = h
    return pd.DataFrame(rows, index=pd.Index(MID, name='theta_deg'))


# --------------------------------------------------------------------------- #
def fig_shapes(out, n):
    plt = _plt()
    band = _band_from_ansatz()
    cols = {}
    with _scaled(plt):
        fig, ax = fs.slide(fs.WIDE)
        lo = band.min(axis=1).to_numpy()
        hi = band.max(axis=1).to_numpy()
        ax.fill_between(MID, np.maximum(lo, 1e-8), hi, color=fs.LINE,
                        alpha=0.9, lw=0, zorder=1,
                        label='what pair_physics.py carries now\n(four ansätze, a factor 38 apart)')
        for kind in ('E0', 'E1', 'M1'):
            d = IB.sample(kind, n, IB.E_TRANSITION)
            y = IB.shape(d, BINS)
            cols[kind] = y
            ax.plot(MID, np.maximum(y, 1e-8), color=KIND_COLOR[kind], zorder=3,
                    label=f'{KIND_LABEL[kind]}   ({100 * IB.wfrac(d, 109):.1f} % above 109°)')
        ax.axvspan(109, 145, color=fs.BAND_SIGNAL, alpha=0.10, lw=0, zorder=0)
        ax.text(127, 4e-4, 'X17 region', ha='center', color=fs.BAND_SIGNAL,
                fontsize=fs.BASE_PT * 0.66)
        ax.set_yscale('log')
        ax.set_xlim(0, 180)
        ax.set_ylim(1e-5, 2e-1)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_xlabel('opening angle at birth  [deg]')
        ax.set_ylabel('normalised  dN/dθ  [1/deg]')
        ax.set_title('The internal-pair continuum is a different curve for every multipole')
        ax.legend(loc='upper right', ncol=1)
        fs.save(fig, out / 'ipc_shapes',
                data=pd.DataFrame(dict(theta_deg=MID, ansatz_lo=lo,
                                       ansatz_hi=hi, **cols)))


def fig_mass(out, n):
    plt = _plt()
    lnb = np.linspace(np.log(2 * IB.M_E), np.log(IB.E_TRANSITION), 60)
    mid = np.exp(0.5 * (lnb[1:] + lnb[:-1]))
    cols = {}
    with _scaled(plt):
        fig, ax = fs.slide(fs.WIDE)
        for kind in ('E0', 'E1', 'M1'):
            d = IB.sample(kind, n, IB.E_TRANSITION)
            h, _ = np.histogram(np.log(d.m_ee), bins=lnb, weights=d.weight)
            h = h / h.sum()
            cols[kind] = h
            ax.plot(mid, h, color=KIND_COLOR[kind], label=KIND_LABEL[kind])
        # dN/dM ~ M^p  =>  dN/dlnM ~ M^(p+1)
        for lbl, p, st in (('Geant generator:  dN/dM ~ 1/M', -1.0, '--'),
                           ('the “extreme” bracket:  dN/dM ~ 1/M³', -3.0, ':')):
            y = mid ** (p + 1)
            y = y / y.sum()
            cols[lbl] = y
            ax.plot(mid, y, color=fs.INK, ls=st, lw=1.8, label=lbl)
        ax.axvline(16.8, color=fs.COPPER, lw=1.6, ls='-.')
        ax.text(16.2, 0.072, 'M = 16.8 MeV\nis 109°', ha='right', va='top',
                color=fs.COPPER, fontsize=fs.BASE_PT * 0.62)
        ax.set_xscale('log')
        ax.set_xlim(1.02, 20.58)
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(['1', '2', '5', '10', '20'])
        ax.set_ylim(0, 0.075)
        ax.set_xlabel('virtual-photon invariant mass  $M_{ee}$  [MeV]')
        ax.set_ylabel('fraction of pairs per bin in ln M')
        ax.set_title('Where the disagreement lives: the mass the virtual photon is given')
        ax.legend(loc='upper left')
        fs.save(fig, out / 'ipc_mass',
                data=pd.DataFrame(dict(m_ee_MeV=mid, **cols)))


def fig_thermal(out, n):
    plt = _plt()
    T = IC.thermal_channels()
    frac = dict(zip(T.channel, T.share_of_pairs))
    band = _band_from_ansatz()
    parts, tot = {}, np.zeros(len(MID))
    for kind in ('M1', 'E0'):
        y = IB.shape(IB.sample(kind, n, IB.E_TRANSITION), BINS) * frac[kind]
        parts[kind] = y
        tot = tot + y
    with _scaled(plt):
        fig, ax = fs.slide(fs.WIDE)
        ax.fill_between(MID, np.maximum(band.min(axis=1), 1e-8),
                        band.max(axis=1), color=fs.LINE, alpha=0.9, lw=0,
                        label='the band this page replaces')
        ax.plot(MID, np.maximum(parts['M1'], 1e-8), color=fs.DET_COLOR['A'],
                ls='--', lw=2.0,
                label=f'M1 from the 1⁺ channel  ({100 * frac["M1"]:.0f} % of pairs)')
        ax.plot(MID, np.maximum(parts['E0'], 1e-8), color=fs.ACCENT,
                ls='--', lw=2.0,
                label=f'E0 from the 0⁺ channel  ({100 * frac["E0"]:.0f} % of pairs)')
        ax.plot(MID, np.maximum(tot, 1e-8), color=fs.INK, lw=3.0,
                label='sum — the ³He prediction below 2 eV')
        ax.axvspan(109, 145, color=fs.BAND_SIGNAL, alpha=0.10, lw=0, zorder=0)
        ax.set_yscale('log')
        ax.set_xlim(0, 180)
        ax.set_ylim(1e-5, 2e-1)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_xlabel('opening angle at birth  [deg]')
        ax.set_ylabel('normalised  dN/dθ  [1/deg]')
        ax.set_title('Below 2 eV only two channels are open, and one of them makes no photons')
        ax.legend(loc='upper right')
        fs.save(fig, out / 'ipc_thermal',
                data=pd.DataFrame(dict(theta_deg=MID, M1=parts['M1'],
                                       E0=parts['E0'], total=tot)))


def fig_energy(out, n):
    plt = _plt()
    S = IB.energy_scan(kinds=('E1', 'M1'),
                       energies=[20.58, 16.0, 12.0, 9.0, IB.AL_SN, 6.0, 4.734,
                                 3.034, 2.590, 1.779], n=n)
    with _scaled(plt):
        fig, ax = fs.slide(fs.WIDE)
        for kind, c in (('E1', fs.DET_COLOR['C']), ('M1', fs.DET_COLOR['A'])):
            s = S[S.multipole == kind].sort_values('w_MeV')
            ax.plot(s.w_MeV, 1e4 * s.pairs_gt109_per_photon, 'o-', color=c,
                    label=f'{kind}  pairs above 109° per photon')
        ax.axvline(IB.AL_SN, color=fs.COPPER, lw=1.8)
        ax.text(IB.AL_SN + 0.3, 2.4, '²⁷Al(n,γ)\n7.73 MeV', color=fs.COPPER,
                fontsize=fs.BASE_PT * 0.62, va='bottom')
        ax.axvline(20.58, color=fs.ACCENT, lw=1.8)
        ax.text(20.3, 2.4, '³He(n,γ)\n20.58 MeV', color=fs.ACCENT, ha='right',
                fontsize=fs.BASE_PT * 0.62, va='bottom')
        ax.set_ylim(0, 5.5)
        ax.set_xlim(0, 22)
        ax.set_xlabel('transition energy  [MeV]')
        ax.set_ylabel('wide-angle pairs per photon  [×10⁻⁴]')
        ax.set_title('A 7.7 MeV capture γ makes almost as many wide-angle pairs as a 20.6 MeV one')
        ax.legend(loc='lower right')
        fs.save(fig, out / 'ipc_energy', data=S)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=1_500_000)
    a = ap.parse_args()
    out = paths.out('ipc') / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig_shapes(out, a.n)
    fig_mass(out, a.n)
    fig_thermal(out, a.n)
    fig_energy(out, max(a.n // 2, 400_000))
    print(f'\nfigures -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
