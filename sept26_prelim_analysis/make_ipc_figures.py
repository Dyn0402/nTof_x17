#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ipc_figures.py -- the figures for the IPC deep-dive page.

Every one of these is a SPECTRUM.  The page used to lead with tables of "what
fraction is beyond 109 / 130 degrees", which is three numbers where there is a
curve, and it invited an argument about the threshold instead of about the
physics.  The thresholds are still marked on the axes; nothing is quoted from
anywhere but the curve.

  ipc_kinematics the master relation -- what virtual-photon mass is needed for
                 what lab opening angle.  Everything else is a weighting of it.
  ipc_density    where each multipole puts its probability in the (mass, angle)
                 plane, which is the mechanism behind the curves below.
  ipc_shapes     dN/dtheta per multipole against the four-ansatz band it
                 replaces.  Log y, because the X17 search lives on a tail.
  ipc_mass       where the difference comes from: the virtual photon's mass.
  ipc_thermal    the >1 ms prediction -- M1 + E0 in the proportion the reaction
                 makes them -- with its E0-fraction band.
  ipc_time       the same prediction against arrival time.  It does not move,
                 and this is the figure that says so.
  ipc_al_lines   which 27Al capture lines make the wide-angle pairs.  Not the
                 7.7 MeV primaries.
  ipc_al_shape   the aluminium continuum against the helium one, at birth and
                 after the capsule wall the aluminium one is born inside.

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
from sept26_prelim_analysis import ipc_aluminium as AL  # noqa: E402

BINS = IB.THETA_BINS
MID = IB.THETA_MID
W = np.diff(BINS)
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


def _median(y):
    return float(np.interp(0.5, np.cumsum(y * W), MID))


def _band_from_ansatz(n=600_000):
    from sept26_prelim_analysis import pair_physics as PP
    rows = {}
    for name, kw in PP.VARIANTS.items():
        a = PP.ipc_angles(n, **kw)
        h, _ = np.histogram(a, bins=BINS, density=True)
        rows[name] = h
    return pd.DataFrame(rows, index=pd.Index(MID, name='theta_deg'))


def _x17_band(ax, y=None):
    ax.axvspan(109, 145, color=fs.BAND_SIGNAL, alpha=0.10, lw=0, zorder=0)
    if y is not None:
        ax.text(127, y, 'X17 region', ha='center', color=fs.BAND_SIGNAL,
                fontsize=fs.BASE_PT * 0.77)


# --------------------------------------------------------------------------- #
def fig_kinematics(out):
    """The master relation: what mass the virtual photon needs for what angle.

    Everything else on the page is a weighting of this one curve.  A pair of
    invariant mass M from a transition of energy W is emitted back-to-back in
    its own frame and boosted by gamma = W/M, so its lab opening angle has a
    hard minimum,

        cos(theta_min) = 1 - 2 M^2 / W^2   (for a symmetric pair)

    and cannot be smaller whatever the decay angle.  Reading it backwards is
    the whole X17 argument: 109 deg needs M >= 16.8 MeV.
    """
    plt = _plt()
    m = np.linspace(2 * IB.M_E, IB.E_TRANSITION, 600)
    w = IB.E_TRANSITION
    cmin = np.clip(1 - 2 * m ** 2 / w ** 2, -1, 1)
    tmin = np.degrees(np.arccos(cmin))
    # the median angle each mass actually produces, over the decay angle
    med = []
    for mm in m:
        c = np.linspace(-1 + 1e-9, 1 - 1e-9, 4000)
        b = np.sqrt(max(1 - 4 * IB.M_E ** 2 / mm ** 2, 0.0))
        k = np.sqrt(max(w ** 2 - mm ** 2, 0.0))
        g, bet, es = w / mm, k / w, mm / 2
        ps = es * b
        pz1, pz2 = g * (ps * c + bet * es), g * (-ps * c + bet * es)
        px = ps * np.sqrt(np.clip(1 - c ** 2, 0, None))
        dot = -px * px + pz1 * pz2
        th = np.degrees(np.arccos(np.clip(
            dot / np.hypot(px, pz1) / np.hypot(px, pz2), -1, 1)))
        med.append(np.median(th))
    med = np.array(med)
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        ax.fill_between(m, tmin, 180, color=fs.LINE, alpha=0.55, lw=0,
                        label='angles this mass can reach')
        ax.plot(m, tmin, color=fs.INK, lw=3.0,
                label='minimum possible opening angle')
        ax.plot(m, med, color=fs.DET_COLOR['C'], lw=2.0, ls='--',
                label='median, over an isotropic decay')
        ax.axhline(109, color=fs.BAND_SIGNAL, lw=1.6)
        ax.axvline(16.8, color=fs.COPPER, lw=1.8)
        ax.plot([16.8], [109], 'o', ms=10, color=fs.COPPER, zorder=5)
        ax.annotate('a 16.8 MeV X17 gives\nexactly 109\u00b0 at its minimum',
                    (16.8, 109), xytext=(-14, 34), textcoords='offset points',
                    ha='right', color=fs.COPPER, fontsize=fs.BASE_PT * 0.77)
        ax.axhspan(109, 145, color=fs.BAND_SIGNAL, alpha=0.08, lw=0, zorder=0)
        ax.text(3.0, 126, 'X17 region', color=fs.BAND_SIGNAL,
                fontsize=fs.BASE_PT * 0.77)
        ax.set_xlim(0.9, 20.6)
        ax.set_ylim(0, 180)
        ax.set_yticks(np.arange(0, 181, 30))
        ax.set_xlabel('virtual-photon invariant mass  $M_{ee}$  [MeV]')
        ax.set_ylabel('opening angle in the lab  [deg]')
        ax.set_title('The whole page in one curve: only a heavy virtual photon '
                     'can make a wide pair')
        ax.legend(loc='upper left')
        fs.save(fig, out / 'ipc_kinematics',
                data=pd.DataFrame(dict(m_ee_MeV=m, theta_min_deg=tmin,
                                       theta_median_deg=med)))


def fig_density(out, n):
    """Where each multipole actually puts its probability, in (M, theta).

    The shapes page shows the projection onto theta; this shows the plane, and
    it is where the difference between the multipoles is visible as a
    mechanism rather than as a curve that happens to be lower.
    """
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    # white at zero so the empty part of the plane is the page, not a colour;
    # one shared log scale across the three panels so they are comparable
    # rather than three separately stretched pictures.
    cmap = LinearSegmentedColormap.from_list(
        'ipc', ['#ffffff', '#e9dcea', '#c79ac9', fs.ACCENT, '#3d1b40'])
    mbins = np.linspace(np.log(2 * IB.M_E), np.log(IB.E_TRANSITION), 90)
    tbins = np.arange(0.0, 181.0, 3.0)
    hs = {}
    for kind in ('M1', 'E1', 'E0'):
        d = IB.sample(kind, n, IB.E_TRANSITION)
        h, _, _ = np.histogram2d(np.log(d.m_ee), d.theta_deg,
                                 bins=[mbins, tbins], weights=d.weight)
        hs[kind] = h / h.sum()
    top = max(np.percentile(h[h > 0], 99.5) for h in hs.values())
    norm = LogNorm(vmin=top * 1e-4, vmax=top)
    with _scaled(plt, 0.66):
        fig, axes = plt.subplots(1, 3, figsize=fs.WIDE, sharey=True)
        cols = {}
        for ax, kind in zip(axes, ('M1', 'E1', 'E0')):
            h = hs[kind]
            cols[kind] = h
            pc = ax.pcolormesh(np.exp(mbins), tbins, np.ma.masked_where(
                h.T <= 0, h.T), cmap=cmap, shading='auto', norm=norm)
            mm = np.exp(0.5 * (mbins[1:] + mbins[:-1]))
            ax.plot(mm, np.degrees(np.arccos(np.clip(
                1 - 2 * mm ** 2 / IB.E_TRANSITION ** 2, -1, 1))),
                color=fs.INK, lw=1.6, ls='--')
            ax.axhspan(109, 145, color=fs.BAND_SIGNAL, alpha=0.16, lw=0)
            ax.set_xscale('log')
            ax.set_xlim(1.02, 20.58)
            ax.set_xticks([1, 2, 5, 10, 20])
            ax.set_xticklabels(['1', '2', '5', '10', '20'])
            ax.set_ylim(0, 180)
            ax.set_yticks(np.arange(0, 181, 30))
            ax.set_title(KIND_LABEL[kind])
            ax.set_xlabel('$M_{ee}$  [MeV]')
        axes[0].set_ylabel('opening angle  [deg]')
        fig.suptitle('Where each multipole puts its pairs. Same colour scale '
                     'on all three; dashed line is the kinematic floor.',
                     fontsize=fs.BASE_PT * 0.89, y=0.99)
        fig.tight_layout(rect=(0, 0, 0.94, 1))
        cax = fig.add_axes([0.955, 0.16, 0.012, 0.66])
        cb = fig.colorbar(pc, cax=cax)
        cb.set_label('fraction of the pairs per cell',
                     fontsize=fs.BASE_PT * 0.64)
        cb.ax.tick_params(labelsize=fs.BASE_PT * 0.57)
        flat = {f'{k}_theta{int(t)}': cols[k][:, i]
                for k in cols for i, t in enumerate(tbins[:-1])}
        fs.save(fig, out / 'ipc_density',
                data=pd.DataFrame(dict(
                    m_ee_MeV=np.exp(0.5 * (mbins[1:] + mbins[:-1])), **flat)))


# --------------------------------------------------------------------------- #
def fig_shapes(out):
    plt = _plt()
    band = _band_from_ansatz()
    cols = {}
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        lo = band.min(axis=1).to_numpy()
        hi = band.max(axis=1).to_numpy()
        ax.fill_between(MID, np.maximum(lo, 1e-8), hi, color=fs.LINE,
                        alpha=0.9, lw=0, zorder=1,
                        label='what pair_physics.py carries now\n(four ansätze)')
        for kind in ('E0', 'E1', 'M1'):
            y = IB.grid_spectrum(kind, IB.E_TRANSITION, BINS)
            cols[kind] = y
            ax.plot(MID, np.maximum(y, 1e-8), color=KIND_COLOR[kind], zorder=3,
                    label=f'{KIND_LABEL[kind]}   (median {_median(y):.0f}°)')
        _x17_band(ax, 4e-4)
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
        fig, ax = fs.figure(fs.WIDE)
        for kind in ('E0', 'E1', 'M1'):
            d = IB.sample(kind, n, IB.E_TRANSITION)
            h, _ = np.histogram(np.log(d.m_ee), bins=lnb, weights=d.weight)
            h = h / h.sum()
            cols[kind] = h
            ax.plot(mid, h, color=KIND_COLOR[kind], label=KIND_LABEL[kind])
        for lbl, p, st in (('Geant generator:  dN/dM ~ 1/M', -1.0, '--'),
                           ('the “extreme” bracket:  dN/dM ~ 1/M³', -3.0, ':')):
            y = mid ** (p + 1)
            y = y / y.sum()
            cols[lbl] = y
            ax.plot(mid, y, color=fs.INK, ls=st, lw=1.8, label=lbl)
        ax.axvline(16.8, color=fs.COPPER, lw=1.6, ls='-.')
        ax.text(16.2, 0.072, 'M = 16.8 MeV\nis 109°', ha='right', va='top',
                color=fs.COPPER, fontsize=fs.BASE_PT * 0.77)
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


def fig_thermal(out):
    """The prediction, with the only free parameter shown as a band."""
    plt = _plt()
    P = IC.thermal_spectrum(BINS)
    band = _band_from_ansatz()
    m1, e0 = P.M1.to_numpy(), P.E0.to_numpy()
    f0 = float(IC.thermal_channels().set_index('channel')
               .loc['E0', 'share_of_pairs'])
    # the E0 fraction is the one thing not settled; the sensitivity table's
    # extremes are 6 % and 52 %, so that is the band.
    lo = 0.064 * e0 + 0.936 * m1
    hi = 0.523 * e0 + 0.477 * m1
    tot = P.total.to_numpy()
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        ax.fill_between(MID, np.maximum(band.min(axis=1), 1e-8),
                        band.max(axis=1), color=fs.LINE, alpha=0.9, lw=0,
                        label='the four-ansatz band this page replaces')
        ax.fill_between(MID, np.maximum(lo, 1e-8), hi, color=fs.ACCENT,
                        alpha=0.20, lw=0,
                        label='E0 fraction 6–52 %, the one open parameter')
        ax.plot(MID, np.maximum(m1, 1e-8), color=fs.DET_COLOR['A'], ls='--',
                lw=2.0, label=f'M1 alone, the 1⁺ channel  (median {_median(m1):.0f}°)')
        ax.plot(MID, np.maximum(e0, 1e-8), color=fs.ACCENT, ls='--', lw=2.0,
                label=f'E0 alone, the 0⁺ channel  (median {_median(e0):.0f}°)')
        ax.plot(MID, np.maximum(tot, 1e-8), color=fs.INK, lw=3.0,
                label=f'the ³He prediction below 2 eV, E0 = {100 * f0:.0f} %'
                      f'  (median {_median(tot):.0f}°)')
        _x17_band(ax, 4e-4)
        ax.set_yscale('log')
        ax.set_xlim(0, 180)
        ax.set_ylim(1e-5, 2e-1)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_xlabel('opening angle at birth  [deg]')
        ax.set_ylabel('normalised  dN/dθ  [1/deg]')
        ax.set_title('Below 2 eV only two channels are open, and one of them makes no photons')
        ax.legend(loc='upper right')
        fs.save(fig, out / 'ipc_thermal', data=P)


def fig_time(out):
    """The spectrum against arrival time.  Four decades, one curve."""
    plt = _plt()
    E = IC.energy_invariance()
    P = IC.thermal_spectrum(BINS)
    tot = P.total.to_numpy()
    with _scaled(plt):
        fig, (ax, bx) = plt.subplots(
            1, 2, figsize=fs.WIDE, gridspec_kw=dict(width_ratios=[1.25, 1]))
        for t, en, c in zip(E.t_ms, E.En_eV,
                            (fs.DET_COLOR['A'], fs.DET_COLOR['B'],
                             fs.DET_COLOR['C'], fs.DET_COLOR['D'],
                             fs.ACCENT, fs.COPPER, fs.INK)):
            w = IB.E_TRANSITION + 0.75 * en * 1e-6
            y = IC.thermal_channels()  # mix is time-independent by construction
            f = dict(zip(y.channel, y.share_of_pairs))
            s = (f['M1'] * IB.grid_spectrum('M1', w, BINS)
                 + f['E0'] * IB.grid_spectrum('E0', w, BINS))
            ax.plot(MID, np.maximum(s, 1e-8), color=c, lw=2.4, alpha=0.85,
                    label=f'{t:g} ms  →  Eₙ = {en:.3g} eV')
        _x17_band(ax, 4e-4)
        ax.set_yscale('log')
        ax.set_xlim(0, 180)
        ax.set_ylim(1e-5, 2e-1)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_xlabel('opening angle at birth  [deg]')
        ax.set_ylabel('normalised  dN/dθ  [1/deg]')
        ax.set_title('Seven arrival times, seven curves, one line')

        bx.loglog(E.En_eV, np.maximum(E.dW_over_W, 1e-16), 'o-',
                  color=fs.DET_COLOR['A'],
                  label='shift in the transition energy  ΔW/W')
        bx.loglog(E.En_eV, E.p_wave_over_s_wave, 's-', color=fs.COPPER,
                  label='p-wave admixture, relative to 0.17 MeV')
        bx.axhline(1.0, color=fs.INK, lw=1.4, ls='--')
        bx.text(E.En_eV.min(), 1.6, 'the E0:M1 mix, and the Al:³He capture '
                                    'ratio\n— both ratios of 1/v channels, so both exactly flat',
                fontsize=fs.BASE_PT * 0.68, color=fs.INK, va='bottom')
        bx.set_ylim(1e-15, 30)
        bx.set_xlabel('neutron energy  [eV]')
        bx.set_ylabel('size of the effect')
        bx.set_title('and nothing that could bend them is bigger than 10⁻⁵')
        bx.legend(loc='lower right', fontsize=fs.BASE_PT * 0.68)
        ax.legend(loc='upper right', fontsize=fs.BASE_PT * 0.68)
        fig.tight_layout()
        fs.save(fig, out / 'ipc_time', data=E)


# --------------------------------------------------------------------------- #
def fig_al_lines(out, lines, assume):
    """Which capture lines make the wide-angle pairs.  Not the hard ones."""
    plt = _plt()
    d = AL.pair_yield(lines, assume)
    d = d[d.pairs_per_capture > 0].copy()
    fr = [IB.frac_above(IB.grid_spectrum(r.mult_used, r.w_MeV, BINS), 109.0)
          for _, r in d.iterrows()]
    d['gt109'] = d.pairs_per_capture * np.array(fr)
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        for mult, c, lbl in (('E1', fs.DET_COLOR['C'], 'E1 — feeds a negative-parity level'),
                             ('M1', fs.DET_COLOR['A'], 'M1 — feeds a positive-parity level'),
                             ('unassigned', fs.MUTED,
                              f'no parity assignment — taken as {assume}')):
            m = d.multipole == mult
            if not m.any():
                continue
            ax.vlines(d.w_MeV[m], 0, 1e6 * d.gt109[m], color=c, lw=3.0,
                      label=lbl)
        for _, r in d.nlargest(5, 'gt109').iterrows():
            ax.annotate(f'{r.e_gam:.0f} keV', (r.w_MeV, 1e6 * r.gt109),
                        textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=fs.BASE_PT * 0.68, color=fs.INK)
        frac25 = float(d.loc[(d.w_MeV > 2) & (d.w_MeV <= 5), 'gt109'].sum()
                       / d.gt109.sum())
        ax.axvspan(2.0, 5.0, color=fs.BAND_CONTROL, alpha=0.10, lw=0, zorder=0)
        ax.text(3.5, 1e6 * d.gt109.max() * 0.60,
                f'{100 * frac25:.0f} % of the wide-angle yield\nis born in this band',
                ha='center', color=fs.MUTED, fontsize=fs.BASE_PT * 0.77)
        ax.set_xlim(1.0, 8.2)
        ax.set_ylim(0, 1e6 * d.gt109.max() * 1.28)
        ax.set_xlabel('γ-line energy  [MeV]')
        ax.set_ylabel('pairs beyond 109° per capture  [×10⁻⁶]')
        ax.set_title('No single line makes this background: the 2–5 MeV group beats the 7.7 MeV one')
        ax.legend(loc='upper left')
        fs.save(fig, out / 'ipc_al_lines',
                data=d[['e_gam', 'intensity', 'multipole', 'mult_used',
                        'alpha_pair', 'pairs_per_capture', 'gt109']])


def fig_al_shape(out, sc):
    """Aluminium against helium, before and after the wall."""
    plt = _plt()
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        pairs = (('he3_birth', fs.ACCENT, '-', 3.0, '³He gas, at birth'),
                 ('he3_after_wall', fs.ACCENT, ':', 2.0,
                  '³He, after crossing the capsule wall'),
                 ('capsule_birth', fs.DET_COLOR['B'], '-', 3.0,
                  'capsule wall (²⁷Al + ¹²C), at birth'),
                 ('capsule_after_wall', fs.DET_COLOR['B'], ':', 2.0,
                  'capsule, after escaping the wall it was born in'))
        for col, c, ls, lw, lbl in pairs:
            y = sc[col].to_numpy()
            ax.plot(MID, np.maximum(y, 1e-8), color=c, ls=ls, lw=lw,
                    label=f'{lbl}   (median {_median(y):.0f}°)')
        _x17_band(ax, 4e-4)
        ax.set_yscale('log')
        ax.set_xlim(0, 180)
        ax.set_ylim(1e-4, 1e-1)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.set_xlabel('opening angle  [deg]')
        ax.set_ylabel('normalised  dN/dθ  [1/deg]')
        ax.set_title('Shape alone will not separate the capsule from the gas')
        ax.legend(loc='upper right')
        fs.save(fig, out / 'ipc_al_shape', data=sc)


def fig_energy(out, n):
    plt = _plt()
    S = IB.energy_scan(kinds=('E1', 'M1'),
                       energies=[20.58, 16.0, 12.0, 9.0, IB.AL_SN, 6.0, 4.734,
                                 3.034, 2.590, 1.779], n=n)
    with _scaled(plt):
        fig, ax = fs.figure(fs.WIDE)
        for kind, c in (('E1', fs.DET_COLOR['C']), ('M1', fs.DET_COLOR['A'])):
            s = S[S.multipole == kind].sort_values('w_MeV')
            ax.plot(s.w_MeV, 1e4 * s.pairs_gt109_per_photon, 'o-', color=c,
                    label=f'{kind}  pairs above 109° per photon')
        ax.axvline(IB.AL_SN, color=fs.COPPER, lw=1.8)
        ax.text(IB.AL_SN + 0.3, 2.4, '²⁷Al(n,γ)\n7.73 MeV', color=fs.COPPER,
                fontsize=fs.BASE_PT * 0.77, va='bottom')
        ax.axvline(20.58, color=fs.ACCENT, lw=1.8)
        ax.text(20.3, 2.4, '³He(n,γ)\n20.58 MeV', color=fs.ACCENT, ha='right',
                fontsize=fs.BASE_PT * 0.77, va='bottom')
        ax.set_ylim(0, 5.5)
        ax.set_xlim(0, 22)
        ax.set_xlabel('transition energy  [MeV]')
        ax.set_ylabel('wide-angle pairs per photon  [×10⁻⁴]')
        ax.set_title('A 3 MeV capture γ makes as many wide-angle pairs as a 20.6 MeV one')
        ax.legend(loc='lower right')
        fs.save(fig, out / 'ipc_energy', data=S)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=1_500_000)
    ap.add_argument('--assume', default='M1', choices=('M1', 'E1'))
    a = ap.parse_args()
    out = paths.out('ipc') / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig_kinematics(out)
    fig_density(out, max(a.n // 2, 500_000))
    fig_shapes(out)
    fig_mass(out, a.n)
    fig_thermal(out)
    fig_time(out)
    fig_energy(out, max(a.n // 3, 400_000))
    lines = AL.line_list()
    fig_al_lines(out, lines, a.assume)
    fig_al_shape(out, AL.shape_comparison(a.assume))
    print(f'\nfigures -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
