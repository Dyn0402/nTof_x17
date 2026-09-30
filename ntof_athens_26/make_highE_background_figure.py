#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_highE_background_figure.py -- X17 against the capsule wall, versus E_n.

    python make_highE_background_figure.py   # -> figures/x17_vs_wall_energy.{png,pdf,csv}

The question (2026-09-30, Dylan): at high neutron energy, how does the X17
yield compare with the aluminium background?  Everything here is PER NEUTRON
ENTERING THE CAPSULE, so the n_TOF flux drops out and the curves say what the
same capsule would see at each energy.

Three yields, all from existing modules -- nothing new is modelled:

  X17        3He radiative captures per neutron (``ganil_background.he3_rates``,
             self-shielded) x IPC pairs per radiative capture x X17/IPC.
             IPC/capture is ipc_channels' M1+E0 value below 2 eV and the rate
             table's 2.1e-3 (Viviani, the p-wave resonance) above;
             X17/IPC = 2.5e-2 is the rate table's own assumption.
  3He IPC    the same, without the 2.5e-2 -- the irreducible background.
  wall       Al + C internal pairs from the capsule wall
             (``ganil_background.capsule_pairs``: capture and every discrete
             inelastic level, ENDF/B-VIII.0), in the X17 window.

Then the ratio: wall pairs / X17, both counted above the X17 minimum opening
angle at that energy (109 deg at thermal, falling at MeV).

ONE GAP IS FILLED BY HAND.  The staged 27Al evaluation carries (n,gamma) in MF3
only from 845 keV; below that it lives in the resonance parameters (MF2),
which ``endf.py`` does not read.  Here it is 1/v from the thermal 0.231 b up to
1 keV, then log-log to the MF3 value at 845 keV (~1 mb: about the
resonance-averaged capture of aluminium).  Rough, and labelled on the figure.

WHAT IS LEFT OUT, and which way it pushes:
  * external conversion of wall photons (84.5 % of the thermal capsule pairs,
    slide 28) -- mostly narrow angle, so it adds less inside the window than
    overall; the wall curve is low by up to ~x6.
  * the inelastic CONTINUUM (MT91) above ~5 MeV -- the wall is low there too
    (ganil_background.inelastic_completeness).
  * everything that is not the capsule: gamma flash, room, (n,p) two-prongs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

np.trapezoid = getattr(np, 'trapezoid', None) or np.trapz   # numpy 1 compat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, 'mpgd26')):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib                                         # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                           # noqa: E402

import plotstyle as P                                     # noqa: E402
from sept26_prelim_analysis import endf as EN             # noqa: E402
from sept26_prelim_analysis import ipc_born as IB         # noqa: E402
from sept26_prelim_analysis import ipc_channels as IC     # noqa: E402
from sept26_prelim_analysis import ganil_background as G  # noqa: E402

OUT = os.path.join(HERE, 'figures')

X17_OVER_IPC = 2.5e-2          # the rate table's header
IPC_PER_CAPTURE_MEV = 2.1e-3   # ... and its Viviani value, p-wave resonance
RESONANCE_EDGE_EV = 2.0        # below: only the s-wave M1 + E0 channels
AL_MF3_START_EV = 8.45e5       # where the staged 27Al (n,g) MF3 begins
AL_THERMAL_B = 0.231
S_N_AL = 7.7255                # 28Al capture line, as ganil_background has it

# the n_TOF windows, for shading: our thermal data, and where the X17 rate is
WIN_DATA_EV = (4.4e-4, 2.0)
WIN_MEV_EV = (2.2e5, 2.2e6)    # make_x17_rate's MeV window, 79 % of the rate

HE3, WALL, X17C = '#e8621f', '#2b5fa8', '#a5308f'


def al_capture_b(e_ev: float) -> float:
    """27Al(n,g): ENDF MF3 where it exists, the hand bridge below it."""
    if e_ev >= AL_MF3_START_EV:
        return float(EN.sigma_at('Al27', 'capture', e_ev))
    s_1kev = AL_THERMAL_B * np.sqrt(0.0253 / 1e3)
    if e_ev <= 1e3:
        return AL_THERMAL_B * np.sqrt(0.0253 / e_ev)
    s_end = float(EN.sigma_at('Al27', 'capture', AL_MF3_START_EV))
    f = np.log(e_ev / 1e3) / np.log(AL_MF3_START_EV / 1e3)
    return float(np.exp(np.log(s_1kev) + f * (np.log(s_end) - np.log(s_1kev))))


def wall_pairs(en_mev: float):
    """(spectrum, pairs/neutron): ganil_background's wall + the Al bridge."""
    bins = IB.THETA_BINS
    spec, tot = G.capsule_pairs(en_mev, bins)
    spec = spec * tot
    if en_mev * 1e6 < AL_MF3_START_EV:
        w = G.N_AL * al_capture_b(en_mev * 1e6) * IB.alpha_pair('E1', S_N_AL)
        spec = spec + w * IB.grid_spectrum('E1', S_N_AL, bins)
        tot += w
    s = (spec * np.diff(bins)).sum()
    return (spec / s if s > 0 else spec), tot


_TH = IC.thermal_channels()
IPC_PER_CAPTURE_THERMAL = float(_TH.sigma_pair_ub.sum()) / IC.SIGMA_NGAMMA_UB


def row(en_mev: float) -> dict:
    he = G.he3_rates(en_mev).iloc[0]
    thermal = en_mev * 1e6 < RESONANCE_EDGE_EV
    ipc = he.radiative_per_neutron * (IPC_PER_CAPTURE_THERMAL if thermal
                                      else IPC_PER_CAPTURE_MEV)
    lo = float(G.x17_min_angle(en_mev))
    if thermal:   # the M1 + E0 mix, as ipc_channels tabulates it above 109 deg
        f_ipc = float((_TH.sigma_pair_ub * _TH.frac_gt109).sum()
                      / _TH.sigma_pair_ub.sum())
    else:
        f_ipc = G._frac_in(G.he3_pair_spectrum(en_mev, 'E1'), lo, 180.0)
    y_w, w = wall_pairs(en_mev)
    f_w = G._frac_in(y_w, lo, 180.0) if w > 0 else 0.0
    f_x = G._frac_in(G.x17_spectrum(en_mev), lo, 180.0)
    x17 = ipc * X17_OVER_IPC
    return dict(En_eV=en_mev * 1e6, theta_min_deg=lo,
                x17_per_n=x17, he3_ipc_per_n=ipc, wall_pairs_per_n=w,
                x17_in_win=x17 * f_x, he3_ipc_in_win=ipc * f_ipc,
                wall_in_win=w * f_w,
                wall_over_x17=w * f_w / (x17 * f_x),
                ipc_over_x17=ipc * f_ipc / (x17 * f_x))


def scan() -> pd.DataFrame:
    e = np.r_[np.logspace(-3, np.log10(1.9), 14),            # eV, thermal
              np.logspace(np.log10(2.1), 6.0, 20),           # 2 eV .. 1 MeV
              np.logspace(np.log10(1.1e6), np.log10(2.25e6), 6),
              np.logspace(np.log10(2.35e6), np.log10(2e7), 14)]
    return pd.DataFrame([row(x * 1e-6) for x in e])


def draw(d: pd.DataFrame):
    P.use()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9),
                             gridspec_kw=dict(wspace=0.26))
    q_hi = G.quiet_band()[1] * 1e6
    for ax in axes:
        P.strip(ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-3, 2e7)
        ax.set_xticks([1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7])
        ax.set_xlabel('neutron energy  [eV]')
        ax.grid(which='minor', visible=False)
        ax.axvspan(*WIN_DATA_EV, color=P.BAND_SIGNAL, alpha=0.09, lw=0, zorder=0)
        ax.axvspan(*WIN_MEV_EV, color=X17C, alpha=0.08, lw=0, zorder=0)
        ax.axvline(q_hi, color=P.MUTED, lw=1.0, ls=':', zorder=1)
    E = d.En_eV.to_numpy()

    # (a) the three yields
    ax = axes[0]
    ax.plot(E, d.wall_in_win, color=WALL, lw=2.6)
    ax.plot(E, d.he3_ipc_in_win, color=HE3, lw=2.2, ls='--')
    ax.plot(E, d.x17_in_win, color=X17C, lw=2.8)
    ax.set_ylim(1e-13, 1e-4)
    ax.set_ylabel('pairs above θ$_{min}$, per neutron')
    P.title(ax, 'What the capsule makes', 'per neutron entering it · '
            'counted above the X17 minimum angle')
    P.end_label(ax, 1.3e-3, float(d.wall_in_win.iloc[0]) * 3.0,
                'Al + C wall pairs', WALL)
    P.end_label(ax, 1.3e-3, float(d.he3_ipc_in_win.iloc[0]) * 4.0,
                '³He IPC', HE3)
    ax.text(3e2, 5e-13, 'X17  (2.5 % of IPC)', ha='left', va='top',
            fontsize=11, color=X17C, fontweight='bold')
    ax.text(np.sqrt(WIN_DATA_EV[0] * WIN_DATA_EV[1]), 0.985, 'our data',
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=9.5, color=P.BAND_SIGNAL, fontweight='bold')
    ax.text(np.sqrt(WIN_MEV_EV[0] * WIN_MEV_EV[1]), 0.985, '79 % of\nX17 rate',
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=9.5, color=X17C, fontweight='bold')

    # (b) the ratio
    ax = axes[1]
    ax.plot(E, d.wall_over_x17, color=WALL, lw=2.8)
    ax.plot(E, d.ipc_over_x17, color=HE3, lw=2.0, ls='--')
    ax.set_ylim(1e0, 3e7)
    ax.set_ylabel('background pairs per X17  (above θ$_{min}$)')
    P.title(ax, 'Wall pairs per X17', 'lower is better')
    at = lambda e: float(np.exp(np.interp(np.log(e), np.log(E),
                                          np.log(d.wall_over_x17))))
    sup = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')

    def rough(v):
        ex = int(np.floor(np.log10(v)))
        return f'{v:.0f}' if ex < 3 else f'~10{str(round(np.log10(v))).translate(sup)}'

    for e_, lab, dy in ((0.0253, 'thermal', 2.5), (7e5, '~1 MeV', 3.2)):
        v = at(e_)
        ax.plot([e_], [v], 'o', color=WALL, ms=6, zorder=5)
        ax.annotate(f'{rough(v)}\nat {lab}', (e_, v), xytext=(e_, v * dy),
                    ha='center', va='bottom', fontsize=11, color=WALL,
                    fontweight='bold')
    ax.text(q_hi / 1.4, 4e6, 'Al / C inelastic γ lines\nabove pair threshold\n'
            f'turn on at {q_hi / 1e6:.1f} MeV →', ha='right', va='center',
            fontsize=9.5, color=P.MUTED, linespacing=1.3)
    P.end_label(ax, 1.3e-3, float(d.ipc_over_x17.iloc[0]) * 3.2,
                '³He IPC, irreducible', HE3)
    fig.text(0.07, -0.02, 'Wall: internal pairs only, no external conversion '
             '(adds up to ~×6, mostly at small angles).  ²⁷Al (n,γ) below 845 keV '
             'bridged by hand (1/v, then log-log to ENDF).\nNot included: γ flash, '
             'room background, (n,p) two-prongs.  X17/IPC = 2.5 % is the rate '
             'table\'s assumption; the ratio scales with it.', ha='left', va='top',
             fontsize=9.0, color=P.MUTED, linespacing=1.35)
    return fig


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    d = scan()
    fig = draw(d)
    base = os.path.join(OUT, 'x17_vs_wall_energy')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}')
    d.to_csv(f'{base}.csv', index=False)
    print(d[['En_eV', 'theta_min_deg', 'x17_in_win', 'wall_in_win',
             'wall_over_x17', 'ipc_over_x17']].to_string(index=False,
                                                         float_format='%.3g'))
    print(f'  -> {base}.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
