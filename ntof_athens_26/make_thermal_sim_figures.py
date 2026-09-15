#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_thermal_sim_figures.py -- slide 40: what a thermal measurement can see.

    python ntof_athens_26/make_thermal_sim_figures.py
    python ntof_athens_26/make_thermal_sim_figures.py --only branching
    python ntof_athens_26/make_thermal_sim_figures.py --only pair_sources

Two figures, each written as .png/.pdf/.csv into `figures/`, plus one .json
carrying every number that is not computed here, with where it came from.

`thermal_branching` -- WHY THE GAS SELF-SHIELDS.  Three panels on one neutron
energy axis.  (a) ³He's cross sections: all of them rise as 1/v towards thermal,
so it is true that the gas gets more opaque.  (b) But that alone is not the
problem -- the RATIO is.  σ(n,p)/σ(n,γ) is ~10⁴ at MeV and ~10⁸ at thermal.
(c) Why the ratio is the number that matters: at 25 meV the 500 atm cell has an
optical depth of ~150, so it absorbs every neutron that enters it and a
radiative capture happens with probability σ(n,γ)/σ(abs) per neutron, however
much gas there is.  The thin-target formula (N·σ(n,γ), what slide 39's rate
table uses) overstates that by the optical depth.  At MeV the cell is thin and
the two agree, which is why the table is right there and wrong here.

`thermal_pair_sources` -- WHERE THE e⁺e⁻ COME FROM.  (a) What is expected, per
neutron entering the capsule in our window: the wall's 7.7 MeV capture cascade
out-produces ³He by 10⁴-10⁶ in pairs.  (b) What fires the trigger in Geant4:
aluminium.  (c) What our coincident two-arm pairs actually are, measured from
the scintillator timing: mostly accidental combinations of those aluminium legs.

WHAT IS COMPUTED HERE AND WHAT IS QUOTED.

* Cross sections: ENDF/B-VIII.0 MF3 through `sept26_prelim_analysis.endf`,
  interpolated log-log.  ³He has no resolved resonances, so MF3 is the whole
  cross section.
* Self-shielding and the capsule bookkeeping: `ipc_aluminium` -- the same
  sphere-equivalent capsule as the rate table's header, so panel (c) and slide
  39 describe the same object.  The analytic shielding factor is ~150; the
  Geant4 thermal note, with the real polycone and the real spectrum, says ×50-100.
  Both are drawn/quoted, neither is tuned to the other.
* Pairs per ³He radiative capture: `ipc_born`/`ipc_channels` (M1 + E0), **not**
  IPC/capture = 2.1e-3, which is Viviani et al.'s MeV number (see CLAUDE.md).
* Pairs per wall capture: `ipc_aluminium` from the EGAF/PGAA line list, 27Al and
  12C weighted by their captures.  Internal conversion only; external conversion
  of the capture γ in the wall is NOT included and only adds to the wall.
* Geant4 trigger provenance and per-day pair yields: quoted from the n_TOF run
  report (`ntof_run_report/make_report.py` §6), whose sources are
  `MX17_Full_Geant/analysis/trigger_provenance/`, `al_pair_background/VERDICT.md`
  and `docs/report/thermal_note.pdf`.  Those live on the Linux box and are not
  re-read here; if they move, update `G4` below.
* Measured true-coincidence fractions: `sept26_prelim_analysis/
  HANDOFF_ACCIDENTAL_TIMING.md` §0 (run_145, unbinned two-component fit to the
  arm1-arm2 scintillator Δt).  PRELIMINARY, and small samples.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(REPO / 'mpgd26')):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import endf as E  # noqa: E402
from sept26_prelim_analysis import ipc_aluminium as IA  # noqa: E402
import plotstyle as P  # noqa: E402

OUT = HERE / 'figures'

#: See make_pair_qa_figures.FOOT_COLS: an unwrapped footnote widens the canvas.
FOOT_COLS = 168

HE3 = P.ACCENT               # the gas, everywhere on both figures
WALL = '#5b6b7d'             # the capsule wall: aluminium + carbon fibre
PALE = '#e3e7ec'

# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
E_THERMAL_EV = 0.0253
#: The window the data actually covers: 1 ms flash veto -> 2.0 eV, down to the
#: slowest recorded neutron.  sept26_prelim_analysis/STATUS.md (neutron_energy).
WIN_LO_EV, WIN_HI_EV, WIN_MEDIAN_EV = 0.44e-3, 2.0, 0.031

#: The gas, from the rate table's own header (x17_rate_3He.txt): 1.977 g of 3He
#: in a 4 cm sphere.  Only used for the mean free path; the optical depth uses
#: IA.N_HE3_ATB, which is the same gas expressed per barn.
HE3_MASS_G, HE3_A, CELL_R_CM = 1.977, 3.016, 2.0
N_HE3_PER_CM3 = HE3_MASS_G / HE3_A * 6.02214076e23 / (4.0 / 3.0 * np.pi * CELL_R_CM ** 3)

#: Slide 39's rate table: neutrons per pulse on the cell, per energy decade, and
#: its nominal pulses per day.  Panel (c) counts over DAYS of that running, so
#: the two slides quote the same beam.
RATE_TABLE = REPO / 'mpgd26' / 'data' / 'x17_rate_3He.txt'
DAYS = 30


def rate_table() -> tuple:
    """``(pulses_per_day, [(e_lo_eV, e_hi_eV, neutrons_per_pulse), ...])``."""
    ppd, bins = None, []
    for line in RATE_TABLE.read_text(encoding='utf-8').splitlines():
        if line.startswith('#pulses'):
            ppd = float(line.split(':')[1])
        elif line.strip() and not line.startswith('#'):
            f = line.split()
            bins.append((float(f[0]), float(f[1]), float(f[3])))
    return ppd, bins

#: Quoted Geant4 results.  See the module docstring for provenance.
G4 = {
    'source': 'ntof_run_report/make_report.py section 6, quoting MX17_Full_Geant '
              '(trigger_provenance, al_pair_background/VERDICT.md, '
              'docs/report/thermal_note.pdf)',
    'np_over_ng_thermal': 1.0e8,
    'shielding_factor': [50, 100],
    'ipc_per_pulse_thin_below_1keV': 1.21e-2,
    'ipc_per_pulse_shielded': [1.1e-4, 2.3e-4],
    'al_capture_gamma_per_pulse': 4121,
    'trigger_per_pulse': [
        # label, per pulse, aluminium fraction
        ['SiPM-wall singles', 2063, 0.80],
        ['plastic singles', 942, 0.57],
        ['arm coincidence\n(one trigger leg)', 205, 0.96],
    ],
    'trigger_note': '10^9 EAR2-flux neutrons, nose-first geometry, 0.5 MIP threshold',
    'per_day_produced': {'Al(n,g) pairs': 5.95e6, '3He IPC': 1.39, 'X17': 0.035},
    'per_day_mm_acceptance': {'Al(n,g) pairs': 6.5e5, '3He IPC': 0.44, 'X17': 0.012},
}

#: Measured: HANDOFF_ACCIDENTAL_TIMING.md section 0.  (label, pairs, f, lo, hi)
TIMING = [
    ('all two-arm pairs', 76, 0.29, 0.17, 0.40),
    ('opposing  (A–C)', 42, 0.42, 0.26, 0.58),
    ('perpendicular', 34, 0.16, 0.00, 0.32),
]
TIMING_SOURCE = ('sept26_prelim_analysis/HANDOFF_ACCIDENTAL_TIMING.md section 0 -- '
                 'run_145, unbinned two-component fit to the arm1-arm2 '
                 'scintillator dt, 68 % intervals')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def he3_xs(mt: int, e_ev: np.ndarray) -> np.ndarray:
    """³He cross section in barns, log-log between the evaluation's points.

    Below a threshold (the (n,d) channel) the evaluation carries zeros, which
    have no logarithm; those energies return 0 rather than an extrapolation.
    """
    d = E.xs('He3', mt)
    x, y = d.E_eV.to_numpy(), d.sigma_b.to_numpy()
    pos = y > 0
    out = np.exp(np.interp(np.log(e_ev), np.log(x[pos]), np.log(y[pos])))
    return np.where(e_ev < x[pos][0], 0.0, out)


def ng_per_neutron(e_ev: np.ndarray) -> tuple:
    """``(thin, shielded)`` ³He(n,γ) per neutron entering the cell.

    Thin is N·σ(n,γ).  Shielded is the absorption probability of a uniform beam
    on the sphere times the radiative share of absorptions, σ(n,γ)/σ(abs).
    """
    s_ng = he3_xs(102, e_ev)
    s_abs = he3_xs(103, e_ev) + s_ng + he3_xs(104, e_ev)
    p_abs = np.array([IA._sphere_absorption(IA.N_HE3_ATB, s) for s in s_abs])
    return IA.N_HE3_ATB * s_ng, p_abs * s_ng / s_abs


def footnote(fig, text: str) -> None:
    fig.text(0.0, -0.01, textwrap.fill(text, FOOT_COLS), ha='left', va='top',
             fontsize=9.0, color=P.MUTED)


def window(ax, label=True) -> None:
    ax.axvspan(WIN_LO_EV, WIN_HI_EV, color=P.BAND_SIGNAL, alpha=0.09, zorder=0,
               lw=0)
    if label:
        ax.text(np.sqrt(WIN_LO_EV * WIN_HI_EV), 0.985, 'our data\n2 eV → 0.4 meV',
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=9.5, color=P.BAND_SIGNAL, fontweight='bold')


def save(fig, stem: str, table: pd.DataFrame) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}')
        print(f'  -> {OUT / f"{stem}.{ext}"}')
    plt.close(fig)
    table.to_csv(OUT / f'{stem}.csv', index=False)
    print(f'  -> {OUT / f"{stem}.csv"}')


def sci(v: float, digits: int = 1) -> str:
    """1.03e-08 -> '1.0×10⁻⁸', for labels."""
    sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    if v == 0:
        return '0'
    ex = int(np.floor(np.log10(abs(float(f'{v:.{digits}e}')))))
    man = v / 10 ** ex
    if ex == -1 and man >= 9.95:
        return '≈ 1'
    if ex in (0, 1, 2, 3):
        return f'{v:,.{max(0, digits - ex)}f}'.replace(',', ' ')
    return f'{man:.{digits}f}×10{str(ex).translate(sup)}'


# --------------------------------------------------------------------------- #
# figure 1 -- the branching ratio and the self-shielding it causes
# --------------------------------------------------------------------------- #
def branching() -> dict:
    e = np.logspace(-4, np.log10(2e7), 400)
    s_tot, s_el = he3_xs(1, e), he3_xs(2, e)
    s_ng, s_np, s_nd = he3_xs(102, e), he3_xs(103, e), he3_xs(104, e)
    s_abs = s_np + s_ng + s_nd
    ratio = s_ng / s_np              # ⁴He* made per proton made

    tau = 1.5 * IA.N_HE3_ATB * s_abs                               # on axis
    thin, shielded = ng_per_neutron(e)
    p_abs = shielded * s_abs / s_ng

    # the same thing as a count: per energy decade, over DAYS at the table's flux
    ppd, bins = rate_table()
    per_run = ppd * DAYS
    d30 = []
    for lo, hi, n_pp in bins:
        g = np.logspace(np.log10(lo), np.log10(min(hi, 2e7)), 120)  # ENDF ends at 20 MeV
        t_, s_ = ng_per_neutron(g)                                  # iso-lethargic in the bin
        d30.append(dict(E_lo_eV=lo, E_hi_eV=hi, neutrons_per_pulse=n_pp,
                        ng_thin=n_pp * t_.mean() * per_run,
                        ng_shielded=n_pp * s_.mean() * per_run))
    d30 = pd.DataFrame(d30)
    win = d30[(d30.E_lo_eV >= 0.01) & (d30.E_hi_eV <= 1.0)]
    he_pairs_per_radcap = IA.rate_comparison('M1').attrs['he_pairs_per_radcap']

    at = lambda arr, ev: float(np.exp(np.interp(np.log(ev), np.log(e), np.log(arr))))
    s_tot_th = at(s_tot, E_THERMAL_EV)
    mfp_mm = 10.0 / (N_HE3_PER_CM3 * s_tot_th * 1e-24)
    num = dict(
        sigma_np_thermal_b=at(s_np, E_THERMAL_EV),
        sigma_ng_thermal_b=at(s_ng, E_THERMAL_EV),
        ratio_thermal=at(ratio, E_THERMAL_EV),
        ratio_1MeV=at(ratio, 1e6),
        mean_free_path_thermal_mm=mfp_mm,
        optical_depth_on_axis_thermal=at(tau, E_THERMAL_EV),
        shielding_factor_thermal=at(thin, E_THERMAL_EV) / at(shielded, E_THERMAL_EV),
        shielding_factor_window_median=at(thin, WIN_MEDIAN_EV) / at(shielded, WIN_MEDIAN_EV),
        ng_per_neutron_shielded_thermal=at(shielded, E_THERMAL_EV),
        ng_per_neutron_thin_thermal=at(thin, E_THERMAL_EV),
        days=DAYS, pulses_per_day=ppd,
        ng_thin_30d_0p01_to_1eV=float(win.ng_thin.sum()),
        ng_shielded_30d_0p01_to_1eV=float(win.ng_shielded.sum()),
        he3_ipc_pairs_30d_0p01_to_1eV=float(win.ng_shielded.sum()) * he_pairs_per_radcap,
    )

    P.use()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9),
                             gridspec_kw=dict(wspace=0.34))
    for ax in axes:
        P.strip(ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-4, 2e7)
        ax.set_xticks([1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7])
        ax.set_xlabel('neutron energy  [eV]')
        ax.grid(which='minor', visible=False)
        window(ax, label=ax is axes[0])

    # (a) the cross sections
    ax = axes[0]
    ax.plot(e, s_np, color=HE3, lw=2.4)
    ax.plot(e, s_el, color=P.MUTED, lw=1.6, ls='--')
    ax.plot(e, s_ng, color=P.COPPER, lw=2.4)
    ax.set_ylim(1e-6, 3e6)
    ax.set_ylabel('³He cross section  [b]')
    P.title(ax, 'Everything rises as 1/v', 'ENDF/B-VIII.0')
    P.end_label(ax, 3e2, at(s_np, 3e2) * 3.5, '(n,p) → p + t', HE3)
    P.end_label(ax, 2e-4, at(s_el, 2e-4) * 4.0, 'elastic', P.MUTED)
    P.end_label(ax, 1e2, 1.5e-3, '(n,γ) → ⁴He*', P.COPPER)
    ax.annotate(f'{sci(num["sigma_np_thermal_b"], 0)} b', (E_THERMAL_EV, num['sigma_np_thermal_b']),
                xytext=(E_THERMAL_EV * 8, num['sigma_np_thermal_b'] * 6), color=HE3,
                fontsize=10, arrowprops=dict(arrowstyle='-', color=HE3, lw=0.8))
    ax.annotate(f'{num["sigma_ng_thermal_b"] * 1e6:.0f} µb', (E_THERMAL_EV, num['sigma_ng_thermal_b']),
                xytext=(E_THERMAL_EV * 8, num['sigma_ng_thermal_b'] * 8), color=P.COPPER,
                fontsize=10, arrowprops=dict(arrowstyle='-', color=P.COPPER, lw=0.8))

    # (b) the ratio
    ax = axes[1]
    ax.plot(e, ratio, color=P.INK, lw=2.4)
    ax.set_ylim(1e-10, 1e-3)
    ax.set_ylabel('σ(n,γ) / σ(n,p)')
    P.title(ax, 'The ratio is what matters', '⁴He* made per proton made')
    for ev, lab, xt, yt, ha in ((E_THERMAL_EV, '25 meV', E_THERMAL_EV, 0.2, 'center'),
                                (1e6, '1 MeV', 2e3, 1.0, 'right')):
        r = at(ratio, ev)
        ax.plot([ev], [r], 'o', color=P.INK, ms=6, zorder=5)
        ax.annotate(f'{sci(r)}\nat {lab}', (ev, r), xytext=(xt, r * yt),
                    ha=ha, va='top' if ha == 'center' else 'center',
                    fontsize=10.5, color=P.INK, fontweight='bold')
    ax.text(0.04, 0.62, f'×{num["ratio_1MeV"] / num["ratio_thermal"]:,.0f} smaller\nat thermal'
            .replace(',', ' '), transform=ax.transAxes, ha='left', va='bottom',
            fontsize=11, color=P.BAND_DEAD, fontweight='bold')

    # (c) what the ratio does to a black absorber
    ax = axes[2]
    edges = np.r_[d30.E_lo_eV.to_numpy(), d30.E_hi_eV.iloc[-1]]
    ax.stairs(d30.ng_thin, edges, color=P.MUTED, lw=2.0, ls='--', baseline=None)
    ax.stairs(d30.ng_shielded, edges, color=P.COPPER, lw=2.6, baseline=None)
    ax.set_ylim(1e2, 1e8)
    ax.set_ylabel(f'³He(n,γ) per {DAYS} days, per energy decade')
    P.title(ax, 'A black cell: every neutron → a proton',
            f'{DAYS} days at slide 39’s flux · '
            + f'{ppd:,.0f} pulses/day'.replace(',', ' '))
    y_thin, y_sh = float(d30.ng_thin.iloc[0]), float(d30.ng_shielded.iloc[0])
    P.end_label(ax, 1.3e-4, y_thin * 5.0, 'thin target (slide 39)', P.MUTED)
    P.end_label(ax, 1.3e-4, y_sh / 4.0, 'self-shielded', P.COPPER)
    x_arrow = np.sqrt(d30.E_lo_eV.iloc[0] * d30.E_hi_eV.iloc[0])
    ax.annotate('', xy=(x_arrow, y_sh), xytext=(x_arrow, y_thin),
                arrowprops=dict(arrowstyle='<->', color=P.BAND_DEAD, lw=1.4))
    ax.text(x_arrow / 1.6, np.sqrt(y_thin * y_sh), f'×{y_thin / y_sh:.0f}',
            ha='right', va='center', fontsize=11, color=P.BAND_DEAD, fontweight='bold')
    ax.text(0.97, 0.03,
            f'0.01–1 eV, {DAYS} days:\n'
            f'{sci(num["ng_thin_30d_0p01_to_1eV"])} → {sci(num["ng_shielded_30d_0p01_to_1eV"])} '
            f'⁴He*\n≈ {num["he3_ipc_pairs_30d_0p01_to_1eV"]:.0f} IPC pairs made\n'
            f'(Geant4, real capsule: ×50–100)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9.8,
            color=P.INK, linespacing=1.3)

    d30.to_csv(OUT / 'thermal_branching_30d.csv', index=False)

    table = pd.DataFrame(dict(E_eV=e, sigma_total_b=s_tot, sigma_elastic_b=s_el,
                              sigma_np_b=s_np, sigma_ng_b=s_ng, sigma_nd_b=s_nd,
                              ng_over_np=ratio, optical_depth_on_axis=tau,
                              p_absorbed=p_abs, ng_per_neutron_thin=thin,
                              ng_per_neutron_shielded=shielded))
    save(fig, 'thermal_branching', table)
    return num


# --------------------------------------------------------------------------- #
# figure 2 -- where the pairs come from
# --------------------------------------------------------------------------- #
def pair_sources() -> dict:
    en = WIN_MEDIAN_EV
    bk = IA.bookkeeping(en)
    row = lambda start: float(bk.loc[bk.what.str.startswith(start), 'per_neutron'].iloc[0])
    wall_floor = row('27Al') + row('12C')                      # single pass
    wall_transport = IA.TABLE_GC_CAPTURES / IA.TABLE_NEUTRONS  # with wall scattering
    he_np = row('3He(n,p)')
    he_ng = row('3He(n,g)')

    cs = IA.capsule_summary('M1')
    wall_pairs_per_cap = float((cs.share_of_captures * cs.pairs_per_capture).sum())
    he_pairs_per_radcap = IA.rate_comparison('M1', en).attrs['he_pairs_per_radcap']

    rows = [
        # label, lo, hi, colour, is_pair
        ('³He(n,p) → p + t', he_np, he_np, HE3, False),
        ('capsule-wall captures\n(²⁷Al, ¹²C → 7.7 MeV γ cascade)',
         wall_floor, wall_transport, WALL, False),
        ('e⁺e⁻ from the wall', wall_floor * wall_pairs_per_cap,
         wall_transport * wall_pairs_per_cap, WALL, True),
        ('³He(n,γ) → ⁴He*', he_ng, he_ng, HE3, False),
        ('e⁺e⁻ from ³He  (IPC)', he_ng * he_pairs_per_radcap,
         he_ng * he_pairs_per_radcap, HE3, True),
    ]
    pair_ratio = (rows[2][1] / rows[4][1], rows[2][2] / rows[4][2])
    num = dict(
        energy_eV=en, he3_np_per_neutron=he_np, he3_ng_per_neutron=he_ng,
        wall_captures_per_neutron=[wall_floor, wall_transport],
        wall_pairs_per_capture=wall_pairs_per_cap,
        he3_pairs_per_radiative_capture=he_pairs_per_radcap,
        wall_pairs_per_neutron=[rows[2][1], rows[2][2]],
        he3_pairs_per_neutron=rows[4][1],
        wall_to_he3_pair_ratio=list(pair_ratio),
        g4_mm_acceptance_ratio=G4['per_day_mm_acceptance']['Al(n,g) pairs']
        / G4['per_day_mm_acceptance']['3He IPC'],
    )

    P.use()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0),
                             gridspec_kw=dict(width_ratios=[1.45, 1.0, 1.0],
                                              wspace=0.62))

    # (a) expected, per neutron entering the capsule
    ax = axes[0]
    P.strip(ax, left=False)
    xmin = 1e-12
    ax.set_xscale('log')
    ax.set_xlim(xmin, 1e3)
    ax.set_xticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
    ax.grid(axis='y', visible=False)
    ax.grid(which='minor', visible=False)
    ys = np.arange(len(rows))[::-1]
    for y, (lab, lo, hi, col, is_pair) in zip(ys, rows):
        h = 0.62 if is_pair else 0.42
        ax.barh(y, lo - xmin, left=xmin, height=h, color=col,
                alpha=1.0 if is_pair else 0.45, lw=0)
        if hi > lo:
            ax.barh(y, hi - lo, left=lo, height=h, color=col, alpha=0.18, lw=0,
                    hatch='///', edgecolor=col)
        txt = sci(lo) if hi == lo else f'{sci(lo)} – {sci(hi)}'
        ax.text(hi * 2.2, y, txt, va='center', ha='left', fontsize=10,
                color=col, fontweight='bold' if is_pair else 'normal')
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(r[3])
        t.set_fontweight('bold' if r[4] else 'normal')
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(f'per neutron entering the capsule  (E = {en * 1e3:.0f} meV)')
    P.title(ax, 'Expected: the wall makes the pairs',
            'the capsule, analytic; pairs are internal conversion only')
    ax.text(2e-6, ys[4] + 0.1,
            f'wall : ³He pairs ≈ {sci(pair_ratio[0], 0)} – {sci(pair_ratio[1], 0)} : 1\n'
            f'Geant4, in MM acceptance, per day:\n'
            f'{sci(G4["per_day_mm_acceptance"]["Al(n,g) pairs"])} Al pairs  vs  '
            f'{G4["per_day_mm_acceptance"]["3He IPC"]} ³He IPC',
            ha='left', va='center', fontsize=9.3,
            color=P.INK, linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.4', fc=P.SURFACE, ec=P.LINE))

    # (b) Geant4: what fires the trigger
    ax = axes[1]
    P.strip(ax, left=False, bottom=False)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    ax.grid(False)
    trig = G4['trigger_per_pulse']
    yb = np.arange(len(trig))[::-1]
    for y, (lab, n, fal) in zip(yb, trig):
        strong = y == 0
        ax.barh(y, fal, height=0.56, color=WALL, alpha=1.0 if strong else 0.7, lw=0)
        ax.barh(y, 1 - fal, left=fal, height=0.56, color=PALE, lw=0)
        ax.text(0.03, y, f'{fal:.0%} Al', va='center', ha='left', color='white',
                fontsize=11.5, fontweight='bold')
        ax.text(0.5, y + 0.36, f'{lab.replace(chr(10), " ")}  ·  {n:,} / pulse'
                .replace(',', ' '), va='bottom', ha='center', fontsize=10,
                color=P.INK, fontweight='bold' if strong else 'normal')
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(trig) - 0.1)
    P.title(ax, 'Geant4: Al fires the trigger',
            'thermal gate, by capture nucleus')

    # (c) measured: what our two-arm pairs are
    ax = axes[2]
    P.strip(ax, left=False, bottom=False)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    ax.grid(False)
    yc = np.arange(len(TIMING))[::-1]
    for y, (lab, n, f, lo, hi) in zip(yc, TIMING):
        ax.barh(y, f, height=0.56, color=P.INK, lw=0)
        ax.barh(y, 1 - f, left=f, height=0.56, color=PALE, lw=0)
        ax.errorbar([f], [y - 0.34], xerr=[[f - lo], [hi - f]], fmt='none',
                    ecolor=P.COPPER, elinewidth=2.0, capsize=3.5, capthick=2.0)
        ax.text(max(f, 0.001) / 2 if f > 0.2 else f + 0.02, y - 0.02,
                f'{f:.0%}', va='center', ha='center' if f > 0.2 else 'left',
                color='white' if f > 0.2 else P.INK, fontsize=11.5,
                fontweight='bold')
        ax.text(0.97, y, f'{1 - f:.0%}', va='center', ha='right', color=P.MUTED,
                fontsize=11.5, fontweight='bold')
        ax.text(0.5, y + 0.36, f'{lab}  ·  {n} pairs', va='bottom', ha='center',
                fontsize=10, color=P.INK)
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(TIMING) - 0.1)
    ax.text(0.0, -0.06, '■ prompt', transform=ax.transAxes, color=P.INK,
            fontsize=10.5, fontweight='bold')
    ax.text(0.40, -0.06, '■ accidental', transform=ax.transAxes, color='#9aa4b0',
            fontsize=10.5, fontweight='bold')
    ax.text(1.0, -0.16, 'PRELIMINARY', transform=ax.transAxes, ha='right',
            color=P.COPPER, fontsize=10, fontweight='bold')
    P.title(ax, 'Data: pairs mostly accidental',
            'run_145, scintillator Δt fit, 68 % intervals')

    footnote(fig,
             '(a) ipc_aluminium.bookkeeping at the window median: wall captures '
             'from a single pass (floor) to the rate table’s transported value '
             f'(hatched); {wall_pairs_per_cap:.2g} internal pairs per wall capture '
             '(EGAF/PGAA line list, Al + C); '
             f'{he_pairs_per_radcap:.2g} pairs per ³He radiative capture (ipc_born, '
             'M1 + E0). External conversion of the 7.7 MeV γ in the wall is not '
             'included and only adds to the wall. (b) Geant4, 10⁹ EAR2-flux '
             'neutrons, 0.5 MIP; 4 121 Al capture γ per pulse (ntof_run_report §6). '
             '(c) HANDOFF_ACCIDENTAL_TIMING §0. An accidental pair is two '
             'uncorrelated legs, and by (b) those are aluminium. A prompt pair is '
             'not identified: the capsule shape fits better than the gas '
             '(χ²/dof 7.4 vs 11.0 perpendicular), but nothing here measures pair '
             'energy.')

    table = pd.DataFrame(
        [dict(panel='a', item=r[0].replace('\n', ' '), value=r[1], value_hi=r[2],
              unit='per neutron entering the capsule',
              source='ipc_aluminium / ipc_born, analytic') for r in rows]
        + [dict(panel='b', item=t[0].replace('\n', ' '), value=t[1], value_hi=t[2],
                unit='per pulse ; aluminium fraction', source=G4['source'])
           for t in G4['trigger_per_pulse']]
        + [dict(panel='c', item=t[0], value=t[2], value_lo=t[3], value_hi=t[4],
                n_pairs=t[1], unit='prompt (true-coincidence) fraction',
                source=TIMING_SOURCE) for t in TIMING])
    save(fig, 'thermal_pair_sources', table)
    return num


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--only', choices=('branching', 'pair_sources'))
    a = ap.parse_args()
    out = {'geant4_quoted': G4, 'measured_timing': dict(rows=TIMING,
                                                        source=TIMING_SOURCE)}
    if a.only in (None, 'branching'):
        out['branching'] = branching()
    if a.only in (None, 'pair_sources'):
        out['pair_sources'] = pair_sources()
    if a.only is None:
        (OUT / 'thermal_sim.json').write_text(json.dumps(out, indent=2,
                                                         default=float))
        print(f'  -> {OUT / "thermal_sim.json"}')
    for k in ('branching', 'pair_sources'):
        if k in out:
            print(f'\n{k}:')
            for kk, v in out[k].items():
                print(f'  {kk:36s} {v}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
