#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_x17_rate_thermal_corrected.py -- the thermal point, corrected.

    ../.venv/bin/python make_x17_rate_thermal_corrected.py

Frame 1 of ``make_x17_rate.py`` (``x17_rate_1_physics``, the slide's X17
statistics spectrum) drawn twice on one axis: the ORIGINAL curve faded, and a
SOLID corrected one where every bin below the p-wave resonance's floor is what
a radiative capture there actually costs, rather than what the table's own
numbers say.

The table's IPC/capture = 2.1e-3 (Viviani et al., the p-wave 1- resonance,
0.17-2 MeV) really does describe that range, and bins inside it (>= 10 eV
here) are untouched.  Below ~2 eV that resonance is gone (CLAUDE.md's IPC-
continuum entry) and two things change at once:

* THIN-TARGET OVERSTATES THE CAPTURE RATE.  The table's He3-captures column is
  N.sigma(n,g), which is only valid while the cell is optically thin.  Below
  ~2 eV the optical depth is tens to ~150 (``ntof_athens_26/make_thermal_sim_figures.py``,
  slide 40's ``thermal_branching`` figure) -- the cell absorbs a growing share
  of neutrons regardless of sigma, and a radiative capture happens with
  probability sigma(n,g)/sigma(abs) per neutron instead.
* THE BRANCHING IS DIFFERENT.  2.1e-3 is IPC/capture for the p-wave resonance;
  below it the only channels are 1+(3S1)->0+ M1 and 0+(1S0)->0+ E0, whose pair
  yield (``sept26_prelim_analysis/ipc_born.py``, M1+E0) is 4.7e-3 pairs per
  RADIATIVE capture -- a different quantity, and larger, but multiplying a much
  smaller capture rate.

THREE BINS, ONE OF THEM SPLIT.  The table bins the neutron spectrum in whole
decades, and the established correction window (``make_thermal_sim_figures.py``'s
own WIN_HI_EV) is 0.44 meV-2 eV, not a decade edge:

* 0.01-0.1 eV and 0.1-1 eV sit entirely inside the window -- corrected in full.
* 1-10 eV straddles 2 eV.  Its 1-2 eV slice (30 % of the bin's log-width,
  assuming the same iso-lethargic flux shape the table and
  ``make_thermal_sim_figures.py`` both already assume) gets the same
  correction; the 2-10 eV slice is left exactly as the table computes it --
  nothing in this repo says what the right branching is there, between the
  resonance's floor and Viviani's own 0.17 MeV.

Suppression: x61, x25 and x7 for the three bins (weakening with energy, since
self-shielding falls off as sigma does) -- the first is close to the Geant4
thermal note's quoted x50-100 for the real capsule (quoted, not rerun here).

X17/IPC (2.5e-2) is left unchanged throughout: no established thermal-specific
version of that fraction exists in this repo, and this figure does not invent
one.

Numbers only -- no arrow, no callout box.  Those go on the slide by hand.
"""
from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import plotstyle as P                                   # noqa: E402
import make_x17_rate as R                                # noqa: E402
from sept26_prelim_analysis import endf as E              # noqa: E402
from sept26_prelim_analysis import ipc_aluminium as IA    # noqa: E402

FIG = os.path.join(REPO, 'ntof_athens_26', 'figures')

#: the table's own header (data/x17_rate_3He.txt) -- unchanged, see docstring.
X17_OVER_IPC = 2.5e-2
IPC_CAPTURE_TABLE = 2.1e-3   # Viviani et al., p-wave 1- resonance, 0.17-2 MeV


# --------------------------------------------------------------------------- #
# the correction, thermal bin only -- same method as
# ntof_athens_26/make_thermal_sim_figures.py (he3_xs / ng_per_neutron),
# reimplemented here in the ~15 lines it actually takes so this figure does
# not import a whole neighbouring deck's script for it.
# --------------------------------------------------------------------------- #
def _he3_xs(mt: int, e_ev: np.ndarray) -> np.ndarray:
    """He3 cross section [b], log-log interpolated. ENDF/B-VIII.0 MF3."""
    d = E.xs('He3', mt)
    x, y = d.E_eV.to_numpy(), d.sigma_b.to_numpy()
    pos = y > 0
    out = np.exp(np.interp(np.log(e_ev), np.log(x[pos]), np.log(y[pos])))
    return np.where(e_ev < x[pos][0], 0.0, out)


def _shielded_ng_per_neutron(e_ev: np.ndarray) -> np.ndarray:
    """P(radiative capture) per neutron entering the cell, self-shielded."""
    s_ng = _he3_xs(102, e_ev)
    s_abs = _he3_xs(103, e_ev) + s_ng + _he3_xs(104, e_ev)
    p_abs = np.array([IA._sphere_absorption(IA.N_HE3_ATB, s) for s in s_abs])
    return p_abs * s_ng / s_abs


def _thin_ng_per_neutron(e_ev: np.ndarray) -> np.ndarray:
    """N.sigma(n,g) per neutron entering the cell -- the table's own formula."""
    return IA.N_HE3_ATB * _he3_xs(102, e_ev)


def _table_row(lo_ev: float, hi_ev: float) -> tuple:
    """``(pulses_per_day, neutrons_per_pulse)`` for one bin of R.TABLE."""
    ppd = None
    for line in open(R.TABLE):
        s = line.strip()
        if s.startswith('#pulses'):
            ppd = float(s.split(':')[1])
        elif s and not s.startswith('#'):
            f = s.split()
            if np.isclose(float(f[0]), lo_ev) and np.isclose(float(f[1]), hi_ev):
                return ppd, float(f[3])
    raise ValueError(f'no row for {lo_ev}-{hi_ev} eV in {R.TABLE}')


#: where the p-wave 1- resonance behind the table's 2.1e-3 is gone (CLAUDE.md;
#: make_thermal_sim_figures.py's WIN_HI_EV) -- the boundary the 1-10 eV bin
#: straddles.
RESONANCE_EDGE_EV = 2.0

#: bins fully inside the established 0.44 meV-2 eV correction window: use the
#: shielded capture rate and ipc_born's M1+E0 channel for the WHOLE bin.
FULLY_CORRECTED_BINS = ((1.0e-2, 1.0e-1), (1.0e-1, 1.0e0))


def _subrange_x17_per_day(lo_ev: float, hi_ev: float, n_pp_sub: float,
                          ppd: float, pairs_per_radcap: float,
                          shielded: bool) -> float:
    g = np.logspace(np.log10(lo_ev), np.log10(hi_ev), 200)
    dens = _shielded_ng_per_neutron(g) if shielded else _thin_ng_per_neutron(g)
    he3_captures_per_pulse = n_pp_sub * dens.mean()      # iso-lethargic in-range
    ipc_per_pulse = he3_captures_per_pulse * pairs_per_radcap
    return ipc_per_pulse * X17_OVER_IPC * ppd


def corrected_bin_x17_per_day(lo_ev: float, hi_ev: float) -> float:
    """The corrected X17/day for one bin of the table.

    Bins entirely inside the 0.44 meV-2 eV window (``FULLY_CORRECTED_BINS``)
    get the shielded capture rate and the ipc_born M1+E0 channel throughout.
    A bin straddling ``RESONANCE_EDGE_EV`` (here, 1-10 eV) is split there:
    the sub-2 eV slice gets the same correction, in proportion to its share
    of the bin's own log-width (iso-lethargic flux, same assumption the rest
    of this table and ``make_thermal_sim_figures.py`` both make); the rest of
    the bin is left exactly as the table computes it -- thin-target, 2.1e-3 --
    because nothing in this repo justifies changing it there.
    """
    ppd, n_pp = _table_row(lo_ev, hi_ev)
    he_pairs_per_radcap = IA.rate_comparison('M1').attrs['he_pairs_per_radcap']

    if (lo_ev, hi_ev) in FULLY_CORRECTED_BINS:
        return _subrange_x17_per_day(lo_ev, hi_ev, n_pp, ppd,
                                     he_pairs_per_radcap, shielded=True)

    if lo_ev < RESONANCE_EDGE_EV < hi_ev:
        frac_below = np.log10(RESONANCE_EDGE_EV / lo_ev) / np.log10(hi_ev / lo_ev)
        below = _subrange_x17_per_day(lo_ev, RESONANCE_EDGE_EV, n_pp * frac_below,
                                      ppd, he_pairs_per_radcap, shielded=True)
        above = _subrange_x17_per_day(RESONANCE_EDGE_EV, hi_ev,
                                      n_pp * (1.0 - frac_below), ppd,
                                      IPC_CAPTURE_TABLE, shielded=False)
        return below + above

    raise ValueError(f'{lo_ev}-{hi_ev} eV is not one of the bins this figure corrects')


#: every bin this figure touches, in table order.
CORRECTED_BINS = FULLY_CORRECTED_BINS + ((1.0, 10.0),)


# --------------------------------------------------------------------------- #
# the figure -- frame 1's axes and spline/point drawing, twice
# --------------------------------------------------------------------------- #
def _spline_and_points(ax, t_mid, t_a, t_b, yv, lit, alpha_mul, point_colors):
    order = np.argsort(t_mid)
    t_mid, t_a, t_b, yv, lit = (a[order] for a in (t_mid, t_a, t_b, yv, lit))
    cs = PchipInterpolator(np.log(t_mid), np.log(yv))
    t_s = np.logspace(np.log10(t_mid.min()), np.log10(t_mid.max()), 800)
    ax.plot(t_s, np.exp(cs(np.log(t_s))), color=P.ACCENT, lw=1.6,
            alpha=0.35 * alpha_mul, zorder=3)
    for m, col, size, lw in ((~lit, point_colors[0], 5.5, 1.2),
                             (lit, point_colors[1], 8.0, 2.0)):
        if not m.any():
            continue
        ax.errorbar(t_mid[m], yv[m],
                    xerr=np.array([t_mid[m] - t_a[m], t_b[m] - t_mid[m]]),
                    fmt='o', ms=size, lw=lw, color=col, ecolor=col,
                    alpha=alpha_mul, capsize=3.0, capthick=lw, zorder=5,
                    markeredgecolor=P.SURFACE, markeredgewidth=0.8)


def draw():
    elo, ehi, y = R.load()
    y_corr = y.copy()
    corrected_idx = []
    for lo_ev, hi_ev in CORRECTED_BINS:
        m = np.isclose(elo, lo_ev) & np.isclose(ehi, hi_ev)
        y_corr[m] = corrected_bin_x17_per_day(lo_ev, hi_ev)
        corrected_idx.append(int(np.flatnonzero(m)[0]))

    t_lo, t_hi = R.t_of_E(ehi) * 1e6, R.t_of_E(elo) * 1e6      # us
    t_mid = 0.5 * (t_lo + t_hi)
    flash_us = R.FLIGHT_M / R.C * 1e6

    P.use()
    fig = plt.figure(figsize=(12.5, 5.25))
    ax = fig.add_axes([0.085, 0.170, 0.895, 0.725])
    ax.set_xscale('log')
    ax.set_xlim(0.05, 4.0e4)
    ax.set_ylim(0.0, 21.0)

    # the same MeV window this frame has always highlighted -- unchanged
    ax.axvspan(R.t_of_E(R.MEV_HI_EV) * 1e6, R.t_of_E(R.MEV_LO_EV) * 1e6,
               color=P.ACCENT, alpha=0.13, zorder=1, lw=0)

    lit = ((t_lo >= R.t_of_E(R.MEV_HI_EV) * 1e6 / 1.01)
           & (t_hi <= R.t_of_E(R.MEV_LO_EV) * 1e6 * 1.01))

    # ---- faded: the original table, thin-target + 2.1e-3 everywhere -------
    _spline_and_points(ax, t_mid, t_lo, t_hi, y, lit, alpha_mul=0.30,
                       point_colors=(P.MUTED, P.ACCENT))

    # ---- solid: corrected thermal bin, everything else identical ----------
    _spline_and_points(ax, t_mid, t_lo, t_hi, y_corr, lit, alpha_mul=1.0,
                       point_colors=(P.MUTED, P.ACCENT))
    # the corrected points themselves, recoloured so the eye finds the bins
    # that moved -- COPPER is the deck's "caution / annotation accent"
    for i in corrected_idx:
        ax.errorbar([t_mid[i]], [y_corr[i]],
                    xerr=[[t_mid[i] - t_lo[i]], [t_hi[i] - t_mid[i]]],
                    fmt='o', ms=8.5, lw=2.2, color=P.COPPER, ecolor=P.COPPER,
                    capsize=3.2, capthick=2.2, zorder=6,
                    markeredgecolor=P.SURFACE, markeredgewidth=0.9)

    # ---- the flash --------------------------------------------------------
    ax.axvline(flash_us, color=P.INK, lw=1.3, zorder=5)
    ax.text(flash_us * 1.3, 20.4, 'γ flash\n(t = 0)', fontsize=10,
            color=P.INK, ha='left', va='top', fontweight='bold', zorder=6)

    # ---- labels -------------------------------------------------------
    n = R.numbers()
    tm = np.sqrt(n['mev_t'][0] * n['mev_t'][1])
    tt = np.sqrt(n['thermal_t'][0] * n['thermal_t'][1]) * 1e3
    ax.text(tm, 7.4, f"{n['mev']:.0f} X17 / day\n"
            f"{n['mev_frac'] * 100:.0f} % of the whole rate",
            ha='center', va='center', fontsize=12.5, fontweight='bold',
            color=P.ACCENT, zorder=6, linespacing=1.45)
    ax.text(tt, 5.3, f"{n['thermal']:.1f} / day", fontsize=10.5,
            color=P.MUTED, alpha=0.55, ha='center', va='bottom', zorder=6)
    # each corrected point labelled on its own -- staggered in y, the three
    # points sit close together in x (one decade) right at the axis floor
    label_y = {(1.0e-2, 1.0e-1): 1.9, (1.0e-1, 1.0e0): 0.6, (1.0, 10.0): 1.9}
    for (lo_ev, hi_ev), i in zip(CORRECTED_BINS, corrected_idx):
        ax.annotate(f'{y_corr[i]:.3f} / day', xy=(t_mid[i], y_corr[i]),
                    xytext=(t_mid[i], label_y[(lo_ev, hi_ev)]),
                    ha='center', va='bottom', fontsize=10.5, fontweight='bold',
                    color=P.COPPER, zorder=6,
                    arrowprops=dict(arrowstyle='-', color=P.COPPER, lw=0.8,
                                    alpha=0.6))

    # ---- axes ---------------------------------------------------------
    ax.set_xlabel(f'neutron flight time over {R.FLIGHT_M:.1f} m  [µs]'
                  '        (10³ µs = 1 ms)')
    ax.set_ylabel('X17 pairs per day\n(nominal ³He cell)')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', alpha=0.20)
    ax.set_axisbelow(False)
    P.strip(ax)

    top = ax.twiny()
    top.set_xscale('log')
    top.set_xlim(*ax.get_xlim())
    ticks_eV = np.array([1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
    tt_us = R.t_of_E(ticks_eV) * 1e6
    keep = (tt_us > ax.get_xlim()[0]) & (tt_us < ax.get_xlim()[1])
    top.set_xticks(tt_us[keep])
    top.set_xticklabels([R._ev(e) for e in ticks_eV[keep]])
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top.set_xlabel('neutron energy', labelpad=7)
    for side in ('right', 'left', 'bottom'):
        top.spines[side].set_visible(False)

    return fig


def main() -> int:
    import csv
    elo, ehi, y = R.load()
    rows = []
    for lo_ev, hi_ev in CORRECTED_BINS:
        m = np.isclose(elo, lo_ev) & np.isclose(ehi, hi_ev)
        table_val = float(y[m][0])
        corr_val = corrected_bin_x17_per_day(lo_ev, hi_ev)
        rows.append((lo_ev, hi_ev, table_val, corr_val))
        print(f'  {lo_ev:6.3g}-{hi_ev:6.3g} eV   table = {table_val:8.4f} /day'
              f'   corrected = {corr_val:9.5f} /day   x{table_val / corr_val:6.1f}')

    os.makedirs(FIG, exist_ok=True)
    fig = draw()
    base = os.path.join(FIG, 'x17_rate_1_physics_thermal_corrected')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', bbox_inches=fig.bbox_inches, pad_inches=0.0)
    print(f'  -> {base}.png')
    plt.close(fig)

    with open(f'{base}.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['E_lo_eV', 'E_hi_eV', 'x17_per_day_table', 'x17_per_day_corrected',
                    'suppression'])
        for lo_ev, hi_ev, tv, cv in rows:
            w.writerow([lo_ev, hi_ev, tv, cv, tv / cv])
    print(f'  -> {base}.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
