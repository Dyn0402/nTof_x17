#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_x17_rate.py -- where the X17 rate actually is, against where we can look.

    ../.venv/bin/python make_x17_rate.py            # both frames
    ../.venv/bin/python make_x17_rate.py --numbers  # just the arithmetic

Writes ``figures/x17_rate_{1_physics,2_window}.png`` and ``.pdf``, and (with
``--slides``) copies the PNGs into ``slides/assets/img/``.

THE FIGURE IS THE ARGUMENT OF THE WHOLE STATUS SECTION.  Frame 1 is what we
came to measure; frames 2 is the same axis with the front end's dead time drawn
on it.  The talk shows frame 1, spends three slides on the flash, and comes
back to frame 2 -- so the two must be the SAME drawing with one thing added,
exactly like the x17 story builds: same limits, same points, same annotations
in the same places, nothing moving between the two.  Both verdict labels sit
UNDER the curve -- the peak is the tallest thing on the figure and a label
above it lands on the interpolation.

WHAT IS PLOTTED
---------------
``data/x17_rate_3He.txt`` -- Dylan's December 2025 rate calculation for the
³He cell (``X17CalculationParser``; the original lives at
``/media/dylan/data/x17/calculation_tables/results_3He`` and the plotting
prototype is the repository root's ``neutron_energy_vs_flight_time.py``).  One
row per decade of neutron energy, and the column taken here is the last one,
``X17 [1/day]``: EAR2 flux -> ³He(n,γ) capture -> internal pair conversion at
2.1e-3 per capture -> X17 at 2.5e-2 per IPC pair, at 1.35e17 protons/day.

The x axis is NEUTRON FLIGHT TIME, not energy, because that is the axis the
detector lives on -- the flash sets t = 0 and everything downstream (dead time,
gate, trigger window) is a time.  Energy rides on top as a second scale.  The
conversion is the relativistic one from ``neutron_energy_vs_flight_time.py``
over the EAR2 flight path, and it is reproduced here rather than imported so
this script runs from the deck directory alone.

TWO THINGS THIS FIGURE IS NOT
-----------------------------
* Not a yield projection for this run.  It is a per-day rate for a nominal
  cell, and the deck's framing decision of 2026-08-10 (RUNNING_ORDER.md) is
  that this talk makes no claim about physics reach.  What the figure is used
  for is the RATIO between two windows, which is a property of the neutron
  spectrum and the capture cross-section and survives everything the absolute
  normalisation does not.
* Not the as-built capsule.  The calculation is for a 40 mm, 500 atm ³He cell
  in 0.5 mm Al + 1.2 mm CF; what was built is a Ø20 mm bore with 40 mm of gas
  at 500 bar in 0.6 mm Al + 0.9 mm CFRP.  The gas column is the same 40 mm and
  the walls are within ~30 %, so the shape is right and the normalisation is a
  nominal one.  Said on the slide, not hidden here.

THE DEAD BAND (frame 2)
-----------------------
Two measurements, both in STATUS_PLAN.md §1.2-1.3:

* run_57's flash-recovery HV map -- at the production operating point (resist
  520-540 V) the per-chamber recovery runs 0.5-8.9 ms, and it is a two-decade
  monotonic function of the resist voltage above that.
* run_79 beam data -- the earliest reconstructed track anywhere is 0.993 ms.

So the band is drawn as a firm edge at 1 ms (nothing has ever been recorded
before it) and a hatched continuation to 9 ms (the slowest chamber at the top
of the production window is still coming back inside the thermal peak).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plotstyle as P  # noqa: E402


def _note(fig, text, x, y, chars):
    """P.note, but inside a canvas that is not going to be tight-cropped."""
    import textwrap
    fig.text(x, y, '\n'.join(textwrap.wrap(text, chars)), ha='left', va='top',
             fontsize=9.5, color=P.MUTED)

FIG = os.path.join(HERE, 'figures')
SLIDES = os.path.join(HERE, 'slides', 'assets', 'img')
TABLE = os.path.join(HERE, 'data', 'x17_rate_3He.txt')

# --- the flight path, and the neutron ------------------------------------
FLIGHT_M = 19.5              # EAR2 target-to-station, the calculation's own
C = 299792458.0
M_N_EV = 939565420.5         # neutron rest energy [eV]

# --- the two windows the talk compares -----------------------------------
# "MeV" is the 0.1-10 MeV pair of decades: the two bins that carry the rate.
MEV_LO_EV, MEV_HI_EV = 1.0e5, 1.0e7
# "thermal" is the bin that arrives after the front end is back.
TH_LO_EV, TH_HI_EV = 1.0e-2, 1.0e-1

# --- the front end, measured (STATUS_PLAN.md 1.2-1.3) --------------------
DEAD_FIRM_MS = 1.0           # nothing recorded before 0.993 ms, ever
DEAD_SOFT_MS = 9.0           # slowest chamber at the top of the production HV


def t_of_E(E_eV):
    """Relativistic neutron flight time [s] over FLIGHT_M."""
    E = np.asarray(E_eV, float)
    gamma = E / M_N_EV + 1.0
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    return FLIGHT_M / (np.clip(beta, 1e-12, 1 - 1e-15) * C)


def load():
    """The first table in the file: one row per decade of neutron energy.

    The file carries a second, commented-out table (a single 0.2-2 MeV band)
    and a totals line; both start with '#' and are skipped, which is also why
    this reads the columns positionally instead of by header -- the header is
    two comment lines and the units line repeats names.
    """
    rows = []
    for line in open(TABLE):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        f = s.split()
        rows.append((float(f[0]), float(f[1]), float(f[-1])))  # elow, eup, X17/day
    rows.sort()
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


def _sum_between(elo, ehi, y, lo, hi):
    m = (elo >= lo * 0.999) & (ehi <= hi * 1.001)
    return float(y[m].sum())


def numbers():
    elo, ehi, y = load()
    tot = float(y.sum())
    mev = _sum_between(elo, ehi, y, MEV_LO_EV, MEV_HI_EV)
    th = _sum_between(elo, ehi, y, TH_LO_EV, TH_HI_EV)
    return dict(
        total=tot, mev=mev, thermal=th,
        mev_frac=mev / tot, thermal_frac=th / tot, ratio=mev / th,
        mev_t=(t_of_E(MEV_HI_EV) * 1e6, t_of_E(MEV_LO_EV) * 1e6),   # us
        thermal_t=(t_of_E(TH_HI_EV) * 1e3, t_of_E(TH_LO_EV) * 1e3),  # ms
        flash_ns=FLIGHT_M / C * 1e9)


def draw(window: bool):
    """One frame.  ``window`` adds the dead band and moves the accent.

    The accent colour is the rhetoric: in frame 1 it is on the MeV decades
    (the rate we came for), in frame 2 it has moved to the thermal bin (the
    rate we could actually record) and the MeV decades are left hollow inside
    the dead band.  Same bins, same limits, same annotation positions.
    """
    elo, ehi, y = load()
    n = numbers()

    # energy bins -> time bins.  High energy is EARLY, so the edges swap.
    t_lo, t_hi = t_of_E(ehi) * 1e6, t_of_E(elo) * 1e6          # us
    flash_us = n['flash_ns'] / 1e3

    P.use()
    # 2.38:1 -- the MEASURED aspect of a figure-solo hole with a caption AND
    # the .figsrc provenance line under it (probe render, 2026-08-20; it was
    # 2.225:1 before the provenance moved into the markup).  Saved with the
    # canvas named explicitly, because savefig.bbox is 'tight' in plotstyle.
    fig = plt.figure(figsize=(12.5, 5.25))
    # the title block is drawn by hand above the axes: the top axis carries
    # the energy scale and its own label, and a matplotlib title would land
    # on top of them
    ax = fig.add_axes([0.085, 0.170, 0.895, 0.725])
    ax.set_xscale('log')
    # LINEAR in y (2026-08-20, Dylan).  A log axis gives the six decades below
    # the peak the same visual weight as the peak itself, which is the exact
    # opposite of this slide's sentence -- on a linear axis the two MeV
    # decades ARE the figure and the rest is a floor, which is what 79 % means.
    # The cost is the eV trough (0.1/day) collapsing onto the axis; that is
    # the honest picture of it, and its numbers are on the previous axis in
    # every earlier version of this figure.
    ax.set_xlim(0.05, 4.0e4)          # 50 ns .. 40 ms
    ax.set_ylim(0.0, 21.0)

    # ---- the dead band, first, so everything else sits on top of it -----
    if window:
        ax.axvspan(flash_us, DEAD_FIRM_MS * 1e3, color=P.BAND_DEAD, alpha=0.15,
                   zorder=0, lw=0)
        # the soft edge is a lighter wash of the same red, not a hatch: a
        # hatch at this width fights the bars it sits behind and reads as the
        # louder statement of the two, which is backwards
        ax.axvspan(DEAD_FIRM_MS * 1e3, DEAD_SOFT_MS * 1e3, color=P.BAND_DEAD,
                   alpha=0.06, zorder=0, lw=0)
        ax.axvline(DEAD_FIRM_MS * 1e3, color=P.BAND_DEAD, lw=1.5, ls='--',
                   zorder=4)

    # ---- the highlighted window ------------------------------------------
    # A band, not a recoloured bar: with points and an interpolation there is
    # no bar to recolour, and the band is what the sentence is about anyway.
    if window:
        ax.axvspan(t_of_E(TH_HI_EV) * 1e6, t_of_E(TH_LO_EV) * 1e6,
                   color=P.ACCENT, alpha=0.13, zorder=1, lw=0)
    else:
        ax.axvspan(t_of_E(MEV_HI_EV) * 1e6, t_of_E(MEV_LO_EV) * 1e6,
                   color=P.ACCENT, alpha=0.13, zorder=1, lw=0)

    # ---- the spectrum: points at the bins, and Dylan's interpolation -----
    # This is `plot_spectrum_vs_time` from the repository root's
    # neutron_energy_vs_flight_time.py, in the deck's colours: one marker per
    # energy decade with the exact bin width as an asymmetric x error bar, and
    # a faint log-log cubic spline through them.  The spline is a reading aid
    # and nothing is derived from it -- ten points over six decades, so it
    # can and does overshoot between them.
    t_mid = 0.5 * (t_lo + t_hi)
    order = np.argsort(t_mid)
    t_mid, t_a, t_b, yv = t_mid[order], t_lo[order], t_hi[order], y[order]

    # PCHIP, not a cubic spline (2026-08-20).  On the old log y axis a cubic
    # overshot the 17.9/day peak to ~23 and it barely showed; on a LINEAR axis
    # that overshoot is a 30 % hump above the highest measured point, sitting
    # exactly where the eye reads the headline.  PCHIP is shape-preserving:
    # it cannot rise above the points it passes through.
    cs = PchipInterpolator(np.log(t_mid), np.log(yv))
    t_s = np.logspace(np.log10(t_mid.min()), np.log10(t_mid.max()), 800)
    ax.plot(t_s, np.exp(cs(np.log(t_s))), color=P.ACCENT, lw=1.6, alpha=0.35,
            zorder=3)

    lit = ((t_a >= t_of_E(MEV_HI_EV) * 1e6 / 1.01)
           & (t_b <= t_of_E(MEV_LO_EV) * 1e6 * 1.01)) if not window else \
          ((t_a >= t_of_E(TH_HI_EV) * 1e6 / 1.01)
           & (t_b <= t_of_E(TH_LO_EV) * 1e6 * 1.01))
    for m, col, size, lw in ((~lit, P.MUTED, 5.5, 1.2),
                             (lit, P.ACCENT, 8.0, 2.0)):
        if not m.any():
            continue
        ax.errorbar(t_mid[m], yv[m],
                    xerr=np.array([t_mid[m] - t_a[m], t_b[m] - t_mid[m]]),
                    fmt='o', ms=size, lw=lw, color=col, ecolor=col,
                    capsize=3.0, capthick=lw, zorder=5,
                    markeredgecolor=P.SURFACE, markeredgewidth=0.8)

    # ---- the flash ------------------------------------------------------
    ax.axvline(flash_us, color=P.INK, lw=1.3, zorder=5)
    # at the TOP of its line, not the bottom: on the linear axis the bottom
    # left corner is where the 10-100 MeV point and its decade-wide error bar
    # sit, and the label landed on them
    ax.text(flash_us * 1.3, 20.4, 'γ flash\n(t = 0)', fontsize=10,
            color=P.INK, ha='left', va='top', fontweight='bold', zorder=6)

    # ---- the two verdicts ----------------------------------------------
    tm = np.sqrt(n['mev_t'][0] * n['mev_t'][1])
    tt = np.sqrt(n['thermal_t'][0] * n['thermal_t'][1]) * 1e3
    # BOTH frames put the MeV verdict UNDER the curve, in the same place
    # (Dylan, 2026-08-19: "move the 79%... text below the curve so it doesn't
    # block").  The peak is the tallest thing on the figure and a label above
    # it sat on the interpolation; the trough under it is empty, and inside
    # the shaded window the label needs no arrow -- the band is the pointer.
    if not window:
        ax.text(tm, 7.4, f"{n['mev']:.0f} X17 / day\n"
                f"{n['mev_frac'] * 100:.0f} % of the whole rate",
                ha='center', va='center', fontsize=12.5, fontweight='bold',
                color=P.ACCENT, zorder=6, linespacing=1.45)
        ax.text(tt, 5.3, '4.4 / day', fontsize=10.5, color=P.MUTED,
                ha='center', va='bottom', zorder=6)
    else:
        ax.text(tm, 7.4, f"{n['mev']:.0f} / day,\nunreachable",
                ha='center', va='center', fontsize=12.5, fontweight='bold',
                color=P.ACCENT, alpha=0.85, zorder=6, linespacing=1.45)
        ax.annotate(f"{n['thermal']:.1f} / day — "
                    f"{n['thermal_frac'] * 100:.0f} %,\nand we can record it",
                    xy=(tt, 5.1), xytext=(tt, 11.4), ha='center', va='bottom',
                    fontsize=12.5, fontweight='bold', color=P.ACCENT,
                    arrowprops=dict(arrowstyle='-|>', color=P.ACCENT,
                                    lw=1.6, shrinkA=5, shrinkB=3), zorder=6)
        # the two dead-band labels ride along the top, inside their own band
        ax.text(np.sqrt(flash_us * DEAD_FIRM_MS * 1e3), 19.4,
                'DREAM front end blind', fontsize=12.5, fontweight='bold',
                color=P.BAND_DEAD, ha='center', va='center', zorder=6)
        # knocked out of the bar behind it, the way make_timeline handles
        # its SPS label: there is no empty place in this band to put it
        ax.text(DEAD_FIRM_MS * 1.06e3, 0.45,
                'still coming back\nto 9 ms', fontsize=8.8,
                color=P.BAND_DEAD, ha='left', va='bottom', zorder=6,
                linespacing=1.35,
                bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=1.6))

    # ---- axes -----------------------------------------------------------
    ax.set_xlabel(f'neutron flight time over {FLIGHT_M:.1f} m  [µs]'
                  '        (10³ µs = 1 ms)')
    ax.set_ylabel('X17 pairs per day\n(nominal ³He cell)')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', alpha=0.20)
    ax.set_axisbelow(False)
    P.strip(ax)

    # ---- energy on top --------------------------------------------------
    top = ax.twiny()
    top.set_xscale('log')
    top.set_xlim(*ax.get_xlim())
    ticks_eV = np.array([1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
    tt_us = t_of_E(ticks_eV) * 1e6
    keep = (tt_us > ax.get_xlim()[0]) & (tt_us < ax.get_xlim()[1])
    top.set_xticks(tt_us[keep])
    top.set_xticklabels([_ev(e) for e in ticks_eV[keep]])
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top.set_xlabel('neutron energy', labelpad=7)
    for side in ('right', 'left', 'bottom'):
        top.spines[side].set_visible(False)

    # NO title block on the figure: both frames live on figure-solo slides
    # whose own title bar says the same thing, and the deck's rule is that
    # a figure never repeats its slide's title.  The height it would have
    # taken goes to the plot instead.

    # NO provenance paragraph on the canvas either (2026-08-20, Dylan:
    # "remove the small text at the bottom and put it in HTML so that I can
    # edit or remove it later").  It is now the .figsrc <div> under each
    # slide's caption; the height it used to take goes to the plot.  Keep
    # _note() around -- it is what would be used for a figure that has to
    # travel outside the deck.
    return fig


def _ev(e):
    if e >= 1e6:
        return f'{e / 1e6:g} MeV'
    if e >= 1e3:
        return f'{e / 1e3:g} keV'
    if e >= 1:
        return f'{e:g} eV'
    return f'{e:g} eV'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--numbers', action='store_true')
    ap.add_argument('--slides', action='store_true')
    args = ap.parse_args()

    n = numbers()
    print(f'  total          {n["total"]:.1f} X17/day')
    print(f'  0.1-10 MeV     {n["mev"]:.1f} /day  = {n["mev_frac"] * 100:.0f} %'
          f'   arriving {n["mev_t"][0]:.2f}-{n["mev_t"][1]:.2f} us')
    print(f'  0.01-0.1 eV    {n["thermal"]:.2f} /day = '
          f'{n["thermal_frac"] * 100:.0f} %   arriving '
          f'{n["thermal_t"][0]:.1f}-{n["thermal_t"][1]:.1f} ms')
    print(f'  MeV / thermal  x{n["ratio"]:.1f}')
    print(f'  gamma flash    {n["flash_ns"]:.1f} ns')
    if args.numbers:
        return

    os.makedirs(FIG, exist_ok=True)
    for window, tag in ((False, '1_physics'), (True, '2_window')):
        fig = draw(window)
        base = os.path.join(FIG, f'x17_rate_{tag}')
        for ext in ('png', 'pdf'):
            # savefig.bbox is 'tight' in plotstyle, and None falls back to it
            fig.savefig(f'{base}.{ext}', bbox_inches=fig.bbox_inches,
                        pad_inches=0.0)
        print(f'  -> {base}.png')
        if args.slides:
            os.makedirs(SLIDES, exist_ok=True)
            import shutil
            shutil.copyfile(f'{base}.png',
                            os.path.join(SLIDES, f'x17_rate_{tag}.png'))
        plt.close(fig)


if __name__ == '__main__':
    main()
