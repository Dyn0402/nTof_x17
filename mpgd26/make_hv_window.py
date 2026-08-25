#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_hv_window.py -- the dead time, put ON the beam's own clock.

    ../.venv/bin/python make_hv_window.py                 # every option, every frame
    ../.venv/bin/python make_hv_window.py --numbers       # just the arithmetic
    ../.venv/bin/python make_hv_window.py --variant a --shape wide
    ../.venv/bin/python make_hv_window.py --contact       # + a review contact sheet

Writes ``figures/hv_window_{variant}_{shape}_{i}_{tag}.png`` and ``.pdf``.  It
does NOT copy anything into ``slides/assets/img`` unless asked with
``--slides``: the deck is being edited elsewhere and these are candidates.

WHY THIS FIGURE EXISTS
----------------------
Slide 21's right panel (``status_deadtime_detA.png``, make_flash_slides.py) is
recovery time against avalanche charge, one point per amplification voltage.
It is a good plot and it carries a real result -- recovery is *roughly
proportional* to the charge the flash delivers, over a factor 20 in charge --
but it asks the audience to hold two axes neither of which is the axis the
talk is on, and then to map the answer onto the *previous* slide's flight-time
axis themselves.  Dylan, 2026-08-23: keep it as backup, and put the recovery
time where it belongs -- as a cut on the X17-rate-versus-time-of-flight plot,
one frame per HV setpoint.

So the drawing is two panels that share a story and, in variant B, an axis:

  strip (short)  what sets the recovery time: the charge, and it is linear.
  main  (tall)   what the recovery time costs: everything to the left of it.

and the build walks the voltage DOWN -- 560 V first, where the front end is
blind for 13.9 ms and essentially the whole spectrum is behind the shading,
then 550, 540, 530, 520 -- with 540 V, where we ran, called out.

TWO VARIANTS, PICK ONE (they are the same main panel)
-----------------------------------------------------
A   the strip is recovery [ms] against charge [nC] on LINEAR axes, with a
    straight line through it.  "Linear" is then something the eye reads in one
    second, which is the whole job of the strip.  The link to the main panel
    is a dotted leader drawn from the highlighted point down to the dead edge
    below it -- an axes-to-axes annotation, so it stays exact.
B   the strip SHARES the main panel's time axis: charge [nC] against the
    recovery time that charge buys, so the highlighted point sits vertically
    above the edge it produces and the leader is a plumb line.  The link is
    exact by construction, but the proportionality is now a slope-1 line on
    log-log over a factor 20, which is a weaker sentence.

THE NUMBERS
-----------
recovery/charge  run_57 detector A, one sub-run per 2 V from 520 to 580 V:
                 charge from the resistive-layer HV supply current, recovery
                 from the flash-random probe, the same sub-runs joined by
                 name.  make_flash_slides.detA_charge_recovery(), which is
                 what slide 21's right panel already plots.
rate vs flight   make_x17_rate.load() -- Dylan's December 2025 ³He rate
                 calculation, one row per decade of neutron energy, on the
                 relativistic flight time over EAR2's 19.5 m.

WHAT SURVIVES A CUT.  The bins are decades of neutron energy and the marker on
the plot carries the decade as its error bar, so a cut that lands mid-bin has
to split one.  This splits it LOG-UNIFORMLY IN TIME (equivalently: uniformly
in log energy inside the decade), which is the same reading of the table the
plotted markers already make.  It is an interpolation inside one bin and
nothing else in the deck depends on it; at 540 V it moves the answer by under
a percentage point either way.  Rounded to 0.1 % on the figure for that
reason, and never quoted finer.

The absolute per-day rate is for a NOMINAL cell and this talk makes no reach
claim (RUNNING_ORDER.md, 2026-08-10) -- what the frames use is the FRACTION
that survives each cut, which is a property of the neutron spectrum and the
capture cross-section and survives everything the normalisation does not.

WHAT THIS FIGURE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not multiply the surviving fraction by a track yield to find an
optimum voltage.  run_55's resist scan has det A's track rate in the 6-14 ms
window at 1.5 / 2.0 / 3.1 / 8.1 / 12.3 % for 520 / 530 / 540 / 550 / 560 V
(mx_july_beam_qa/calib/25_hv_scan_summary.json), and the temptation is to
fold the two curves together.  Don't: that window is itself inside the
recovery at the top of the scan, so the yield number is not independent of
the quantity being cut with -- at 560 V the front end is blind for 13.9 ms
and the 6-14 ms rate is the HIGHEST of the scan, which is by itself proof
that the two axes are entangled.  The gain trade stays a sentence the speaker
says (the deck already quotes the ~4x), not a curve on this figure.
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

import plotstyle as P                 # noqa: E402
import make_x17_rate as X             # noqa: E402
import make_flash_slides as F         # noqa: E402

# The yield half of the trade is an ANALYSIS, not a deck figure: the gas map,
# the electronics ledger and the figure of merit live in
# ntof_july_analysis/hv_tradeoff/ and have their own report.  The deck imports
# the numbers rather than re-deriving them, so the slide and the report can
# never disagree.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'ntof_july_analysis',
                                'hv_tradeoff'))
import hv_tradeoff as T               # noqa: E402

FIG = os.path.join(HERE, 'figures')
SLIDES = os.path.join(HERE, 'slides', 'assets', 'img')

OP_RESIST = 540                       # where we ran
# The build order Dylan asked for: start where the gain wants to be, walk down.
FRAMES = (560, 550, 540, 530, 520)
# ...and then come back to the setpoint (Dylan, 2026-08-24: "repeat the 540 V
# working last so that I can show we chose this and discuss a bit").  The
# repeat is the same drawing as frame 3 with one difference that carries the
# whole point: by then every other voltage's edge is on the axis behind it, so
# the frame says *of all of these* rather than *this one*.
SEQUENCE = FRAMES + (OP_RESIST,)

# figure holes, measured 2026-08-23 by the probe recipe in slides/NOTES.md
# ("Measuring the hole"), on a slide with a kicker, a .title-sm, and NO
# .caption and NO .figsrc -- which is the point of the rebuild.
SHAPES = {
    'wide': (12.5, 6.38),             # .figure-solo, text-free   1.961 : 1
    'col': (6.60, 7.10),              # right column of .cols-2   0.930 : 1
}


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #

_RESULTS = {}


def analysis():
    """hv_tradeoff.results(), read once -- it opens several JSON products."""
    if not _RESULTS:
        _RESULTS.update(T.results())
    return _RESULTS


def hv_points():
    """{V: (charge nC, recovery ms)} for every run_57 sub-run of detector A."""
    return {int(round(v)): (q, ms) for q, ms, v in F.detA_charge_recovery()}


def surviving(t_cut_us):
    """Fraction of the X17 rate arriving after ``t_cut_us``.

    Whole bins count whole; the bin the cut lands in is split log-uniformly in
    time (see the module docstring).  Returns (fraction, rate per day).
    """
    elo, ehi, y = X.load()
    t_lo, t_hi = X.t_of_E(ehi) * 1e6, X.t_of_E(elo) * 1e6      # us, low..high
    tot = float(y.sum())
    kept = 0.0
    for a, b, w in zip(t_lo, t_hi, y):
        if t_cut_us <= a:
            kept += w
        elif t_cut_us < b:
            kept += w * np.log(b / t_cut_us) / np.log(b / a)
    return kept / tot, kept


def numbers():
    hv = hv_points()
    out = []
    for v in FRAMES:
        q, ms = hv[v]
        frac, rate = surviving(ms * 1e3)
        out.append(dict(volts=v, charge_nC=q, recovery_ms=ms,
                        frac=frac, rate=rate))
    return out


# --------------------------------------------------------------------------- #
# the strip: what sets the recovery time
# --------------------------------------------------------------------------- #

def _strip(ax, variant, volts, shape):
    """The recovery-vs-charge result, cut down to its one sentence.

    Every sub-run is drawn, because the scatter IS the error bar here (the
    recovery is quantised to the probe's log-time bins, which is the vertical
    stepping), but only the five build voltages are labelled and only the
    frame's own point is lit.
    """
    hv = hv_points()
    q = np.array([hv[v][0] for v in sorted(hv)])
    m = np.array([hv[v][1] for v in sorted(hv)])
    small = 10.0 if shape == 'wide' else 9.2

    ok = m > 0.25                     # the prompt floor is a limit, not a point

    if variant == 'a':
        # LINEAR axes.  The claim is "proportional", and proportional is a
        # straight line through the origin on linear paper and nothing at all
        # on log paper.
        c = np.polyfit(q[ok], m[ok], 1)
        xs = np.array([0.0, q.max() * 1.06])
        ax.plot(xs, np.polyval(c, xs), color=P.MUTED, lw=1.3, ls='--', zorder=2)
        ax.plot(q, m, 'o', ms=4.2, color=P.DET_COLOR['A'], alpha=0.55, lw=0,
                zorder=3, markeredgecolor='none')
        ax.set_xlim(0, q.max() * 1.06)
        ax.set_ylim(0, m.max() * 1.14)
        ax.set_xlabel('avalanche charge per beam pulse  [nC]', fontsize=small,
                      labelpad=2)
        ax.set_ylabel('recovery\n[ms]', fontsize=small, linespacing=1.15)
        ax.tick_params(labelsize=small - 0.8)
        ax.grid(axis='y', alpha=0.18)
        xh, yh = hv[volts]
    else:
        # SHARED TIME AXIS.  x is the recovery time, in the main panel's own
        # microseconds, so the lit point stands directly above the edge it
        # makes.  y is the charge that bought it.
        ax.set_xscale('log')
        ax.set_yscale('log')
        cl = np.polyfit(np.log10(m[ok]), np.log10(q[ok]), 1)
        xs = np.logspace(np.log10(m.min() * 0.8), np.log10(m.max() * 1.3), 30)
        ax.plot(xs * 1e3, 10 ** np.polyval(cl, np.log10(xs)), color=P.MUTED,
                lw=1.3, ls='--', zorder=2)
        ax.plot(m * 1e3, q, 'o', ms=4.2, color=P.DET_COLOR['A'], alpha=0.55,
                lw=0, zorder=3, markeredgecolor='none')
        # Compressed, and read on the RIGHT (Dylan, 2026-08-24).  The strip
        # has to span the whole canvas because its x axis is the main panel's,
        # but its DATA occupies only the right third; ticks and label on the
        # left would sit on top of the yield panel that now fills the
        # empty end.  The labelled ticks start at 10^2 as asked -- the floor
        # itself is 28 nC, not 100, because 520 / 530 / 540 V put 35 / 74 /
        # 93 nC on the chamber and a hard 10^2 floor drops three of the five
        # build points off the panel.
        ax.set_ylim(28, 1200)
        ax.set_yticks([1e2, 1e3])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_ylabel('charge  [nC]', fontsize=small, labelpad=4)
        ax.tick_params(labelsize=small - 0.8, labelbottom=False)
        ax.grid(axis='y', alpha=0.18)
        xh, yh = hv[volts][1] * 1e3, hv[volts][0]

    # The other four build voltages are marked but NOT labelled: their names
    # are already on the main panel, on the edges they left behind, and five
    # labels inside a 1.5 in strip is four more than it can hold.  What the
    # audience needs from this panel is the trend and where the frame's own
    # point sits on it.
    for v in FRAMES:
        qq, mm = hv[v]
        x, y = (qq, mm) if variant == 'a' else (mm * 1e3, qq)
        if v == volts:
            continue
        ax.plot(x, y, 'o', ms=6.0, color=P.DET_COLOR['A'], zorder=4,
                markeredgecolor=P.SURFACE, markeredgewidth=0.8)

    ax.plot(xh, yh, marker='*', ms=17, color=P.DET_COLOR['A'], zorder=6,
            markeredgecolor=P.BAND_DEAD, markeredgewidth=1.5)
    # 520 V sits hard against the y axis, and a centred label there hangs off
    # the canvas -- so the label goes beside the star, not over it, whenever
    # the star is in the left tenth of the strip
    x0, x1 = ax.get_xlim()
    frac = ((xh - x0) / (x1 - x0) if ax.get_xscale() == 'linear'
            else (np.log10(xh / x0) / np.log10(x1 / x0)))
    beside = frac < 0.06
    ax.annotate(f'{volts} V', xy=(xh, yh),
                xytext=(11, 2) if beside else (0, 9),
                textcoords='offset points', fontsize=small,
                ha='left' if beside else 'center',
                va='center' if beside else 'baseline',
                color=P.BAND_DEAD, fontweight='bold', zorder=6)

    P.strip(ax)
    # Short, because in variant a this headline shares its row with the yield
    # strip's.  The two are written as a pair: one cost each.
    lead = ('the charge sets how long we are blind'
            if variant == 'a' else
            'det A  ·  run_57  ·  one sub-run per 2 V')
    # variant b's scoreboard sits in the strip's empty left end, so its
    # headline goes to the right end where the points are
    ax.text(0.0 if variant == 'a' else 1.0, 1.06, lead, transform=ax.transAxes,
            ha='left' if variant == 'a' else 'right', va='bottom',
            fontsize=small, color=P.MUTED)
    return xh, yh


def _eff_panel(ax, volts, shape, compact=False):
    """What the gain buys: the chamber's OWN efficiency against voltage.

    Dylan, 2026-08-25: *"this plot in the top left needs to be exactly the
    detector efficiency measured by the cosmic bench, which is relatively flat
    though decreases a bit at low voltage -- translated of course to 90/10
    Ar/iso."*  Until then the panel carried run_55's MIP tracks per trigger,
    which is a reconstructability ladder and not an efficiency at all; it fell
    100 -> 29 % across 560 -> 540 V and said the opposite of what the bench
    measured.  That ladder is backup only now.

    WHICH BENCH SCAN.  The **27 June saturday det3 scan**, both interleaved
    passes -- the only one that reaches below the plateau (0.49 at 425 V, up to
    0.81 by 455 V).  The 22 June overnight scan starts at 450 V, already flat,
    so it cannot show a turn-on; it agrees on the plateau's flatness and sits
    ~10 points higher only because det3 was in the bottom slot there, half the
    M3 lever arm into the same fixed 5 mm box.  The saturday scan is also the
    run ``mesh_ladder.csv`` comes from, so this curve and the gain slope that
    maps it are the same measurement.

    HOW IT IS PLACED on the n_TOF axis: the full ledger, not the gas term
    alone -- an efficiency is a threshold quantity, so the CSA range and the
    per-channel noise belong in the shift as much as the gas does.  Both eras
    are drawn, because they differ by 22 V and that is half this panel:
    **solid = production**, after the 23 July noise step, which is where we
    actually ran, and **dashed = July**, run_55's own configuration.  The
    23 July step is the gap between them.

    Measured points are plotted as markers, always.  Where the production
    placement runs off the left of the scan the LINE continues as a straight
    fit to the three lowest points, drawn dashed and faded, and it is labelled
    as an extrapolation on the canvas.
    """
    small = 9.2 if compact else (10.0 if shape == 'wide' else 9.2)
    x0, x1 = 517, 563

    # July first, so production draws over it
    vj, ej, _dj, vjl, ejl, nxj = T.bench_eff_on_ntof_axis('run_55', v_min=x0)
    ax.plot(vjl, np.asarray(ejl) * 100, '-', color=P.MUTED, lw=1.1, ls=(0, (4, 2)),
            alpha=0.85, zorder=2)

    vp, ep, dep, vpl, epl, nxp = T.bench_eff_on_ntof_axis('production', v_min=x0)
    vpl, epl = np.asarray(vpl), np.asarray(epl) * 100
    if nxp:
        ax.plot(vpl[:nxp + 1], epl[:nxp + 1], '-', color=P.DET_COLOR['A'],
                lw=1.4, ls=(0, (2, 2)), alpha=0.5, zorder=3)
    ax.plot(vpl[nxp:], epl[nxp:], '-', color=P.DET_COLOR['A'], lw=1.7, zorder=3)
    ax.errorbar(vp, np.asarray(ep) * 100, yerr=np.asarray(dep) * 100, fmt='o',
                ms=3.6, color=P.DET_COLOR['A'], ecolor=P.DET_COLOR['A'],
                elinewidth=0.9, capsize=0, markeredgecolor=P.SURFACE,
                markeredgewidth=0.5, zorder=4)

    e_here = float(np.interp(volts, vpl, epl))
    ax.plot(volts, e_here, marker='*', ms=17, color=P.DET_COLOR['A'], zorder=6,
            markeredgecolor=P.BAND_DEAD, markeredgewidth=1.5)
    # beside the star, never over it: high on the plateau there is no room
    # above (the headline is there), low on the turn-on there is none below
    # on the plateau the curve is flat, so a label beside the star lands ON
    # the line -- go above it, where the only thing overhead is empty panel
    high = e_here > 75
    ax.annotate(f'{e_here:.0f} %', xy=(volts, e_here),
                xytext=(0, 12) if high else (12, 6),
                textcoords='offset points', fontsize=small,
                ha='center' if high else 'left', va='bottom',
                color=P.BAND_DEAD, fontweight='bold', zorder=6)

    if nxp:
        # the band's own top-left corner is the only empty part of the panel:
        # the extrapolated line runs 35 -> 49 % underneath it
        ax.axvspan(x0, float(vpl[nxp]), color=P.MUTED, alpha=0.07, lw=0, zorder=0)
        ax.text(x0 + 0.8, 112 if compact else 124, 'extrap.', ha='left',
                va='top', fontsize=small - 2.0, color=P.MUTED)

    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 118 if compact else 130)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=small - 0.8, pad=1.5 if compact else 3)
    ax.set_xlabel('amplification voltage  [V]', fontsize=small,
                  labelpad=1 if compact else 2)
    ax.set_ylabel('%' if compact else 'efficiency\n[%]', fontsize=small,
                  linespacing=1.15, labelpad=2 if compact else 4)
    ax.grid(axis='y', alpha=0.18)
    P.strip(ax)
    ax.text(0.0, 1.06,
            'bench efficiency, mapped to 90/10  ·  solid: as we ran  ·  dashed: July'
            if compact else 'and the gain sets what we see',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=small + (0.6 if compact else 0), color=P.MUTED)
    return e_here / 100.0


# --------------------------------------------------------------------------- #
# the main panel: the rate, on the beam's clock, with the cut on it
# --------------------------------------------------------------------------- #

def eff_at(volts):
    """(efficiency, is_extrapolated) at an n_TOF setpoint, production placement.

    One definition, used by the panel and by the scoreboard, so the star and
    the number can never disagree.  Production and not July because that is
    where the campaign ran; the panel draws both so the 22 V between them is
    on the canvas rather than in a footnote.

    The flag matters for exactly one frame: 520 V maps to bench 417 V, below
    the scan's lowest point (425 V), so its number is read off the straight
    continuation and the scoreboard has to say so rather than print it like a
    measurement.
    """
    vm, _e, _de, vl, el, _n = T.bench_eff_on_ntof_axis('production', v_min=517)
    return float(np.interp(volts, vl, el)), bool(volts < vm[0])


def _readout(target, volts, x, y, dy, size, ha='left'):
    """The frame's scoreboard: the voltage, the blindness, what is left.

    Fixed position on every frame -- it is the only thing that changes as the
    build walks the voltage down, so it must not also move while it changes.
    """
    hv = hv_points()
    ms = hv[volts][1]
    frac, _ = surviving(ms * 1e3)
    role = {560: 'where the gain wants to be',
            OP_RESIST: 'where we ran'}.get(volts, '')
    kw = dict(ha=ha, va='top', zorder=9)
    if hasattr(target, 'transAxes'):
        kw['transform'] = target.transAxes
    rel, extrap = eff_at(volts)
    lines = [
        (0.00, f'{volts} V', size + 10.0, 'bold', P.BAND_DEAD),
        (1.55, role, size, 'normal', P.MUTED),
        (2.30, f'blind for {ms:.1f} ms after every flash',
         size + 1.0, 'bold', P.BAND_DEAD),
        (3.00, f'{frac * 100:.1f} % of the X17 rate is left',
         size + 1.0, 'bold', P.ACCENT if frac > 0.01 else P.MUTED),
        (3.70, f'{"~" if extrap else ""}{rel * 100:.0f} % efficient, from the '
         f'bench{" (extrapolated)" if extrap else ""}',
         size + 1.0, 'bold', P.DET_COLOR['A']),
    ]
    for k, txt, fs, weight, col in lines:
        if not txt:
            continue
        target.text(x, y - k * dy, txt, fontsize=fs, fontweight=weight,
                    color=col, **kw)


def _readout_stats(fig, volts, x, y, dy, size):
    """The scoreboard as three short stat rows, for variant b.

    Dylan, 2026-08-24: the large letters are the VOLTAGE, the per cent of the
    X17 rate left, and the track yield -- the last one because in this variant
    the strip is spoken for by the shared time axis, so the yield panel is the
    small inset on the left and the number is what carries it.

    NOT called an efficiency, deliberately (Dylan, 2026-08-24, having gone
    looking for the bench curve and not recognised it).  This is run_55's MIP
    tracks per trigger normalised to its own best point; the bench's actual
    efficiency is FLAT at 91 % across every voltage this panel shows, and
    eff_anyhit is ~100 % at all of them.  Calling the ladder an efficiency puts
    a number on the slide that the bench measurement contradicts.

    Numbers right-aligned in one column, descriptors left-aligned in the next,
    so the block does not re-flow as the digits change between frames.  The
    recovery time is the fourth fact and it is deliberately small: the strip
    puts it on the axis and the main panel draws it as the wall, so writing it
    large would be the third time the same number appears on one canvas.
    """
    hv = hv_points()
    ms = hv[volts][1]
    frac, _ = surviving(ms * 1e3)
    rel, extrap = eff_at(volts)
    role = {560: 'where the gain wants to be',
            OP_RESIST: 'where we ran'}.get(volts, '')

    gap = 0.012
    rows = ((f'{volts} V', role, size + 15.0, P.BAND_DEAD, P.MUTED),
            (f'{frac * 100:.1f} %', 'of the X17 rate left', size + 6.0,
             P.ACCENT if frac > 0.01 else P.MUTED, P.MUTED),
            (f'{"~" if extrap else ""}{rel * 100:.0f} %',
             'efficient  (cosmic bench, extrapolated)' if extrap
             else 'efficient  (cosmic bench)', size + 6.0,
             P.DET_COLOR['A'], P.MUTED))
    for k, (num, lab, fs, cnum, clab) in enumerate(rows):
        yy = y - k * dy
        fig.text(x, yy, num, fontsize=fs, fontweight='bold', color=cnum,
                 ha='right', va='center', zorder=9)
        fig.text(x + gap, yy, lab, fontsize=size, color=clab, ha='left',
                 va='center', zorder=9)
    fig.text(x + gap, y - (len(rows) - 0.42) * dy,
             f'blind for {ms:.1f} ms after every flash', fontsize=size - 0.5,
             color=P.BAND_DEAD, ha='left', va='center', zorder=9)


def _readout_row(fig, volts, left, right, y, size):
    """The scoreboard as ONE line, for the narrow shape.

    A column-shaped main panel has no free corner -- the peak reaches the top
    of the axes and the gamma-flash label owns the only gap -- so in that
    shape the scoreboard lives in the band between the two panels.  Three
    fixed anchors, so the line does not re-flow as the numbers change.
    """
    hv = hv_points()
    ms = hv[volts][1]
    frac, _ = surviving(ms * 1e3)
    role = {560: ' — where the gain wants to be',
            OP_RESIST: ' — where we ran'}.get(volts, '')
    fig.text(left, y, f'{volts} V', fontsize=size + 6.0, fontweight='bold',
             color=P.BAND_DEAD, ha='left', va='center')
    fig.text(left + 0.115, y, role.lstrip(' —'), fontsize=size,
             color=P.MUTED, ha='left', va='center')
    fig.text(right, y + 0.028, f'blind for {ms:.1f} ms after every flash',
             fontsize=size + 0.5, fontweight='bold', color=P.BAND_DEAD,
             ha='right', va='center')
    fig.text(right, y - 0.014, f'{frac * 100:.1f} % of the X17 rate is left',
             fontsize=size + 0.5, fontweight='bold',
             color=P.ACCENT if frac > 0.01 else P.MUTED, ha='right',
             va='center')


def _main(ax, volts, shape, shown):
    """``shown`` = the voltages already built, drawn as spent edges."""
    hv = hv_points()
    t_cut_us = hv[volts][1] * 1e3
    elo, ehi, y = X.load()
    n = X.numbers()
    t_lo, t_hi = X.t_of_E(ehi) * 1e6, X.t_of_E(elo) * 1e6
    flash_us = n['flash_ns'] / 1e3
    big = 12.5 if shape == 'wide' else 10.5

    ax.set_xscale('log')
    ax.set_xlim(0.05, 4.0e4)          # 50 ns .. 40 ms, same as slides 18/22
    ax.set_ylim(0.0, 21.0)

    # ---- the blindness ---------------------------------------------------
    ax.axvspan(flash_us, t_cut_us, color=P.BAND_DEAD, alpha=0.17, zorder=0,
               lw=0)
    ax.axvline(t_cut_us, color=P.BAND_DEAD, lw=2.2, zorder=6)

    # every edge already built stays on the axis as a spent tick, so the
    # audience can see the wall walking left
    for v in shown:
        if v == volts:
            continue
        te = hv[v][1] * 1e3
        col = P.INK if v == OP_RESIST else P.MUTED
        ax.axvline(te, color=col, lw=1.1, ls=':', zorder=5,
                   alpha=0.9 if v == OP_RESIST else 0.55)
        # At MID-HEIGHT, not on the axis (Dylan, 2026-08-24).  Down there the
        # labels sat across the thermal marker and its decade-wide error bar,
        # which is the one measurement on the right of the plot; the middle of
        # the axis is empty at every voltage in the build.  Knocked out of
        # whatever is behind them, the way the main panel's other labels are.
        ax.annotate(f'{v} V', xy=(te, 0.42),
                    xycoords=('data', 'axes fraction'),
                    rotation=90, ha='center', va='center',
                    fontsize=big - 3.0, color=col,
                    fontweight='bold' if v == OP_RESIST else 'normal',
                    bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=1.4,
                              alpha=0.85), zorder=6)

    # ---- the spectrum ----------------------------------------------------
    t_mid = 0.5 * (t_lo + t_hi)
    order = np.argsort(t_mid)
    t_mid, t_a, t_b, yv = t_mid[order], t_lo[order], t_hi[order], y[order]
    cs = PchipInterpolator(np.log(t_mid), np.log(yv))
    t_s = np.logspace(np.log10(t_mid.min()), np.log10(t_mid.max()), 800)
    ax.plot(t_s, np.exp(cs(np.log(t_s))), color=P.ACCENT, lw=1.6, alpha=0.30,
            zorder=3)

    # a bin is drawn lit when the CENTRE of its arrival window clears the cut.
    # Not its trailing edge: at 560 V the thermal decade's tail pokes 220 us
    # past the wall, and lighting the whole marker for that would say "this
    # one survives" while the scoreboard beside it says 0.1 % -- the two have
    # to agree.  Half a decade of a decade-wide bin is the honest threshold.
    lit = t_mid > t_cut_us
    for m_, col, size, lw, al in ((~lit, P.MUTED, 5.5, 1.2, 0.45),
                                  (lit, P.ACCENT, 8.0, 2.0, 1.0)):
        if not m_.any():
            continue
        ax.errorbar(t_mid[m_], yv[m_],
                    xerr=np.array([t_mid[m_] - t_a[m_], t_b[m_] - t_mid[m_]]),
                    fmt='o', ms=size, lw=lw, color=col, ecolor=col, alpha=al,
                    capsize=3.0, capthick=lw, zorder=5,
                    markeredgecolor=P.SURFACE, markeredgewidth=0.8)

    # ---- the flash -------------------------------------------------------
    ax.axvline(flash_us, color=P.INK, lw=1.3, zorder=5)
    ax.text(flash_us * 1.3, 20.4, 'γ flash\n(t = 0)', fontsize=big - 2.5,
            color=P.INK, ha='left', va='top', fontweight='bold', zorder=7)

    # ---- the live edge, named at the edge --------------------------------
    # The scoreboard says what the voltage costs; this says WHICH line is the
    # voltage's, and it is the only label that moves between frames.
    # ALWAYS to the left of the line, never to the right: the label is ~0.3 of
    # the axis wide and the earliest edge in the build (520 V, 0.86 ms) is
    # already at 0.72 of it, so a right-hand label would run off the canvas on
    # every frame -- and left of the line is inside the blindness, which is
    # what the sentence is about anyway.
    ax.annotate(f'{volts} V — the front end wakes up here ',
                xy=(t_cut_us, 0.955), xycoords=('data', 'axes fraction'),
                ha='right', va='top', fontsize=big - 1.5, color=P.BAND_DEAD,
                fontweight='bold', zorder=8,
                bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=1.8,
                          alpha=0.88))

    # the MeV peak is named once, under the curve, in the same place on every
    # frame -- it never comes out from behind the shading, and that is the
    # point of starting at 560 V
    tm = np.sqrt(n['mev_t'][0] * n['mev_t'][1])
    ax.text(tm, 6.9, f"{n['mev_frac'] * 100:.0f} % of the rate is here",
            ha='center', va='center', fontsize=big - 2.0, color=P.BAND_DEAD,
            zorder=7, alpha=0.95,
            bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=1.8,
                      alpha=0.55))

    ax.set_xlabel(f'neutron flight time over {X.FLIGHT_M:.1f} m  [µs]'
                  '        (10³ µs = 1 ms)')
    ax.set_ylabel('X17 pairs per day\n(nominal ³He cell)')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', alpha=0.20)
    ax.set_axisbelow(False)
    P.strip(ax)
    return t_cut_us


def _energy_axis(ax):
    top = ax.twiny()
    top.set_xscale('log')
    top.set_xlim(*ax.get_xlim())
    ticks_eV = np.array([1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
    tt = X.t_of_E(ticks_eV) * 1e6
    keep = (tt > ax.get_xlim()[0]) & (tt < ax.get_xlim()[1])
    top.set_xticks(tt[keep])
    top.set_xticklabels([X._ev(e) for e in ticks_eV[keep]])
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top.set_xlabel('neutron energy', labelpad=7)
    for side in ('right', 'left', 'bottom'):
        top.spines[side].set_visible(False)
    return top


# --------------------------------------------------------------------------- #
# the closing frame: the two costs, multiplied
# --------------------------------------------------------------------------- #

def draw_trade(shape='wide'):
    """One drawing of the whole argument: the product has a maximum.

    Not a frame of the build -- the build's geometry is two strips over the
    beam's clock, and this is a different statement about the same numbers, so
    it gets the whole canvas.  Three curves, each normalised to its own
    maximum, because both factors are relative and their product therefore is
    too: it has a shape and no units.

    The dashed copy is the same product built on the 8-12 ms window.  It is
    shown because it is the window with the statistics, and it is dashed
    because above 550 V it sits inside the chamber's own recovery -- the
    quantity being traded against -- so its top points are suppressed by
    construction.  Where the two disagree, the solid one is right.
    """
    P.use()
    w, h = SHAPES[shape]
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.085, 0.135, 0.895, 0.735])

    v, vis, rel, prod = T.figure_of_merit('b2')
    _, _, _, prod1 = T.figure_of_merit('b1')
    big = 12.5 if shape == 'wide' else 10.5

    ax.plot(v, vis / vis.max(), '-o', color=P.BAND_DEAD, lw=2.0, ms=6,
            markeredgecolor=P.SURFACE, markeredgewidth=0.8, zorder=3)
    ax.plot(v, rel, '-o', color=P.DET_COLOR['A'], lw=2.0, ms=6,
            markeredgecolor=P.SURFACE, markeredgewidth=0.8, zorder=3)
    ax.plot(v, prod / prod.max(), '-o', color=P.ACCENT, lw=3.4, ms=9,
            markeredgecolor=P.SURFACE, markeredgewidth=1.0, zorder=5)
    ax.plot(v, prod1 / prod1.max(), '--', color=P.ACCENT, lw=1.3, alpha=0.55,
            zorder=2)

    # direct labels, never a legend box: three curves, three colours, three
    # sentences, each written where its curve is alone on the axis
    ax.text(521.5, vis[0] / vis.max() - 0.09, 'X17 rate that arrives\nafter the '
            'chamber is alive', color=P.BAND_DEAD, fontsize=big - 1.0,
            ha='left', va='top', fontweight='bold', linespacing=1.35)
    # at the START of its curve, in the empty strip under it: at the other end
    # it runs off the canvas, and the middle is where the dashed copy lives
    ax.text(520.4, 0.02, 'tracks the chamber reconstructs',
            color=P.DET_COLOR['A'], fontsize=big - 1.0, ha='left', va='bottom',
            fontweight='bold')
    kb = int(np.argmax(prod))
    ax.annotate('their product — what the campaign\nwas actually optimising',
                xy=(v[kb], 1.0), xytext=(v[kb] - 1.5, 1.30),
                ha='center', va='bottom', color=P.ACCENT, fontsize=big,
                fontweight='bold', linespacing=1.35,
                arrowprops=dict(arrowstyle='-|>', color=P.ACCENT, lw=1.8,
                                shrinkB=8))

    ax.axvline(OP_RESIST, color=P.INK, lw=1.8, zorder=6)
    ax.text(OP_RESIST - 0.6, 0.03, 'we ran here ', color=P.INK, ha='right',
            va='bottom', fontsize=big, fontweight='bold', zorder=7)

    ax.set_xlim(518.0, 562.5)
    ax.set_ylim(0, 1.52)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '½', 'best'])
    ax.set_xlabel('amplification voltage, Ar/iso 90/10  [V]')
    ax.set_ylabel('relative to its own best')
    ax.grid(axis='y', alpha=0.20)
    P.strip(ax)
    ax.text(0.0, 1.035, 'Detector A. Left: the flash outlasts the thermal '
            'neutrons. Right: no gain, no tracks. The window between them is '
            'about 20 V wide.',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=big - 1.5,
            color=P.MUTED)
    return fig


# --------------------------------------------------------------------------- #
# one frame
# --------------------------------------------------------------------------- #

def draw(volts, variant='a', shape='wide', step=None):
    P.use()
    w, h = SHAPES[shape]
    fig = plt.figure(figsize=(w, h))
    ax_y = None                       # the yield strip, wide + variant a only

    if shape == 'wide':
        # Variant b reads its strip's charge axis on the RIGHT, so both axes
        # stop short of the canvas edge to leave the tick labels room.  What
        # has to match is strip-to-main, not either-to-canvas: they share an x
        # axis and the plumb line only works while their boxes are identical.
        L, R = (0.077, 0.985) if variant == 'a' else (0.077, 0.945)
        # variant a: two strips across the top -- what sets the recovery time
        # on the left, what the gain buys in the middle -- and the scoreboard
        # on the right.  variant b: the strip needs the whole width, because
        # it is the main panel's axis, so the scoreboard moves inside the plot
        # and the yield strip is dropped (there is nowhere left to put it).
        ax_s = fig.add_axes([L, 0.780 if variant == 'a' else 0.815,
                             (R - L) * (0.30 if variant == 'a' else 1.0),
                             0.150 if variant == 'a' else 0.115])
        ax_m = fig.add_axes([L, 0.115, R - L, 0.520])
        if variant == 'a':
            ax_y = fig.add_axes([0.445, 0.780, 0.185, 0.150])
            _readout(fig, volts, R, 0.965, 0.045, 11.0, ha='right')
        else:
            # Variant b's strip has to span the FULL width -- its x axis is the
            # main panel's, and that is the whole point of the variant -- but
            # every recovery time the chamber can reach lands in the last
            # decade and a half of a six-decade axis, so its left two thirds
            # are empty.  That is where the other two things go: the
            # efficiency panel (which is tied to no axis on this canvas) as an
            # opaque inset, and the three numbers beside it.  Nothing overlaps
            # the strip's own data, which starts at x = 0.70 of the canvas.
            # widened 2026-08-24 into the gap that used to sit between it and
            # the numbers; the numbers' own column starts at ~0.43
            ax_y = fig.add_axes([0.088, 0.792, 0.300, 0.140])
            ax_y.set_facecolor(P.SURFACE)
            ax_y.patch.set_alpha(1.0)
            ax_y.set_zorder(4)
            _readout_stats(fig, volts, 0.500, 0.902, 0.058, 11.5)
    else:
        # In the column shape the strip always takes the full width and the
        # scoreboard always goes inside the main panel: there is no free
        # corner beside a 6.6 in strip, and a scoreboard that overlapped its
        # headline is what the first draft did.
        L, R = 0.150, 0.985
        ax_s = fig.add_axes([L, 0.855, R - L, 0.105])
        ax_m = fig.add_axes([L, 0.075, R - L, 0.565])
        _readout_row(fig, volts, L, R, 0.727, 10.5)

    shown = (SEQUENCE[:step + 1] if step is not None
             else FRAMES[:FRAMES.index(volts) + 1])
    if ax_y is not None:
        _eff_panel(ax_y, volts, shape, compact=(variant == 'b'))
    xh, yh = _strip(ax_s, variant, volts, shape)
    t_cut = _main(ax_m, volts, shape, shown)
    _energy_axis(ax_m)

    if variant == 'b':
        # The strip shares the time axis, so the link between the two panels is
        # a plumb line and needs no explanation.
        #
        # It does NOT have to span the whole canvas to do that (Dylan,
        # 2026-08-24: "stop the x axis at the first point on the left").  What
        # the plumb line needs is that a time maps to the same figure x in both
        # axes -- so the strip is cut back to just before its first point and
        # its BOX is moved right by exactly the fraction of the axis that
        # removes, which leaves the mapping identical and hands the empty left
        # end over to the efficiency panel and the numbers for good.
        x0, x1 = ax_m.get_xlim()
        if shape == 'wide':
            t_first = min(ms for _, ms in hv_points().values()) * 1e3 / 1.30
            frac = np.log(t_first / x0) / np.log(x1 / x0)
            box = ax_s.get_position()
            left = box.x0 + (box.x1 - box.x0) * frac
            ax_s.set_position([left, box.y0, box.x1 - left, box.height])
            ax_s.set_xlim(t_first, x1)
        else:
            # the column shape has no efficiency panel to hand the freed space
            # to, so its strip stays full width
            ax_s.set_xlim(x0, x1)
        ax_s.axvline(t_cut, color=P.BAND_DEAD, lw=1.4, ls=':', zorder=1)
        ax_m.annotate('', xy=(t_cut, 21.0), xycoords=ax_m.transData,
                      xytext=(xh, yh), textcoords=ax_s.transData,
                      arrowprops=dict(arrowstyle='-|>', color=P.BAND_DEAD,
                                      lw=1.2, ls=':', alpha=0.75, shrinkA=8,
                                      shrinkB=1), zorder=8)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--numbers', action='store_true')
    ap.add_argument('--variant', default='ab', help='a, b or ab')
    ap.add_argument('--shape', default='wide', help='wide, col or both')
    ap.add_argument('--contact', action='store_true',
                    help='also write a review contact sheet per variant')
    ap.add_argument('--slides', action='store_true',
                    help='copy the PNGs into slides/assets/img (NOT the default:'
                         ' the deck is being edited elsewhere)')
    args = ap.parse_args()

    for r in numbers():
        print(f'  {r["volts"]} V   Q = {r["charge_nC"]:6.1f} nC   '
              f'recovery {r["recovery_ms"]:6.2f} ms   '
              f'surviving {r["frac"] * 100:5.2f} %  ({r["rate"]:.2f} /day)')
    if args.numbers:
        return

    os.makedirs(FIG, exist_ok=True)
    shapes = ('wide', 'col') if args.shape == 'both' else (args.shape,)
    for variant in args.variant:
        for shape in shapes:
            paths = []
            for i, v in enumerate(SEQUENCE, 1):
                fig = draw(v, variant, shape, step=i - 1)
                base = os.path.join(FIG, f'hv_window_{variant}_{shape}_{i}_{v}')
                for ext in ('png', 'pdf'):
                    fig.savefig(f'{base}.{ext}', bbox_inches=fig.bbox_inches,
                                pad_inches=0.0)
                plt.close(fig)
                paths.append(f'{base}.png')
                print(f'  -> {base}.png')
                if args.slides:
                    import shutil
                    os.makedirs(SLIDES, exist_ok=True)
                    shutil.copyfile(f'{base}.png', os.path.join(
                        SLIDES, f'hv_window_{i}_{v}.png'))
            # the closing frame does not depend on the variant -- it is the
            # same two costs multiplied either way -- but it is written once
            # per variant so a --variant b run ships a complete set
            if True:
                fig = draw_trade(shape)
                base = os.path.join(
                    FIG, f'hv_window_{variant}_{shape}_{len(SEQUENCE) + 1}_trade')
                for ext in ('png', 'pdf'):
                    fig.savefig(f'{base}.{ext}', bbox_inches=fig.bbox_inches,
                                pad_inches=0.0)
                plt.close(fig)
                paths.append(f'{base}.png')
                print(f'  -> {base}.png')
                if args.slides:
                    import shutil
                    shutil.copyfile(
                        f'{base}.png',
                        os.path.join(SLIDES,
                                     f'hv_window_{len(SEQUENCE) + 1}_trade.png'))
            if args.contact:
                _contact(paths, os.path.join(
                    FIG, f'hv_window_{variant}_{shape}_contact.png'))


def _contact(paths, out):
    """All five frames on one canvas, for looking at the build as a build."""
    import matplotlib.image as mpimg
    n = len(paths)
    im0 = mpimg.imread(paths[0])
    ar = im0.shape[1] / im0.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 7.0 / ar * n))
    for ax, p in zip(np.atleast_1d(axes), paths):
        ax.imshow(mpimg.imread(p))
        ax.axis('off')
    fig.subplots_adjust(hspace=0.02, left=0, right=1, top=1, bottom=0)
    fig.savefig(out, dpi=110, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  -> {out}')


if __name__ == '__main__':
    main()
