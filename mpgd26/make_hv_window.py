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

plus, top left, what the recovery time is BEING PAID FOR -- see ``_gain_panel``
and the entry for 2026-08-28 (evening) below.

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

FORMATTED FOR THE ROOM, 2026-08-28
----------------------------------
The build is slides 50-55 of ``mpgd26_talk.pptx`` (21.1-21.6 in the footer),
and it is the densest drawing in the talk: three panels and a scoreboard, the
top three of them crammed into a band an inch and a quarter deep.  Dylan asked
for width and for that band to use it.  What changed:

  canvas       12.5 -> 13.5 -> 14.25 in at the SAME 6.38 in height.  The
               picture frame on the slide went with it, 11.691 -> 12.625 ->
               13.327 in wide at the same 5.962 in tall -- a 13.333 in slide,
               so that is full bleed but for a hairline.  Because neither
               height ever moved, every point size on the slide is unchanged
               through all three passes: the projection scale is frame height
               over canvas height whatever the width.  1.75 in of new canvas,
               and all of it went to the top band.
  top band     both small panels sit on the same baseline (0.740) and reach
               the same 0.975, and they DOUBLED: 0.115 -> 0.235 of the canvas
               for the charge strip, 0.140 -> 0.235 for the efficiency panel,
               i.e. 0.73 -> 1.50 in and 0.89 -> 1.50 in.  The first pass took
               the gap over the main panel's energy axis; the second took the
               band the two headlines were in, which is why they went.
  headlines    both removed.  They carried the panels' provenance, so it moved
               to the y label (``Efficiency [%]``, spelt out now) and to the
               speaker -- see the notes in ``_eff_panel`` and ``_strip``, and
               the row for this slide in slides/RUNNING_ORDER.md.
  efficiency   the grey dashed July placement is gone too, and with it the key
               in the headline that explained it.  See ``_eff_panel``.
  scoreboard   on a faint post-it now (the one warm surface on a cool-white
               canvas -- warm reads before saturation does, so it can be
               nearly the background colour and still separate),
               so that the three numbers read as the third element of the band
               rather than as a caption that lost its figure.  Measured from
               the drawn text; it clears both panels by better than 0.2 in on
               every frame and ``draw`` prints the clearance each time.

THE TOP-LEFT PANEL, 2026-08-28 (EVENING): GAIN, NOT EFFICIENCY
--------------------------------------------------------------
Dylan: *"instead show a scaled gain curve ... aim for the peak strip in the
median event to be saturated ... we'll call this 100 % optimal gain ... then we
can show percentages of this optimal gain instead of efficiency."*

The efficiency was re-derived that morning and came out FLAT at 93-95 % with no
turn-on -- correct, and a poor left half of a trade: a panel that does not move
cannot be what the milliseconds are being paid for.  The gain does move, and the
same 27 June scan measures it best.  ``_gain_panel`` replaces ``_eff_panel``,
which is kept, still works, and comes back with ``--panel eff``.

  100 %        the median track's peak strip just fills the readout --
               frac_sat = 0.5, bench 497 V, bracketed by two measured points
               5 V apart in BOTH views.  Over it, the median track is clipping.
               That is the 200 fC DREAM the SCAN ran; the 600 fC one n_TOF ran
               needs 3x the charge and would put 100 % at n_TOF 586 V instead
               (``T.bench_gain_on_ntof_axis(ref='ntof600')``, and its top is
               extrapolated).  Dylan, 2026-08-29: 100 % goes at 497 V.
  the curve    TOTAL COLLECTED CHARGE (deconvolved forward fit, rail censored),
               referred to its value at 497 V.  LINEAR axis (Dylan, same day):
               nothing on it is more than a factor 7 from anything else once
               the curve stops at 100 %, and linear shows the run-away at the
               top, which is the point.
  the numbers  560/550/540/530/520 V = 78/48/32/20/14 % of optimal, and the
               100 % crossing is at n_TOF 565 +- 20 V, just past 560.
  the map      ONLY gas + pressure move the voltage, -67.85 V, so n_TOF 560 V
               is read at bench 492 V -- the plain gas equivalence.  The CSA
               change (200 -> 600 fC) then DIVIDES the ADC by three; it is a
               factor, not a voltage.  The efficiency map's +102.7 V does not
               apply: it carries the per-channel noise, and a rail is a fixed
               ADC count however noisy the channel is.

CORRECTED 2026-08-28 (Dylan: *"in my head 560 corresponded to 490, did I have
this wrong?"* -- he did not).  The first version folded the factor 3 into the
voltage axis as ln(3)/slope = +26.3 V, one shift of +94.1 V.  That is exact
only for a straight ladder, and this one is curved: it read the wrong part of
the ladder, by -13 % at 520 V and +6 % at 560 V, and it hid a 26 V slide inside
what looked like a fully measured curve.  The panel now dashes the ~13 V of
continuation the 100 % crossing genuinely needs.

The scoreboard's third row changed with it.  Everything else on the canvas is
untouched.

The closing "two costs multiplied" frame (``draw_trade``) is NOT part of the
build and did NOT move -- it keeps ``TRADE_WIDE``.  See the note there: the
backup section it used to live in was trimmed away the same day.

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
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch
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
# .caption and NO .figsrc -- which is the point of the rebuild.  'wide' is no
# longer that hole: since 2026-08-28 it is the .pptx picture frame, which is
# what the deck of record actually uses (index.html has been frozen since
# 2026-08-26 -- slides/NOTES.md).
SHAPES = {
    # 2026-08-28, in two passes: widened 12.5 -> 13.5 -> 14.25 in at an
    # UNCHANGED height, on the way to the full slide (Dylan: "widen the figure
    # while keeping the height to use the width of the slide", then "can we
    # make it even wider").  2280 x 1020 px at 160 dpi, 2.235 : 1, which is a
    # picture 13.327 x 5.962 in on a 13.333 in slide -- a hairline of margin,
    # essentially full bleed, which is where the rest of the deck has been
    # going (slide 56 is 13.338 wide at x = 0).
    #
    # Because the HEIGHT never moved, every point size in the drawing projects
    # at exactly the size it did before, through both passes: the on-slide
    # scale is frame-height / canvas-height = 5.962 / 6.38 whatever the width.
    # The 1.75 in of new canvas is pure room and all of it went to the top
    # band, which was the crowded part.
    'wide': (14.25, 6.38),            # .figure-solo, text-free   2.235 : 1
    'col': (6.60, 7.10),              # right column of .cols-2   0.930 : 1
}
# The closing "two costs multiplied" frame is NOT part of the build, and it
# keeps the shape it was drawn at.  It was backup slide 46 (deck slide 102)
# until 2026-08-28, when the deck's whole backup section was cut from 111
# slides to 70 and took it with it -- so today there is no frame to match and
# no frame to stretch, and widening it would be a change with no reader.  If
# it goes back on a slide, THAT frame sets its shape: measure the picture hole
# first, the way the build's 13.5 x 6.38 was set from slides 50-55.
TRADE_WIDE = (12.5, 6.38)

# The scoreboard's card.  The three numbers sit between two plots with nothing
# but canvas around them, and at 2.235 : 1 there is enough width that a panel
# behind them reads as a third element rather than as a patch over one of the
# plots (Dylan, 2026-08-28).
#
# A dull post-it, on request, and the colour is doing a second job: every
# other surface on this canvas is the same cool near-white (P.SURFACE
# #fbfcfe), so a warm one is the one thing the eye separates at 20 m without
# reading it.  It does NOT have to be a strong yellow to do that -- the eye
# reads warm-against-cool long before it reads saturation, so the tint can sit
# almost at the background and still divide the band into three.
#
# Toned down hard on 2026-08-28 ("make the yellow background much more
# subtle"): #faf3cf -> #fdfbef, which is 4/5 of the way back to the canvas.
# What holds the card together now is the EDGE, not the fill, so the edge was
# NOT lightened by as much -- lighten both and the card stops being an object
# and becomes a smudge.  Anything stronger than this competes with the two
# things that are meant to carry the frame: the 540 V and the shaded blind
# region.
CARD_FACE = '#fdfbef'
CARD_EDGE = '#e9e0bf'

# Which small panel goes top-left: ``gain`` (the default since 2026-08-28) or
# ``eff``, the efficiency curve it replaced.  Both functions are kept and both
# still work; the efficiency one is backup, and the reason for the swap is in
# ``_gain_panel``'s docstring.  ``--panel eff`` restores the old drawing
# without editing anything.
PANEL = 'gain'


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
    # Variant b has no headline since 2026-08-28 (Dylan: "remove the titles
    # above the two top plots all together to recover some vertical space").
    # It said ``det A  ·  run_57  ·  one sub-run per 2 V``, which is the
    # panel's whole provenance, so this is a real trade: the room bought the
    # panel 0.24 in of height and the sentence moved into the speaker's mouth.
    # It is in RUNNING_ORDER.md under this slide, and in the deck's alt text.
    if variant == 'a':
        ax.text(0.0, 1.06, 'the charge sets how long we are blind',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=small, color=P.MUTED)
    return xh, yh


def _gain_panel(ax, volts, shape, compact=False):
    """What the gain IS, as a fraction of the gain the readout could take.

    Dylan, 2026-08-28: *"instead show a scaled gain curve ... show the full
    charge collected, and aim for the peak strip in the median event to be
    saturated (just barely saturating is probably ideal gain).  We'll call
    this 100 % optimal gain (can show over 100 % and indicate probably too
    much if we end up there on the scale).  Then we can show percentages of
    this optimal gain instead of efficiency."*

    WHY THIS PANEL REPLACED THE EFFICIENCY ONE.  The efficiency curve was
    re-derived the same day and came out **flat at 93-95 % across the whole
    n_TOF window, with no turn-on** (see ``_eff_panel`` below, kept as
    backup).  A flat panel beside a panel that walks 13.9 -> 1.0 ms says
    "nothing to see here" -- true, and useless as the left half of a trade.
    The gain is the quantity that actually moves, it is what the recovery time
    is paid for, and it is what the same 27 June scan measures best.

    WHAT 100 % MEANS.  The mesh voltage at which the peak strip of the MEDIAN
    track just fills the readout: ``frac_sat`` = 0.5, bracketed by two measured
    points 5 V apart (0.39 at 495 V, 0.66 at 500 V, both views) at **bench
    497 V**, which is **n_TOF 565 V**.  Above that the median track is clipping
    and the extra gain goes into the rail rather than into the measurement --
    hence the shaded "too much" band, and hence "just barely saturating"
    being the ideal.

    ``peak_amp`` is the tallest SAMPLE of the tallest STRIP of the event -- the
    max strip, the thing that clips first.  The 50 % point is stable to 0.7 V
    over any sane clipping threshold (496.4 / 497.0 / 497.1 V at 0.88 / 0.92 /
    0.95 of the rail).  Note the *median amplitude* only reaches the nominal
    rail near 500 V, which is what the eye reads off ``gain_vs_hv.png``; both
    are right and they are different statements.

    WHICH READOUT, AND IT IS WORTH 3x.  497 V fills the **200 fC** DREAM the
    bench scan ran.  n_TOF ran **600 fC** -- 3x less ADC per electron -- so
    filling *that* needs 3x the charge, at n_TOF 586 V.  Dylan, 2026-08-29,
    chose 497 V, and it is the right lead: it is the scan's own measured
    saturation point, it leaves nothing on the panel extrapolated, and the
    600 fC setting was forced by the gamma flash rather than chosen for
    tracking.  ``T.bench_gain_on_ntof_axis(ref='ntof600')`` gives the other
    scale; ``results()['gain_scale']['ntof600']`` carries its numbers.

    WHAT IS PLOTTED is the **total collected charge** per track, not the peak
    sample -- the deconvolved forward-fit charge from the same 18 sub-runs,
    which censors railed samples and so keeps measuring where the peak sample
    cannot.  It is normalised to its own value at 497 V.  Charge and peak
    amplitude are proportional to +-5 % across this range, so a charge ladder
    referred to a saturation voltage is self-consistent; the model-free window
    sum gives the same percentages to 5 % (10.2 vs 10.7 % at 540 V).

    THE MAP.  Only the GAS and PRESSURE terms move the voltage: n_TOF W is
    read off the bench ladder at **W - 67.85 V**, so 560 V is bench 492 V --
    the equivalence everyone already carries in their head.  The readout change
    (CSA 200 -> 600 fC) then DIVIDES the ADC by three; it is not a voltage.
    The efficiency panel's +102.7 V map does not apply here at all: it carries
    the per-channel noise, and a rail sits at a fixed ADC count however noisy
    the channel is.

    ``T.adc_shift()``'s +94.1 V is the same statement collapsed into one shift
    (67.85 + ln 3 / slope).  It is a fair way to SAY which bench voltage makes
    the same ADC, and it is not used to evaluate the ladder: that would be
    exact only for a straight ladder, and this one is curved (0.33 per 10 V
    near 440 V, 0.52 near 495).  Corrected 2026-08-28 after Dylan queried the
    mapping -- it had been reading the ladder 26 V too low, by -13 % at 520 V
    and +6 % at 560 V.

NOTHING ON THIS PANEL IS EXTRAPOLATED, and that is a consequence of
    where 100 % was put.  The measured bench charge runs 425-505 V = n_TOF
    493-573 V, and the 100 % crossing is at n_TOF 565 V -- inside it.  (Refer
    the scale to the 600 fC range instead and 100 % moves to 586 V, ~13 V past
    the last trustworthy bench point, and the top of the curve has to be
    dashed.  That is the other reason to lead with this one.)  The
    horizontal placement carries the gas map's +-20 V (``T.bracket()``), which
    slides the whole curve and takes the 100 % voltage with it; it does NOT
    touch the ratios between setpoints, which is all the percentages compare.
    The axis runs past the crossing so that 100 % is a place on the plot rather
    than an assertion in a caption.
    """
    small = 9.2 if compact else (10.0 if shape == 'wide' else 9.2)
    # x stops just past the last measured point (n_TOF 573 V) and y just past
    # where the curve gets there, so the whole measured ladder is on the canvas
    # and it exits through the top-right corner rather than the top edge.
    x0, x1 = 517, 572
    y0, y1 = 0.0, 150.0

    g = T.bench_gain_on_ntof_axis()
    vn, pct, v_opt, nm = g['v'], g['pct'], g['v_opt'], g['n_meas']
    vl, pl = g['v_line'], g['pct_line']

    # LINEAR since 2026-08-29 (Dylan).  It works now and would not have before:
    # with 100 % at 565 V the drawn range is 14-100 %, a factor 7, where the
    # 600 fC scale ran 4.5-100 over a factor 22 and needed a log axis to be
    # anything but a hockey stick.  Linear also puts the run-away at the top
    # back on the canvas, which is what the slide is about.
    # over-gain.  Copper is the deck's caution accent and it is the only warm
    # thing in the panel, so it reads as a boundary without a key.
    ax.axhspan(100.0, y1, color=P.COPPER, alpha=0.10, lw=0, zorder=0)
    ax.axhline(100.0, color=P.COPPER, lw=1.2, ls=(0, (4, 2)), zorder=2)
    # Both annotations are placed in AXES fractions, not data: this panel is
    # drawn at 3.16 x 1.50 in in variant b and at 2.64 x 0.96 in in variant a,
    # and a data-space offset that clears the curve in one lands on it in the
    # other.
    ax.text(0.025, 0.845, 'too much gain', transform=ax.transAxes, ha='left',
            va='center', fontsize=small - 1.6, color=P.COPPER)

    ax.plot(vl[:nm], pl[:nm], '-', color=P.DET_COLOR['A'], lw=1.7, zorder=3)
    ax.plot(vn, pct, 'o', ms=3.4, color=P.DET_COLOR['A'],
            markeredgecolor=P.SURFACE, markeredgewidth=0.5, zorder=4)
    if nm < len(vl):                      # only the ref='ntof600' scale needs it
        ax.plot(vl[nm - 1:], pl[nm - 1:], color=P.DET_COLOR['A'], lw=1.4,
                ls=(0, (2, 2)), alpha=0.55, zorder=3)
        ax.axvspan(g['v_last_meas'], x1, color=P.MUTED, alpha=0.07, lw=0,
                   zorder=0)

    # Where the curve meets 100 %.  A RING, not a filled marker: it is a
    # crossing the chamber never ran at.  The +-20 V is the gas map's own
    # bracket and it is written rather than drawn -- a whisker that long
    # reaches past the right edge of a 3.2 in panel, and at this size a
    # systematic whisker reads as scatter anyway.
    ax.plot([v_opt], [100.0], 'o', ms=5.5, mfc=P.SURFACE, color=P.COPPER,
            markeredgewidth=1.4, zorder=5)
    # Under the curve at the right edge -- the only empty corner of the panel
    # once the line has climbed past 50 %.  Two short lines rather than one
    # long one: right-aligned at the axis edge, a single line would reach back
    # under the curve at 573 V.
    # bottom right: on a linear axis the curve leaves that corner empty until
    # the last few volts, and the ring at 565 V is directly above it
    ax.text(0.985, 0.13, f'100 % at\n{v_opt:.0f} $\\pm$ 20 V',
            transform=ax.transAxes, ha='right', va='center', linespacing=1.25,
            fontsize=small - 1.6, color=P.COPPER)

    g_here = float(np.exp(np.interp(volts, vn, np.log(pct))))
    assert volts <= g['v_last_meas'], 'a setpoint has left the measured ladder'
    ax.plot(volts, g_here, marker='*', ms=17, color=P.DET_COLOR['A'], zorder=6,
            markeredgecolor=P.BAND_DEAD, markeredgewidth=1.5)
    # Above the star: the curve is a rising straight line on a log axis, so
    # everything below-right of it is under the line and everything above-left
    # is empty panel.  Up and slightly left is the only clean corner, and it
    # stays clean for every frame because the star only ever moves along the
    # line.
    # ...except at the bottom of the build, where 520 V sits 3 V from the
    # left spine and a centred label hangs over the tick labels.  There it goes
    # up and to the RIGHT, which is still above the line.
    # LEFT of the star at its own height.  The curve is convex on a linear
    # axis, so at every setpoint the space immediately to the left is above the
    # line and empty; going UP instead runs the 560 V label into the 100 % rule.
    # Except at the bottom of the build, where 520 V sits 3 V from the spine.
    lab = (dict(ha='left', va='bottom', xytext=(7, 9)) if volts - x0 < 8
           else dict(ha='right', va='center', xytext=(-9, 3)))
    ax.annotate(f'{g_here:.0f} %', xy=(volts, g_here),
                textcoords='offset points', fontsize=small,
                color=P.BAND_DEAD, fontweight='bold', zorder=6, **lab)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_xticks([520, 530, 540, 550, 560, 570])
    ax.tick_params(labelsize=small - 0.8, pad=1.5 if compact else 3)
    ax.set_xlabel('amplification voltage  [V]', fontsize=small,
                  labelpad=1 if compact else 2)
    ax.set_ylabel('Gain [% of optimal]' if compact else 'gain\n[% of optimal]',
                  fontsize=small, linespacing=1.15,
                  labelpad=2 if compact else 4)
    ax.grid(axis='y', alpha=0.18, which='major')
    P.strip(ax)
    if not compact:
        ax.text(0.0, 1.06, 'and the gain is what it is paid for',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=small, color=P.MUTED)
    return g_here / 100.0


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
    passes -- the only one that reaches below 450 V.  It is also the run
    ``mesh_ladder.csv`` comes from, so this curve and the gain slope that maps
    it are the same measurement.

    RE-DERIVED 2026-08-28, AND THE PANEL'S STORY CHANGED.  Until then this read
    a scan written on 29 June that carried none of July's basis changes -- the
    golden M3 recipe, the significance floor, the reprocessed hits -- and took
    its active box from the highest-HV sub-run, where half the events are
    discharges.  It plateaued at **81 %** where the same chamber, the same
    night, at the same 490 V reads **93.3 %** off its own long run.  Both
    scans were rebuilt by ``mx_june_cosmic_qa/10b_hv_scan_efficiency.py``,
    whose ``--closure`` reproduces that published breakdown to the third
    decimal.  Two consequences for what this panel may say:

      * **The plateau is 93-95 %** (455-500 V mean: 93.5 %), so the star reads
        ~93 % at 540 V, not 69 %.
      * **There is no turn-on in the scan.**  425 V reads 89.6 %, not 49 %.
        The old rise was the pre-reprocessing analyzer's amplitude threshold:
        this scan's own gain ladder puts the weakest 2 % of events at 69 ADC on
        the peak strip at 425 V, ~10 sigma over the 6.85 ADC bench pedestal.
        So the panel is FLAT across the whole n_TOF window and the low edge was
        never reached.  Do not describe it as a turn-on, and do not lean on the
        extrapolation below the scan for anything but "still no turn-on yet".

    The 22 June overnight scan starts at 450 V, already flat.  It used to sit
    ~10 points higher, which was explained here by the top slot doubling the M3
    lever arm into the same fixed 5 mm box; on the re-derived chain the two
    scans AGREE and that explanation is withdrawn.  The lever arm shows up in
    the core residual instead (0.34-0.41 mm bottom slot, 0.44-0.59 mm top) and
    never cost efficiency at a 5 mm match.

    HOW IT IS PLACED on the n_TOF axis: the full ledger, not the gas term
    alone -- an efficiency is a threshold quantity, so the CSA range and the
    per-channel noise belong in the shift as much as the gas does.  The
    placement drawn is **production**, after the 23 July noise step, which is
    where the campaign actually ran.

    ONE CURVE, NOT TWO (Dylan, 2026-08-28: *"remove the gray dashed line"*).
    Until 2026-08-28 run_55's own placement was drawn beside it as a grey
    dashed line, 22 V to the left, so that the 23 July noise step was on the
    canvas rather than in a footnote -- it is what turns 540 V from 81 % into
    69 %.  That is a real point, but it needed a key in the headline
    (``solid: as we ran · dashed: July``) to be readable at all, and the
    headline sits over the most crowded band of the drawing.  The step is the
    speaker's line now, and ``ntof_july_analysis/hv_tradeoff/report.html``
    carries it; ``T.bench_eff_on_ntof_axis('run_55', ...)`` still returns the
    other placement for anyone who wants the curve back.

    Measured points are plotted as markers, always.  Where the production
    placement runs off the left of the scan the LINE continues as a straight
    fit to the three lowest points, drawn dashed and faded, and it is labelled
    as an extrapolation on the canvas.
    """
    small = 9.2 if compact else (10.0 if shape == 'wide' else 9.2)
    x0, x1 = 517, 563

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
    # beside the star, never over it.  The curve is flat, so a label beside the
    # star lands ON the line -- go above it, where the only thing overhead is
    # empty panel.  EXCEPT inside the extrapolation band: since the 2026-08-28
    # re-derivation the star sits near 90 % at every voltage, including the
    # leftmost frame, and there 'above the star' is exactly where the 'extrap.'
    # tag lives.  So in the band, go to the right at the star's own height.
    high = e_here > 75
    in_extrap = bool(nxp) and volts < float(vpl[nxp])
    if high and in_extrap:
        off, ha, va = (13, 0), 'left', 'center'
    elif high:
        off, ha, va = (0, 12), 'center', 'bottom'
    else:
        off, ha, va = (12, 6), 'left', 'bottom'
    ax.annotate(f'{e_here:.0f} %', xy=(volts, e_here), xytext=off,
                textcoords='offset points', fontsize=small, ha=ha, va=va,
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
    # Spelt out since 2026-08-28 (Dylan).  It used to be a bare ``%``, which
    # only worked while the headline overhead said what the axis was; the
    # headline is gone and the panel now has to name itself.  There is room
    # for it: the panel is 1.5 in tall, and the rotated label is ~1.0 in.
    ax.set_ylabel('Efficiency [%]' if compact else 'efficiency\n[%]',
                  fontsize=small, linespacing=1.15,
                  labelpad=2 if compact else 4)
    ax.grid(axis='y', alpha=0.18)
    P.strip(ax)
    # No headline in the compact (variant b) panel since 2026-08-28 -- see the
    # matching note in _strip.  It said ``bench efficiency, mapped to 90/10``;
    # the axis label now carries "efficiency" and the speaker carries "bench,
    # mapped to 90/10".  That mapping is not a detail, so it is worth saying
    # out loud: this is the 27 June cosmic bench in 95/5, moved onto the n_TOF
    # 90/10 axis by the full ledger.  It is also in the deck's alt text.
    if not compact:
        ax.text(0.0, 1.06, 'and the gain sets what we see',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=small, color=P.MUTED)
    return e_here / 100.0


# --------------------------------------------------------------------------- #
# the main panel: the rate, on the beam's clock, with the cut on it
# --------------------------------------------------------------------------- #

def gain_at(volts):
    """(gain as a fraction of optimal, is_over) at an n_TOF setpoint.

    One definition, used by the panel and by the scoreboard, so the star and
    the number can never disagree -- the same contract ``eff_at`` had.
    Interpolated on the MEASURED points only; every setpoint in the build is
    inside them (they reach n_TOF 573 V).

    ``is_over`` is the flag the scoreboard needs to say "past the rail" rather
    than print a number over 100 % as if more were better.  On the drawn axis
    it is never set: 560 V, the top of the build, is at 26 %.
    """
    d = T.bench_gain_on_ntof_axis()
    g = float(np.exp(np.interp(volts, d['v'], np.log(d['pct']))))
    return g / 100.0, bool(g > 100.0)


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
    rel, over = gain_at(volts)
    lines = [
        (0.00, f'{volts} V', size + 10.0, 'bold', P.BAND_DEAD),
        (1.55, role, size, 'normal', P.MUTED),
        (2.30, f'blind for {ms:.1f} ms after every flash',
         size + 1.0, 'bold', P.BAND_DEAD),
        (3.00, f'{frac * 100:.1f} % of the X17 rate is left',
         size + 1.0, 'bold', P.ACCENT if frac > 0.01 else P.MUTED),
        (3.70, f'{rel * 100:.0f} % of the gain the readout could take'
         f'{" -- past the rail" if over else ""}',
         size + 1.0, 'bold', P.COPPER if over else P.DET_COLOR['A']),
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

    Returns the Text artists, so ``_readout_card`` can measure what was
    actually drawn rather than guess at it.  The block is meant to be the same
    width on every frame -- see the note on ``extrap.`` below -- and measuring
    is how that stays true when someone edits a line.
    """
    hv = hv_points()
    ms = hv[volts][1]
    frac, _ = surviving(ms * 1e3)
    rel, over = gain_at(volts)
    role = {560: 'where the gain wants to be',
            OP_RESIST: 'where we ran'}.get(volts, '')

    gap = 0.012
    rows = ((f'{volts} V', role, size + 15.0, P.BAND_DEAD, P.MUTED),
            (f'{frac * 100:.1f} %', 'of the X17 rate left', size + 6.0,
             P.ACCENT if frac > 0.01 else P.MUTED, P.MUTED),
            (f'{rel * 100:.0f} %',
             # The width-setting line of the block is ``blind for NN.N ms
             # after every flash``, which is the same length on every frame,
             # so the card behind the scoreboard is the same rectangle
             # throughout the build.  Keep this label shorter than that one or
             # the card starts changing size mid-build and stops clearing the
             # charge strip -- which is what happened when the old efficiency
             # row spelt out ``extrapolated``.
             'of optimal gain  (past the rail)' if over
             else 'of optimal gain  (cosmic bench)', size + 6.0,
             P.COPPER if over else P.DET_COLOR['A'], P.MUTED))
    art = []
    for k, (num, lab, fs, cnum, clab) in enumerate(rows):
        yy = y - k * dy
        art.append(fig.text(x, yy, num, fontsize=fs, fontweight='bold',
                            color=cnum, ha='right', va='center', zorder=9))
        art.append(fig.text(x + gap, yy, lab, fontsize=size, color=clab,
                            ha='left', va='center', zorder=9))
    art.append(fig.text(x + gap, y - (len(rows) - 0.42) * dy,
                        f'blind for {ms:.1f} ms after every flash',
                        fontsize=size - 0.5, color=P.BAND_DEAD, ha='left',
                        va='center', zorder=9))
    return art


def _readout_card(fig, art, padx=0.17, pady=0.10):
    """A light panel behind the scoreboard (Dylan, 2026-08-28).

    The three numbers are the only thing on this canvas that is neither a plot
    nor a label on one, and between two plots with nothing but canvas around
    them they read as a caption that lost its figure.  A tinted card says they
    are the third element.

    Measured from the drawn text and padded in INCHES, not in figure fraction:
    the canvas is 2.1 : 1, so an equal fraction is twice as much room
    horizontally as vertically and the corners would come out elliptical.

    It has to clear the efficiency panel on its left and the charge strip on
    its right, and there are only 5.13 in between the two.  Measured on the
    six frames it lands 0.35 in clear on the left and 0.22-0.34 in on the
    right, the tight one being 520 V (the ``extrap.`` line) and 560 V (the
    13.9 ms one).  ``draw`` prints both clearances for every frame, so a
    layout change that eats them says so in the build log instead of in
    Prague.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = mtransforms.Bbox.union([a.get_window_extent(r) for a in art])
    bb = bb.transformed(fig.dpi_scale_trans.inverted())
    card = FancyBboxPatch((bb.x0 - padx, bb.y0 - pady),
                          bb.width + 2 * padx, bb.height + 2 * pady,
                          boxstyle='round,pad=0,rounding_size=0.12',
                          transform=fig.dpi_scale_trans, clip_on=False,
                          facecolor=CARD_FACE, edgecolor=CARD_EDGE, lw=1.0,
                          zorder=8)
    fig.patches.append(card)
    return mtransforms.Bbox.from_extents(bb.x0 - padx, bb.y0 - pady,
                                         bb.x1 + padx, bb.y1 + pady)


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
    w, h = TRADE_WIDE if shape == 'wide' else SHAPES[shape]
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
    ax_card = None                    # the scoreboard's text, wide + variant b
    card_bb = None

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
        # 2026-08-28, in two passes: variant b's strip grew 0.115 -> 0.198 ->
        # 0.235 of the canvas and dropped 0.815 -> 0.740 (Dylan: "increase the
        # height of both of these small plots ... to use the full vertical
        # space", then "remove the titles ... to recover some vertical space
        # as well and stretch there").  The first pass took the room between
        # the main panel's energy axis and the panels; the second took the
        # band the two headlines had been sitting in, which is why they had to
        # go for it.  Its charge decade now has somewhere to go -- 0.73 in of
        # axis for a factor 43 in charge, and 1.50 in now.  Both top panels
        # sit on the same baseline, 0.740, and reach the same 0.975, so the
        # band reads as a row.
        ax_s = fig.add_axes([L, 0.780 if variant == 'a' else 0.740,
                             (R - L) * (0.30 if variant == 'a' else 1.0),
                             0.150 if variant == 'a' else 0.235])
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
            # the strip's own data, which begins at x = 0.683 of the canvas
            # whatever the canvas is (the strip's box is cut to the axis, so
            # that fraction is a property of the time axis, not of the width).
            #
            # 2026-08-28: taller with the strip, and narrower in inches than
            # it started -- 0.222 x 14.25 = 3.16 in where it was 0.300 x 12.5
            # = 3.75.  The scoreboard between the two panels is what sets
            # this: it must clear both, so the width the panel gives up is the
            # width the card needs.  The panel does not miss it -- it is a
            # turn-on and a plateau, and it gains far more than it loses
            # (0.89 -> 1.50 in tall; 4.2 : 1 was a letterbox, 2.1 : 1 is a
            # plot).
            ax_y = fig.add_axes([0.077, 0.740, 0.222, 0.235])
            ax_y.set_facecolor(P.SURFACE)
            ax_y.patch.set_alpha(1.0)
            ax_y.set_zorder(4)
            # x is set by the card, not by the type: the block is centred
            # in the 5.47 in of canvas between the efficiency panel's right
            # edge and the charge strip's left one.  draw() prints both
            # clearances.  The type went up with the band -- 11.5 -> 12.5,
            # the only size on this canvas that changed on 2026-08-28 -- so
            # that the card fills the height the headlines gave back instead
            # of floating in it.
            ax_card = _readout_stats(fig, volts, 0.424, 0.925, 0.062, 12.5)
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
    if ax_card is not None:
        card_bb = _readout_card(fig, ax_card)
    if ax_y is not None:
        (_eff_panel if PANEL == 'eff' else _gain_panel)(
            ax_y, volts, shape, compact=(variant == 'b'))
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
    if card_bb is not None:
        # Both must stay positive.  The scoreboard is the widest thing on the
        # canvas and it is the ONLY one whose width the data can change.
        wi = fig.get_size_inches()[0]
        print(f'     card clears the panels by '
              f'{card_bb.x0 - ax_y.get_position().x1 * wi:+.2f} in left, '
              f'{ax_s.get_position().x0 * wi - card_bb.x1:+.2f} in right')
    return fig


def main():
    global PANEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--numbers', action='store_true')
    ap.add_argument('--panel', default=PANEL, choices=('gain', 'eff'),
                    help='top-left panel: scaled gain curve (default) or the '
                         'efficiency curve it replaced on 2026-08-28')
    ap.add_argument('--variant', default='ab', help='a, b or ab')
    ap.add_argument('--shape', default='wide', help='wide, col or both')
    ap.add_argument('--contact', action='store_true',
                    help='also write a review contact sheet per variant')
    ap.add_argument('--slides', action='store_true',
                    help='copy the PNGs into slides/assets/img (NOT the default:'
                         ' the deck is being edited elsewhere)')
    args = ap.parse_args()
    PANEL = args.panel

    _g = T.bench_gain_on_ntof_axis()
    _g6 = T.bench_gain_on_ntof_axis(ref='ntof600')
    print(f'  100 % of optimal gain = the median peak strip fills the readout'
          f' = bench {T.saturating_voltage(0.5)[0]:.0f} V'
          f' = n_TOF {_g["v_opt"]:.0f} V  (200 fC, the range the scan ran;'
          f' the 600 fC range n_TOF ran needs x3 and lands at'
          f' {_g6["v_opt"]:.0f} V)')
    print(f'  ladder read at V - {_g["shift"]:.2f} V (gas+pressure);'
          f' measured to n_TOF {_g["v_last_meas"]:.0f} V, so nothing drawn is'
          f' extrapolated')
    for r in numbers():
        g, _over = gain_at(r['volts'])
        print(f'  {r["volts"]} V   Q = {r["charge_nC"]:6.1f} nC   '
              f'recovery {r["recovery_ms"]:6.2f} ms   '
              f'surviving {r["frac"] * 100:5.2f} %  ({r["rate"]:.2f} /day)'
              f'   gain {g * 100:5.1f} % of optimal')
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
