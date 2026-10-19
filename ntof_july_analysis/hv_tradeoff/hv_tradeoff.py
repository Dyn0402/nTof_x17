#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hv_tradeoff.py -- what the amplification voltage buys and what it costs.

    ../../.venv/bin/python hv_tradeoff.py            # the whole ledger
    ../../.venv/bin/python hv_tradeoff.py --json     # + results.json

THE QUESTION.  The n_TOF campaign ran detector A at 540 V because the flash
recovery time grows with the avalanche charge and 560 V is blind for 13.9 ms
-- past the thermal neutrons.  That is half a trade.  The other half is what
lowering the voltage gives up, and until now the deck asserted it with one
number (``560 V finds ~4x the tracks 540 V does``).  This module builds the
other half properly, from three independent measurements, and multiplies the
two halves together.

    what we lose by going UP    post-flash recovery -> how much of the X17
                                rate arrives after the front end is back
    what we lose by going DOWN  gas gain -> cluster quality -> track yield
    where the two cross         the product, as a function of voltage

THE THREE LEGS
--------------
1. **The bench, at Ar/iso 95/5** (June, Saclay).  det3 == chamber A, so this
   is the same physical chamber.  TWO scans, and which one you read matters:
   the **27 June saturday scan** (``bench_efficiency_saturday()``, both
   passes) runs **425-525 V**, the only one reaching below 450 V, so it is the
   one the deck plots; the **22 June overnight scan** (``bench_efficiency()``)
   runs 450-525 V.  BOTH WERE RE-DERIVED 2026-08-28 (see BENCH_EFF_SAT): the
   plateau is **93-95 %**, not 81 %, and the two scans **agree** -- they used
   to differ by ~10 points and that gap was explained by the top slot's M3
   lever arm; the gap is gone and the explanation is withdrawn.  **Neither
   scan shows a turn-on**: 425 V reads 89.6 %, and the chamber's own gain
   ladder says why (69 ADC on the peak strip in the weakest 2 % of events,
   ~10 sigma over the bench pedestal).  Same efficiency definition in both,
   and it is the one 02_efficiency.py publishes for this chamber.
   M3-referenced, drift 1000 V.  The 27 June
   saturday scan (``mesh_ladder.csv``) gives the MEASURED gain ladder --
   median strip amplitude against voltage, 425-525 V -- whose slope,
   0.418 per 10 V, is the number this module divides by whenever a gain ratio
   has to become a voltage.

2. **Garfield++, to cross the gas boundary.**  The bench ran 95/5 and n_TOF
   ran 90/10, so the bench curve cannot be read on the n_TOF voltage axis
   without a map.  The repository already has one and it is the authority
   here: ``garfield_sim/results/hv_equivalence.json`` via
   ``ntof_july_analysis/gain_map.GainMap`` -- per-mixture ln G = a + bV + c2V²
   fits at two site pressures, inverted to match GAIN rather than to shift by
   a constant.  It says 90/10 needs **+72.6 V at CERN** (+75.5 at Saclay) for
   the gain 95/5 has, and it is flat to ±0.6 V across 400-590 V.

3. **n_TOF itself, at Ar/iso 90/10** (run_55, 18 July).  A cyclical resist
   scan 560->520 V on all four chambers with a scintillator-doubles trigger,
   ``calib/25_hv_scan_summary.json``.  Its MIP-track rate per trigger is the
   only measurement of yield against voltage taken in the production gas, on
   the production chamber.  Two time windows: b1 = 8-12 ms, b2 = 16-28 ms.

   The recovery ladder is run_57, two days later, SAME chamber and SAME drift
   voltage (600 V on A in both), so the two halves of the trade are joined
   without a conditions jump.

WHAT THE LEDGER SAYS (see ``report.html`` for the figures)
---------------------------------------------------------
Bench-equivalent voltage of an n_TOF setpoint, term by term:

    gas 95/5 -> 90/10                        +72.6 V   (CERN pressure)
    Saclay -> CERN pressure                   -4.6 V   (thinner air, more gain)
    electronics, run_55 era                  +12.8 V   (200 fC -> 600 fC CSA
                                                        range, and sigma 6.85
                                                        -> 3.90 ADC)
    electronics, production (post 23 July)   +34.8 V   (same range, sigma 9.80)

so **n_TOF 540 V sits at bench ~459 V in the run_55 configuration and at bench
~437 V in the production one**, and the bench's own efficiency plateau
(91-92.5 %, 450-485 V) maps to **n_TOF 518-553 V** for run_55.

TWO THINGS THIS DOES NOT CLAIM
------------------------------
* **The n_TOF ladder is not an efficiency.**  Its denominator is a
  doubles-trigger whose geometric ceiling per arm is ~50 %, and its numerator
  requires a 3-20 strip, <=25 mm MIP-like cluster in both views.  A cluster
  loses strips over threshold as the gain falls, so the ladder turns on far
  later than DETECTION does -- which is why it can rise x4 across 540-560 V
  while the mapped bench curve says the chamber was already detecting
  everything.  The two are not in conflict; they measure different things, and
  the honest reading is that what the voltage bought at n_TOF was
  RECONSTRUCTABILITY, not detection.  Only the SHAPE of the ladder is used
  here, never its normalisation.
* **The map is worth about ±20 V, not ±2 V.**  Cross-checks in
  ``bracket()``: the same ratio taken from the T6 meshfield ladders gives
  +86 V and from the uniform-field slopehunt pair +103 V, against the blessed
  map's +72.6.  And the bench gain slope is 1.4x the simulated one
  ([[mx17-hv-slope-test]]), so a gain ratio turned into volts with the sim's
  own slope comes out ~35 % larger.  Everything downstream is quoted with
  that bracket attached.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                  # ntof_july_analysis
REPO = os.path.dirname(PKG)
sys.path[:0] = [PKG, os.path.join(REPO, 'mpgd26')]

from gain_map import GainMap                        # noqa: E402
import make_status_plots as S                       # noqa: E402
import make_x17_rate as X                           # noqa: E402

# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
BENCH_EFF = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
             'mx17_det2_det3_overnight_6-22-26/hv_scan/mx17_3/'
             'efficiency_vs_hv.csv')
# The 27 June saturday det3 scan -- the SAME chamber, five days later, in the
# TOP slot (FEU 7 x / 8 y, z = 702) instead of the bottom one.  Two interleaved
# passes, 425-525 V and 460-520 V.  It is the only bench scan that reaches
# below 450 V, which is why the panel uses it.
#
# REDERIVED 2026-08-28.  Both this scan and the 22 June one above were rebuilt
# by mx_june_cosmic_qa/10b_hv_scan_efficiency.py on the current chain -- golden
# M3 recipe (chi2<1.0 & NClus=4), the 2026-07-25 significance floor, the
# reprocessed hits, and the 02_efficiency.py accounting inside the long run's
# own box.  The files this used to read were written on 29 JUNE and carried
# none of that; they plateaued at 81 % where the same chamber at the same
# 490 V, the same night, reads 93.3 %.  They are parked under
# ``<scan>/mx17_3/_superseded_20260629/`` with a README.  Two things the
# correction changed, and both are load-bearing here:
#
#   * THE PLATEAU IS 93-95 %, not 81 %, and it agrees with the chamber's
#     published headline (93.1 % hits / 93.5 % wft on long_run at 490 V).
#   * THERE IS NO TURN-ON INSIDE THIS SCAN.  425 V reads 89.6 %, not 49 %.
#     The old rise from 0.49 to 0.81 across 425-455 V was the pre-reprocessing
#     analyzer's amplitude threshold, not the chamber: this scan's own gain
#     ladder (mesh_ladder.csv, same sub-runs) puts the peak strip at 69 ADC in
#     the weakest 2 % of events at 425 V, ~10 sigma over the 6.85 ADC bench
#     pedestal.  Anything downstream that leans on a steep low-V turn-on is
#     leaning on an artefact.
#
# The 22 June scan's plateau also moved, 91 -> 91-94 %, and the two scans now
# AGREE.  The old note here explained their ~10-point gap by the top slot
# doubling the M3 lever arm into the same fixed 5 mm box; that gap is gone, so
# the explanation is withdrawn.  The lever arm is still visible where it
# belongs -- in the core residual, 0.34-0.41 mm bottom slot against
# 0.44-0.59 mm top -- it just never cost efficiency at a 5 mm match.
#
# This is also the run that produced ``mesh_ladder.csv``, so the efficiency and
# the gain ladder are the same scan.
BENCH_EFF_SAT = [
    ('/media/dylan/data/x17/cosmic_bench/Analysis/'
     'mx17_det3_saturday_scan_6-27-26/hv_scan/mx17_3/efficiency_vs_hv.csv',
     'pass 1'),
    ('/media/dylan/data/x17/cosmic_bench/Analysis/'
     'mx17_det3_saturday_scan_6-27-26/hv_scan2/mx17_3/'
     'efficiency_vs_hv.csv', 'pass 2'),
]
MESH_LADDER = os.path.expanduser('~/x17/response_sim/hv_slope/mesh_ladder.csv')
# The SAME 18 sub-runs, read a second way: total collected charge per track,
# from mx_june_cosmic_qa/10e_hv_scan_charge_angle.py.  ``q_sum`` is the
# deconvolved forward-fit charge, which CENSORS railed samples and so keeps
# measuring to 505 V where the peak strip stopped at ~495; ``q_win`` is a
# model-free sum of every sample over +-10 strips.  The two, and the peak
# sample, agree on d ln Q / dV to under 2 % over 425-505 V, so the charge
# ladder and the mesh ladder are one measurement seen three ways.
CHARGE_LADDER = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
                 'mx17_det3_saturday_scan_6-27-26/hv_scan/mx17_3/'
                 'charge_angle_vs_hv.csv')
SLOPES = os.path.expanduser('~/x17/response_sim/hv_slope/slopes.json')
NTOF_SCAN = os.path.join(REPO, 'mx_july_beam_qa', 'calib',
                         '25_hv_scan_summary.json')
CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'mpgd26_status')

OP_V = 540                      # where production ran
GAIN_V = 560                    # where the gain wants to be
DET = 'A'                       # == bench det3

# Pedestal noise, median CNS sigma of the FEU carrying chamber A's x view
# (FEU 3 in both setups).  Bench: recomputed 2026-08-23 with
# ntof_pedestal_qa/lxplus/extract_pedestals.stats_for_file on
# cosmic_bench/pedestals/pedestals_06-22-26_19-14-37 -- the pedestal of THIS
# scan.  n_TOF: ntof_pedestal_qa/data/ped_stats.npz, 260718_14H07 (run_55's
# own week) and 260723_16H55 (the first pedestal of the noisy production
# configuration).  See ntof_pedestal_qa/README.md for the 23 July step.
SIGMA_BENCH_ADC = 6.85
SIGMA_NTOF_RUN55_ADC = 3.90
SIGMA_NTOF_PROD_ADC = 9.80

# DREAM charge-sensitive-preamplifier full scale, from the configurations
# actually loaded.  Bench: `Feu * Dream * 6 0xAAAA` = 200 fC in every saved
# CosmicTb_MX17.cfg.  n_TOF: `"6": "0xffff ..."` = 600 fC in ALL 56 pedestal
# contexts, 1 July to 10 August (ntof_pedestal_qa/data/ped_context.json) --
# which also settles the deck's open question about which range production
# ran on.  Signal per electron scales as 1/full-scale.
CSA_BENCH_FC = 200.0
CSA_NTOF_FC = 600.0

# The 12-bit DREAM sample rails at 3871.5 ADC (measured: the median peak
# amplitude at 525 V, where >99 % of tracks clip -- 10c_hv_scan_gain.py).
# ``SAT_ADC`` is the level 10e counts a sample as railed at, 92 % of that; the
# 0.6 V between the two definitions of "saturated" is far inside the map's own
# bracket and nothing here is sensitive to the choice.
ADC_RAIL = 3871.5
SAT_ADC = 3550.0


# --------------------------------------------------------------------------- #
# the bench, at 95/5
# --------------------------------------------------------------------------- #

def bench_efficiency():
    """(V, eff, err, spark_frac) from the 22 June det3 overnight scan.

    Re-derived 2026-08-28 on the current chain (see BENCH_EFF_SAT).  Plateau
    91-94 %, which now agrees with the 27 June scan; the two used to differ by
    ~10 points and no longer do.  `eff` and `spark_frac` are FRACTIONS -- the
    CSV carries the 02_efficiency percent columns too, under their own names.
    """
    v, e, de, sp = [], [], [], []
    with open(BENCH_EFF) as fh:
        for r in csv.DictReader(fh):
            v.append(float(r['hv']))
            e.append(float(r['eff_reco']))
            de.append(float(r['eff_reco_err']))
            sp.append(float(r['spark_frac']))
    o = np.argsort(v)
    return tuple(np.asarray(a, float)[o] for a in (v, e, de, sp))


def bench_efficiency_saturday():
    """(V, eff, err, spark) from the 27 June det3 scan, both passes merged.

    The two passes interleave (pass 1 at 425/435/445/455/465/..., pass 2 at
    460/470/480/...), were taken the same day on the same slot at the same
    drift voltage, and share no voltage, so nothing is averaged -- they are one
    curve sampled twice.  On the re-derived chain adjacent points from opposite
    passes agree to 1-2 points (465 -> 94.1 against 460 -> 94.8 and 470 ->
    94.9; 485 -> 93.7 against 480 -> 93.8 and 490 -> 93.7), which is the
    cross-check that the two passes are one measurement.

    Plateau 93-95 %.  Statistical errors are ~0.9 points, so the point-to-point
    scatter on the plateau is a little wider than statistics alone; treat a
    single point to better than ~1.5 points as unmeasured.
    """
    v, e, de, sp = [], [], [], []
    for path, _tag in BENCH_EFF_SAT:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                key = 'hv' if 'hv' in r else 'x'
                v.append(float(r[key]))
                e.append(float(r['eff_reco']))
                de.append(float(r['eff_reco_err']))
                sp.append(float(r['spark_frac']))
    o = np.argsort(v)
    return tuple(np.asarray(a, float)[o] for a in (v, e, de, sp))


def eff_turn_on_slope(n=3):
    """(slope per V, intercept) of a straight line through the LOWEST n points.

    Only for continuing the curve below 425 V, where the panel's axis reaches
    and the scan does not.

    ON THE RE-DERIVED CHAIN THIS IS ESSENTIALLY FLAT: three points
    (425/435/445 = 89.6/93.1/91.8 %) give **0.0011 per V**, two give 0.0035.
    It used to be 0.0141 per V, off a curve that rose 0.49 -> 0.81 over the
    same three points -- and that rise was the old analyzer's amplitude
    threshold, not the chamber (mesh_ladder.csv, same sub-runs, puts the weakest
    2 % of events at 69 ADC on the peak strip at 425 V, ~10 sigma over the
    6.85 ADC bench pedestal).

    So the continuation below 425 V is now a nearly horizontal line near 90 %,
    and it is a WEAK statement: the fit is through three points whose own
    scatter (+-1.5) is larger than the slope times the extrapolated span.  It
    says only "no turn-on is visible yet by 425 V", which is what the gain
    ladder independently says.  It must not be read as a measured plateau below
    425 V -- the chamber has to turn off eventually, this scan just never
    reaches low enough to see it.  Draw it dashed and label it extrapolated.
    """
    v, e, _, _ = bench_efficiency_saturday()
    k = np.argsort(v)[:n]
    m, c = np.polyfit(v[k], e[k], 1)
    return float(m), float(c)


def bench_eff_on_ntof_axis(era='production', n_extrap=3, v_min=None):
    """The saturday curve, placed on the n_TOF setpoint axis.

    Returns ``(v_meas, eff, err, v_line, eff_line, n_measured)``: the measured
    points (mapped, plotted as markers) and a line that is those points plus,
    if ``v_min`` reaches below the scan, a straight continuation off the
    lowest ``n_extrap`` of them.  ``n_measured`` says how many leading samples
    of the line are extrapolated, so the caller can dash exactly those.
    """
    shift = total_shift(era)['total']
    v, e, de, _ = bench_efficiency_saturday()
    vm = v + shift
    if v_min is None or v_min >= vm[0]:
        return vm, e, de, vm, e, 0
    m, c = eff_turn_on_slope(n_extrap)
    vx = np.arange(np.floor(v_min), vm[0], 1.0)
    ex = np.clip(m * (vx - shift) + c, 0.0, None)
    return vm, e, de, np.concatenate([vx, vm]), np.concatenate([ex, e]), len(vx)


def bench_gain_ladder(view='x', est='p10'):
    """(V, median strip amplitude [ADC]) from the 27 June saturday scan.

    ``p10`` -- the 10th percentile of the peak-strip amplitude -- is the
    estimator whose ladder stays unsaturated highest (to 500 V); ``p50``
    saturates from 495 V.  The rows carry their own ``<est>_ok`` flag and the
    saturated ones are dropped rather than fitted through.
    """
    v, a = [], []
    with open(MESH_LADDER) as fh:
        for r in csv.DictReader(fh):
            if r['view'] != view or r.get(f'{est}_ok') != 'True':
                continue
            v.append(float(r['volt']))
            a.append(float(r[est]))
    o = np.argsort(v)
    return np.asarray(v, float)[o], np.asarray(a, float)[o]


def bench_gain_slope():
    """d ln(gain)/dV per 10 V, MEASURED, averaged over the two views.

    slopes.json's ``p10_full`` -- the widest unsaturated span (425-500 V) of
    the estimator with the latest saturation onset.  This is the number that
    turns every gain ratio below into volts.  The simulated slope over the
    same window is 0.31/10 V, i.e. 1.4x shallower; that difference is a known,
    unexplained bench-vs-Garfield discrepancy and it is carried as a bracket,
    not resolved here.
    """
    sl = json.load(open(SLOPES))['data']
    per = [sl[v]['p10_full']['slope10'] for v in ('x', 'y')]
    err = [sl[v]['p10_full']['err10'] for v in ('x', 'y')]
    return float(np.mean(per)), float(np.mean(err))


def bench_charge_ladder(est='q_sum', v_max=505.0):
    """(V, Q, ln-error) -- TOTAL collected charge per track, 27 June scan.

    Both views averaged in the log (they are the same tracks seen twice and
    their ladders are parallel), one row per mesh voltage.  ``q_sum`` is the
    deconvolved forward-fit charge; ``q_win`` the model-free window sum.

    Cut at ``v_max`` = 505 V: above it the WINDOW SUM clips as well (tens of
    railed cells per track) and the fit is censoring most of the pulse, so
    nothing up there is a charge measurement.  See the grey band that
    10e_hv_scan_charge_angle.py draws on the same ladder.
    """
    rows = {}
    with open(CHARGE_LADDER) as fh:
        for r in csv.DictReader(fh):
            v = float(r['hv'])
            if v > v_max:
                continue
            rows.setdefault(v, []).append((float(r[est]),
                                           float(r[f'{est}_lnerr'])))
    v = np.array(sorted(rows))
    q = np.array([np.exp(np.mean([np.log(a) for a, _ in rows[k]])) for k in v])
    e = np.array([float(np.mean([b for _, b in rows[k]])) / np.sqrt(len(rows[k]))
                  for k in v])
    return v, q, e


def saturating_voltage(frac=0.5):
    """The bench mesh voltage at which ``frac`` of tracks rail the peak strip.

    ``peak_amp`` is the tallest SAMPLE of the tallest STRIP of the event -- the
    max strip, which is what the question is about, not a per-strip average.

    Dylan's definition of the ideal gain, 2026-08-28: *"aim for the peak strip
    in the median event to be saturated (just barely saturating is probably
    ideal gain)"*.  The median event is ``frac = 0.5``, and the scan measures
    it directly -- ``frac_sat`` goes 0.39 at 495 V to 0.66 at 500 V in BOTH
    views, so the crossing is bracketed by two measured points 5 V apart and
    is not an extrapolation.  Linear in the fraction inside that bracket.

    HOW MUCH THE DEFINITION MATTERS, measured 2026-08-29 on the 23,774 per-event
    ``peak_amp`` values, when Dylan said he remembered ~500 V off the gain plot:

      clipped at >= 0.88 x rail (3407)   V50 = 496.4 V
      clipped at >= 0.92 x rail (3550)   V50 = 497.0 V   <- what this returns
      clipped at >= 0.95 x rail (3678)   V50 = 497.1 V
      clipped at >= 0.98 x rail (3800)   V50 = 508.6 V   <- NOT a clipping test

    Stable to 0.7 V over any sane clipping threshold.  0.98 breaks because the
    railed population is not a delta function: per-channel pedestal subtraction
    spreads it over ~3700-3900, so a cut at 3800 stops asking "is this event
    clipped" and starts asking "did this channel's rail land high".

    **Both readings of the gain plot are right.**  Half the events have the max
    strip clipped at 497 V; the MEDIAN AMPLITUDE only reaches the nominal
    3871.5 rail near 500 V, because at the 50 % point half the sample is still
    below it.  500 V is what the eye reads off ``gain_vs_hv.png``, where the
    p50 marker visibly lies on the rail line -- there, 67 % of events are
    clipped and the median sits at 97 % of the rail.  Moving the anchor from
    497 to 500 V would lower every percentage on the gain scale by ~13 %.

    The spark veto is not what sets this: on the full M3-golden fiducial set
    V50 is 496.8 V against 497.0 spark-free.

    Returns (V, per-view dict).  The two views agree to 0.04 V.
    """
    per = {}
    for view in ('x', 'y'):
        v, f = [], []
        with open(CHARGE_LADDER) as fh:
            for r in csv.DictReader(fh):
                if r['view'] != view:
                    continue
                v.append(float(r['hv']))
                f.append(float(r['frac_sat']))
        o = np.argsort(v)
        per[view] = float(np.interp(frac, np.asarray(f)[o], np.asarray(v)[o]))
    return float(np.mean(list(per.values()))), per


def saturation_ladder():
    """(V, fraction of tracks whose MAX STRIP is railed), both views averaged.

    Straight off ``frac_sat`` in the charge ladder -- the spark-free,
    M3-golden, fiducial set, at the 3550 ADC censoring level.  This is the
    "when does the max strip start to clip" ladder; ``saturating_voltage``
    is just its 50 % point.
    """
    rows = {}
    with open(CHARGE_LADDER) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(float(r['hv']), []).append(float(r['frac_sat']))
    v = np.array(sorted(rows))
    return v, np.array([float(np.mean(rows[k])) for k in v])


def sim_gain_slope():
    """The simulated 90/10 ladder's slope per 10 V, for the bracket."""
    return float(json.load(open(SLOPES))['sim']['iso10_full']['slope10'])


# --------------------------------------------------------------------------- #
# the map: a bench voltage -> the n_TOF voltage with the same signal
# --------------------------------------------------------------------------- #

def gas_shift(v_ref_95_5=460.0):
    """Volts to add to a 95/5 voltage to keep the gain in 90/10, at CERN.

    Uses the blessed equivalence table by INVERSION: find the 90/10 voltage
    whose simulated gain equals the 95/5 gain at ``v_ref_95_5``.  GainMap maps
    the other way, so this scans.  Flat to ±0.6 V over 400-590 V, which is
    why the callers treat it as a constant.
    """
    gm = GainMap(pressure='CERN_450m')
    grid = np.arange(400.0, 620.0, 0.05)
    ref = gm.to_ref_voltage('Ar/Iso 90/10', grid)
    k = int(np.nanargmin(np.abs(ref - v_ref_95_5)))
    return float(grid[k] - v_ref_95_5)


def pressure_shift(v=460.0):
    """Volts to add going Saclay (160 m) -> CERN (450 m), in 95/5.

    Thinner air at CERN means fewer molecules in the gap, so the same voltage
    makes MORE gain and the equivalent voltage is LOWER: this is negative.
    """
    a = GainMap(pressure='Saclay_160m').ln_gain('Ar/Iso 95/5', v)
    b = GainMap(pressure='CERN_450m')
    grid = np.arange(400.0, 620.0, 0.05)
    k = int(np.argmin(np.abs(b.ln_gain('Ar/Iso 95/5', grid) - a)))
    return float(grid[k] - v)


def electronics_shift(era='run_55'):
    """Volts of extra gas gain n_TOF needs for the bench's signal-to-noise.

    Two factors, both documented rather than fitted: the CSA full-scale range
    (200 fC on the bench, 600 fC at n_TOF -> 3x less ADC per electron) and the
    per-channel noise the threshold rides on.  Their product is a threshold
    ratio in CHARGE, and the measured bench gain slope turns it into volts.
    """
    sigma = {'run_55': SIGMA_NTOF_RUN55_ADC,
             'production': SIGMA_NTOF_PROD_ADC}[era]
    ratio = (sigma * CSA_NTOF_FC) / (SIGMA_BENCH_ADC * CSA_BENCH_FC)
    slope10, _ = bench_gain_slope()
    return float(np.log(ratio) / (slope10 / 10.0)), float(ratio)


def total_shift(era='run_55'):
    """Volts to add to a BENCH (95/5, Saclay, 200 fC) voltage to get the
    n_TOF (90/10, CERN, 600 fC) voltage with the same signal-to-noise."""
    el, ratio = electronics_shift(era)
    terms = dict(gas=gas_shift(), pressure=pressure_shift(), electronics=el)
    terms['total'] = sum(terms.values())
    terms['threshold_ratio'] = ratio
    return terms


def adc_shift():
    """Volts to add to a BENCH voltage to get the n_TOF voltage with the same
    number of ADC COUNTS on the peak strip.

    This is NOT ``total_shift()``, and the difference is the whole reason this
    function exists.  ``total_shift`` answers *where does the bench's
    signal-to-noise reappear*, so it carries the per-channel noise; that is
    the right map for an EFFICIENCY, which is a threshold quantity.
    Saturation is not a threshold quantity -- the rail sits at a fixed number
    of ADC counts however noisy the channel is -- so the noise term must come
    OUT and only the CSA range stays in:

        gas 95/5 -> 90/10        +72.6 V   (same as always)
        Saclay -> CERN            -4.7 V   (same as always)
        CSA 200 fC -> 600 fC     +26.3 V   (3x less ADC per electron, so 3x
                                            the avalanche to reach the rail)
                                 --------
                                 +94.1 V

    against +102.7 V for the production threshold map.  The 8.6 V between them
    is exactly ln(9.80/6.85) worth of noise, and putting it in would be saying
    that a noisier channel saturates sooner.
    """
    slope10, _ = bench_gain_slope()
    terms = dict(gas=gas_shift(), pressure=pressure_shift(),
                 csa=float(np.log(CSA_NTOF_FC / CSA_BENCH_FC) / (slope10 / 10.0)))
    terms['total'] = float(sum(terms.values()))
    terms['adc_ratio'] = CSA_NTOF_FC / CSA_BENCH_FC
    return terms


def bench_gain_on_ntof_axis(est='q_sum', ref='bench200', n_top=5):
    """The charge ladder on the n_TOF axis, as a PER CENT OF OPTIMAL GAIN.

    100 % is the gain at which the peak strip of the median track just fills
    the readout.  WHICH readout is a choice, and it is worth 3x:

      ``bench200``  the DREAM the scan was taken with, 200 fC full scale.  The
                    measured point, ``saturating_voltage(0.5)`` = **bench
                    497 V**, which lands at **n_TOF 565 V**.  DEFAULT since
                    2026-08-29 (Dylan: *"I think 497 V is fine ... can we put
                    100 % at 497 V instead?"*).  Everything on the panel is
                    then measured: bench 425-505 V covers n_TOF 493-573 V, so
                    the setpoints AND the 100 % crossing sit inside data.
      ``ntof600``   the DREAM n_TOF actually ran, 600 fC full scale -- 3x less
                    ADC per electron, so 3x the avalanche to reach the same
                    rail.  That is bench ~518 V, n_TOF ~586 V, and it needs
                    ~13 V of continuation past the last trustworthy bench
                    point, so its top is extrapolated.

    Both are honest; they answer "full scale of what".  ``bench200`` is the
    one to lead with, for three reasons: it is the scan's own measured
    saturation point, nothing on the curve is extrapolated, and the 600 fC
    setting was forced by the gamma flash (668 pC on a strip -- 1113x the
    DREAM range) rather than chosen for tracking, so referring a TRACKING gain
    to it asks for 3x more avalanche than a MIP measurement needs.  Say which
    one a number came from; do not mix them.

    THE ARITHMETIC.  Only gas and site pressure move the VOLTAGE:

        pct(W)  =  Q_bench(W - 67.85) / Q_bench(497)            [bench200]
        pct(W)  =  Q_bench(W - 67.85) / (3 * Q_bench(497))      [ntof600]

    so n_TOF 560 V is read off the bench ladder at bench 492 V -- the plain gas
    equivalence.  **Corrected 2026-08-28**: the first version folded the factor
    3 into the voltage axis as ln(3)/slope = +26.3 V, one shift of +94.1 V.
    That is exact only for a straight ladder, and this one is CURVED (0.33 per
    10 V near 435-445 V, 0.52 near 485-505), so it read the wrong part of the
    ladder -- by -13 % at n_TOF 520 V and +6 % at 560 V.  Keep ``adc_shift()``
    for SAYING which bench voltage makes the same ADC; never for evaluating.

    Returns a dict: ``v``/``pct``/``lnerr`` the measured points, ``v_line`` and
    ``pct_line`` the drawn line (its first ``n_meas`` entries are the measured
    ones; on ``bench200`` that is all of them), ``v_opt`` the 100 % crossing,
    ``v_last_meas`` where the measurement stops, and the continuation slopes.
    """
    if ref not in ('bench200', 'ntof600'):
        raise ValueError(ref)
    shift = gas_shift() + pressure_shift()
    v, q, e = bench_charge_ladder(est)
    lnq = np.log(q)
    v_sat, _ = saturating_voltage(0.5)
    q_opt = float(np.exp(np.interp(v_sat, v, lnq)))
    if ref == 'ntof600':
        q_opt *= CSA_NTOF_FC / CSA_BENCH_FC

    k = v >= v[-n_top]
    s_top = float(np.polyfit(v[k], lnq[k], 1)[0])
    s_all = float(np.polyfit(v, lnq, 1)[0])
    v_end = v[-1] + (np.log(q_opt) - lnq[-1]) / s_top          # 100 % crossing
    if v_end <= v[-1]:                                          # inside the data
        v_end = float(np.interp(np.log(q_opt), lnq, v))
        vx, qx = np.array([]), np.array([])
    else:
        vx = np.arange(v[-1] + 1.0, v_end + 4.0, 1.0)
        qx = q[-1] * np.exp(s_top * (vx - v[-1]))

    return dict(
        ref=ref, v=v + shift, pct=100.0 * q / q_opt, lnerr=e,
        v_line=np.concatenate([v, vx]) + shift,
        pct_line=100.0 * np.concatenate([q, qx]) / q_opt,
        n_meas=len(v), v_last_meas=float(v[-1] + shift),
        v_opt=float(v_end + shift),
        v_opt_alt=float(v[-1] + (np.log(q_opt) - lnq[-1]) / s_all + shift)
        if len(vx) else float(v_end + shift),
        slope10=s_top * 10, slope10_all=s_all * 10, shift=shift)


def bracket():
    """The map's systematic, from three independent Garfield determinations.

    Each is a gain RATIO between the two mixtures turned into volts by a
    slope.  The spread, not any one of them, is the honest uncertainty.
    """
    meas10, _ = bench_gain_slope()
    sim10 = sim_gain_slope()
    out = {}

    # 1. the blessed equivalence table (its own logquad fits, both mixtures)
    out['equivalence_table'] = dict(dlnG=None, dV=gas_shift(),
                                    note='inverted gain match, CERN pressure')

    # 2. the T6 meshfield ladders -- both mixtures, same field maps, one
    #    overlapping voltage (530 V)
    p = json.load(open(os.path.expanduser(
        '~/x17/response_sim/avalanche/aval_calib_meshfield_hvscan.json')))['points']
    g = {k: v['polya']['gain_mean'] for k, v in p.items()}
    dln = np.log(g['Ar_iC4H10_95_5_Saclay_160m.gas@530V']
                 / g['Ar_iC4H10_90_10_Saclay_160m.gas@530V'])
    out['meshfield_530V'] = dict(dlnG=float(dln), dV_sim_slope=float(dln / (sim10 / 10)),
                                 dV_meas_slope=float(dln / (meas10 / 10)))

    # 3. the uniform-field slope hunt -- both mixtures at three voltages
    q = json.load(open(os.path.expanduser(
        '~/x17/response_sim/avalanche/aval_calib_slopehunt.json')))['points']
    r = [q[f'Ar_iC4H10_95_5_Saclay_160m.gas@{v}V@auto']['polya']['gain_mean']
         / q[f'Ar_iC4H10_90_10_Saclay_160m.gas@{v}V']['polya']['gain_mean']
         for v in (460, 490, 520)]
    dln = float(np.log(np.mean(r)))
    out['uniform_field'] = dict(dlnG=dln, dV_sim_slope=dln / (sim10 / 10),
                                dV_meas_slope=dln / (meas10 / 10))
    vals = [out['equivalence_table']['dV'],
            out['meshfield_530V']['dV_sim_slope'],
            out['meshfield_530V']['dV_meas_slope'],
            out['uniform_field']['dV_sim_slope'],
            out['uniform_field']['dV_meas_slope']]
    out['span_V'] = [float(min(vals)), float(max(vals))]
    return out


# --------------------------------------------------------------------------- #
# n_TOF: the yield ladder and the recovery ladder
# --------------------------------------------------------------------------- #

def ntof_yield(det=DET):
    """(V, b1 %, b2 %) -- MIP-track rate per trigger, run_55.

    b1 = 8-12 ms, b2 = 16-28 ms after the flash.  **b2 is the one to trust**:
    at 555-560 V the recovery reaches 8-14 ms, so b1's top points sit inside
    the chamber's own dead time and are suppressed by the very quantity this
    module is trading against -- using them would be circular.  b1 is kept
    because it has more statistics and because the two agreeing on the SHAPE
    below 550 V is worth seeing.
    """
    d = json.load(open(NTOF_SCAN))['track_rate_pct']
    b1, b2 = d[f'{det}_b1'], d[f'{det}_b2']
    v = np.array(sorted(int(k) for k in b1), float)
    return v, np.array([b1[str(int(x))] for x in v]), \
        np.array([b2[str(int(x))] for x in v])


def recovery_ladder(det=DET, run='run_57'):
    """(V, charge nC, recovery ms) per sub-run -- the flash-random probe.

    Same loaders the deck's flash slides use.  Chamber A ran drift 600 V in
    both run_55 and run_57, so the yield ladder and this one are at the same
    field.
    """
    rows = S.load_charge(run)
    rec = S.load_recovery(CACHE)
    q_by_sub = {r['subrun']: (r['q_per_pulse_nc'], r['resist_v'])
                for r in rows if r['det'] == det}
    out = []
    for sub, ms in rec.get(det, {}).items():
        if sub not in q_by_sub:
            continue
        q, v = q_by_sub[sub]
        if np.isfinite(q) and q > 0 and ms > 0.25:
            out.append((float(v), float(q), float(ms)))
    out.sort()
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2]


def recovery_fit():
    """ln(recovery ms) = c0·V + c1, with the scatter that survives it.

    The per-sub-run recovery is quantised to the analysis' log-time bins and
    scatters by x1.33 rms about this line, so the FIGURE OF MERIT is built on
    the fit and never on individual points -- one bin of scatter moves the
    cliff by several volts.
    """
    v, _, ms = recovery_ladder()
    c = np.polyfit(v, np.log(ms), 1)
    rms = float((np.log(ms) - np.polyval(c, v)).std())
    return c, rms


def recovery_at(V):
    c, _ = recovery_fit()
    return np.exp(np.polyval(c, np.asarray(V, float)))


def visible_fraction(V):
    """Fraction of the X17 rate arriving after the front end is back at V.

    The rate calculation and the log-uniform in-bin split are
    ``mpgd26/make_hv_window.surviving``; reproduced here through the same
    table so this module does not depend on the deck.
    """
    elo, ehi, y = X.load()
    t_lo, t_hi = X.t_of_E(ehi) * 1e6, X.t_of_E(elo) * 1e6
    tot = float(y.sum())
    out = []
    for t_cut in np.atleast_1d(recovery_at(V) * 1e3):
        kept = 0.0
        for a, b, w in zip(t_lo, t_hi, y):
            if t_cut <= a:
                kept += w
            elif t_cut < b:
                kept += w * np.log(b / t_cut) / np.log(b / a)
        out.append(kept / tot)
    out = np.asarray(out)
    return out if np.ndim(V) else float(out[0])


def figure_of_merit(window='b2'):
    """(V, visible fraction, relative yield, product) on the run_55 grid.

    The product is what the campaign was actually optimising: X17 rate that
    arrives after the chamber is alive, times the fraction of those tracks it
    reconstructs.  Both factors are relative, so the product is too -- it has
    a maximum and no units.
    """
    v, b1, b2 = ntof_yield()
    y = {'b1': b1, 'b2': b2}[window]
    rel = y / y.max()
    vis = visible_fraction(v)
    return v, vis, rel, vis * rel


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #

def results():
    meas10, meas_err = bench_gain_slope()
    bv, be, bde, bsp = bench_efficiency()
    out = dict(
        bench=dict(slope10=meas10, slope10_err=meas_err,
                   sim_slope10=sim_gain_slope(),
                   eff_max=float(be.max()), eff_max_V=float(bv[np.argmax(be)]),
                   eff_at_450=float(be[0]),
                   spark_5pct_V=float(np.interp(0.05, bsp, bv)),
                   spark_10pct_V=float(np.interp(0.10, bsp, bv))),
        shift=dict(run_55=total_shift('run_55'),
                   production=total_shift('production')),
        bracket=bracket(),
    )
    s55 = out['shift']['run_55']['total']
    sprod = out['shift']['production']['total']
    sv, se, _sde, _ssp = bench_efficiency_saturday()
    plat = se[(sv >= 455) & (sv <= 500)]
    out['saturday'] = dict(
        v_min=float(sv.min()), v_max=float(sv.max()),
        eff_min=float(se[np.argmin(sv)]),
        plateau=float(np.mean(plat)), plateau_lo_V=455.0, plateau_hi_V=500.0,
        # where the setpoints land, in each noise era
        eff_at_op={era: float(np.interp(OP_V - total_shift(era)['total'], sv, se))
                   for era in ('run_55', 'production')},
        eff_at_gain={era: float(np.interp(GAIN_V - total_shift(era)['total'], sv, se))
                     for era in ('run_55', 'production')},
    )
    out['mapped'] = dict(
        # where the n_TOF setpoints sit on the bench's own voltage axis
        op_bench_V_run55=OP_V - s55, op_bench_V_prod=OP_V - sprod,
        gain_bench_V_run55=GAIN_V - s55,
        # and where the bench's features land on the n_TOF axis
        plateau_lo_ntof_V=float(bv[0]) + s55,
        plateau_hi_ntof_V=float(bv[np.argmax(be)]) + s55,
        spark_10pct_ntof_V=float(np.interp(0.10, bsp, bv)) + s55,
    )
    vsat, vsat_per = saturating_voltage(0.5)
    g = bench_gain_on_ntof_axis('q_sum')
    gw = bench_gain_on_ntof_axis('q_win')
    g6 = bench_gain_on_ntof_axis('q_sum', ref='ntof600')
    grid = list(range(520, 565, 5))

    def _at(d, x):
        return float(np.exp(np.interp(x, d['v'], np.log(d['pct']))))

    sv, sf = saturation_ladder()
    out['gain_scale'] = dict(
        v_sat50_bench=vsat, v_sat50_per_view=vsat_per,
        onset={int(a): float(b) for a, b in zip(sv, sf)},
        onset_quartile=float(np.interp(0.25, sf, sv)),
        onset_5pct=float(np.interp(0.05, sf, sv)),
        onset_90pct=float(np.interp(0.90, sf, sv)),
        shift=g['shift'], adc_shift=adc_shift(),
        v_opt_ntof=g['v_opt'], v_opt_ntof_alt=g['v_opt_alt'],
        v_meas_ntof=[float(g['v'].min()), float(g['v'].max())],
        slope10=g['slope10'], slope10_all=g['slope10_all'],
        pct={int(v): _at(g, v) for v in grid},
        pct_qwin={int(v): _at(gw, v) for v in grid},
        # the same scale referred to the 600 fC range n_TOF actually ran
        ntof600=dict(v_opt_ntof=g6['v_opt'],
                     pct={int(v): _at(g6, v) for v in grid}),
    )
    fom = {}
    for w in ('b1', 'b2'):
        v, vis, rel, p = figure_of_merit(w)
        fom[w] = dict(V=v.tolist(), visible=vis.tolist(), rel_yield=rel.tolist(),
                      product=p.tolist(), best_V=float(v[int(np.argmax(p))]),
                      at_op=float(p[list(v).index(OP_V)]),
                      best=float(p.max()))
    out['fom'] = fom
    c, rms = recovery_fit()
    out['recovery'] = dict(doubling_V=float(np.log(2) / c[0]),
                           ln_rms=rms,
                           at_op_ms=float(recovery_at(OP_V)),
                           at_gain_ms=float(recovery_at(GAIN_V)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true', help='write results.json')
    a = ap.parse_args()
    r = results()

    print('--- the map: bench (95/5, Saclay, 200 fC) -> n_TOF (90/10, CERN, 600 fC)')
    for era in ('run_55', 'production'):
        s = r['shift'][era]
        print(f'  {era:11s} gas {s["gas"]:+6.1f}  pressure {s["pressure"]:+5.1f}'
              f'  electronics {s["electronics"]:+6.1f}'
              f' (threshold x{s["threshold_ratio"]:.2f})'
              f'   TOTAL {s["total"]:+6.1f} V')
    b = r['bracket']
    print(f'  bracket on the gas term: {b["span_V"][0]:.0f} to {b["span_V"][1]:.0f} V'
          f'   (table {b["equivalence_table"]["dV"]:.1f},'
          f' meshfield {b["meshfield_530V"]["dV_meas_slope"]:.0f}-'
          f'{b["meshfield_530V"]["dV_sim_slope"]:.0f},'
          f' uniform {b["uniform_field"]["dV_meas_slope"]:.0f}-'
          f'{b["uniform_field"]["dV_sim_slope"]:.0f})')

    m = r['mapped']
    print('\n--- where that puts us')
    print(f'  n_TOF {OP_V} V  == bench {m["op_bench_V_run55"]:.0f} V (run_55 era)'
          f' / {m["op_bench_V_prod"]:.0f} V (production)')
    print(f'  bench plateau {r["bench"]["eff_at_450"] * 100:.0f}-'
          f'{r["bench"]["eff_max"] * 100:.1f} % maps to n_TOF '
          f'{m["plateau_lo_ntof_V"]:.0f}-{m["plateau_hi_ntof_V"]:.0f} V')
    sat = r['saturday']
    print(f'  27 Jun scan {sat["v_min"]:.0f}-{sat["v_max"]:.0f} V: turn-on '
          f'{sat["eff_min"] * 100:.0f} % -> plateau {sat["plateau"] * 100:.0f} %; '
          f'540 V is worth {sat["eff_at_op"]["run_55"] * 100:.0f} % (July) / '
          f'{sat["eff_at_op"]["production"] * 100:.0f} % (production)')
    print(f'  bench 10 % spark fraction maps to n_TOF '
          f'{m["spark_10pct_ntof_V"]:.0f} V')

    g = r['gain_scale']
    print('\n--- how much gain we actually had  (100 % = median peak strip'
          ' just fills the 12-bit sample)')
    print(f'  max strip clips in 5 % of tracks by bench'
          f' {g["onset_5pct"]:.0f} V, a quarter by {g["onset_quartile"]:.0f} V,'
          f' 90 % by {g["onset_90pct"]:.0f} V')
    print(f'  median peak strip rails at bench {g["v_sat50_bench"]:.1f} V'
          f' (x {g["v_sat50_per_view"]["x"]:.1f},'
          f' y {g["v_sat50_per_view"]["y"]:.1f})')
    am = g['adc_shift']
    print(f'  100 % = fills the 200 fC DREAM the scan ran  ->  n_TOF'
          f' {g["v_opt_ntof"]:.0f} V.  For the 600 fC range n_TOF ran, 3x the'
          f' charge  ->  n_TOF {g["ntof600"]["v_opt_ntof"]:.0f} V')
    print(f'  the ladder is evaluated at V - {g["shift"]:.2f} V (gas+pressure);'
          f' the one-shift form {am["total"]:+.1f} V is equivalent only for a'
          f' straight ladder, so it is not used')
    print(f'  [same-gas-gain {g["shift"]:+.1f} | same-ADC {am["total"]:+.1f}'
          f' | same-S/N {r["shift"]["production"]["total"]:+.1f}]')
    print(f'  measured bench charge covers n_TOF '
          f'{g["v_meas_ntof"][0]:.0f}-{g["v_meas_ntof"][1]:.0f} V;'
          f' above that the curve is a continuation on'
          f' {g["slope10"]:.3f}/10 V')
    print(f'  {"V":>5} {"% optimal":>10} {"(window sum)":>13}'
          f' {"vs 600 fC":>11}')
    for v in sorted(g['pct']):
        print(f'  {v:5d} {g["pct"][v]:9.1f} % {g["pct_qwin"][v]:12.1f} %'
              f' {g["ntof600"]["pct"][v]:10.1f} %')

    print('\n--- the trade, on the run_55 grid')
    v, vis, rel, p = figure_of_merit('b2')
    _, vis1, rel1, p1 = figure_of_merit('b1')
    print(f'  {"V":>5} {"recovery":>9} {"visible":>8} {"yield b2":>9}'
          f' {"product":>8} | {"yield b1":>9} {"product":>8}')
    for i, vv in enumerate(v):
        print(f'  {vv:5.0f} {recovery_at(vv):8.2f}ms {vis[i] * 100:7.2f}%'
              f' {rel[i]:9.3f} {p[i]:8.4f} | {rel1[i]:9.3f} {p1[i]:8.4f}')
    for w in ('b2', 'b1'):
        f = r['fom'][w]
        print(f'  optimum ({w}): {f["best_V"]:.0f} V'
              f'   -- {OP_V} V is at {f["at_op"] / f["best"] * 100:.0f} % of it')

    if a.json:
        path = os.path.join(HERE, 'results.json')
        json.dump(r, open(path, 'w'), indent=1)
        print(f'\n  -> {path}')


if __name__ == '__main__':
    main()
