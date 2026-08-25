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
   passes) runs **425-525 V** and is the only one that reaches below the
   plateau -- 49 % at 425 V climbing to 81 % by 455 -- so it is the one the
   deck plots; the **22 June overnight scan** (``bench_efficiency()``) runs
   450-525 V, starts already flat, and cannot show a turn-on at all.  They
   agree on the plateau's flatness and on the discharge collapse; 27 June's
   level is ~10 points lower because det3 sat in the top slot there (z 702,
   FEU 7/8) instead of the bottom (z 232, FEU 3/4), twice the M3 lever arm
   into the same fixed 5 mm match box.  Same efficiency definition in both.
   Spark fraction and the older mapping numbers still come from 22 June.
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
# passes, 425-525 V and 460-520 V, and it is the only bench scan that reaches
# BELOW the plateau: 0.49 at 425 V climbing to 0.81 by 455 V.  The 22 June scan
# starts at 450 V, already on the plateau, so it cannot show a turn-on at all.
# Its plateau sits ~10 points lower than 22 June's because the top slot doubles
# the M3 lever arm into the same fixed 5 mm match box -- same chamber, same
# efficiency definition, different pointing.  Use it for SHAPE, and say which
# scan any absolute number came from.  This is also the run that produced
# ``mesh_ladder.csv``, so the efficiency and the gain ladder are the same scan.
BENCH_EFF_SAT = [
    ('/media/dylan/data/x17/cosmic_bench/Analysis/'
     'mx17_det3_saturday_scan_6-27-26/hv_scan/mx17_3/efficiency_vs_hv_scan.csv',
     'pass 1'),
    ('/media/dylan/data/x17/cosmic_bench/Analysis/'
     'mx17_det3_saturday_scan_6-27-26/hv_scan2/mx17_3/'
     'efficiency_vs_hv_scan2.csv', 'pass 2'),
]
MESH_LADDER = os.path.expanduser('~/x17/response_sim/hv_slope/mesh_ladder.csv')
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


# --------------------------------------------------------------------------- #
# the bench, at 95/5
# --------------------------------------------------------------------------- #

def bench_efficiency():
    """(V, eff, err, spark_frac) from the 22 June det3 overnight scan."""
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

    The two passes interleave (425/435/... and 460/470/...), were taken the
    same day on the same slot at the same drift voltage, and agree to about
    0.02 in the overlap, so they are one curve sampled twice.  Sorted by
    voltage; duplicates are not averaged because there are none.
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
    and the scan does not.  Three points (425/435/445) give 0.0141 per V; two
    give 0.0176.  Extrapolated values are drawn dashed and clipped at zero --
    a turn-on is not a straight line and this one would cross zero at 410 V.
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
