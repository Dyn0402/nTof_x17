#!/usr/bin/env python3
"""make_model_deck.py -- the physical detector forward model for MX17, and how it differs from the
production (r06) sharing kernel, as a figure-first slide note.

Part 1 explains the new model on its own, as the chain from a track to the strip waveforms
(ionisation and drift, the readout footprint, the RC spread along the resistive strips, the
electronics template, the gas by setup, the parameter list). Part 2 compares it with the old
copy kernel (c1, c2/c1, kY, tau_s, sigma_s, sigma_p0, Dp) and shows the held-out bench result.
The last slide lists what is open; nothing in it needs new data (the detectors are gone).

Every number is read from the analysis outputs, never typed in, except where noted (a few FINDINGS
text values that no JSON carries; each is marked ``# FINDINGS §n``):

  results/bench_uniform.json         joint X+Y head-on fit per bench chamber (sigma0, Dd, D_rc) = the rcm arms
  results/compare_bench_rc.json      held-out bench, paired, scale-corrected s68 (rcm, rcf3, rcf4x, rcfD)
  results/compare_bench_lor.json     the same for the footprint-tail arms (rcmlor, rcmlor2)
  results/rc_diffusion.json          RC fit per view, bench + beam (D, prompt-only vs +RC rms)
  results/rc_drift.json              beam det4 RC fit with drift diffusion (D_rc, sigma0 per view)
  results/iso_fit.json, x_width_beam.json   the X width excess
  results/decompose_tmax*.json       prompt + delayed decomposition (beam Y tau)
  results/headon_masked_k12.json, bench_driftscan.json   beam / bench late-charge stacks
  results/beam_comp_fit_air_hs.json, driftscan_fit_air_hs_x.json   gas compositions
  results/magboltz_bench_dry_amp.json, gasmodel.GasGrid  Magboltz transport
  results/bench_tanscan_det{3,4}.json  sigma0 with a tight |tan| cut
  production bundles (bundle.json + arrays.npz) on the data drive: old hypers and the X/Y templates

    ../../../.venv/bin/python make_model_deck.py [--out PATH]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug mx17-detector-model-slides --force --deploy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                   # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,      # noqa: E402,F401
                      INK, MUT, RULE, DBLUE, DRED, DGREY, DGREEN, DMUT, DINK, CARD)
import gasmodel as M                                                    # noqa: E402

RES = os.path.join(HERE, 'results')
OUT = os.path.expanduser('~/x17/cosmic_bench/cloud_basics/model_deck')
DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
PITCH = 0.78                      # readout pitch [mm] (wft.model.PITCH)
RPITCH, RWIDTH = 0.80, 0.55       # resistive strips: pitch, width [mm] (FINDINGS §3, from the gerbers)
AMP_GAP_CM = 0.015                # 150 µm amplification gap
GAP = 30.0                        # drift gap [mm]
OLD_C, NEW_C = RED, GREEN         # production kernel / physical kernel, on every slide
X_C, Y_C = BLUE, PURPLE           # readout views
DRY_C, WET_C = INK, ORANGE        # Magboltz dry / Magboltz with the water v implies
S_C = {0: INK, 1: ORANGE, 2: PURPLE}   # strip offsets 0, ±1, ±2

# production bundles the held-out bench compared against (FLEET_DIGEST "Which calibration each column is")
BROOT = '/media/dylan/data/x17/cosmic_bench/Analysis'
PROD = {
    'det2': ('o22_long_det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2/wft/calib_bundle_r06'),
    'det3': ('sat_det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/calib_bundle_r06'),
    'det4': ('g_det4', 'mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle_lp_t0p'),
    'det6': ('g_det6_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6/wft/calib_bundle_lp'),
    'det7': ('g_det7_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7/wft/calib_bundle_r06'),
}
# Fallback copy of the production hypers (read from the bundles above on 2026-10-16) for machines
# without the data drive. Used only when a bundle.json is missing; the build prints a warning.
PROD_FALLBACK = {
    'det2': dict(c1=0.051325, c2_over_c1=0.6, kY=5.40466, tau_s=133.94, sigma_s=172.22, sigma_p0=0.39252, Dp=0.015137, v=39.94),
    'det3': dict(c1=0.050449, c2_over_c1=0.6, kY=2.88252, tau_s=166.29, sigma_s=9.148, sigma_p0=0.42459, Dp=0.014361, v=36.6),
    'det4': dict(c1=0.054343, c2=0.036538, kY=2.97628, tau_s=129.32, sigma_s=47.26, sigma_p0=0.26090, Dp=0.016346, v=34.16),
    'det6': dict(c1=0.064216, c2=0.052821, kY=3.32840, tau_s=104.77, sigma_s=165.86, sigma_p0=0.038802, Dp=0.0016051, v=26.7),
    'det7': dict(c1=0.050724, c2_over_c1=0.6, kY=5.60257, tau_s=134.73, sigma_s=52.14, sigma_p0=0.43396, Dp=0.015313, v=36.6),
}
# Values that live only in the FINDINGS text (no JSON carries them)
X_CHI2 = {'det3': (61, 84), 'det2': (110, 189)}            # FINDINGS §12: X chi2/dof, production -> rcm
C2C1_INVERTED = {'det3': 1.14, 'det2': 1.53, 'det7': 1.75, 'det4': 2.12}   # CLAUDE.md / calib.py: shipped before 2026-08-21
C2C1_BEAM, C2C1_BENCH3 = (0.45, 0.03), (0.63, 0.10)        # calib.py: H4 head-on model-free; det3 near-vertical cosmics
TAU_DIRECT_DET3 = {'x': 230, 'y': 410}                     # FINDINGS §4 (HANDOFF_2026-07-30): direct waveform, det3 bench
COMPOSITIONS = [                                           # FINDINGS §25 table (run_71 and bench det3 are read from JSON)
    ('run_56 CO₂ gas', 291, '1 Aug, CO₂ period: 1.56 % water, 0.139 % air; ZS 5σ, rough (257–355 ppm)'),
    ('run_63 25.6°, 142 V/cm', 230, '3 Aug 00:22: 0.110 % air; ZS'),
    ('run_63 25.6°, 108 V/cm', 199, '3 Aug 00:30: 0.095 % air; ZS'),
    ('run_63 flat, 243 V/cm', 147, '3 Aug 01:00–01:54: 1.59 % water, 0.070 % air; ZS'),
]

G = dict(
    headon=('Head-on in a view: |tan θ| < 0.03 in that view, so the charge from every depth lands on the same few '
            'strips. The neighbour signal is then pure lateral spread, with no drift ladder mixed in.'),
    s68=('s68: the half-width of the central 68 % of (reconstructed − reference) track angle, in degrees, on the same '
         '2000 held-out bench events for both kernels (paired bootstrap error). Scale-corrected: each arm\'s tan is '
         'first divided by its own fitted slope against the M3 reference, so a kernel that reads angles a few % flat '
         'is not rewarded or punished for it.'),
    rcm=('rcm: the physical kernel with every constant taken from the head-on measurement (bench_uniform.json, '
         'joint X+Y fit), no χ² refit: σ_p0 = σ0, Dp = √(2·Dd), rc_D_y = D_rc, rc_D_x = 0, X template for both views.'),
    magboltz=('Magboltz (via Garfield++): electron transport from cross sections. Gives v, D_T, D_L and the attachment '
              'coefficient η for any mixture and field.'),
    bench=('The June 2026 cosmic bench: MX17 chambers det2/3/4/6/7 under the M3 reference telescope, Ar/iso 95/5, '
           'five "golden" long runs (sat_det3, o22_long_det2, g_det4, g_det6_long, g_det7_long).'),
    beam=('det4 at SPS H4 (31 Jul – 3 Aug 2026), Ar/CF₄/iso 88/10/2. run_71 is the RAW dataset (every sample read) at '
          'three drift fields: 243 / 150 / 92 V/cm.'),
    r06=('r06: the production calibration of the waveform reconstruction (wft) on the bench, c2 = 0.6·c1. det4 and '
         'det6 run their own bundles (calib_bundle_lp_t0p, calib_bundle_lp).'),
    dt=('D_T, transverse diffusion coefficient: σ_drift = D_T·√z after drifting a depth z. Quoted in µm/√cm. '
        'The model\'s Dd is the same thing per unit time: σ² = 2·Dd·u after a drift time u, so D_T² = 2·Dd / v.'),
)


def J(name, base=RES):
    return json.load(open(os.path.join(base, name)))


def frac(o, sig):
    """Fraction of a Gaussian of width sig (mm) centred on strip 0 that lies over strip offset o."""
    sig = np.maximum(np.asarray(sig, float), 1e-6)
    z = 1.0 / (np.sqrt(2) * sig)
    return 0.5 * (erf((o * PITCH + PITCH / 2) * z) - erf((o * PITCH - PITCH / 2) * z))


def dt_from_dd(dd, v_um_ns):
    """Dd [mm²/ns] -> D_T [µm/√cm] at drift velocity v."""
    return math.sqrt(2 * dd / (v_um_ns * 1e-3)) * 1e3 * math.sqrt(10)


def dt_from_dp(dp, v_um_ns):
    """old Dp [mm/√ns] (σ² += Dp²·u) -> D_T [µm/√cm]."""
    return dp / math.sqrt(v_um_ns * 1e-3) * 1e3 * math.sqrt(10)


def errbar(P, x, y, e, color, w=2.5, cap=7, horizontal=False):
    if horizontal:
        a, b, yy = P.X(x - e), P.X(x + e), P.Y(y)
        return (sd.line(a, yy, b, yy, color, w) + sd.line(a, yy - cap, a, yy + cap, color, w)
                + sd.line(b, yy - cap, b, yy + cap, color, w))
    xx, a, b = P.X(x), P.Y(y - e), P.Y(y + e)
    return (sd.line(xx, a, xx, b, color, w) + sd.line(xx - cap, a, xx + cap, a, color, w)
            + sd.line(xx - cap, b, xx + cap, b, color, w))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self):
        self.U = J('bench_uniform.json')
        self.CR = J('compare_bench_rc.json')
        self.CL = J('compare_bench_lor.json')
        self.RD = J('rc_diffusion.json')
        self.RDr = J('rc_drift.json')
        self.ISO = J('iso_fit.json')
        self.XB = J('x_width_beam.json')
        self.DC1, self.DC3 = J('decompose_tmax1300.json'), J('decompose_tmax3500.json')
        self.H = J('headon_masked_k12.json')
        self.BD = J('bench_driftscan.json')
        self.BC = J('beam_comp_fit_air_hs.json')
        self.DF = J('driftscan_fit_air_hs_x.json')
        self.AMP = J('magboltz_bench_dry_amp.json')
        self.WE = J('bench_water_e.json')
        self.TS = {d: J(f'bench_tanscan_{d}.json')[d] for d in ('det3', 'det4')}
        self.Gn = M.GasGrid('bench', 'air_hs')
        self.Gb = M.GasGrid('beam', 'air_hs')
        self.warn = []
        self.prod, self.tmpl = {}, {}
        for det, (key, rel) in PROD.items():
            p = os.path.join(BROOT, rel)
            try:
                b = json.load(open(os.path.join(p, 'bundle.json')))
                h = {k: float(v) for k, v in b['hyper'].items()}
                h['v'] = float(b['v_drift'])
                h['share_mode'] = b.get('share_mode') or 'delay (loader default)'
                h['bundle'] = os.path.basename(rel)
                self.prod[det] = h
            except OSError:
                self.warn.append(f'{det}: bundle not found, using the embedded copy of its hypers')
                self.prod[det] = dict(PROD_FALLBACK[det], share_mode='delay', bundle=os.path.basename(rel))
            if det == 'det3':
                try:
                    z = np.load(os.path.join(p, 'arrays.npz'))
                    self.tmpl = dict(grid=z['grid'].astype(float), x=z['tmpl_x'] / z['tmpl_x'].max(),
                                     y=z['tmpl_y'] / z['tmpl_y'].max())
                except OSError:
                    T = J('template_det3_x.json')
                    tx = np.array(T['tmpl_x'])
                    self.tmpl = dict(grid=np.array(T['grid'], float), x=tx / tx.max(), y=None)
                    self.warn.append('det3 arrays.npz not found: Y template unavailable, X from template_det3_x.json')
        for det in DETS:
            h = self.prod[det]
            h['c2eff'] = h['c2_over_c1'] * h['c1'] if 'c2_over_c1' in h else h['c2']
            h['r21'] = h['c2eff'] / h['c1']
        # new-model constants per bench chamber (the rcm arms)
        self.new = {}
        for det in DETS:
            f = self.U[det]['Dd_free']
            v = self.prod[det]['v']
            self.new[det] = dict(sig0=f['sig0'], Dd=f['Dd'], Drc=f['Drc'], v=v, DT=dt_from_dd(f['Dd'], v),
                                 E=self.WE[det]['run']['E_Vcm'])
        # beam det4 D_rc: rc_drift with Magboltz drift diffusion, loss-corrected (/0.94, FINDINGS §27)
        self.beam_drc = [self.RDr[k]['y']['magboltz']['D'] for k in ('sps700', 'sps450', 'sps275')]
        self.beam_drc_corr = float(np.mean(self.beam_drc)) / 0.94
        self.beam_s0 = {v: [self.RDr[k][v]['magboltz']['sig0'] for k in ('sps700', 'sps450', 'sps275')] for v in 'xy'}
        # avalanche spread over the 150 µm amplification gap, dry bench gas at 25-40 kV/cm
        self.aval = [p['DT_um_rtcm'] * math.sqrt(AMP_GAP_CM) for p in self.AMP['points']]
        # beam gas: run_71 composition -> attachment rate and survival over the gap
        self.beam_gas = {}
        for E in (243, 150, 92):
            g = self.Gb(self.BC['water'], self.BC['air'], float(E))
            self.beam_gas[E] = dict(v=float(g['v']), etav=float(g['etav']), DT=float(g['DT']),
                                    surv=math.exp(-float(g['etav']) * GAP * 1e3 / float(g['v'])))

    def cmp(self, det, arm, view, src=None):
        q = (src or self.CR)[f'{det}_{arm}_{view}']['scaled']
        return dict(all=q['d'][0], head=q['d'][1], all_e=q['err'][0], head_e=q['err'][1],
                    prod_all=q['prod'][0], arm_all=q['arm'][0], prod_head=q['prod'][1], arm_head=q['arm'][1],
                    slope_prod=q['prod'][3], slope_arm=q['arm'][3])


# --------------------------------------------------------------------------- #
# model waveforms (a unit charge landing on the centre of strip 0, head-on)
# --------------------------------------------------------------------------- #
TT = np.arange(-200.0, 1400.0, 10.0)        # the templates end at +1.39 µs, as in wft


def _h(d, view, shift=0.0, smooth=0.0):
    g, h = d.tmpl['grid'], d.tmpl[view]
    if smooth > 0:
        h = gaussian_filter1d(h, max(smooth, 1.0) / 10.0)
    return np.interp(TT - shift, g, h, left=0, right=0)


def old_wave(d, det, view, o):
    """wft.model.build_matrix (delay form) for one landing: strip o gets its own prompt fraction through the
    view's template, plus c1·kY copies of its ±1 neighbours' fractions delayed by tau_s and c2·kY copies of its
    ±2 neighbours' delayed by 2·tau_s, both smeared by sigma_s."""
    h = d.prod[det]
    k = h['kY'] if view == 'y' else h.get('cX', 1.0)
    c1, c2 = h['c1'] * k, h['c2eff'] * k
    F = lambda j: float(frac(j, h['sigma_p0']))          # noqa: E731
    w = F(o) * _h(d, view)
    w = w + c1 * (F(o - 1) + F(o + 1)) * _h(d, view, h['tau_s'], h['sigma_s'])
    w = w + c2 * (F(o - 2) + F(o + 2)) * _h(d, view, 2 * h['tau_s'], h['sigma_s'])
    return w


def new_wave(d, sig0, drc, o, smax=3000.0, ds=10.0):
    """wft.model.build_matrix_rc for one landing (on a fine time-since-landing grid): the strip sees each
    increment of the charge above it through the X template."""
    w = float(frac(o, sig0)) * _h(d, 'x')
    if drc > 0:
        s = np.arange(0.0, smax + ds, ds)
        F = frac(o, np.sqrt(sig0 ** 2 + 2 * drc * s))
        dF = np.diff(F)
        mids = 0.5 * (s[1:] + s[:-1])
        for a, m in zip(dF, mids):
            if abs(a) > 1e-7:
                w = w + a * _h(d, 'x', m)
    return w


# --------------------------------------------------------------------------- #
# Part 0: answer + setup
# --------------------------------------------------------------------------- #
def s_cover(D, d):
    ya = [d.cmp(det, 'rcm', 'y') for det in DETS]
    nbetter = sum(1 for c in ya if c['all'] + 2 * c['all_e'] < 0)
    lo_a, hi_a = max(c['all'] for c in ya), min(c['all'] for c in ya)
    lo_h, hi_h = max(c['head'] for c in ya), min(c['head'] for c in ya)
    drc = [d.new[k]['Drc'] for k in DETS]
    tip_y = 'Held-out bench, 2000 events per chamber, scale-corrected s68 (all / head-on):\n' + '\n'.join(
        f'{k}: {c["all"]:+.2f} ± {c["all_e"]:.2f} / {c["head"]:+.2f} ± {c["head_e"]:.2f}°' for k, c in zip(DETS, ya))
    nums = ''.join([
        sd.bignum(f'{nbetter} / 5', 'chambers with a better Y angle, no refit', DGREEN,
                  f'Y s68 {lo_a:+.2f} to {hi_a:+.2f}° on all tracks, {lo_h:+.2f} to {hi_h:+.2f}° head-on.',
                  tip=tip_y, size=64),
        sd.bignum('5 → 1', 'fitted sharing constants replaced by one measured D_rc', DBLUE,
                  'c1, c2, kY, τ_s, σ_s become a resistive-strip diffusion constant on Y; X has none.',
                  tip='Per chamber D_rc (Y, joint head-on fit), 10⁻⁴ mm²/ns:\n' + '\n'.join(
                      f'{k}: {d.new[k]["Drc"] * 1e4:.1f}' for k in DETS)
                      + f'\ndet4 at the beam: {d.beam_drc_corr * 1e4:.1f} (loss-corrected)', size=64),
        sd.bignum('X: open', 'the X width grows faster than the gas allows', DRED,
                  'det4 X is worse, X χ² is worse everywhere, and the kernel is not frozen.',
                  tip='See slides §§heldout§§, §§x-arms§§ and §§xwidth§§.', size=64),
    ])
    body = (sd.kicker('MX17 resistive-strip micromegas · the reconstruction\'s forward model · 16 Oct 2026')
            + '<h1 style="font-size:74px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1660px">'
              'A physical detector model replaces the fitted copy kernel: drift diffusion, a readout footprint, '
              'and RC spreading along the resistive strips.</h1>'
            + f'<p style="font-size:30px;color:{DMUT};width:1560px;line-height:1.35">Every constant is measured on '
              f'head-on tracks (bench and beam), none is fitted to the angle. With no refit, the Y angle improves on '
              f'all five bench chambers. X does not close yet.</p>'
            + '<div style="flex:1"></div>'
            + f'<div style="display:flex;gap:56px">{nums}</div>')
    D.slide('cover', body, f'''
<p><b>What this note is.</b> The waveform reconstruction (<code>wft/</code>) fits every strip's sampled signal with a forward
model: given a track's position, inclination and start time, it predicts each strip's waveform and solves the charge per
drift-depth slice by NNLS. How charge reaches a strip's neighbours is the part of that model that sets the angle. Until
October it was a fitted <i>copy kernel</i> (c1, c2, kY, τ_s, σ_s on top of a Gaussian σ_p0 and a drift term Dp). The
cloud_basics study (FINDINGS §1–29) replaced it, piece by piece, with physics measured directly on head-on tracks.</p>
<p><b>Part 1</b> (slides §§chamber§§–§§params§§) explains the new model on its own, in the order the charge travels. <b>Part 2</b>
(§§old-new§§–§§heldout§§) sets it against the old kernel and shows what the held-out bench says. Slides §§x-arms§§–§§open§§ are what is
not settled.</p>
<p><b>Status (16 Oct 2026).</b> The physical kernel is implemented as <code>wft.model.build_matrix_rc</code> (opt-in through the
<code>rc_D_y</code> hyper; production bundles are untouched and bit-identical). It is not frozen and the campaign re-pass is on
hold until the X width question is answered or stated as a limitation. The detectors are no longer available: everything
here, including what is open, works from data and simulation already in hand.</p>
<p>Generated by <code>sps_beam_test_26/analysis/cloud_basics/make_model_deck.py</code>.</p>''', dark=True, short='Answer')


def s_chamber(D, d):
    """Side cross-section of the readout (not to scale) + top view of the resistive strips."""
    W, H = 1000, 640
    o = []
    x0, x1 = 40, 960
    # drift region (compressed)
    o.append(f'<rect x="{x0}" y="20" width="{x1 - x0}" height="200" fill="#eef1f6" stroke="{RULE}"/>')
    o.append(sd.line(x0, 20, x1, 20, INK, 4))
    o.append(sd.T(x0 + 10, 48, 'drift cathode', 22, INK, 'start', 600))
    o.append(sd.T(x1 - 10, 48, f'drift gap {GAP:.0f} mm (det3: 27.9 mm)', 22, MUT, 'end',
                  tip='The 30 mm drift gap, where the track ionises the gas. det3\'s cathode is dished: 27.9 mm.'))
    tx = 300
    o.append(sd.line(tx, 0, tx + 60, 220, INK, 4))
    o.append(sd.T(tx + 70, 130, 'track', 22, INK, 'start', 600))
    for k in range(7):
        y = 40 + k * 26
        xx = tx + 60 * (y / 220)
        o.append(f'<circle cx="{xx:.1f}" cy="{y}" r="6" fill="{BLUE}"/>')
    o.append(sd.arrow(820, 70, 820, 200, MUT, 3, 12))
    o.append(sd.T(832, 140, 'electrons drift down', 20, MUT, 'start'))
    # mesh
    o.append(sd.line(x0, 222, x1, 222, INK, 3, '7 5'))
    o.append(sd.T(x0 + 10, 248, 'mesh', 22, INK, 'start', 600))
    # amplification gap
    o.append(f'<rect x="{x0}" y="224" width="{x1 - x0}" height="70" fill="#fbf1e6" stroke="none"/>')
    o.append(sd.T(x1 - 10, 266, 'amplification gap 150 µm (avalanche)', 22, ORANGE, 'end', 600,
                  tip='~30–40 kV/cm: each electron multiplies. The avalanche spreads laterally by only ~25 µm '
                      '(Magboltz D_T 180–225 µm/√cm over 150 µm).'))
    # resistive strips (cut across x: segments 550 wide on 800 pitch)
    sc = 150.0                      # px per mm for the readout layers
    cx = 500
    for j in range(-3, 4):
        xc = cx + j * RPITCH * sc
        o.append(f'<rect x="{xc - RWIDTH * sc / 2:.1f}" y="296" width="{RWIDTH * sc:.1f}" height="16" fill="{GREY}"'
                 f'{sd.tipattr("Resistive strip: 550 µm wide on a 0.80 mm pitch, running along y (out of the page). Charge on it can move along y, never across to the next strip in x.")}/>')
    o.append(f'<rect x="{x0}" y="314" width="{x1 - x0}" height="40" fill="#f6f4ef" stroke="none"/>')
    o.append(sd.T(x0 + 10, 340, 'resistive strips (run along y, into the page)', 21, INK, 'start', 600))
    o.append(sd.T(x1 - 10, 342, 'insulator ~50 µm · pads', 20, MUT, 'end'))
    # Y strips (layer 5): run along x -> one long bar in this cut
    o.append(f'<rect x="{x0 + 20}" y="362" width="{x1 - x0 - 40}" height="14" fill="{Y_C}" fill-opacity="0.75"'
             f'{sd.tipattr("Y readout strips (board layer 5): they run along x and measure y. In this cut one strip is a long bar.")}/>')
    o.append(sd.T(x0 + 10, 400, 'Y readout strips (run along x, measure y)', 21, Y_C, 'start', 600))
    # X strips (layer 6): cut across -> segments at 0.78 pitch
    for j in range(-3, 4):
        xc = cx + j * PITCH * sc
        o.append(f'<rect x="{xc - 0.45 * PITCH * sc:.1f}" y="410" width="{0.9 * PITCH * sc:.1f}" height="14" '
                 f'fill="{X_C}" fill-opacity="0.8"{sd.tipattr("X readout strips (layer 6, one layer below Y): pitch 0.78 mm, they run along y and measure x.")}/>')
    o.append(sd.T(x0 + 10, 450, 'X readout strips (run along y, measure x), one layer deeper', 21, X_C, 'start', 600))
    # induced footprint on X
    xs = np.linspace(cx - 2.2 * sc, cx + 2.2 * sc, 90)
    ys = 520 - 50 * np.exp(-0.5 * ((xs - cx) / (0.42 * sc)) ** 2)
    o.append(sd.poly(list(xs), list(ys), GREEN, 3.5, tip='The prompt footprint: the induced charge on the readout strips '
                     'from one avalanche, σ0 ≈ 0.4 mm. It is set by the board (resistive strip width, induction through '
                     'the layers), not by the gas.'))
    o.append(sd.line(cx - 2.4 * sc, 520, cx + 2.4 * sc, 520, RULE, 1.5))
    o.append(sd.T(cx + 2.5 * sc, 512, 'footprint σ0 ≈ 0.4 mm', 21, GREEN, 'start', 600))
    o.append(sd.T(W / 2, H - 30, 'Cross-section across x. Vertical scale not uniform: the drift gap is 30 mm, '
                                 'everything below the mesh is under 0.5 mm.', 19, MUT))
    side = sd.svg(W, H, ''.join(o), 'readout cross-section')

    # top view
    W2, H2 = 560, 640
    q = []
    s2 = 70.0
    ox, oy = 280, 300
    for j in range(-3, 4):
        xc = ox + j * RPITCH * s2
        q.append(f'<rect x="{xc - RWIDTH * s2 / 2:.1f}" y="40" width="{RWIDTH * s2:.1f}" height="460" fill="{GREY}" fill-opacity="0.35"/>')
    for r, a in ((0.6, 0.55), (1.3, 0.35), (2.2, 0.2)):
        q.append(f'<ellipse cx="{ox}" cy="{oy}" rx="{RWIDTH * s2 / 2:.1f}" ry="{r * s2:.1f}" fill="{PURPLE}" fill-opacity="{a}"/>')
    q.append(f'<circle cx="{ox}" cy="{oy}" r="7" fill="{INK}"/>')
    q.append(sd.arrow(ox + 150, oy + 40, ox + 150, oy - 120, PURPLE, 3, 12))
    q.append(sd.T(ox + 165, oy - 40, 'charge spreads', 20, PURPLE, 'start', 600))
    q.append(sd.T(ox + 165, oy - 14, 'along the strip', 20, PURPLE, 'start', 600))
    q.append(sd.arrow(40, 560, 160, 560, MUT, 2.5, 11))
    q.append(sd.T(170, 567, 'x', 22, MUT, 'start'))
    q.append(sd.arrow(40, 560, 40, 440, MUT, 2.5, 11))
    q.append(sd.T(40, 428, 'y', 22, MUT))
    q.append(sd.T(W2 / 2, 30, 'top view: resistive strips', 22, INK, 'middle', 600))
    q.append(sd.arrow(ox + 20, 545, ox + 110, 545, RED, 3, 11))
    q.append(sd.T(ox + 65, 538, '×', 30, RED, 'middle', 600))
    q.append(sd.T(ox + 125, 552, 'no path across (x)', 20, RED, 'start', 600))
    top = sd.svg(W2, H2, ''.join(q), 'top view of resistive strips',)
    body = sd.title('The chamber: a drift gap, a 150 µm amplification gap, and a resistive layer cut into strips along y',
                    f'MX17 readout board (from the gerbers). Hover the layers. '
                    f'{sd.term("Dotted terms", "Like this one. Chart points, curves and boxes work the same way.")} and chart points have tooltips.')
    body += sd.row(side, top, gap=60)
    D.slide('chamber', body, '''
<p>The MX17 chambers are bulk micromegas with a 30 mm drift gap and two orthogonal readout views. A track ionises the gas;
the electrons drift down to the mesh, multiply in the 150 µm amplification gap, and the avalanche charge lands on a
resistive film. The film is not continuous: it is cut into strips 550 µm wide on a 0.80 mm pitch that run along y
(<code>mpgd26/scenes_chamber.py</code>, from the board gerbers). Below it, through ~50 µm of insulator and the pads, are the Y
readout strips (layer 5, they measure y) and the X readout strips (layer 6, one layer deeper, they measure x), both on a
0.78 mm pitch.</p>
<p>Two consequences drive the whole model. (1) The charge that lands on a resistive strip can flow along it, i.e. along
y, as an RC line, but it cannot cross to the next resistive strip in x. So the Y view sees charge spread with time and the
X view does not. (2) The footprint of one avalanche on the readout strips is set by the board (strip width, induction
through the layers), not by the gas, and X, one layer deeper, sees it slightly wider than Y.</p>''',
            short='Chamber')


def s_chain(D, d):
    steps = [
        dict(label='1 · Ionisation', sub='uniform along the track, clustered (Landau); charge q_k per 60 ns depth slice, solved by NNLS',
             color=INK, tip='The forward model does not assume a charge profile: the charge per drift-depth slice is a free, '
                            'non-negative vector solved inside every fit. Head-on bench tracks give a flat (box) current.'),
        dict(label='2 · Drift', sub='velocity v(E) per chamber; transverse diffusion σ² += 2·Dd·u', color=BLUE,
             tip='u = drift time of the slice. v comes from the calibration bundle (per chamber and run condition). '
                 'Dd is taken from the measured X widths (≈ dry gas), not from the water fraction v implies.'),
        dict(label='3 · Avalanche + footprint', sub='prompt Gaussian σ0 ≈ 0.4 mm per view: the readout, not the gas', color=GREEN,
             tip='The avalanche itself spreads ~25 µm. The 0.35–0.5 mm footprint is the board.'),
        dict(label='4 · Resistive spread', sub='Y only: 1-D RC line along the strip, σ² += 2·D_rc·s; X: none', color=PURPLE,
             tip='s = time since landing. D_rc 2–9×10⁻⁴ mm²/ns per chamber; det4 2.2×10⁻⁴ from the beam.'),
        dict(label='5 · Electronics', sub='every charge increment through the X template h(t), both views', color=GOLD,
             tip='The X template is the clean DREAM response (−2 % undershoot). Y\'s own template carries part of the RC '
                 'spreading (−8 % undershoot), so it is not used.'),
        dict(label='6 · Strip waveforms', sub='compared with the 60 ns samples; fit p0, w (angle), t0', color=GREY,
             tip='tan θ = w / v. χ² over all strips and samples, saturated and dead samples censored.'),
    ]
    formula = (f'<div style="background:{CARD};border:1px solid {RULE};border-radius:16px;padding:26px 34px;'
               f'font-size:34px;line-height:1.7;font-family:\'IBM Plex Mono\',monospace">'
               f'W<sub>strip</sub>(t) = Σ<sub>k</sub> q<sub>k</sub> Σ<sub>s</sub> ΔF<sub>strip</sub>[ σ²<sub>k</sub>(s) ] · '
               f'h<sub>X</sub>(t − t0 − u<sub>k</sub> − s)<br>'
               f'σ²<sub>k</sub>(s) = {sd.term("σ0²", "Prompt footprint, per view per chamber (readout).")} + '
               f'{sd.term("2·Dd·u<sub>k</sub>", "Drift diffusion of depth slice k (drift time u_k).")} + '
               f'{sd.term("2·D_rc·s", "RC spread along the resistive strip, s = time since landing. Y only; D_rc,x = 0.")} '
               f'&nbsp;&nbsp;centred at p0 + w·u<sub>k</sub></div>')
    gas = sd.row(
        sd.callout(f'Gas enters through v, Dd and, on the {sd.term("beam", G["beam"])}, an attachment loss of late (deep) '
                   f'charge to O₂. The {sd.term("bench", G["bench"])} gas is only moist: no loss.', RED, 25),
        sd.callout('ΔF = the increase, over one time step, of the fraction of a Gaussian of width σ that lies over the '
                   'strip. The prompt landing is the first step.', GREY, 25),
        sd.callout('Each constant has its own measurement on head-on tracks: slides §§drift§§–§§params§§ take them one '
                   'at a time.', GREEN, 25), gap=40)
    body = sd.title('From a track to the strip waveforms in six steps, each one measured on its own',
                    'The new forward model (wft.model.build_matrix_rc), in the order the charge travels. Hover each box.')
    body += sd.flow(steps, size=27)
    body += formula + gas
    D.slide('chain', body, '''
<p>This is the whole model. Each 60 ns depth slice k of the track drifts for u_k and lands at p0 + w·u_k with a Gaussian
width σ0² + 2·Dd·u_k. On Y, the landed charge then keeps spreading along the resistive strip with σ² growing by 2·D_rc per
ns. The readout sees each increment of the charge above a strip through the electronics response; in the code
(<code>build_matrix_rc</code>) the spread is evaluated at the time-since-landing edges <code>RC_EDGES</code> (0, 30, 60, …, 3000 ns),
fine early, coarse late. The charge profile q_k is not modelled: NNLS solves it for every trial (p0, w, t0).</p>
<p>The hypers that switch it on are <code>rc_D_y</code> (presence enables the kernel), <code>rc_D_x</code> (default 0),
<code>sigma_p0_x/_y</code> or <code>sigma_p0</code>, <code>Dp</code> = √(2·Dd), <code>rc_tmpl</code> (1 = X template for both views). Optional:
<code>lor_frac_&lt;view&gt;</code>/<code>lor_gamma_&lt;view&gt;</code> (footprint tails, resolution-neutral, slide §§x-arms§§), <code>rc_drain</code>.
None of c1, c2, kY, τ_s, σ_s is read.</p>
<p>What the model does not contain: attachment (needed on the beam only, slide §§gas§§), longitudinal diffusion, a field
gradient, and anything that would explain the excess X width growth (slide §§xwidth§§).</p>''', short='The chain')


# --------------------------------------------------------------------------- #
# Part 1: the new model, step by step
# --------------------------------------------------------------------------- #
def s_drift(D, d):
    rows = {r['key']: r for r in d.ISO['chambers']['det3']['rows']}
    r3 = rows['sat_det3']
    E = float(r3['E'])
    ws = np.linspace(0, 1.0, 21)
    vv = [float(d.Gn(w, 0.0, E)['v']) for w in ws]
    dd = [float(d.Gn(w, 0.0, E)['DT']) for w in ws]
    dry = dd[0]
    # left: schematic of drift diffusion with depth
    W, H = 700, 700
    o = []
    gx0, gx1, gz0, gz1 = 40, 640, 40, 500
    o.append(f'<rect x="{gx0}" y="{gz0}" width="{gx1 - gx0}" height="{gz1 - gz0}" fill="#eef1f6" stroke="{RULE}"/>')
    o.append(sd.line(gx0, gz0, gx1, gz0, INK, 4))
    o.append(sd.T(gx0 + 8, gz0 - 12, 'cathode, z = 30 mm', 21, INK, 'start', 600))
    o.append(sd.line(gx0, gz1, gx1, gz1, INK, 3, '7 5'))
    o.append(sd.T(gx0 + 8, gz1 + 28, 'mesh, z = 0', 21, INK, 'start', 600))
    tx = 340
    o.append(sd.line(tx, gz0 - 20, tx, gz1, INK, 4))
    depths = [(27, RED, 'deep'), (15, ORANGE, 'mid'), (4, BLUE, 'shallow')]
    pxmm = 120.0
    for z, c, lab in depths:
        y = gz1 - (gz1 - gz0) * z / GAP
        o.append(f'<circle cx="{tx}" cy="{y:.1f}" r="8" fill="{c}"/>')
        o.append(sd.T(tx - 18, y + 7, f'{lab}, z = {z} mm', 20, c, 'end', 600))
        sig = r3['DT'] * 1e-3 * math.sqrt(z / 10.0)   # mm
        xs = np.linspace(tx - 2.6 * pxmm, tx + 2.6 * pxmm, 120)
        smin = r3['DT'] * 1e-3 * math.sqrt(4 / 10.0)
        ys = gz1 + 140 - 105 * np.exp(-0.5 * ((xs - tx) / (sig * pxmm)) ** 2) * (smin / sig)
        o.append(sd.poly(list(xs), list(ys), c, 3.2,
                         tip=f'From z = {z} mm: σ_drift = D_T·√z = {sig:.2f} mm at the measured det3 D_T = {r3["DT"]:.0f} µm/√cm '
                             f'(arrives after z / v = {z / (r3["v"] * 1e-3):.0f} ns).'))
        o.append(sd.line(tx, y + 10, tx, gz1 - 4, c, 1.5, '3 5'))
    o.append(sd.line(gx0, gz1 + 140, gx1, gz1 + 140, RULE, 1.5))
    o.append(sd.T(W / 2, gz1 + 178, 'lateral profile at the mesh from each depth (equal charge)', 20, MUT))
    schem = sd.svg(W, H, ''.join(o), 'drift diffusion with depth')

    vm, ve = r3['v'], r3['ve']
    P1 = sd.Plot(900, 330, x=(0, 1.0), y=(30, 45), ylabel='v [µm/ns]', margin=(14, 30, 50, 104))
    P1.xticks([(w, '') for w in (0, 0.25, 0.5, 0.75, 1.0)]).yticks([(v, f'{v}') for v in (30, 35, 40, 45)])
    P1.band([0, 1.0], [vm - ve] * 2, [vm + ve] * 2, BLUE, 0.15, tip=f'det3 measured v = {vm:.1f} ± {ve:.1f} µm/ns ({E:.0f} V/cm, 27.9 mm gap)')
    P1.line(list(ws), vv, INK, 4, markers=False, tip=f'Magboltz Ar/iso 95/5 + water, {E:.0f} V/cm')
    P1.text(0.02, vm + ve + 0.6, f'det3 measured v {vm:.1f} µm/ns', 21, BLUE, weight=600)
    P1.text(0.62, 41.8, 'Magboltz v', 21, INK)
    P1.vline(r3['water_95'], WET_C, '6 5', 2.5,
             tip='The water fraction that reproduces the measured v at this field (grid edge at 1 %).')
    P1.text(r3['water_95'] - 0.015, 43.6, f'v ⇒ {r3["water_95"]:.1f} % water', 21, WET_C, 'end', 600)
    dm, de = r3['DT'], r3['DTe']
    P2 = sd.Plot(900, 380, x=(0, 1.0), y=(250, 520), xlabel='water in Ar/iso 95/5 [%]', ylabel='D_T [µm/√cm]',
                 margin=(14, 30, 92, 104))
    P2.xticks([(w, f'{w:g}') for w in (0, 0.25, 0.5, 0.75, 1.0)]).yticks([(v, f'{v}') for v in (300, 400, 500)])
    P2.band([0, 1.0], [dm - de] * 2, [dm + de] * 2, GREEN, 0.18,
            tip=f'det3 X head-on stack: D_T = {dm:.0f} ± {de:.0f} µm/√cm (x_width.py / iso_fit.json)')
    P2.line(list(ws), dd, INK, 4, markers=False, tip='Magboltz D_T for the same mixtures')
    P2.points([0.0], [dry], DRY_C, 9, tips=[f'dry: {dry:.0f} µm/√cm'])
    P2.points([r3['water_95']], [r3['DT_at_v95']], WET_C, 10,
              tips=[f'at the water v implies: {r3["DT_at_v95"]:.0f} µm/√cm'])
    P2.text(0.3, dm + de + 10, f'X widths measure {dm:.0f} ± {de:.0f} µm/√cm ≈ dry', 21, GREEN, weight=600)
    P2.text(r3['water_95'] - 0.02, r3['DT_at_v95'] - 22, f'v-implied: {r3["DT_at_v95"]:.0f}', 21, WET_C, 'end', 600)
    right = sd.col(P1.svg('v vs water'), P2.svg('DT vs water'), gap=0)
    body = sd.title('Drift: electrons arrive in order of depth, and spread as √depth',
                    f'Left: the mechanism. Right: det3 at {E:.0f} V/cm, Magboltz against what the chamber measures.')
    body += sd.row(schem, right, gap=40)
    D.slide('drift', body, f'''
<p><b>Velocity.</b> Each depth slice arrives after u = z / v, so arrival time is depth. v(E) comes from the calibration
bundle of each chamber and run condition (bench 24.6–39.9 µm/ns; beam det4 {d.beam_gas[243]["v"]:.1f} / {d.beam_gas[150]["v"]:.1f} /
{d.beam_gas[92]["v"]:.1f} µm/ns at 243 / 150 / 92 V/cm from the run_71 composition).</p>
<p><b>Diffusion.</b> The transverse spread grows as σ² = D_T²·z = 2·Dd·u. Magboltz says water slows the drift and cools it:
at {E:.0f} V/cm, the {r3["water_95"]:.1f} % water that reproduces det3's measured v ({vm:.1f} µm/ns) would give D_T = {r3["DT_at_v95"]:.0f} µm/√cm.
The X head-on widths, where nothing else widens the cluster, say {dm:.0f} ± {de:.0f}, close to dry ({dry:.0f}). The same holds on
every bench chamber (slide §§xwidth§§).</p>
<p><b>The model's choice, for now:</b> Dd is taken empirically from the X stack (≈ dry gas), per chamber, and a v-derived water
fraction is never carried into D_T (FINDINGS §28–29). This is a decision forced by an open problem, not a resolution of it.</p>
<p>The water grid for Ar/iso 95/5 ends at 1 %, so det3's v-implied water sits on the grid edge (<code>iso_fit.json</code> row
<code>sat_det3</code>: v {vm:.1f}, D_T {dm:.0f} ± {de:.0f}, Magboltz at that v {r3["DT_at_v95"]:.0f}). Gap 27.9 mm (dished cathode) is
taken into account in v and D_T.</p>''', short='Drift')


def s_footprint(D, d):
    av_lo, av_hi = min(d.aval), max(d.aval)
    rows = [('avalanche, 150 µm gap', float(np.mean(d.aval)) / 1000, ORANGE,
             f'Magboltz dry bench gas at 25–40 kV/cm: D_T {min(p["DT_um_rtcm"] for p in d.AMP["points"]):.0f}–'
             f'{max(p["DT_um_rtcm"] for p in d.AMP["points"]):.0f} µm/√cm over 150 µm = {av_lo:.0f}–{av_hi:.0f} µm',
             f'{av_lo:.0f}–{av_hi:.0f} µm')]
    for det in DETS:
        n = d.new[det]
        rows.append((f'bench {det}', n['sig0'], GREEN,
                     f'{det}: joint X+Y head-on fit (|tan| < 0.03), σ0 = {n["sig0"]:.3f} mm. This cut lets ~0.1 mm of '
                     'track walk in (next rows).', ''))
    for det in ('det3', 'det4'):
        s = d.TS[det]['0.01']['sig0']
        rows.append((f'bench {det}, |tan| < 0.01', s, DGREEN,
                     f'{det}, the same fit with the head-on cut tightened to |tan| < 0.01: σ0 = {s:.3f} mm '
                     f'(n = {d.TS[det]["0.01"].get("n", "?")}).', ''))
    bx, by = d.beam_s0['x'], d.beam_s0['y']
    rows.append(('beam det4, Y', float(np.mean(by)), Y_C,
                 'run_71 RC fit with drift diffusion, 243/150/92 V/cm: ' + ' / '.join(f'{v:.2f}' for v in by) + ' mm', ''))
    rows.append(('beam det4, X', float(np.mean(bx)), X_C,
                 'run_71, X: ' + ' / '.join(f'{v:.2f}' for v in bx) + ' mm. X is one board layer deeper than Y.', ''))
    bars = sd.hbars([(a, b, c, t, '') for a, b, c, t, _ in rows], 0.6, width=600, h=46, label_w=360,
                    fmt=lambda v: f'{v * 1000:.0f} µm' if v < 0.1 else f'{v:.2f} mm', size=26)
    side = sd.col(
        sd.p(f'The avalanche spreads by {av_lo:.0f}–{av_hi:.0f} µm. The footprint the strips see is 14–22 times wider.', 28),
        sd.callout('So σ0 is a readout constant: resistive-strip width, induction through the layers, inter-strip coupling. '
                   'Water or dry, beam or bench, the gas cannot change it.', GREEN, 27),
        sd.p('det4 reads the same on the bench and at the beam once the track inclination is cut out (0.36–0.40 mm). '
             'X is a little wider than Y, as the deeper X layer should be.', 26, MUT),
        gap=30, w=540)
    body = sd.title('The prompt footprint σ0 ≈ 0.4 mm is the readout, not the gas',
                    'Width of the charge one avalanche puts on the readout strips, against the avalanche itself. Hover each bar.')
    body += sd.row(bars, side, gap=50)
    D.slide('footprint', body, f'''
<p>The bench and beam calibrations used to disagree on the prompt width (σ_p0 0.39–0.43 mm on the bench, 0.15 mm on the beam
with det4's beam kernel). That was bookkeeping: each fit put the same neighbour charge in a different parameter (FINDINGS §1).
Measured model-free on head-on tracks, the leading-edge (first-charge) neighbour fraction is a detector property that does
not move with the drift field.</p>
<p>The avalanche cannot make a 0.4 mm footprint: at 25–40 kV/cm Magboltz gives D_T {min(p["DT_um_rtcm"] for p in d.AMP["points"]):.0f}–{max(p["DT_um_rtcm"] for p in d.AMP["points"]):.0f} µm/√cm,
so the 150 µm gap spreads it by {av_lo:.0f}–{av_hi:.0f} µm (<code>magboltz_bench_dry_amp.json</code>). The width comes from the board.</p>
<p><b>What the model takes:</b> one σ0 per view per chamber. The joint fit at |tan| &lt; 0.03 (the rcm arms) carries ~0.1 mm of
track walk; tightening to 0.01 brings det4 to {d.TS["det4"]["0.01"]["sig0"]:.3f} and det3 to {d.TS["det3"]["0.01"]["sig0"]:.3f} mm, i.e. to det4's beam Y
value (FINDINGS §28). X's footprint is wider than Y's by the board geometry (Y strips are layer 5, X strips layer 6), so the
X/Y ratio should be a fixed geometric number rather than a free per-chamber knob. The X footprint also has tails beyond a
Gaussian (pseudo-Voigt, FINDINGS §23); they are real but do not move the resolution.</p>''', short='Footprint')


def s_rc(D, d):
    """Mechanism of the Y RC spread + the strip fractions it produces (det4 beam constants)."""
    s0 = float(np.mean(d.beam_s0['y']))
    Drc = d.beam_drc_corr
    W, H = 760, 700
    o = []
    x0, x1 = 60, 720
    pxmm = 90.0
    cx = (x0 + x1) / 2
    yb0 = 330                                     # baseline of the charge profiles = top of the resistive strip
    o.append(sd.T(x0, 40, 'charge on one resistive strip, at three times after landing', 22, INK, 'start', 600))
    times = [(0, INK, 1.25), (500, ORANGE, 1.6), (2000, PURPLE, 2.1)]
    for t, c, lx in times:
        sg = math.sqrt(s0 ** 2 + 2 * Drc * t)
        xs = np.linspace(x0, x1, 160)
        ys = yb0 - 230 * np.exp(-0.5 * ((xs - cx) / (sg * pxmm)) ** 2) * (s0 / sg)
        o.append(sd.poly(list(xs), list(ys), c, 3.8,
                         tip=f't = {t} ns after landing: σ_y = √(σ0² + 2·D_rc·t) = {sg:.2f} mm '
                             f'(σ0 = {s0:.2f} mm, D_rc = {Drc * 1e4:.1f}×10⁻⁴ mm²/ns, det4 beam)'))
        k = [tt for tt, _, _ in times].index(t)
        o.append(sd.T(x1, 80 + 28 * k, f'{t} ns after landing: σ = {sg:.2f} mm', 20, c, 'end', 600))
    o.append(f'<rect x="{x0}" y="{yb0}" width="{x1 - x0}" height="30" fill="{GREY}" fill-opacity="0.45" rx="5"/>')
    o.append(sd.T(x1, yb0 + 21, 'resistive strip, along y →', 19, INK, 'end', 600))
    yb = yb0 + 52
    for j in range(-4, 5):
        xc = cx + j * PITCH * pxmm
        lab = {0: 'centre', 1: '+1', 2: '+2', -1: '−1', -2: '−2'}.get(j, '')
        o.append(f'<rect x="{xc - 0.45 * PITCH * pxmm:.1f}" y="{yb}" width="{0.9 * PITCH * pxmm:.1f}" height="22" '
                 f'fill="{S_C.get(abs(j), GREY)}" fill-opacity="{0.85 if abs(j) <= 2 else 0.3}"/>')
        if lab:
            o.append(sd.T(xc, yb + 48, lab, 20, S_C.get(abs(j), MUT), 'middle', 600))
    o.append(sd.T(x0, yb + 90, 'Y readout strips under it (pitch 0.78 mm) catch the spread', 21, Y_C, 'start', 600))
    o.append(sd.T(x0, yb + 135, 'The X strips run along the resistive strip: spreading along y', 21, X_C, 'start'))
    o.append(sd.T(x0, yb + 163, 'keeps the charge over the same X strip, so X sees no RC term.', 21, X_C, 'start'))
    o.append(sd.T(x0, yb + 215, 'RC line: σ² grows as 2·D_rc·t, D_rc = 1 / (R_s·c)', 22, INK, 'start', 600,
                  tip='A resistive film of surface resistivity R_s (a few MΩ/□) over a capacitance per area c '
                      '(~50 µm insulator) is a 2-D diffusion equation for the charge; on a strip it is 1-D.'))
    schem = sd.svg(W, H, ''.join(o), 'RC spread along a resistive strip')

    ts = np.linspace(0, 3000, 61)
    P = sd.Plot(860, 640, x=(0, 3.0), y=(0, 0.75), xlabel='time since landing [µs]',
                ylabel='fraction of the charge over the strip', margin=(20, 30, 92, 104))
    P.xticks([(v, f'{v:g}') for v in (0, 0.5, 1, 1.5, 2, 2.5, 3)]).yticks([(v, f'{v:g}') for v in (0, 0.25, 0.5, 0.75)])
    for o_ in (0, 1, 2):
        f = frac(o_, np.sqrt(s0 ** 2 + 2 * Drc * ts))
        lab = {0: 'centre', 1: 'each ±1', 2: 'each ±2'}[o_]
        P.line(list(ts / 1e3), list(f), S_C[o_], 4.5, markers=False,
               tip=f'{lab}: {f[0]:.3f} at landing → {f[20]:.3f} at 1 µs → {f[-1]:.3f} at 3 µs')
        P.text(3.0, float(f[-1]) + 0.02, lab, 22, S_C[o_], 'end', 600)
    P.text(0.12, 0.70, f'det4, beam constants: σ0 = {s0:.2f} mm, D_rc = {Drc * 1e4:.1f}×10⁻⁴ mm²/ns', 21, INK)
    body = sd.title('Y: the landed charge keeps spreading along the resistive strip, as an RC line',
                    'Left: the mechanism. Right: what each Y strip holds as time passes, at det4\'s beam values.')
    body += sd.row(schem, P.svg('strip fractions vs time'), gap=40)
    D.slide('rc', body, f'''
<p>The avalanche charge lands on a resistive strip and drains through it. Along the strip the film is a distributed RC
line: the charge density obeys a diffusion equation with D_rc = 1/(R_s·c), so a landed Gaussian keeps its shape and widens
as σ² = σ0² + 2·D_rc·s, s the time since landing (Dixit et al., NIM A 518 (2004) 721; for strips, Galan et al., JINST 7 (2012)
C04009; references in <code>wft/REFERENCES.md</code>). Each readout strip sees its fraction of that charge change with time, and
the electronics see every increment.</p>
<p>The resistive strips run along y. The Y readout strips cross them, so they see the spread. The X readout strips run
along the resistive strips, so moving charge along y keeps it over the same X strip, and charge cannot jump to the next
resistive strip in x. X therefore has no RC term, which is what the data say (next slide).</p>
<p>Under this mechanism there are no discrete copies. Charge reaches ±2 only by passing ±1, so ±2 always holds less than
±1 and starts later, while late in the pulse it comes close (curves). That is the ordering the old c2 &lt; c1 rule tried to
impose by hand (slide §§c2c1§§).</p>
<p>Constants drawn: det4 at the beam, σ0,y = {s0:.2f} mm (mean of three fields) and D_rc = {Drc * 1e4:.2f}×10⁻⁴ mm²/ns (run_71 RC fit
with Magboltz drift diffusion, {" / ".join(f"{v * 1e4:.2f}" for v in d.beam_drc)}×10⁻⁴ at 243/150/92 V/cm, ÷ 0.94 for the attachment
loss, FINDINGS §27).</p>''', short='RC spread (Y)')


def s_rc_evidence(D, d):
    keys = list(DETS) + ['sps700', 'sps450', 'sps275']
    labs = [k if k.startswith('det') else {'sps700': 'beam 243', 'sps450': 'beam 150', 'sps275': 'beam 92'}[k] for k in keys]
    Dx = [d.RD[k]['x']['fits']['x']['p'][1] for k in keys]
    Dy = [d.RD[k]['y']['fits']['y']['p'][1] for k in keys]
    n = len(keys)
    P = sd.Plot(960, 660, x=(-0.6, n - 0.4), y=(0, 10), ylabel='D_rc fitted [10⁻⁴ mm²/ns]', margin=(20, 20, 92, 104))
    P.xticks([(i, lab) for i, lab in enumerate(labs)]).yticks([(v, f'{v}') for v in (0, 2, 4, 6, 8, 10)])
    for i, k in enumerate(keys):
        P.vbar(i - 0.18, Dy[i] * 1e4, 40, Y_C, tip=f'{labs[i]} Y: D_rc = {Dy[i] * 1e4:.2f}×10⁻⁴ mm²/ns '
                                                 f'(n = {d.RD[k]["y"]["n"]} head-on events)')
        P.vbar(i + 0.18, max(Dx[i] * 1e4, 0.05), 40, X_C, tip=f'{labs[i]} X: D_rc = {Dx[i] * 1e4:.2f}×10⁻⁴ mm²/ns '
                                                           f'(n = {d.RD[k]["x"]["n"]})')
    P.vline(4.5, RULE, '4 5', 2)
    P.text(2.0, 9.4, 'cosmic bench', 22, MUT, 'middle')
    P.text(6.0, 9.4, 'det4 at the beam', 22, MUT, 'middle')
    leg = sd.legend([('Y', Y_C, 'box'), ('X', X_C, 'box')], 22)
    # rms gain table
    rows = []
    tips = []
    for k, lab in zip(keys, labs):
        ry0, ry = d.RD[k]['y']['fits']['x0']['rms'], d.RD[k]['y']['fits']['y']['rms']
        rx0, rx = d.RD[k]['x']['fits']['x0']['rms'], d.RD[k]['x']['fits']['x']['rms']
        rows.append([lab, f'{ry0:.3f} → <b>{ry:.3f}</b>', f'{rx0:.3f} → {rx:.3f}'])
        tips.append(f'{lab}: stack rms with the prompt footprint only → with the RC term. Y ×{ry0 / ry:.1f}, X ×{rx0 / rx:.2f}.')
    tbl = sd.table(['', 'Y rms: prompt → +RC', 'X rms'], rows, size=21, widths=[130, 250, 200], tips=tips)
    gy = [d.RD[k]['y']['fits']['x0']['rms'] / d.RD[k]['y']['fits']['y']['rms'] for k in keys]
    rxy = [Dy[i] / max(Dx[i], 1e-9) for i in range(n)]
    side = sd.col(tbl, sd.callout(f'One constant per chamber fits Y {min(gy):.1f}–{max(gy):.1f}× better. On X the RC term buys '
                                  f'almost nothing and fits {min(rxy):.0f}–{max(rxy):.0f}× smaller than on Y: there is no '
                                  'resistive spread across the strips.', PURPLE, 24),
                  gap=18, w=600)
    body = sd.title('The data: Y spreads with time on every chamber, X does not',
                    'RC fit of each view\'s head-on (strip offset × time) stack, W_o = S ⊗ dF_o/ds, S = the measured strip sum.')
    body += sd.row(sd.col(leg, P.svg('D_rc by view'), gap=6), side, gap=40)
    D.slide('rc-evidence', body, f'''
<p><code>rc_diffusion.py</code> fits each view of each dataset with W_o(t) = S(t) ⊗ dF_o/ds, where S is the all-strip sum (it carries
the electronics and the arrival current, taken from the data, no template), and F_o the fraction over strip offset o of a
Gaussian with σ² = σ0² + 2·D_rc·s. One free D_rc per view. Unbiased stacks (fixed threshold, no per-event normalisation,
FINDINGS §10).</p>
<p>This fit has no drift-diffusion term. On the bench, where the drift spreads the charge too, it absorbs part of the drift
growth into D_rc, so its bench Y numbers (5–9×10⁻⁴) are higher than the joint X+Y fit the model actually uses
(2.8–7.9×10⁻⁴, slide §§params§§), where X pins the drift diffusion. On the beam the drift spread is 2–3× smaller and the
two agree (2.0–2.3 vs 1.9–2.1×10⁻⁴). The point of this slide is the contrast between views, which survives either way.</p>
<p>D_rc differs from chamber to chamber (det4 lowest, det7 highest): a construction constant of each board's resistive
film. det4 bench and beam agree within 1.2–1.4× once the bench drift diffusion is fitted jointly (FINDINGS §28).</p>''',
            short='RC: evidence')


def s_template(D, d):
    g = d.tmpl['grid']
    tx, ty = d.tmpl['x'], d.tmpl['y']
    mnx, mny = float(tx.min()), (float(ty.min()) if ty is not None else float('nan'))

    def mk(lim_y, ttl, zoom):
        P = sd.Plot(810, 540, x=(-0.2, 1.4) if not zoom else (0.5, 1.4), y=lim_y, xlabel='time [µs]',
                    ylabel='template (peak = 1)' if not zoom else '', margin=(10, 30, 92, 104), title=ttl)
        xt = (0, 0.4, 0.8, 1.2) if not zoom else (0.6, 0.8, 1.0, 1.2, 1.4)
        P.xticks([(v, f'{v:g}') for v in xt])
        P.yticks([(v, f'{v:g}') for v in ((0, 0.5, 1) if not zoom else (-0.08, -0.04, 0, 0.04))])
        P.hline(0, MUT, None, 1.5)
        sel = (g / 1e3 >= P.xs[0]) & (g / 1e3 <= P.xs[1])
        if zoom:
            sel = sel & (tx <= lim_y[1]) & ((ty if ty is not None else tx) <= lim_y[1])
        P.line(list(g[sel] / 1e3), list(tx[sel]), X_C, 4.5, markers=False,
               tip=f'X template, det3 r06 bundle: undershoot {mnx:+.3f} at {g[tx.argmin()]:.0f} ns')
        if ty is not None:
            P.line(list(g[sel] / 1e3), list(ty[sel]), Y_C, 4.5, markers=False,
                   tip=f'Y template, same bundle: undershoot {mny:+.3f} at {g[ty.argmin()]:.0f} ns')
        if zoom and ty is not None:
            P.text(g[ty.argmin()] / 1e3, mny - 0.012, f'Y {mny * 100:.0f} %', 24, Y_C, 'middle', 600)
            P.text(g[tx.argmin()] / 1e3 + 0.1, mnx + 0.012, f'X {mnx * 100:.0f} %', 24, X_C, 'middle', 600)
            P.hline(mny, Y_C, '4 5', 1.5)
            P.hline(mnx, X_C, '4 5', 1.5)
        return P
    P1 = mk((-0.15, 1.05), 'whole response', False)
    P2 = mk((-0.1, 0.05), 'the tail, magnified', True)
    leg = sd.legend([('X template', X_C), ('Y template', Y_C)], 22)
    body = sd.title('Electronics: the clean X template drives both views',
                    'The measured single-strip impulse responses of det3 (production bundle), normalised to their peak.')
    body += leg + sd.row(P1.svg('templates'), P2.svg('template tails'), gap=44)
    body += sd.row(
        sd.callout('Both views are read by the same DREAM chips with the same settings: their electronics are identical.', GREY, 25),
        sd.callout(f'Yet Y dips to {mny * 100:.0f} % and X only to {mnx * 100:.0f} %: the extra dip is charge leaving the strip along '
                   'the resistive layer, already inside Y\'s "impulse response".', Y_C, 25),
        sd.callout('So the model uses the X template for both views (rc_tmpl = 1) and adds the RC spread explicitly, '
                   'instead of counting it twice.', GOLD, 25), gap=36)
    D.slide('template', body, f'''
<p>The template is the response of one strip to charge arriving on it at t = 0: DREAM preamplifier and shaper (peaking
~180 ns, the same on bench and beam) plus the ion tail. It is measured, not modelled (<code>wft.calibrate.measure_templates</code>):
the brightest strip of inclined bench tracks, which also contains the strip's own short depth segment. Beam bundles reused
the bench template.</p>
<p>A clean electronics response for a 60 ns sampled DREAM channel has a small undershoot; X shows {mnx * 100:.1f} %. Y shows
{mny * 100:.1f} %, and the difference has the sign and size of charge spreading away from the strip after it lands. Since the
electronics are the same, that part belongs to the RC term, and the physical kernel uses the X template for both views
(<code>rc_tmpl</code> = 1 or 'x'). Templates are per chamber (each bundle has its own); det3 is drawn.</p>
<p>Source: <code>calib_bundle_r06/arrays.npz</code> of sat_det3 (<code>tmpl_x</code>, <code>tmpl_y</code>, 10 ns grid).</p>''',
            short='Electronics')


def s_gas(D, d):
    t = np.array(d.H['t'], float)
    FC = {243: BLUE, 150: ORANGE, 92: PURPLE}
    P1 = sd.Plot(540, 580, x=(1.0, 3.8), y=(0.4, 1.15), title='beam det4 (run_71, RAW), X',
                 xlabel='time after trigger [µs]', ylabel='stacked signal / plateau', margin=(10, 20, 92, 96))
    P1.xticks([(v, f'{v:g}') for v in (1, 2, 3)]).yticks([(v, f'{v:g}') for v in (0.5, 0.75, 1.0)])
    for lab, E in (('raw700', 243), ('raw450', 150), ('raw275', 92)):
        S = np.array(d.H[lab]['x']['sum']['2'], float)
        m = (t >= 1080) & (t <= 1260)
        S = S / np.nanmean(S[m])
        tend = 2600 if E == 243 else 3800
        sel = (t >= 1000) & (t <= tend)
        g = d.beam_gas[E]
        P1.line(list(t[sel] / 1e3), list(S[sel]), FC[E], 4, markers=False,
                tip=f'{E} V/cm: X ±2-strip head-on stack, n = {d.H[lab]["n"]}. Model at the run_71 composition: '
                    f'η·v = {g["etav"] * 1e4:.2f}×10⁻⁴/ns, {g["surv"]:.0%} of the cathode electrons survive.')

    P1.hline(1.0, MUT, '6 5', 1.5)
    P1.vline(2.65, BLUE, '3 5', 1.5, tip='At 243 V/cm the drift ends inside the window (~2.7 µs): the stack is cut there.')
    for k_, E in enumerate((243, 150, 92)):
        P1.raw(sd.line(P1.X(2.85), P1.Y(1.1 - 0.05 * k_), P1.X(3.05), P1.Y(1.1 - 0.05 * k_), FC[E], 4))
        P1.text(3.1, 1.1 - 0.05 * k_ - 0.012, f'{E} V/cm', 20, FC[E], 'start', 600)
    P2 = sd.Plot(500, 580, x=(0.4, 2.0), y=(0.4, 1.15), title='bench det3, drift scan, X',
                 xlabel='time after arrival start [µs]', margin=(10, 20, 92, 50))
    P2.xticks([(v, f'{v:g}') for v in (0.5, 1, 1.5, 2)]).yticks([(v, '') for v in (0.5, 0.75, 1.0)])
    gr = np.array(d.BD['grid'], float)
    for V, E, c in (('100', 34.7, GREEN), ('300', 104.2, DGREEN)):
        S = np.array(d.BD[V]['x']['stack'], float)
        m = (gr >= 600) & (gr <= 1200)
        S = S / np.nanmean(S[m])
        fin = np.where(np.isfinite(S))[0]
        sel = (gr >= 400) & np.isfinite(S) & (np.arange(len(S)) <= fin[-1] - 3)
        P2.line(list(gr[sel] / 1e3), list(S[sel]), c, 4, markers=False,
                tip=f'{E:.0f} V/cm (drift {V} V): n = {d.BD[V]["x"]["n"]} events, normalised at 0.6–1.2 µs. '
                    'A beam-like loss would remove ~14 % over this span.')
        P2.text(0.45, 1.09 if V == '300' else 0.9, f'{E:.0f} V/cm', 21, c, 'start', 600)
    P2.hline(1.0, MUT, '6 5', 1.5)
    rows = [('run_71 RAW, 3 fields', d.BC['o2_ppm'], RED,
             f'3 Aug 05:22: {d.BC["water"]:.2f} % water + {d.BC["air"]:.3f} % air; all 3 fields × X/Y + the ladder v (beam_comp_fit_air_hs.json)')]
    rows += [(a, b, RED, c) for a, b, c in COMPOSITIONS]
    rows.append(('bench det3, 6 fields', 0.0, GREEN, f'{d.DF["water"]:.2f} % water, no air: ≲ 10 ppm O₂ (driftscan_fit)'))
    rows = sorted(rows, key=lambda r: -r[1])
    bars = sd.hbars([(a, max(b, 3), c, t_) for a, b, c, t_ in rows], 300, width=150, h=30, label_w=270,
                    fmt=lambda v: '≲ 10' if v <= 3 else f'{v:.0f}', size=21)
    side = sd.col(sd.p('O₂ in the gas, ppm (Magboltz composition fits)', 21, INK, 600), bars,
                  sd.p('Per run: the beam O₂ moved over hours. Water does not attach.', 20, MUT), gap=12, w=520)
    body = sd.title('Gas by setup: the beam gas lost late electrons to O₂; the bench gas is only moist',
                    'Head-on stacks normalised to their plateau. Beam: one loss rate in time at every field. Bench: flat.')
    body += sd.row(P1.svg('beam stacks'), P2.svg('bench stacks'), side, gap=24)
    g243 = d.beam_gas[243]
    D.slide('gas', body, f'''
<p><b>Beam.</b> The late charge in det4's run_71 falls at the same rate per unit time at all three drift fields, in both
views, with no electronics undershoot after the drift ends (FINDINGS §16). Only loss of drifting electrons does that, and
Magboltz puts it on O₂: one composition, {d.BC["water"]:.2f} % water and {d.BC["air"]:.3f} % air ({d.BC["o2_ppm"]:.0f} ppm O₂), fits all three fields
and both views with no free loss rate. At 243 V/cm that is η·v = {g243["etav"] * 1e4:.2f}×10⁻⁴/ns, and {g243["surv"]:.0%} of the electrons from the
cathode arrive ({d.beam_gas[92]["surv"]:.0%} at 92 V/cm). The O₂ was not constant: run_63 the night before went from 230 to 147 ppm over hours,
and the CO₂ period on 1 Aug was ~290 ppm. The absolute ppm scale depends on Magboltz's three-body O₂ attachment and is
uncertain by a factor of a few; the beam/bench contrast is not.</p>
<p><b>Bench.</b> The det3 drift scan (35–382 V/cm) stays flat to ±2 % for 1.2 µs of drift at the two lowest fields, where a
beam-like loss would remove ~14 %. The same physics model gives {d.DF["water"]:.2f} % water and no air (≲ 10 ppm O₂).</p>
<p><b>In the model.</b> The forward model has no attachment term today. The bench needs none. A beam reconstruction would
need one per run (η·v from the composition, or the measured rate). The SPS constants the bench model takes (σ0, D_rc) are
shifted by the beam loss by ≤ 7 % (injection test, FINDINGS §27).</p>
<p>Full study: notes/mx17-attachment-slides and notes/mx17-beam-attachment; FINDINGS §16–27.</p>''', short='Gas by setup')


def s_params(D, d):
    def cell(v):
        return f'<span style="white-space:nowrap">{v}</span>'
    hdr = ['', 'what', 'det2', 'det3', 'det4', 'det6', 'det7', 'det4 beam', 'measured from']
    n = d.new
    bx = d.XB
    beam_v = ' / '.join(f'{bx[k]["v"]:.1f}' for k in ('sps700', 'sps450', 'sps275'))
    rows = [
        ['v', 'drift velocity [µm/ns]'] + [cell(f'{n[k]["v"]:.1f}') for k in DETS] + [cell(beam_v),
         'bundle per run (bench); drift end + ladder (beam)'],
        ['E', 'drift field [V/cm]'] + [cell(f'{n[k]["E"]:.0f}') for k in DETS] + ['243 / 150 / 92', 'run HV / gap'],
        ['σ0', 'footprint [mm], |tan|&lt;0.03'] + [cell(f'{n[k]["sig0"]:.2f}') for k in DETS]
        + [cell(f'X {np.mean(d.beam_s0["x"]):.2f} · Y {np.mean(d.beam_s0["y"]):.2f}'), 'joint X+Y head-on fit; beam RC fit'],
        ['Dd → D_T', 'drift diffusion [µm/√cm]'] + [cell(f'{n[k]["DT"]:.0f}') for k in DETS]
        + [cell(' / '.join(f'{d.beam_gas[E]["DT"]:.0f}' for E in (243, 150, 92)) + ' Magboltz'),
           'X head-on stack (bench); Magboltz (beam)'],
        ['D_rc,y', 'RC spread [10⁻⁴ mm²/ns]'] + [cell(f'{n[k]["Drc"] * 1e4:.1f}') for k in DETS]
        + [cell(f'{d.beam_drc_corr * 1e4:.1f}'), 'joint fit (bench); run_71 RC fit (beam)'],
        ['D_rc,x', 'across the strips'] + ['0'] * 5 + ['0', 'board geometry; fits ≈ 0'],
        ['h(t)', 'electronics template'] + ['X tmpl'] * 5 + ['bench X tmpl', 'each chamber\'s bundle'],
        ['η·v', 'attachment [10⁻⁴/ns]'] + ['0'] * 5
        + [cell(' / '.join(f'{d.beam_gas[E]["etav"] * 1e4:.1f}' for E in (243, 150, 92))), 'composition fit (per run)'],
    ]
    tips = [
        'Bench: the calibration bundle v of each golden run. Beam: x_width_beam.json v at 243/150/92 V/cm.',
        'Bench drift HV: det2/det3 1000 V, det6/det7 700 V, det4 600 V over the gap.',
        'σ0 from bench_uniform.json (Dd free). With |tan| < 0.01 det3/det4 drop to 0.37/0.36 mm (track walk).',
        'Bench: Dd fitted on the X head-on stack, converted with the bundle v. Beam: Magboltz at the run_71 composition '
        '(the beam X widths read wider, slide xwidth).',
        'Bench: joint X+Y fit (bench_uniform.json). Beam: rc_drift.json, loss-corrected. det4 can take the beam value on the '
        'bench at no cost (FINDINGS §28).',
        'rc_D_x = 0: X has no resistive spread (refits put it at 0–1.2×10⁻⁵).',
        'rc_tmpl = 1: the X template drives both views.',
        'Bench: no attachment (det3 six-field scan, ≲ 10 ppm O₂). Beam: Magboltz η·v at 1.51 % water + 0.078 % air.',
    ]
    tbl = sd.table(hdr, rows, size=21, widths=[110, 290, 80, 80, 80, 80, 80, 270, 400],
                   align=['left', 'left'] + ['right'] * 6 + ['left'], tips=tips)
    steps = [
        dict(label='per chamber', sub='σ0 (per view), D_rc,y, Dd, template', color=GREEN,
             tip='Board and chamber constants. The footprint and D_rc are construction properties.'),
        dict(label='per run condition', sub='v, drift field, gas (water; O₂ on the beam)', color=BLUE,
             tip='CLAUDE.md: anything calibrated is per detector AND per run condition.'),
        dict(label='fixed by geometry', sub='D_rc,x = 0; X template for both views; σ0,X > σ0,Y', color=GREY,
             tip='From the board: resistive strips along y; X strips one layer deeper.'),
        dict(label='not fitted to angles', sub='every constant from head-on stacks; held-out test with no refit', color=GOLD,
             tip='No χ² refit against the reference angle (the refits in slide x-arms do not change the result).'),
    ]
    chain = [
        dict(label='ionisation', sub='q_k: solved (NNLS)', color=INK),
        dict(label='drift', sub='v · Dd', color=BLUE),
        dict(label='footprint', sub='σ0,x · σ0,y', color=GREEN),
        dict(label='RC spread', sub='D_rc,y (D_rc,x = 0)', color=PURPLE),
        dict(label='electronics', sub='h(t) = X template', color=GOLD),
        dict(label='gas loss', sub='η·v (beam only)', color=RED),
        dict(label='fit', sub='p0 · w · t0 per track', color=GREY),
    ]
    body = sd.title('The parameters: what each is, its value, where it comes from',
                    'The chain with its constants, then their values: the held-out bench arms (rcm) and det4 at the beam. Hover a row for its source.')
    body += sd.flow(chain, size=22) + tbl + sd.flow(steps, size=20, arrow='')
    D.slide('params', body, f'''
<p>The bench constants are those of the <code>rcm</code> arms (<code>make_rc_arms.py</code> from <code>results/bench_uniform.json</code>, Dd free),
which is what the held-out comparison in slide §§heldout§§ ran. Converted: σ_p0 = σ0, Dp = √(2·Dd), rc_D_y = D_rc. The latest joint
fits (FINDINGS §28, <code>bench_pinrc.py</code>) agree: D_rc det2 4.2, det3 2.8, det6 7.9, det7 5.7×10⁻⁴; det4 2.57×10⁻⁴ free or 2.2 pinned to the
beam at no cost in the stack (+0.3 % rms) or the held-out angles (±0.03°).</p>
<p><b>Decided, not frozen:</b> det4 takes D_rc from the beam, the other chambers keep their bench value; Dd per chamber from the
joint fit (≈ dry); σ0 should come from the tight-|tan| stack (it carries ~0.1 mm of track walk at 0.03). None of these choices
moves det4's held-out resolution by more than ±0.04°. Freezing waits on the X width question (slide §§xwidth§§).</p>
<p>The beam column is for completeness. No beam bundle has been rebuilt with this kernel; the beam bundles still carry the old
copy kernel and the H4 v/Dp.</p>''', short='Parameters')


# --------------------------------------------------------------------------- #
# Part 2: old vs new
# --------------------------------------------------------------------------- #
def _mech_svg(old, h, n):
    """Schematic of one landing on strips -2..+2: old = prompt + delayed copies; new = prompt + continuous spread."""
    W, H = 800, 360
    o = []
    cx, sw = W / 2, 120
    base = 190
    for j in range(-2, 3):
        x = cx + j * sw
        o.append(f'<rect x="{x - sw * 0.42:.1f}" y="{base}" width="{sw * 0.84:.1f}" height="22" fill="{S_C[abs(j)]}" fill-opacity="0.85"/>')
        o.append(sd.T(x, base - 8, {0: 'centre', 1: '+1', 2: '+2', -1: '−1', -2: '−2'}[j], 19, S_C[abs(j)], 'middle', 600))
    xs = np.linspace(cx - 2.5 * sw, cx + 2.5 * sw, 120)
    if old:
        ys = base - 30 - 120 * np.exp(-0.5 * ((xs - cx) / (h['sigma_p0'] / PITCH * sw)) ** 2)
        o.append(sd.poly(list(xs), list(ys), INK, 3.5, tip=f'Prompt Gaussian σ_p0 = {h["sigma_p0"]:.2f} mm (+ Dp²·u with depth)'))
        for j, dep, dash, lab in ((1, 45, '7 5', f'c1·kY to ±1, delayed τ_s = {h["tau_s"]:.0f} ns'),
                                  (2, 75, '2 6', f'c2·kY to ±2, delayed 2·τ_s = {2 * h["tau_s"]:.0f} ns')):
            for sgn in (-1, 1):
                xe = cx + sgn * j * sw
                xm = (cx + xe) / 2
                o.append(f'<path d="M {cx + sgn * 10:.1f} {base + 24} Q {xm:.1f} {base + 24 + dep * 1.6:.1f} {xe:.1f} {base + 26}" '
                         f'fill="none" stroke="{OLD_C}" stroke-width="3" stroke-dasharray="{dash}"/>')
                o.append(f'<polygon points="{xe - 7:.1f},{base + 38} {xe + 7:.1f},{base + 38} {xe:.1f},{base + 25}" fill="{OLD_C}"/>')
            yl = base + 125 + 26 * (j - 1)
            o.append(sd.line(cx - 260, yl - 6, cx - 210, yl - 6, OLD_C, 3, dash))
            o.append(sd.T(cx - 200, yl, lab, 19, OLD_C, 'start', 600,
                          tip=('A discrete copy of the impulse response, delayed and smeared by σ_s = '
                               f'{h["sigma_s"]:.0f} ns. c1 = {h["c1"]:.3f}, kY = {h["kY"]:.2f} on Y (1 on X), '
                               f'c2/c1 = {h["r21"]:.2f}.')))
        o.append(sd.T(16, 24, 'OLD: a prompt Gaussian + discrete delayed copies to ±1 and ±2', 22, OLD_C, 'start', 600))
    else:
        for k, (t, c) in enumerate(((0, INK), (500, ORANGE), (2000, PURPLE))):
            sg = math.sqrt(n['sig0'] ** 2 + 2 * n['Drc'] * t)
            ys = base - 30 - 120 * (n['sig0'] / sg) * np.exp(-0.5 * ((xs - cx) / (sg / PITCH * sw)) ** 2)
            o.append(sd.poly(list(xs), list(ys), c, 3.5,
                             tip=f't = {t} ns: σ = {sg:.2f} mm (σ0 = {n["sig0"]:.2f}, D_rc = {n["Drc"] * 1e4:.1f}×10⁻⁴ mm²/ns)'))
            o.append(sd.T(W - 20, 60 + 26 * k, f'{t} ns after landing: σ = {sg:.2f} mm', 19, c, 'end', 600))
        o.append(sd.T(cx, base + 70, 'no copies: the share of each strip changes continuously,', 19, NEW_C, 'middle', 600))
        o.append(sd.T(cx, base + 96, '±2 is reached only by passing ±1', 19, NEW_C, 'middle', 600))
        o.append(sd.T(16, 24, 'NEW: one Gaussian that keeps widening (Y only)', 22, NEW_C, 'start', 600))
    return sd.svg(W, H, ''.join(o), 'old kernel' if old else 'new kernel')


def s_old_new(D, d):
    det = 'det3'
    h, n = d.prod[det], d.new[det]
    P = {}
    lim = (-0.2, 1.4)
    for which in ('old', 'new'):
        Pl = sd.Plot(800, 400, x=lim, y=(-0.08, 0.4), xlabel='time after landing [µs]',
                     ylabel='signal / centre peak' if which == 'old' else '', margin=(10, 20, 92, 96))
        Pl.xticks([(v, f'{v:g}') for v in (0, 0.4, 0.8, 1.2)]).yticks([(v, f'{v:g}') for v in (0, 0.1, 0.2, 0.3, 0.4)])
        Pl.hline(0, MUT, None, 1.2)
        if which == 'old':
            ws = {o_: old_wave(d, det, 'y', o_) for o_ in (0, 1, 2)}
        else:
            ws = {o_: new_wave(d, n['sig0'], n['Drc'], o_) for o_ in (0, 1, 2)}
        pk = ws[0].max()
        for o_ in (0, 1, 2):
            y = ws[o_] / pk
            lab = {0: 'centre ÷ 3', 1: '±1', 2: '±2'}[o_]
            yy = y / 3 if o_ == 0 else y
            i = int(np.argmax(y))
            Pl.line(list(TT / 1e3), list(yy), S_C[o_], 4, markers=False,
                    tip=f'{which} kernel, det3 Y, strip {lab}: peak {y.max():.3f} of the centre at {TT[i]:.0f} ns; '
                        f'area {y.sum() / (ws[0] / pk).sum():.2f} of the centre\'s (to 1.4 µs)')
            Pl.text(TT[i] / 1e3 + (0.08 if o_ else -0.04), float(yy.max()) + (0.0 if o_ == 1 else 0.015), lab, 21, S_C[o_],
                    'start' if o_ else 'end', 600)
        P[which] = Pl
    body = sd.title('Old vs new: delayed copies, or one cloud that keeps spreading',
                    'det3, Y view, one unit charge landing on the centre strip: the mechanism and the strip waveforms it predicts.')
    body += sd.row(sd.col(_mech_svg(True, h, n), P['old'].svg('old waveforms'), gap=0),
                   sd.col(_mech_svg(False, h, n), P['new'].svg('new waveforms'), gap=0), gap=64)
    D.slide('old-new', body, f'''
<p><b>Old (production r06, delay form; <code>wft.model.build_matrix</code>).</b> Each strip receives its own prompt fraction of a
Gaussian (σ_p0 = {h["sigma_p0"]:.3f} mm, plus Dp²·u with drift time) through its own view's template, plus a copy of each ±1 neighbour's
fraction scaled by c1·kY ({h["c1"]:.3f} × {h["kY"]:.2f} on Y) and delayed by τ_s = {h["tau_s"]:.0f} ns, and of each ±2 neighbour's by
c2·kY (c2/c1 = {h["r21"]:.2f}) at 2·τ_s, both smeared by σ_s = {h["sigma_s"]:.0f} ns. On X the copies are the same with kY → 1. The
<code>lp</code> share mode (an RC-dispersed copy instead of a delayed one) was used for the beam studies.</p>
<p><b>New (<code>build_matrix_rc</code>).</b> One Gaussian of σ0 = {n["sig0"]:.3f} mm (+ 2·Dd·u) lands; on Y it widens as 2·D_rc·s with
D_rc = {n["Drc"] * 1e4:.2f}×10⁻⁴ mm²/ns, and every strip sees its share change through the X template. ±1 and ±2 start at once
(their prompt share) and keep rising for microseconds; ±2 always stays below ±1. In the old kernel the neighbours' extra
charge arrives as a short pulse at a fixed delay.</p>
<p>The centre strip is drawn at a third of its height. Both columns use det3's production and rcm constants. The waveforms
are a single landing; a real track adds up 18 such landings along its depth.</p>''', short='Old vs new')


def s_replaced(D, d):
    def v(k, f):
        return ' · '.join(f(d.prod[det]) for det in DETS)
    rows = [
        ['c1', 'amplitude of the ±1 copy', v('c1', lambda h: f'{h["c1"]:.3f}'), '<b>D_rc,y</b>',
         'neighbour charge is continuous RC spread, one constant per chamber, measured'],
        ['c2 (c2/c1)', 'amplitude of the ±2 copy', v('r21', lambda h: f'{h["r21"]:.2f}'), '— (built in)',
         '±2 is reached through ±1 by the diffusion itself; no ratio, no gate'],
        ['kY', 'Y/X copy multiplier', v('kY', lambda h: f'{h["kY"]:.2f}'), 'D_rc,x = 0',
         'X has no resistive spread; the view difference is geometry, not a factor'],
        ['τ_s [ns]', 'delay of the copies', v('tau_s', lambda h: f'{h["tau_s"]:.0f}'), '— (√t spread)',
         'a fixed delay cannot represent a √t spread; the fit found a compromise'],
        ['σ_s [ns]', 'smearing of the copies', v('sigma_s', lambda h: f'{h["sigma_s"]:.0f}'), '—', 'no copies to smear'],
        ['σ_p0 [mm]', 'prompt Gaussian', v('sigma_p0', lambda h: f'{h["sigma_p0"]:.2f}'), '<b>σ0</b> per view',
         'the readout footprint, measured on head-on stacks'],
        ['Dp [mm/√ns]', 'drift diffusion', v('Dp', lambda h: f'{h["Dp"]:.4f}'), '<b>√(2·Dd)</b>',
         'from the X width growth (≈ dry), ~2× the old value'],
        ['template Y', 'Y impulse response', f'own ({float(d.tmpl["y"].min()) * 100:.0f} % undershoot, det3)' if d.tmpl['y'] is not None else 'own',
         'X template', 'the Y template contains part of the RC spread'],
    ]
    tips = [None] * len(rows)
    tips[0] = 'Production c1 per chamber (det2 · det3 · det4 · det6 · det7). Fitted on bench cosmics by wft.calibrate.fit_hypers.'
    tips[1] = ('Effective c2/c1: r06 bundles carry c2 = 0 and c2_over_c1 = 0.6 (read with wft.calib.effective_c2); det4/det6 bundles '
               'carry c2 directly.')
    tbl = sd.table(['old hyper', 'role', 'production: det2 · det3 · det4 · det6 · det7', 'new', 'why'], rows, size=21,
                   widths=[170, 270, 520, 200, 560], align=['left', 'left', 'left', 'left', 'left'], tips=tips)
    body = sd.title('Seven fitted hypers become three measured constants',
                    'The production kernel the held-out bench compared against, value by value, and what replaces each.')
    body += tbl
    body += sd.row(
        sd.callout('The old constants were fitted on bench cosmics by the angle χ² (<code>wft.calibrate.fit_hypers</code>), '
                   'not taken from the detector physics. Several directions in that χ² are flat, so the fit could wander '
                   '(c2 &gt; c1 on every bundle shipped before 21 Aug).', OLD_C, 23),
        sd.callout('The new constants come from head-on stacks, where every depth lands on the same strips and the neighbour '
                   'signal is pure lateral spread. None is tuned against the reference angle.', NEW_C, 23), gap=40)
    sm = ', '.join(f'{k} {d.prod[k]["share_mode"]} ({d.prod[k]["bundle"]})' for k in DETS)
    D.slide('replaced', body, f'''
<p>Production bundles and share modes: {sm}. The det4/det6 bundles are named "_lp" but carry no <code>share_mode</code> field, so
the loader's default ('delay') applies; their <code>share_lp</code> hyper is not read by the model.</p>
<p>The old σ_p0 and Dp are not one-to-one with σ0 and Dd: the copy kernel also put neighbour charge in c1·kY, so the three
pieces were traded against each other by the fit (det6's production bundle has σ_p0 = {d.prod["det6"]["sigma_p0"]:.3f} mm and
Dp = {d.prod["det6"]["Dp"]:.4f}, i.e. almost all its spread in the copies). The new kernel has one place for each piece of
physics.</p>
<p>Code: <code>wft/model.py</code> (<code>build_matrix</code> old, <code>build_matrix_rc</code> new, selected by the presence of
<code>rc_D_y</code>), <code>wft/calib.py</code> (<code>HYPER_NAMES</code>, <code>effective_c2</code>, <code>check_kernel_ordering</code>).</p>''',
            short='What is replaced')


def s_c2c1(D, d):
    P1 = sd.Plot(800, 640, x=(-0.6, 8.6), y=(0, 2.4), ylabel='c2 / c1', margin=(20, 20, 120, 100))
    items = []
    for det in ('det3', 'det2', 'det7', 'det4'):
        items.append((det, C2C1_INVERTED[det], OLD_C, f'{det}: shipped before 21 Aug with c2/c1 = {C2C1_INVERTED[det]:.2f}. Retired.'))
    for det in ('det4', 'det6'):
        h = d.prod[det]
        items.append((det, h['r21'], GOLD, f'{det} current production ({h["bundle"]}): c2/c1 = {h["r21"]:.2f}'))
    items.append(('r06', 0.6, GREY, 'r06 bundles (det2, det3, det7): c2 pinned to 0.6·c1 after the retirement.'))
    items.append(('H4', C2C1_BEAM[0], BLUE, f'H4 head-on, model-free: {C2C1_BEAM[0]} ± {C2C1_BEAM[1]} over a 2.6× range of field.'))
    items.append(('det3', C2C1_BENCH3[0], BLUE, f'det3 near-vertical cosmics: {C2C1_BENCH3[0]} ± {C2C1_BENCH3[1]}.'))
    P1.xticks([(i, it[0]) for i, it in enumerate(items)]).yticks([(v, f'{v:g}') for v in (0, 0.5, 1, 1.5, 2)])
    for i, (lab, v, c, tp) in enumerate(items):
        P1.vbar(i, v, 50, c, tip=tp, label=f'{v:.2f}')
    P1.hline(1.0, INK, '8 6', 2.5, tip='Above this line the ±2 copy is stronger than the ±1: physically impossible.')
    P1.text(8.5, 1.04, 'c2 = c1', 21, INK, 'end')
    for x, lab, c in ((1.5, 'shipped, retired', OLD_C), (4.5, 'production', GOLD), (6, 'pinned', GREY), (7.5, 'measured', BLUE)):
        P1.raw(sd.T(P1.X(x), P1.y0 + P1.ph + 68, lab, 21, c, 'middle', 600))
    # right: RC fractions ±2 / ±1 vs time, every chamber
    ts = np.linspace(0, 3000, 61)
    P2 = sd.Plot(800, 640, x=(0, 3.0), y=(0, 1.0), xlabel='time since landing [µs]', ylabel='charge on ±2 / charge on ±1',
                 margin=(20, 120, 92, 100))
    P2.xticks([(v, f'{v:g}') for v in (0, 1, 2, 3)]).yticks([(v, f'{v:g}') for v in (0, 0.25, 0.5, 0.75, 1)])
    P2.hline(1.0, INK, '8 6', 2.5)
    cols = {'det2': BLUE, 'det3': ORANGE, 'det4': RED, 'det6': PURPLE, 'det7': GREEN}
    for det in DETS:
        n = d.new[det]
        sg = np.sqrt(n['sig0'] ** 2 + 2 * n['Drc'] * ts)
        r = frac(2, sg) / frac(1, sg)
        P2.line(list(ts / 1e3), list(r), cols[det], 3.5, markers=False,
                tip=f'{det}: σ0 {n["sig0"]:.2f} mm, D_rc {n["Drc"] * 1e4:.1f}×10⁻⁴: ratio {r[20]:.2f} at 1 µs, {r[-1]:.2f} at 3 µs')
        P2.text(3.02, float(r[-1]) + {'det3': -0.02, 'det4': 0.02}.get(det, 0.0), det, 20, cols[det], 'start', 600)
    body = sd.title('The c2 < c1 problem disappears: there is no ratio left to fit',
                    'Left: the ±2/±1 copy ratio the bundles carried. Right: under RC spreading, ±2 always holds less than ±1.')
    body += sd.row(P1.svg('c2/c1 history'), P2.svg('RC ratio'), gap=50)
    db = d.DC1
    D.slide('c2c1', body, f'''
<p>Physically, charge reaches the ±2 strip only through the ±1, so a copy kernel needs c2 &lt; c1. The cosmic χ² is flat in that
direction, and every bundle shipped before 21 Aug 2026 had walked to c2 &gt; c1 (det3 1.14, det2 1.53, det7 1.75, det4 2.12); all of
them and every product built on them were retired (<code>mx_june_wft/RETIRE_C2GTC1_2026-08-21.md</code>). Since then
<code>wft.calib.check_kernel_ordering</code> refuses an inverted bundle at load, save and install, and r06 pins c2 = 0.6·c1.</p>
<p>That gate is a rule imposed on a parameterisation that cannot express the physics. The data themselves are not a fixed
ratio: on the beam, Y's delayed component reaches ±2 about as strongly as ±1 (decomposition b1 = {db["sps700_y"]["b1"]:.2f}, b2 =
{db["sps700_y"]["b2"]:.2f} at 243 V/cm, FINDINGS §2), which a copy kernel with c2 &lt; c1 cannot draw and RC spreading draws naturally:
late in the pulse ±2 approaches ±1 (right), while the total on ±2 is always below ±1 because a widening Gaussian centred on
strip 0 always puts less charge further out.</p>
<p>In the physical kernel there is no c2: the gate has nothing to check. (The per-view ratio study of 9 Oct, which pinned
c2/c1_y = 0.95, is superseded by this.)</p>''', short='c2 < c1')


def s_dp(D, d):
    det = 'det3'
    h, n = d.prod[det], d.new[det]
    rows = {r['key']: r for r in d.ISO['chambers'][det]['rows']}
    r3 = rows['sat_det3']
    v = h['v']
    dry = float(d.Gn(0.0, 0.0, float(r3['E']))['DT'])
    zs = np.linspace(0, GAP, 61)
    P = sd.Plot(860, 620, x=(0, 30), y=(0, 1.2), xlabel='drift depth [mm]', ylabel='lateral σ at landing [mm]',
                margin=(20, 30, 92, 104))
    P.xticks([(z, f'{z:g}') for z in (0, 5, 10, 15, 20, 25, 30)]).yticks([(s, f'{s:g}') for s in (0, 0.25, 0.5, 0.75, 1.0)])

    def sig(s0, DT):
        return np.sqrt(s0 ** 2 + (DT * 1e-3) ** 2 * zs / 10.0)
    old_DT = dt_from_dp(h['Dp'], v)
    curves = [
        ('old: σ_p0, Dp', sig(h['sigma_p0'], old_DT), OLD_C, None,
         f'production: σ_p0 = {h["sigma_p0"]:.3f} mm, Dp = {h["Dp"]:.4f} mm/√ns = {old_DT:.0f} µm/√cm'),
        ('new: σ0, Dd', sig(n['sig0'], n['DT']), NEW_C, None,
         f'physical: σ0 = {n["sig0"]:.3f} mm, Dd = {n["Dd"] * 1e4:.2f}×10⁻⁴ mm²/ns = {n["DT"]:.0f} µm/√cm'),
        ('Magboltz dry', sig(n['sig0'], dry), DRY_C, '8 6', f'σ0 + dry Ar/iso 95/5 at {r3["E"]:.0f} V/cm: {dry:.0f} µm/√cm'),
        ('Magboltz, v-implied water', sig(n['sig0'], r3['DT_at_v95']), WET_C, '8 6',
         f'σ0 + the water det3\'s v implies: {r3["DT_at_v95"]:.0f} µm/√cm'),
    ]
    for lab, s, c, dash, tp in curves:
        P.line(list(zs), list(s), c, 4.5 if dash is None else 3, dash, markers=False, tip=tp)
    k25 = int(np.searchsorted(zs, 25.0))
    P.text(25, float(curves[0][1][k25]) - 0.07, 'old (production)', 22, OLD_C, 'middle', 600)
    P.text(14, float(curves[1][1][int(np.searchsorted(zs, 14.0))]) + 0.1, 'new ≈ Magboltz dry', 22, NEW_C, 'middle', 600)
    P.text(25, float(curves[3][1][k25]) + 0.05, 'Magboltz, v-implied water', 21, WET_C, 'middle', 600)
    # right: D_T equivalents per chamber
    Q = sd.Plot(760, 620, x=(150, 560), y=(-0.6, 4.6), xlabel='D_T equivalent [µm/√cm]', margin=(20, 20, 92, 110))
    Q.xticks([(x, f'{x}') for x in (200, 300, 400, 500)]).yticks([(i, det) for i, det in enumerate(DETS[::-1])])
    for i, det_ in enumerate(DETS[::-1]):
        hh, nn = d.prod[det_], d.new[det_]
        od = dt_from_dp(hh['Dp'], hh['v'])
        E = d.WE[det_]['run']['E_Vcm']
        dr = float(d.Gn(0.0, 0.0, float(E))['DT'])
        Q.raw(sd.line(Q.X(max(od, 160)), Q.Y(i), Q.X(nn['DT']), Q.Y(i), RULE, 3))
        Q.points([max(od, 160)], [i], OLD_C, 10, tips=[f'{det_} old Dp = {hh["Dp"]:.4f} mm/√ns → {od:.0f} µm/√cm'
                                                       + (' (drawn at the axis edge)' if od < 160 else '')])
        Q.points([nn['DT']], [i], NEW_C, 10, tips=[f'{det_} new Dd → {nn["DT"]:.0f} µm/√cm (v = {nn["v"]:.1f})'])
        Q.points([dr], [i], DRY_C, 8, marker='diamond', tips=[f'{det_} Magboltz dry at {E:.0f} V/cm: {dr:.0f}'])
    leg = sd.legend([('old Dp', OLD_C, 'dot'), ('new Dd', NEW_C, 'dot'), ('Magboltz dry', DRY_C, 'box')], 21)
    ratio = [dt_from_dp(d.prod[k]['Dp'], d.prod[k]['v']) / d.new[k]['DT'] for k in ('det2', 'det3', 'det4', 'det7')]
    body = sd.title(f'The old Dp was about half the measured diffusion; the fit made up the width elsewhere',
                    f'Left: det3, lateral width against depth under each description. Right: D_T implied by each, all chambers.')
    body += sd.row(P.svg('sigma vs depth'), sd.col(leg, Q.svg('DT per chamber'), gap=8), gap=40)
    D.slide('dp', body, f'''
<p>The production Dp (0.014–0.016 mm/√ns on det2/3/4/7) converts to D_T = Dp/√v ≈ {min(dt_from_dp(d.prod[k]["Dp"], d.prod[k]["v"]) for k in ("det2", "det3", "det4", "det7")):.0f}–{max(dt_from_dp(d.prod[k]["Dp"], d.prod[k]["v"]) for k in ("det2", "det3", "det4", "det7")):.0f} µm/√cm, {min(ratio):.2f}–{max(ratio):.2f}×
of what the X head-on widths measure and about half of dry Magboltz (FINDINGS §6). det6's production bundle has almost
no drift term at all (Dp = {d.prod["det6"]["Dp"]:.4f}).</p>
<p>The bench calibration put c1 near its floor (0.05) and loaded the neighbour charge into a constant σ_p0 (0.39–0.43 mm) and
the copies. On head-on bench data alone, the prompt width, the drift diffusion and the drift length are degenerate
(det3 σ0 moves 0.36 ↔ 0.48 while Dd moves ×5, FINDINGS §6), so the angle χ² could settle anywhere along that valley. The new
model breaks the degeneracy with a joint X+Y fit: X has no RC term, so X's width growth with depth is drift diffusion
alone.</p>
<p>Left panel: the old model is too narrow for deep charge and similar for shallow charge; the new one follows the dry
curve. The orange curve is what the gas model would predict from v: that is the open X width excess (slide §§xwidth§§).</p>''',
            short='Dp')


def s_tau(D, d):
    rows = []
    for det in DETS:
        rows.append((f'{det} τ_s (prod)', d.prod[det]['tau_s'], None, OLD_C,
                     f'{det} production τ_s = {d.prod[det]["tau_s"]:.0f} ns, fitted by the angle χ² on bench cosmics'))
    rows.append(('det3 bench Y, direct', TAU_DIRECT_DET3['y'], None, BLUE,
                 f'det3 bench, direct waveform measurement of Y spreading: {TAU_DIRECT_DET3["y"]} ns (X {TAU_DIRECT_DET3["x"]} ns)'))
    ty = [d.DC1[k]['tau'] for k in ('sps700_y', 'sps450_y', 'sps275_y')] + \
         [d.DC3[k]['tau'] for k in ('sps700_y', 'sps450_y', 'sps275_y')]
    rows.append(('beam det4 Y, decomposition', min(ty), max(ty), BLUE,
                 'run_71 Y: W(±1) = a·W0 + b·LP_τ(W0), fitted to 1.3 µs and to 3.5 µs, three fields: τ = '
                 + ', '.join(f'{t:.0f}' for t in ty) + ' ns'))
    for det in DETS:
        n = d.new[det]
        tp = (PITCH ** 2) / (2 * n['Drc'])
        rows.append((f'{det} pitch² / 2·D_rc', tp, None, NEW_C,
                     f'{det}: time for the RC spread alone to reach one readout pitch: 0.78² / (2 × {n["Drc"] * 1e4:.1f}×10⁻⁴) = {tp:.0f} ns'))
    tpb = PITCH ** 2 / (2 * d.beam_drc_corr)
    rows.append(('det4 beam pitch² / 2·D_rc', tpb, None, NEW_C, f'det4 beam D_rc {d.beam_drc_corr * 1e4:.1f}×10⁻⁴: {tpb:.0f} ns'))
    nrow = len(rows)
    P = sd.Plot(1100, 700, x=(0, 1500), y=(-0.7, nrow - 0.3), xlabel='time scale [ns]', margin=(16, 30, 92, 330))
    P.xticks([(x, f'{x}') for x in (0, 250, 500, 750, 1000, 1250, 1500)])
    P.yticks([(nrow - 1 - i, r[0]) for i, r in enumerate(rows)])
    lo_old = min(d.prod[k]['tau_s'] for k in DETS)
    hi_old = max(d.prod[k]['tau_s'] for k in DETS)
    P.raw(f'<rect x="{P.X(lo_old):.1f}" y="{P.y0}" width="{P.X(hi_old) - P.X(lo_old):.1f}" height="{P.ph}" fill="{OLD_C}" '
          f'fill-opacity="0.08"{sd.tipattr(f"Production τ_s range: {lo_old:.0f}–{hi_old:.0f} ns")}/>', back=True)
    for i, (lab, a, b, c, tp) in enumerate(rows):
        y = nrow - 1 - i
        if b is not None:
            P.raw(sd.line(P.X(a), P.Y(y), P.X(b), P.Y(y), c, 8))
            P.raw(f'<rect class="hit" x="{P.X(a):.1f}" y="{P.Y(y) - 12:.1f}" width="{P.X(b) - P.X(a):.1f}" height="24" '
                  f'fill="transparent"{sd.tipattr(tp)}/>')
        else:
            P.points([a], [y], c, 10, tips=[tp])
    dlo, dhi = min(TAU_DIRECT_DET3['y'], min(ty)), max(ty)
    side = sd.col(
        sd.p(f'The fitted copy delay τ_s sits at {lo_old:.0f}–{hi_old:.0f} ns on every chamber.', 26),
        sd.p(f'Measured directly, the Y neighbour signal builds up over {dlo:.0f}–{dhi:.0f} ns, and the RC constants put one pitch of '
             'spreading at hundreds of ns to over a microsecond.', 26),
        sd.callout('A single delayed copy cannot follow a √t spread. The χ² picked the compromise that fits the early '
                   'part of the pulse; the late part was misfit, seen as "Y copies too early on the beam".', OLD_C, 24),
        gap=24, w=520)
    body = sd.title(f'τ_s was a fitted compromise, {dlo / hi_old:.1f}–{dhi / lo_old:.0f}× faster than the spreading it stood for',
                    'Time scales of the neighbour signal: the old kernel\'s delay against direct measurements and the RC constants.')
    body += sd.row(P.svg('time scales'), side, gap=30)
    D.slide('tau', body, f'''
<p><code>tau_s</code> and <code>sigma_s</code> (and c1, c2, kY) were fitted on bench cosmics by <code>wft.calibrate.fit_hypers</code>; they do not
come from simulation. One value serves both views. Direct waveform measurements give much slower spreading: 410 ns on the det3
bench for Y (230 ns for X, HANDOFF_2026-07-30), and {min(ty):.0f}–{max(ty):.0f} ns on the beam (decomposition into a prompt part plus a
low-passed copy of the centre strip, <code>decompose.py</code>; X has no delayed part at all on the beam).</p>
<p>The two closure failures logged earlier, "Y copies 13–15σ too early on the beam" and "Y ±2 delay too short on the bench", are
the same defect seen twice: the copy form has one delay, the physics has a continuum (FINDINGS §4).</p>
<p>The "pitch² / 2·D_rc" rows are not a fitted time constant: they are the time it takes the RC spread alone to widen the cloud
by one readout pitch in quadrature, a natural scale of the mechanism. The bench decomposition is not shown: there the drift
diffusion makes the neighbour/centre ratio grow through the pulse and the decomposition is unphysical for X (FINDINGS §2).</p>''',
            short='τ_s')


def s_heldout(D, d):
    def panel(view, title):
        P = sd.Plot(810, 660, x=(-0.6, 4.6), y=(-1.05, 0.35), title=title,
                    ylabel='Δ s68, new − old [°]' if view == 'y' else '', margin=(10, 20, 80, 104))
        P.xticks([(i, k) for i, k in enumerate(DETS)]).yticks([(v, f'{v:+.2f}' if v else '0') for v in (-1.0, -0.75, -0.5, -0.25, 0, 0.25)])
        P.raw(f'<rect x="{P.x0}" y="{P.Y(0):.1f}" width="{P.pw}" height="{P.Y(-1.05) - P.Y(0):.1f}" fill="{NEW_C}" fill-opacity="0.05"/>', back=True)
        P.hline(0, INK, None, 2)
        P.text(4.55, -1.0, 'new is better ↓', 20, NEW_C, 'end', 600)
        for i, det in enumerate(DETS):
            c = d.cmp(det, 'rcm', view)
            for dx, key, mk in ((-0.14, 'all', 'circle'), (0.14, 'head', 'open')):
                val, e = c[key], c[key + '_e']
                col = NEW_C if val + 2 * e < 0 else (OLD_C if val - 2 * e > 0 else GREY)
                P.raw(errbar(P, i + dx, val, e, col))
                pa, aa = c['prod_' + key], c['arm_' + key]
                P.points([i + dx], [val], col, 9, marker=mk,
                         tips=[f'{det} {view.upper()} {"all tracks" if key == "all" else "head-on"}: s68 {pa:.2f} → {aa:.2f}°, '
                               f'Δ = {val:+.3f} ± {e:.3f}° (n = 2000 held-out events). Angle scale: slope '
                               f'{c["slope_prod"]:.2f} → {c["slope_arm"]:.2f}.'])
        return P
    leg = sd.legend([('filled: all tracks · open: head-on', INK, 'dot'),
                     ('better by > 2σ', NEW_C, 'dot'), ('worse by > 2σ', OLD_C, 'dot'), ('within 2σ', GREY, 'dot')], 21)
    chi = ', '.join(f'{k} {a} → {b}' for k, (a, b) in X_CHI2.items())
    body = sd.title('Held out, no refit: Y better on all five chambers, X mixed',
                    f'Paired change in the {sd.term("scale-corrected s68", G["s68"])} angle resolution, '
                    f'{sd.term("physical kernel (rcm)", G["rcm"])} − {sd.term("production", G["r06"])}, 2000 events per chamber.')
    body += leg + sd.row(panel('y', 'Y view').svg('Y'), panel('x', 'X view').svg('X'), gap=44)
    D.slide('heldout', body, f'''
<p><b>Test.</b> <code>make_rc_arms.py</code> → <code>plane_bench.py</code> (2000 held-out events per golden run, the same events for both kernels)
→ <code>compare_bench.py --start blind</code> (<code>results/compare_bench_rc.json</code>). The physical kernel takes every constant from
the head-on measurement and is not refitted against the reference. The comparison is scale-corrected: each arm's tan is
divided by its own slope against M3 before s68 is taken, because the rc arms read angles 3–8 % flat (the bundle v and the
per-chamber angle scale kw were calibrated under the old kernel).</p>
<p><b>Y</b> improves everywhere: {"; ".join(f"{k} {d.cmp(k, 'rcm', 'y')['all']:+.2f} / {d.cmp(k, 'rcm', 'y')['head']:+.2f}°" for k in DETS)} (all / head-on).
The head-on gain is the largest, where the old kernel's copy timing mattered most.</p>
<p><b>X</b>: neutral on det2/det3, better on det6/det7, worse on det4 over all tracks ({d.cmp("det4", "rcm", "x")["all"]:+.2f} ± {d.cmp("det4", "rcm", "x")["all_e"]:.2f}°).
The X χ²/dof is worse than production on every chamber ({chi}): the model's X is missing something that production's small X
copies were absorbing. The ref start agrees (det2 Y −0.10/−0.30, det6 Y −0.27/−0.70, det4 X +0.09).</p>''',
            short='Held-out bench')


def s_x_arms(D, d):
    arms = [('rcm', d.CR, 'rcm: measured, no refit'), ('rcf3', d.CR, 'rcf3: σ0,x, σ0,y, D_rc,y refitted'),
            ('rcf4x', d.CR, 'rcf4x: + D_rc,x free'), ('rcfD', d.CR, 'rcfD: + Dp free'),
            ('rcmlor', d.CL, 'rcmlor: + footprint tails'), ('rcmlor2', d.CL, 'rcmlor2: tails, variant')]
    ac = [NEW_C, BLUE, PURPLE, GOLD, ORANGE, GREY]
    P = sd.Plot(1060, 620, x=(-0.6, 4.6), y=(-0.55, 0.3), ylabel='X: Δ s68, all tracks, vs production [°]',
                margin=(16, 20, 80, 110))
    P.xticks([(i, k) for i, k in enumerate(DETS)]).yticks([(v, f'{v:+.1f}' if v else '0') for v in (-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3)])
    P.hline(0, INK, None, 2)
    for j, ((arm, src, lab), c) in enumerate(zip(arms, ac)):
        dx = -0.3 + j * 0.12
        for i, det in enumerate(DETS):
            q = d.cmp(det, arm, 'x', src)
            P.raw(errbar(P, i + dx, q['all'], q['all_e'], c, 2, 5))
            P.points([i + dx], [q['all']], c, 7, tips=[f'{det} X, {lab}: Δ = {q["all"]:+.3f} ± {q["all_e"]:.3f}° '
                                                       f'(head-on {q["head"]:+.3f} ± {q["head_e"]:.3f})'])
    leg = sd.legend([(a[2], c, 'dot') for a, c in zip(arms, ac)], 20)
    rdx = [d.CR[f'{k}_rcf4x_x'] for k in DETS]
    side = sd.col(
        sd.p('Refitting the physical constants, freeing an X RC term, freeing the drift diffusion, or adding footprint '
             'tails: none moves X by more than its error, except rcfD on det4.', 25),
        sd.callout(f'det4 X stays {d.cmp("det4", "rcm", "x")["all"]:+.2f}° worse. det4\'s amplification stripes run across X, '
                   'a gain pattern a uniform-gain model cannot represent: an untested hypothesis.', OLD_C, 24),
        sd.p('Freeing Dp brings back the degeneracy: Dp drops to 0.013–0.022 and σ0 inflates to 0.45–0.75 mm for a 1–2 % χ² '
             'gain. The fit uses the footprint to absorb a deficiency.', 22, MUT),
        gap=22, w=560)
    body = sd.title('X is not closed: refits, an X RC term and footprint tails barely move it',
                    'Paired X change against production on the held-out bench, for every X variant tried (condor 4410535, 4410788).')
    body += sd.row(sd.col(leg, P.svg('X arms'), gap=8), side, gap=40)
    D.slide('x-arms', body, '''
<p>Arms (all from <code>recal.py</code> seeded from rcm, benched paired against production; <code>results/compare_bench_rc.json</code>,
<code>compare_bench_lor.json</code>): rcf3 refits σ0,x, σ0,y and D_rc,y; rcf4x adds D_rc,x (it fits to 0–1.2×10⁻⁵, against 3.6–5.9×10⁻⁴ for Y,
so rcf4x ≡ rcf3); rcfD frees Dp too; rcmlor/rcmlor2 replace part of the prompt Gaussian by a Lorentzian (pseudo-Voigt tails,
η 0.10–0.36, γ 0.53–0.91 mm), which improves the head-on profile χ² by 17–37 % but not the angle.</p>
<p>Also rejected for X: charge snapping to the 550 µm resistive-strip centres (no 0.80 mm periodicity in X centroids, Rayleigh
power 0.2–1.7 where ~1 is noise, FINDINGS §23).</p>
<p>What remains open on X: (1) the excess width growth (next slide), which the model takes empirically but does not explain;
(2) det4's X deficit. Both can be pursued on data in hand: det4's gain map against its stripes exists from the bench, and a
position-dependent gain can be written into the model and benched on the same held-out events.</p>''', short='X: arms tried')


def s_xwidth(D, d):
    rows = []
    for det, c in d.ISO['chambers'].items():
        for r in c['rows']:
            k = r['key']
            if k.startswith('sat_det3__hv_scan') or k.startswith('sat_det3__drift'):
                continue
            rows.append((det, k, r))
    rows.sort(key=lambda r: (r[0], r[1]))
    n = len(rows)
    P = sd.Plot(1000, 640, x=(200, 600), y=(-0.7, n - 0.3), xlabel='X transverse diffusion D_T [µm/√cm]',
                margin=(16, 20, 92, 300), title='bench, Ar/iso 95/5')
    P.xticks([(x, f'{x}') for x in (200, 300, 400, 500, 600)])
    P.yticks([(n - 1 - i, f'{det} {k.replace("__long_run_resist_490V_drift_1000V", " long (ref-only)")}') for i, (det, k, _) in enumerate(rows)])
    for i, (det, k, r) in enumerate(rows):
        y = n - 1 - i
        dry = float(d.Gn(0.0, 0.0, float(r['E']))['DT'])
        P.raw(sd.line(P.X(r['DT_at_v95']), P.Y(y), P.X(r['DT']), P.Y(y), RULE, 3))
        P.points([dry], [y], DRY_C, 7, marker='diamond', tips=[f'dry Magboltz at {r["E"]:.0f} V/cm: {dry:.0f}'])
        P.points([r['DT_at_v95']], [y], WET_C, 9, marker='open',
                 tips=[f'Magboltz with the {r["water_95"]:.2f} % water that gives v = {r["v"]:.1f}: {r["DT_at_v95"]:.0f}'])
        P.raw(errbar(P, r['DT'], y, r['DTe'], NEW_C, 2.5, 6, horizontal=True))
        P.points([r['DT']], [y], NEW_C, 9, tips=[f'{k}: measured {r["DT"]:.0f} ± {r["DTe"]:.0f} µm/√cm at {r["E"]:.0f} V/cm'])
    Q = sd.Plot(560, 640, x=(100, 420), y=(-0.7, 2.7), xlabel='D_T [µm/√cm]', margin=(16, 20, 92, 120),
                title='beam det4, Ar/CF₄/iso')
    Q.xticks([(x, f'{x}') for x in (100, 200, 300, 400)])
    bl = [('sps700', 243), ('sps450', 150), ('sps275', 92)]
    Q.yticks([(2 - i, f'{E} V/cm') for i, (_, E) in enumerate(bl)])
    for i, (k, E) in enumerate(bl):
        r = d.XB[k]
        y = 2 - i
        Q.raw(sd.line(Q.X(r['DT_magboltz']), Q.Y(y), Q.X(r['DT']), Q.Y(y), RULE, 3))
        Q.points([r['DT_magboltz']], [y], WET_C, 9, marker='open', tips=[f'Magboltz at the run_71 composition: {r["DT_magboltz"]:.0f}'])
        Q.raw(errbar(Q, r['DT'], y, r['DT_err'], NEW_C, 2.5, 6, horizontal=True))
        Q.points([r['DT']], [y], NEW_C, 9, tips=[f'{E} V/cm: measured {r["DT"]:.0f} ± {r["DT_err"]:.0f} (template-free fit); '
                                                 f'excess in quadrature {r["Dx"]:.0f} ± {r["Dx_err"]:.0f}'])
    leg = sd.legend([('measured, X head-on', NEW_C, 'dot'), ('Magboltz at the gas v implies', WET_C, 'dot'),
                     ('Magboltz dry', DRY_C, 'box')], 21)
    b = d.ISO['chambers']['det3']['fits']['both']
    body = sd.title('Open: X widens like dry gas, though v says the gas is wet',
                    f'X diffusion from head-on stacks against Magboltz, every dataset. No Ar/iso/water mix fits det3\'s field scan (χ² {b["chi2"]:.0f}/{b["ndf"]}).')
    body += leg + sd.row(P.svg('bench DT'), Q.svg('beam DT'), gap=30)
    D.slide('xwidth', body, f'''
<p><b>The observation</b> (FINDINGS §29, report <code>~/x17/cosmic_bench/cloud_basics/xwidth/report.html</code>). On every bench dataset X widens
with depth at D_T 400–520 µm/√cm. Ar/iso 95/5 with the water each run's v needs gives 270–325. The excess, in quadrature, is
~300–430 µm/√cm on every chamber. It does not depend on gain (det3, mesh 425–505 V), covers the whole active area with no hot
spot, lands at the anode (no X undershoot, D_rc,x → 0) and is not track inclination. It is closer to per-depth than per-time
(det6/det7 0.86 ± 0.11 against 1.16 for per-time). The beam has it too ({", ".join(f"+{d.XB[k]['Dx']:.0f}" for k, _ in bl)} in quadrature at 243 / 150 / 92 V/cm;
indicative only, at 92 V/cm the drift is 3× the window and no beam template exists, so the fit is template-free).</p>
<p><b>Not the gas at any composition tried</b>: Ar/iso 0.5–8 % × water 0–1.5 %, seven fields (condor 4440949/4440956). An iso-poor gas
can match one single-field chamber, but not det6 and not det3's five fields together (best χ² {b["chi2"]:.0f}/{b["ndf"]} at iso {b["iso"]} %, water {b["water"]} %).</p>
<p><b>Two readings remain:</b> (1) Magboltz's water model: water slows the drift as predicted but cools the transverse energy
less; both setups are wet. (2) A non-gas lateral term linear in depth, which no current model has.</p>
<p><b>For the model, meanwhile:</b> Dd from the X stack (≈ dry), never from v.</p>''', short='Open: X width')


def s_open(D, d):
    items = [
        ('The X width excess.', 'Unexplained. Checks left, all on data or simulation in hand: a joint X+Y fit on the det3 '
                                'drift-scan Y view (the caches hold it), where Dd is shared with the RC term; the same mixtures '
                                'with another cross-section set or published Ar/iso/H₂O D_T data; other contaminants (N₂, CO₂) in '
                                'the Magboltz grid; the beam X with a proper template re-extracted from run_71, and the 25.6° '
                                'runs as a known-angle cross-check. If they cannot separate the two readings, it is stated as '
                                'a limitation and Dd stays empirical.'),
        ('det4\'s X deficit.', f'{d.cmp("det4", "rcm", "x")["all"]:+.2f} ± {d.cmp("det4", "rcm", "x")["all_e"]:.2f}° against production. '
                               'Hypothesis: its amplification stripes across X. Testable with the bench gain map and a '
                               'position-dependent gain in the model, benched on the same held-out events.'),
        ('X χ² worse everywhere.', 'Even where the X angle is unchanged or better: something production\'s small X copies '
                                   'absorbed is missing (likely the same physics as the width excess).'),
        ('Angle scale.', 'The rc arms read angles 3–8 % flat; the bundle v and kw were calibrated under the old kernel and '
                         'must be re-derived with the new one before any absolute angle is quoted.'),
        ('Edge widths.', 'The early (prompt) X width doubles in ⟨x²⟩ at the chamber edges: a separate footprint / field '
                         'effect, visible in the existing position maps.'),
    ]
    lis = ''.join(f'<li style="margin-bottom:14px"><b style="color:{DINK}">{a}</b> {b}</li>' for a, b in items)
    dec = [
        ('The kernel is not frozen', 'det4 D_rc from the beam, others from the bench; Dd from the joint fit; σ0 from the '
                                     'tight-|tan| stack. Then the full gate.'),
        ('The re-pass is on hold', 'until the X width question is answered or written down as a limitation.'),
        ('Beam bundles', 'still carry the old copy kernel and H4 v/Dp; a beam model would also need attachment per run.'),
    ]
    dl = ''.join(f'<li style="margin-bottom:12px"><b style="color:{DINK}">{a}</b>: {b}</li>' for a, b in dec)
    body = (sd.kicker('What is open, and what this does not rule out')
            + '<div style="display:flex;gap:64px">'
            + f'<div style="flex:1.45"><ul style="font-size:26px;line-height:1.38;color:{DMUT};padding-left:28px">{lis}</ul></div>'
            + f'<div style="flex:1"><p style="font-size:30px;color:{DINK};font-weight:600;margin-bottom:18px">Not yet settled</p>'
            + f'<ul style="font-size:27px;line-height:1.4;color:{DMUT};padding-left:28px">{dl}</ul>'
            + f'<p style="font-size:22px;color:{DMUT};margin-top:24px">The detectors are no longer available: every item above is '
              f'answered from existing data and simulation, or stated as a limitation.</p></div></div>')
    D.slide('open', body, '''
<p>What the held-out result does not show: that the physical kernel is better in absolute angle (it is compared after
scale correction), that it transfers to the n_TOF campaign (different chambers' conditions, gas line and trigger; it has not
been run there), or that the X description is right (it is not yet).</p>
<p>Retracted along the way and not to be revived: bench attachment (a max-normalised stacking artefact, FINDINGS §10), the
field-ordered "leading spike" (threshold alignment, §14), the per-view c2/c1_y = 0.95 re-pass (superseded).</p>
<p>All numbers: <code>sps_beam_test_26/analysis/cloud_basics/FINDINGS.md</code> §1–29; HANDOFF.md, "Detector model basics" and
"X width excess".</p>''', dark=True, short='Open')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT, 'mx17-detector-model-slides.html'))
    a = ap.parse_args()
    d = Data()
    for w in d.warn:
        print('WARNING:', w, file=sys.stderr)
    D = sd.Deck('MX17 detector model',
                'The physical forward model of the MX17 resistive-strip micromegas (drift, readout footprint, RC spread '
                'along the resistive strips, electronics, gas), and how it compares with the old fitted copy kernel.')
    s_cover(D, d)
    s_chamber(D, d)
    s_chain(D, d)
    s_drift(D, d)
    s_footprint(D, d)
    s_rc(D, d)
    s_rc_evidence(D, d)
    s_template(D, d)
    s_gas(D, d)
    s_params(D, d)
    s_old_new(D, d)
    s_replaced(D, d)
    s_c2c1(D, d)
    s_dp(D, d)
    s_tau(D, d)
    s_heldout(D, d)
    s_x_arms(D, d)
    s_xwidth(D, d)
    s_open(D, d)
    p = D.write(a.out, note_meta=dict(
        title='MX17 detector model: the physical forward model, and the old copy kernel',
        summary='Track to waveform in physical steps (drift, σ0 ≈ 0.4 mm readout footprint, RC spread along the resistive '
                'strips on Y only, X template, gas by setup), compared with the r06 copy kernel: Y better on all five '
                'chambers with no refit; X open. Slides.',
        tags='X17, micromegas, detector model, reconstruction',
        date='2026-10-16'))
    ids = [s_[0] for s_ in D.slides]
    txt = p.read_text()
    missing = sorted(set(re.findall(r'§§([a-z0-9-]+)§§', txt)) - set(ids))
    if missing:
        raise SystemExit(f'unknown slide references: {missing}')
    p.write_text(re.sub(r'§§([a-z0-9-]+)§§', lambda m: str(ids.index(m.group(1)) + 1), txt))
    print('wrote', p, f'({len(ids)} slides)')


if __name__ == '__main__':
    main()
