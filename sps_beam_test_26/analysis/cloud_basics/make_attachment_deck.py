#!/usr/bin/env python3
"""make_attachment_deck.py -- the late-charge (attachment) study as a figure-first slide note.

The same results as make_report.py (notes/mx17-beam-attachment), told in slides with hover
tooltips and a Details drop-down per slide, for dylan-neff.web.cern.ch/notes. It opens with
what the observable is and what each candidate mechanism would do to it, then the beam
measurement, the gas model, and the consistency of the one model across beam and bench.
Built with dylan-cern-site/scripts/slidedoc.py. Every chart is drawn from results/*.json and
<OUT>/fits.json + compositions.json (run make_figures.py and make_comp_figures.py first);
model curves are recomputed with the same physics code the fits used.

    ../../../../.venv/bin/python make_attachment_deck.py [--out PATH]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug mx17-attachment-slides --force --deploy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                   # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,      # noqa: E402,F401
                      INK, MUT, RULE, DBLUE, DRED, DGREY, DGREEN, DMUT, DINK)
import gasmodel as M                                                    # noqa: E402
import beam_comp_fit as BC                                              # noqa: E402
import driftscan_fit as DS                                              # noqa: E402
import bench_fit as BF                                                  # noqa: E402
from make_figures import model as box_model, OUT                       # noqa: E402

RES = os.path.join(HERE, 'results')
PLAT = (('raw700', 243), ('raw450', 150), ('raw275', 92))
FC = {243: BLUE, 150: ORANGE, 92: PURPLE}
BEAM_C, BENCH_C, MODEL_C, NOAIR_C = RED, GREEN, INK, GREY
O2_PER_AIR = 2095.0                      # ppm O2 per % air
BENCH_AIR_CONTRAST = 0.04                # % air: the bench grid's top, ~half the beam's O2


def J(name, base=RES):
    return json.load(open(os.path.join(base, name)))


def win(t, y, lo, hi):
    t = np.asarray(t)
    return float(np.nanmean(np.asarray(y)[(t >= lo) & (t <= hi)]))


# --------------------------------------------------------------------------- #
# glossary
# --------------------------------------------------------------------------- #
G = dict(
    headon=('Head-on in a view: the track is perpendicular to that view\'s strips (|tan θ| < 0.03), so '
            'charge from every depth lands on the same few strips. Their summed signal is then the '
            'arrival current, with no drift ladder mixed in.'),
    R=('R = level(2.4–2.7 µs) / level(1.08–1.26 µs) after the trigger: how much of the early plateau '
       'is still arriving 1.3 µs later. 1 = nothing lost.'),
    r=('r: the loss rate per unit drift time, from a forward fit of box(T)·e^(−rt) ⊗ the measured '
       'electronics response. For attachment r = η·v (attachment coefficient × drift velocity).'),
    raw=('RAW mode: every sample of every strip is read out (no zero suppression). The beam\'s run_71 is '
         'the only RAW dataset; the FEU drops ~20–25 % of its samples in ~5-sample packets, which are '
         'kept as missing (NaN), never zero.'),
    zs=('Zero suppression: only samples above ~4–5σ of the pedestal are kept. It removes small and '
        'negative samples, which distorts a stack additively.'),
    trig=('Trigger-placed: each event is placed by the beam trigger (DREAM sample index), never by its own '
          'pulse. Aligning on the pulse puts the biggest ionisation cluster at t = 0 and fakes a spike '
          'and a sag.'),
    magboltz=('Magboltz (via Garfield++): electron transport in a gas mixture from cross sections. Gives '
              'drift velocity v, attachment coefficient η and diffusion for any composition and field. '
              'High-statistics grid: 3×10⁸ collisions per point (condor 4410759/4410787).'),
    o2=('O₂ attaches electrons through a three-body process; H₂O or isobutane can be the third body, '
        'which Magboltz models poorly. The absolute ppm scale is uncertain by a factor of a few; the '
        'beam/bench contrast and the time trend are not.'),
    det4=('det4: the MX17 chamber that went to the SPS H4 beam (31 Jul – 3 Aug 2026), parasitic in the '
          'banco P2 setup. Ar/CF₄/iso 88/10/2, 30 mm drift gap.'),
    bench=('The June 2026 cosmic bench: MX17 chambers det2/3/4/6/7 under the M3 reference telescope, '
           'Ar/iso 95/5. Same DREAM electronics settings as the beam.'),
    shaper=('Effective shaper: a parametric electronics response fitted per setup. It includes the '
            'gas-dependent ion tail, so it is not the literal DREAM CR-RCⁿ. The DREAM registers are '
            'identical on bench and beam.'),
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self, out):
        self.H = J('headon_masked_k12.json')
        self.t = np.array(self.H['t'], float)
        self.F = J('fits.json', out)
        self.C = J('compositions.json', out)
        self.B = J('beam_comp_fit_air_hs.json')
        self.L = J('ladder_profile.json')
        self.BL = J('bench_ladder.json')
        self.Gv = J('gain_vs_loss.json')
        self.S = J('headon_split.json')
        self.T = J('split_toy.json')
        self.Zt = J('zs_timestack.json')
        self.Dd = J('bench_driftscan.json')
        self.DF = J('driftscan_fit_air_hs_x.json')
        self.BS = J('bench_stack.json')
        self.BFj = J('bench_fit_x_free.json')
        self.Gb = M.GasGrid('beam', 'air_hs')
        self.Gn = M.GasGrid('bench', 'air_hs')
        self.comp = (self.B['water'], self.B['air'])
        self.rates = {E: self.F[l]['r'] for l, E in PLAT}
        self.rerr = {E: self.F[l]['r_err'] for l, E in PLAT}

    def curve(self, lab, v, h):
        S = np.array(self.H[lab][v]['sum'][str(h)])
        ref = win(self.t, S, 1080, 1260)
        return S / ref, np.maximum(np.array(self.H[lab][v]['band'][str(h)]), 1e-3)


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D, d):
    r = d.rates
    rmin, rmax = min(r.values()) * 1e4, max(r.values()) * 1e4
    bm = d.C['beam']
    nums = ''.join([
        sd.bignum(f'{rmin:.2f}–{rmax:.2f}', '×10⁻⁴ of the drifting charge lost per ns', DRED,
                  'Beam, three drift fields, both views: the same per unit time.',
                  tip='run_71 RAW, X ±2 forward fit:\n' + '\n'.join(
                      f'{E} V/cm: r = {r[E] * 1e4:.2f} ± {d.rerr[E] * 1e4:.2f} ×10⁻⁴/ns' for E in (243, 150, 92)),
                  size=64),
        sd.bignum(f'{bm["o2_ppm"]:.0f} ppm', 'O₂ in the beam gas (run_71)', DRED,
                  f'{bm["water"]:.2f} % water + {bm["air"]:.3f} % air, one model, no free loss rate.',
                  tip=G['o2'], size=64),
        sd.bignum('≲ 10 ppm', 'O₂ on the cosmic bench', DGREEN,
                  f'{d.DF["water"]:.2f} % water, no air: six drift fields, same model. '
                  'The bench rejects even half the beam\'s air.',
                  tip='det3 drift scan, 27 Jun, 35–382 V/cm. Statistical limit ≲ 2 ppm; ≲ 10 ppm allowing '
                      'for the model.', size=64),
    ])
    body = (sd.kicker('MX17 · det4 at SPS H4 (Aug 2026) against the June cosmic bench · 10 Oct 2026')
            + '<h1 style="font-size:78px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1640px">'
              'The beam gas lost late drift electrons to oxygen; the bench gas did not. '
              'One gas model explains both.</h1>'
            + f'<p style="font-size:30px;color:{DMUT};width:1540px;line-height:1.35">The observable is the shape '
              'of the arriving charge in time. On the beam it falls at the same rate per nanosecond at every '
              'drift field, in both readout views, with no electronics undershoot. Only attachment does that.</p>'
            + '<div style="flex:1"></div>'
            + f'<div style="display:flex;gap:56px">{nums}</div>')
    D.slide('cover', body, f'''
<p><b>The question.</b> The det4 beam data (SPS H4, early August) showed less charge arriving late in the drift than
early. If real, the reconstruction's forward model needs an attachment term on the beam and none on the bench; if
an artefact, it needs none anywhere. The answer reversed four times in two days (FINDINGS §5, §8–11, §13–15) before
the observable was rebuilt from the raw samples with every stacking trap removed (§16) and the gas was fitted as a
composition rather than a free loss rate (§24–25).</p>
<p><b>What is settled (10 Oct 2026).</b> Every beam dataset with waveforms loses late charge, in both views. In run_71
(RAW, the cleanest dataset) the loss is {rmin:.2f}–{rmax:.2f}×10⁻⁴ per ns at 243, 150 and 92 V/cm, which is the same in time
while the depth reached differs threefold. One physics model (Magboltz transport for the actual gas, no free loss rate)
describes the beam at all three fields with {bm["water"]:.2f} % water and {bm["o2_ppm"]:.0f} ppm O₂, and the bench at six
fields with {d.DF["water"]:.2f} % water and no O₂.</p>
<p><b>How the note runs:</b> the observable and what each mechanism would do to it (slides §§geometry§§–§§method§§); the beam measurement and
the tests that single out attachment (§§run71§§–§§ladder§§); from a gas composition to a waveform (§§gas-model§§–§§beam-comp§§); <b>consistency</b>: the bench
with the same model, its sensitivity, the loss rate against field for both setups, a ledger of every observation (§§consistency§§–§§ledger§§);
where the oxygen comes from and when (§§map§§–§§source§§); what it changes and what it does not rule out (§§consequences§§–§§open§§).</p>
<p>The long-form report with the same numbers stays at <a href="mx17-beam-attachment.html">notes/mx17-beam-attachment</a>.
Generated by <code>sps_beam_test_26/analysis/cloud_basics/make_attachment_deck.py</code> (branch mx17-paper-status).</p>''',
            dark=True, short='Answer')


def s_geometry(D, d):
    """Schematic: head-on track in the drift gap; depth -> arrival time; attachment."""
    W, H = 980, 760
    o = []
    gx0, gx1, gz0, gz1 = 70, 600, 70, 600            # gap: cathode at gz0, mesh at gz1
    o.append(f'<rect x="{gx0}" y="{gz0}" width="{gx1 - gx0}" height="{gz1 - gz0}" fill="#eef1f6" stroke="{RULE}"/>')
    o.append(sd.line(gx0, gz0, gx1, gz0, INK, 4))
    o.append(sd.T(gx1, gz0 - 14, 'drift cathode (z = 30 mm)', 22, INK, 'end', 600))
    o.append(sd.line(gx0, gz1, gx1, gz1, INK, 3, '6 4'))
    o.append(sd.T(gx0, gz1 + 70, 'mesh + amplification (150 µm) + strips', 22, INK, 'start', 600))
    for i in range(18):
        sx = gx0 + 8 + i * 29.2
        c = '#c63d4f' if 4 <= i <= 7 else '#9aa3b2'
        o.append(f'<rect x="{sx:.1f}" y="{gz1 + 14}" width="21" height="12" fill="{c}"/>')
    o.append(sd.T(gx0 + 8 + 5.5 * 29.2 + 10, gz1 + 46, 'X ±2 strips summed', 19, RED))
    # field arrow
    o.append(sd.arrow(gx1 + 34, gz0 + 30, gx1 + 34, gz1 - 30, MUT, 3, 14))
    o.append(sd.T(gx1 + 50, (gz0 + gz1) / 2, 'E (drift field)', 21, MUT, 'start', rot=None))
    o.append(sd.T(gx1 + 50, (gz0 + gz1) / 2 + 28, 'electrons drift down', 19, MUT, 'start'))
    # the track: vertical, head-on in this view
    tx = 250
    o.append(sd.line(tx, gz0 - 50, tx, gz1 + 6, INK, 5))
    o.append(sd.T(tx - 12, gz0 - 34, 'beam track', 22, INK, 'end', 600))
    rng = np.random.default_rng(3)
    zs = np.linspace(gz0 + 20, gz1 - 20, 14)
    for k, z in enumerate(zs):
        dx = rng.normal(0, 3)
        lost = k in (2, 5, 9)
        col = '#c63d4f' if lost else BLUE
        o.append(f'<circle cx="{tx + dx:.1f}" cy="{z:.1f}" r="7" fill="{col}"/>')
        if lost:
            o.append(sd.T(tx + dx - 22, z + 8, '×', 30, RED, 'middle', 600,
                          tip='Attached: the electron is captured by an O₂ molecule during the drift and '
                              'never reaches the mesh. Deeper electrons drift longer, so they are more '
                              'likely to be lost.'))
    # depth labels
    o.append(sd.T(tx - 40, zs[1] + 6, 'deep: arrives last', 20, MUT, 'end'))
    o.append(sd.T(tx - 40, zs[-2] + 6, 'shallow: arrives first', 20, MUT, 'end'))
    o.append(sd.arrow(tx + 30, zs[3], tx + 30, zs[3] + 120, BLUE, 2.5, 11))
    o.append(sd.T(tx + 42, zs[3] + 70, 't = z / v', 26, BLUE, 'start', 600,
                  tip='An electron from depth z arrives after z / v. v is the drift velocity '
                      '(4–14 µm/ns on the beam at 92–243 V/cm). The gap is 30 mm, so the drift lasts '
                      '2–6 µs; the DREAM window is 3.84 µs.'))
    # legend
    o.append(f'<circle cx="{gx0 + 14}" cy="{H - 26}" r="7" fill="{BLUE}"/>')
    o.append(sd.T(gx0 + 30, H - 19, 'ionisation electron', 20, MUT, 'start'))
    o.append(sd.T(gx0 + 290, H - 18, '×', 28, RED, 'middle', 600))
    o.append(sd.T(gx0 + 308, H - 19, 'lost to attachment (O₂)', 20, MUT, 'start'))
    schem = sd.svg(W, H, ''.join(o), 'head-on track in the drift gap')

    right = sd.col(
        sd.p(f'A beam track crosses the 30 mm gap and leaves ionisation <b>uniformly in depth</b>. '
             f'For a {sd.term("head-on", G["headon"])} track, every depth lands on the same strips.', 27),
        sd.p('The electrons drift to the mesh at a constant speed v, so <b>arrival time is depth</b>: '
             'the charge arriving at time t started at depth v·t.', 27),
        sd.callout('So the current reaching the mesh is a flat box, lasting T = 30 mm / v. '
                   'Anything that removes electrons <i>while they drift</i> makes the box fall with time.',
                   BLUE, 27),
        sd.p(f'Hover the {sd.term("dotted terms", "Like this one. Chart points and curves work the same way.")} '
             'and any chart point for definitions, counts and sources.', 22, MUT),
        gap=28, w=620)
    body = sd.title('On a head-on track, arrival time is drift depth',
                    'What a micromegas drift gap does with a straight track, in one readout view.')
    body += sd.row(schem, right, gap=60)
    D.slide('geometry', body, '''
<p>MX17 chambers are 30 mm drift-gap micromegas with two orthogonal strip views (X and Y) under a resistive layer.
A track inclined in a view spreads its depths across strips (the micro-TPC drift ladder used for angles); a track
<b>head-on</b> in a view puts every depth on the same strips, so summing a few strips around the track gives the
arrival current directly, free of the ladder and of the per-strip charge sharing.</p>
<p>The beam (det4 at H4) is nearly perpendicular to the chamber: most tracks are head-on in both views at once,
which is why the beam is the natural place to measure this. On the cosmic bench, tracks arrive at all angles and the
head-on subset is small; slides §§consistency§§–§§bench-chambers§§ show how the bench is tested instead.</p>
<p>X is summed over ±2 strips (X has no resistive spread: the RC fits put its spread at zero). Y needs ±8 to contain its
resistive-strip spread; at ±8 it equals X (slide §§run71§§).</p>''', short='Geometry')


def s_observable(D, d):
    """current -> electronics -> recorded signal, with and without attachment (243 V/cm fit)."""
    f = d.F['raw700']
    tt = np.linspace(0, 3840, 385)
    t0, T, r = f['t0'], f['T'], f['r']
    cur0 = ((tt >= t0) & (tt < t0 + T)).astype(float)
    cur1 = cur0 * np.exp(-r * np.clip(tt - t0, 0, None))
    P1 = sd.Plot(520, 470, x=(0, 3.84), y=(-0.1, 1.15), title='1 · current at the mesh',
                 xlabel='time after trigger [µs]', ylabel='current (arb.)', margin=(10, 20, 90, 96))
    P1.xticks([(v, f'{v:g}') for v in (0, 1, 2, 3)]).yticks([(0, '0'), (0.5, '0.5'), (1, '1')])
    P1.line(list(tt / 1e3), list(cur0), NOAIR_C, 3.5, '10 7', markers=False,
            tip='No loss: a flat box from the first to the last electron (T = 30 mm / v).')
    P1.line(list(tt / 1e3), list(cur1), BEAM_C, 4, markers=False,
            tip=f'With attachment: e^(−rt), r = {r * 1e4:.2f}×10⁻⁴/ns (the beam at 243 V/cm).')
    P1.text(t0 / 1e3 + 0.05, 1.07, 'no loss', 20, NOAIR_C)
    P1.text(1.9, 0.55, 'e^(−rt)', 24, BEAM_C, weight=600)
    P1.vline((t0 + T) / 1e3, MUT, '4 5', 1.5, label='drift ends',
             tip=f'T = {T:.0f} ns at 243 V/cm: v ≈ {30e3 / T:.1f} µm/ns over 30 mm.')
    # template
    TM = J('template_det3_x.json')
    tg, tm = np.array(TM['grid']), np.array(TM['tmpl_x'])
    tm = tm / tm.max()
    P2 = sd.Plot(420, 470, x=(-0.4, 1.4), y=(-0.25, 1.1), title='2 · electronics response',
                 xlabel='time [µs]', margin=(10, 20, 90, 60))
    P2.xticks([(0, '0'), (0.5, '0.5'), (1, '1')]).yticks([(0, '0'), (1, '1')])
    P2.line(list(tg / 1e3), list(tm), INK, 4, markers=False,
            tip='The measured single-electron response (bench det3, X): DREAM preamplifier + shaper, '
                'plus the ion tail. Peaking ~250 ns, with a small negative lobe.')
    P2.text(0.45, 0.8, '⊗', 60, MUT)
    ys = box_model(tt, f['amp'], t0, T, r)
    y0 = box_model(tt, f['amp'], t0, T, 0.0)
    P3 = sd.Plot(640, 470, x=(0.4, 3.84), y=(-0.1, 1.15), title='3 · recorded signal (stacked, normalised)',
                 xlabel='time after trigger [µs]', margin=(10, 20, 90, 70))
    P3.xticks([(v, f'{v:g}') for v in (1, 2, 3)]).yticks([(0, '0'), (0.5, '0.5'), (1, '1')])
    P3.raw(f'<rect x="{P3.X(1.08):.1f}" y="{P3.y0}" width="{P3.X(1.26) - P3.X(1.08):.1f}" height="{P3.ph}" '
           f'fill="{BLUE}" fill-opacity="0.13"{sd.tipattr("Reference window, 1.08–1.26 µs: the plateau just after the rise. Everything is normalised here.")}/>', back=True)
    P3.raw(f'<rect x="{P3.X(2.4):.1f}" y="{P3.y0}" width="{P3.X(2.7) - P3.X(2.4):.1f}" height="{P3.ph}" '
           f'fill="{RED}" fill-opacity="0.13"{sd.tipattr("Late window, 2.4–2.7 µs. R = late / reference.")}/>', back=True)
    P3.band(list(tt / 1e3), list(ys), list(y0), GREY, 0.18, tip='The charge that did not arrive.')
    P3.line(list(tt / 1e3), list(y0), NOAIR_C, 3, '10 7', markers=False, tip='No loss, shaped.')
    P3.line(list(tt / 1e3), list(ys), BEAM_C, 4, markers=False, tip='With the beam\'s loss rate, shaped.')
    P3.text(1.1, 1.08, 'ref', 20, BLUE, weight=600)
    P3.text(2.42, 1.08, 'late', 20, RED, weight=600)
    R = win(tt, ys, 2400, 2700) / win(tt, ys, 1080, 1260)
    R0 = win(tt, y0, 2400, 2700) / win(tt, y0, 1080, 1260)
    P3.text(2.0, 0.42, f'R = {R:.2f} (no loss: {R0:.2f})', 22, INK, weight=600, tip=G['R'])
    figs = sd.row(P1.svg('current'), P2.svg('template'), P3.svg('recorded'), gap=42, align='end')
    body = sd.title('Attachment turns the flat current into a decaying one; the electronics only smooth it',
                    'The observable: the stacked signal\'s late level relative to its early plateau. '
                    'Curves: the 243 V/cm beam fit.')
    body += figs
    body += sd.row(
        sd.callout(f'Two numbers summarise a stack: the ratio {sd.term("R", G["R"])} (model-free) and the '
                   f'loss rate {sd.term("r", G["r"])} from a forward fit. For attachment r = η·v: '
                   'the attachment coefficient times the drift velocity.', BLUE, 25),
        sd.callout('Without any loss R is not exactly 1: the response\'s tail and the jitter round the box. '
                   'The forward fit carries that; the no-loss curve is drawn on every data slide.', GREY, 25),
        gap=48)
    D.slide('observable', body, f'''
<p><b>The current.</b> Uniform ionisation, constant drift velocity: the current at the mesh is constant from the trigger-defined
start t₀ until the deepest electrons arrive at t₀ + T, T = gap / v. An electron that drifts for a time t survives with
probability e^(−η v t), η the attachment coefficient per unit length. So the current becomes I(t) ∝ e^(−rt) with r = η·v.</p>
<p><b>The electronics.</b> The recorded signal is the current convolved with the electronics' impulse response (panel 2: the
measured det3 X template, i.e. DREAM shaping plus the ion tail). The forward fit used for r is box(T)·e^(−rt) ⊗ template,
averaged over the 60 ns sampling phase, with amplitude, t₀, r (and T at 243 V/cm, where the drift ends inside the window) free.</p>
<p><b>The two summaries.</b> R = level(2.4–2.7 µs)/level(1.08–1.26 µs) needs no model; it depends a little on the shaping
(here {R0:.2f} with no loss). r is the physical quantity, and the one compared with Magboltz.</p>
<p>Source: <code>make_figures.model</code>, <code>fits.json</code> (243 V/cm: t₀ = {t0:.0f} ns, T = {T:.0f} ns,
r = {r * 1e4:.2f}×10⁻⁴/ns), <code>results/template_det3_x.json</code>.</p>''', short='Observable')


def s_fingerprints(D, d):
    """What each candidate mechanism would do: attachment, per-depth loss, high-pass, charging."""
    r = d.rates[243]
    vel = {E: d.Gb(*d.comp, float(E))['v'] for E in (243, 150, 92)}
    lam = vel[243] / r          # um: attenuation length if the loss were per depth
    tt = np.linspace(0, 3.6, 181)

    def panel(ttl, mode):
        bottom = mode == 'depth'
        P = sd.Plot(800, 370 if bottom else 320, x=(0, 3.6), y=(0, 1.12), title=ttl,
                    margin=(10, 24, 86 if bottom else 44, 90), xlabel='drift time [µs]' if bottom else '')
        P.xticks([(v, f'{v:g}') for v in (0, 1, 2, 3)]).yticks([(0, '0'), (0.5, '0.5'), (1, '1')])
        for E in (243, 150, 92):
            T = 30e3 / vel[E] / 1e3
            if mode == 'att':
                y = np.where(tt < T, np.exp(-r * tt * 1e3), 0)
                tip = f'{E} V/cm: e^(−rt), the same r at every field. Drift ends at {T:.1f} µs.'
            else:
                y = np.where(tt < T, np.exp(-vel[E] * tt * 1e3 / lam), 0)
                tip = (f'{E} V/cm: e^(−z/λ) with z = v·t, λ = {lam / 1e3:.0f} mm fixed. Slower drift reaches '
                       f'less depth per ns, so the decay in time is {vel[243] / vel[E]:.1f}× slower than at 243 V/cm.')
            P.line(list(tt), list(y), FC[E], 4, markers=False, tip=tip)
        return P

    Pa = panel('Attachment: same loss per unit TIME at every field', 'att')
    Pb = panel('Per-depth loss (field/geometry): same per unit DEPTH', 'depth')
    Pa.text(1.2, 0.86, 'all three fields on one curve', 20, MUT)
    Pa.text(2.15, 0.12, '243 V/cm drift ends', 19, FC[243])
    # high-pass: same sag, then an undershoot when the current stops
    f = d.F['raw700']
    t2 = np.linspace(0, 3840, 769)
    flat = box_model(t2, f['amp'], f['t0'], f['T'], 0.0)
    att = box_model(t2, f['amp'], f['t0'], f['T'], f['r'])
    tau = 1.0 / f['r']
    hp = flat.copy(); acc = 0.0; dtt = t2[1] - t2[0]
    for i in range(1, len(t2)):
        acc = acc * np.exp(-dtt / tau) + flat[i - 1] * dtt / tau
        hp[i] = flat[i] - acc
    Pc = sd.Plot(800, 320, x=(0.4, 3.84), y=(-0.4, 1.12), title='Readout high-pass: same sag, then an UNDERSHOOT',
                 margin=(10, 24, 44, 90))
    Pc.xticks([(v, f'{v:g}') for v in (1, 2, 3)]).yticks([(-0.3, '−0.3'), (0, '0'), (0.5, '0.5'), (1, '1')])
    Pc.hline(0, MUT, None, 1)
    Pc.line(list(t2 / 1e3), list(hp), GOLD, 4, markers=False,
            tip=f'An AC coupling with τ = 1/r = {tau / 1e3:.1f} µs sags the plateau the same way, but must swing to '
                f'≈ {d.F["undershoot"]["highpass"]:+.2f} after the drift ends: the charge it removed comes back negative.')
    Pc.line(list(t2 / 1e3), list(att), BEAM_C, 3, '10 7', markers=False,
            tip='Attachment: the charge is gone, so the signal returns to zero and stays there.')
    Pc.text(2.05, -0.27, 'undershoot →', 21, GOLD, weight=600)
    Pc.text(1.3, 0.55, 'attachment (dashed)', 20, BEAM_C)
    # charging: R vs gain
    Pd = sd.Plot(800, 370, x=(0.4, 2.2), y=(0.4, 1.0), title='Charging / space charge: scales with GAIN or RATE',
                 margin=(10, 24, 86, 90), xlabel='local gain / mean (or beam rate)')
    Pd.xticks([(0.5, '0.5'), (1, '1'), (1.5, '1.5'), (2, '2')]).yticks([(0.5, '0.5'), (0.75, '0.75'), (1, '1')])
    gg = [0.5, 2.1]
    Pd.line(gg, [1 - 0.23 * g for g in gg], GOLD, 4, markers=False,
            tip='Resistive-layer charging that produced the whole loss: R ≈ 1 − 0.23·(gain/mean). '
                'Space charge would grow with beam rate the same way.')
    Pd.line(gg, [0.77, 0.77], BEAM_C, 3, '10 7', markers=False,
            tip='Attachment happens in the drift gas before amplification: R does not care about gain or rate.')
    Pd.text(1.75, 0.66, 'charging', 21, GOLD, weight=600)
    Pd.text(1.45, 0.80, 'attachment (flat)', 20, BEAM_C)
    grid = (f'<div style="display:grid;grid-template-columns:800px 800px;gap:6px 64px">'
            f'{Pa.svg("attachment")}{Pc.svg("highpass")}{Pb.svg("per depth")}{Pd.svg("charging")}</div>')
    leg = sd.legend([('243 V/cm', FC[243]), ('150 V/cm', FC[150]), ('92 V/cm', FC[92]),
                     ('attachment', BEAM_C, 'dash'), ('alternative', GOLD)], size=22)
    body = sd.title('Four ways to lose late charge leave four different fingerprints',
                    'Predictions, not data. The measurement on the next slides is designed to tell them apart.')
    body += leg + grid
    D.slide('fingerprints', body, f'''
<p>Each panel is what a candidate mechanism predicts for the observable, built with the beam's own numbers (r at 243 V/cm,
Magboltz drift velocities {vel[243]:.1f} / {vel[150]:.1f} / {vel[92]:.1f} µm/ns at 243 / 150 / 92 V/cm).</p>
<ol>
<li><b>Attachment</b> (electrons captured in the gas). Loss per unit time r = η·v, and η·v is nearly field-independent for O₂
in this gas over 75–150 V/cm: the three fields fall together in time. Drift ends at different times because v differs.</li>
<li><b>A loss per unit depth</b> (a field gradient near the cathode, a geometry effect, a depth-dependent collection).
Fixed attenuation length λ = {lam / 1e3:.0f} mm (the value that would give the 243 V/cm loss). At slower drift the same depth
takes longer to reach, so the decay in time slows in proportion to v.</li>
<li><b>A readout high-pass</b> (AC coupling, baseline restoration). It can sag the plateau identically, but a high-pass passes
no DC: when the current stops it must undershoot by the charge it removed.</li>
<li><b>Gain-side effects</b> (resistive-layer charging-up, space charge in the amplification gap). They act after the drift,
on the avalanche, so they track the local gain and the beam rate. The −0.23 slope is what charging would need to make the
whole observed loss across det4's ×1.4–2.8 gain stripes.</li>
</ol>''', short='Fingerprints')


def s_method(D, d):
    good = [
        dict(label='Place by the trigger', sub='DREAM sample index', color=GREEN, tip=G['trig']),
        dict(label='Keep missing samples missing', sub='average each strip-sample over events that have it',
             color=GREEN, tip=G['raw']),
        dict(label='Sum strips', sub='X ±2, Y ±8 (contains the RC spread)', color=GREEN,
             tip='Widths 0…12 were all computed; X converges at ±2, Y by ±8.'),
        dict(label='Normalise at 1.08–1.26 µs', sub='forward-fit r, bootstrap bands', color=GREEN,
             tip='100 bootstrap resamples of events give the shaded bands and the errors on R and r.'),
    ]
    bad = [
        dict(label='Align on the pulse', sub='fake spike + field-ordered sag (§11, §13)', color=RED, strike=True,
             tip='Threshold or peak alignment puts the largest Landau cluster at t = 0. This produced the '
                 '"field-ordered spike" and the 5× O₂ spread that were retracted.'),
        dict(label='Zero-fill dropped RAW packets', sub='~6 % fake late loss (§14)', color=RED, strike=True,
             tip='The FEU drops ~20–25 % of RAW samples, slightly more late in the window. Counting them as 0 '
                 'reads as a loss.'),
        dict(label='Ratio-correct zero suppression', sub='ZS is additive, breaks past the drift end', color=RED,
             strike=True, tip=G['zs']),
        dict(label='Raw Magboltz η per field', sub='±10–15 % MC noise between fields (§24)', color=RED, strike=True,
             tip='Even at 3×10⁸ collisions; smoothed in log E per mixture (gasmodel.GasGrid).'),
    ]
    body = sd.title('How the stack is built, and the four traps that reversed the answer',
                    'run_71 RAW: 20 000 head-on events per field. Every earlier pass fell into at least one of the bottom row.')
    body += sd.p('<b>The stack, as built for every result in this note</b>', 26, GREEN)
    body += sd.flow(good, size=24)
    body += sd.p('<b>What earlier passes did, and what each one faked</b>', 26, RED)
    body += sd.flow(bad, size=24)
    body += sd.row(
        sd.callout('A fifth trap is about the bench: a single-field bench plateau lasts only ~800 ns, and an '
                   'undershoot, a field gradient and a small loss all tilt it the same way. The bench is therefore '
                   'tested with a <b>drift scan</b> (six fields) and a template-free ladder, not one field.', GOLD, 24),
        sd.callout('Before quoting any late/early ratio from stacked DREAM waveforms, check all five. They are '
                   'recorded as a standing rule (memory: <i>stack artefacts in DREAM waveforms</i>).', GREY, 24),
        gap=40)
    D.slide('method', body, '''
<p><b>Trigger placement.</b> On the beam the DREAM window opens on the trigger, so each sample index is a fixed time after the
particle. Events are summed at fixed sample index with per-strip pre-trigger baselines and a common mode computed from
strips masked away from the track (<code>extract_det4_only.py --cm masked --keep 12</code>). The earlier block-median
common mode at ±4 strips biases R by &lt; 1 %.</p>
<p><b>Missing samples.</b> RAW run_71 has 80 % of samples present early and 75 % late. <code>headon_stack.py</code> keeps
them NaN and averages each strip-sample only over the events that have it. The same missingness also biased an
event-charge split (a sum over present samples selects on missingness): the "faint-tercile excess" of §16, resolved in §19.</p>
<p><b>Zero suppression</b> (run_63, run_56). The distortion is measured on RAW run_71 by emulating ZS at 4σ and 5σ, and
applied additively (ZS-emulated − RAW) with a 1 % systematic. run_63's real ZS 4σ R (X ±1) matches RAW run_71 emulated at 4σ.</p>
<p>The reversals: §5 (loss) → §10 (no loss: stacking artefact) → §11 (common to X and Y) → §13/§14 (spike = alignment) →
§16 (rebuilt from raw samples). Each retraction is kept, struck through, in <code>FINDINGS.md</code>.</p>''',
            short='Method')


def s_run71(D, d):
    panels = []
    for i, (lab, E) in enumerate(PLAT):
        f = d.F[lab]
        yx, ex = d.curve(lab, 'x', 2)
        yy, ey = d.curve(lab, 'y', 8)
        k = d.t >= 400
        t, yx, ex, yy, ey = d.t[k] / 1e3, yx[k], ex[k], yy[k], ey[k]
        P = sd.Plot(548, 590, x=(0.4, 3.84), y=(-0.1, 1.15), title=f'{E} V/cm',
                    xlabel='time after trigger [µs]', ylabel='signal / 1.08–1.26 µs level' if i == 0 else '',
                    margin=(10, 18, 90, 96 if i == 0 else 40))
        P.xticks([(v, f'{v:g}') for v in (1, 2, 3)]).yticks(
            [(0, '0'), (0.25, '0.25'), (0.5, '0.5'), (0.75, '0.75'), (1, '1')] if i == 0 else
            [(v, '') for v in (0, 0.25, 0.5, 0.75, 1)])
        tt = np.linspace(400, 3840, 300)
        m1 = box_model(tt, f['amp'], f['t0'], f['T'], f['r'])
        m0 = box_model(tt, f['amp'], f['t0'], f['T'], 0.0)
        P.band(list(tt / 1e3), list(m1), list(m0), GREY, 0.2, tip='Charge that did not arrive: the same fit with r → 0.')
        P.line(list(tt / 1e3), list(m0), NOAIR_C, 3, '10 7', markers=False, tip='Same fit, r set to 0.')
        P.band(list(t), list(yx - ex), list(yx + ex), FC[E], 0.2)
        tips = [f'{E} V/cm, X ±2\nt = {a * 1e3:.0f} ns\n{b:.3f} ± {c:.3f}\n{d.H[lab]["n"]} events' for a, b, c in zip(t, yx, ex)]
        P.line(list(t), list(yx), FC[E], 4, r=3, tips=tips, tip='X, ±2 strips')
        P.line(list(t), list(yy), FC[E], 3, '9 6', markers=False, tip='Y, ±8 strips')
        P.line(list(tt / 1e3), list(m1), INK, 2, markers=False,
               tip=f'Fit: r = {f["r"] * 1e4:.2f} ± {f["r_err"] * 1e4:.2f}×10⁻⁴/ns, χ² {f["chi2"]:.0f} '
                   f'(r = 0: {f["chi2_r0"]:.0f}), {f["ndf"]} points')
        rx = d.H[lab]['x']['metrics']['2']['r2400']
        ry = d.H[lab]['y']['metrics']['8']['r2400']
        P.text(1.5, 0.2, f'r = {f["r"] * 1e4:.2f} ± {f["r_err"] * 1e4:.2f}', 24, INK, weight=600)
        P.text(1.5, 0.08, f'R: X {rx:.3f} · Y {ry:.3f}', 21, MUT)
        panels.append(P.svg(f'run_71 {E}'))
    leg = sd.legend([('X ±2 strips', INK), ('Y ±8 strips', INK, 'dash'), ('fit, e^(−rt)', INK),
                     ('same fit, no loss', GREY, 'dash'), ('charge not arrived', '#d9dbde', 'box')], size=22)
    rr = ' / '.join(f'{d.rates[E] * 1e4:.2f}' for E in (243, 150, 92))
    body = sd.title(f'Late charge is missing in both views at every field: r = {rr} ×10⁻⁴/ns',
                    'run_71 RAW, det4 head-on, Ar/CF₄/iso, 3 Aug 05:22–05:52. Loss rate r in units of 10⁻⁴/ns.')
    body += leg + sd.row(*panels, gap=10)
    D.slide('run71', body, f'''
<p>run_71 is the one beam dataset taken in <b>RAW</b> mode (no zero suppression), at three drift voltages: 700, 450 and 275 V
(243, 150, 92 V/cm). 20 000 head-on events per field. X is summed over ±2 strips, Y over ±8; shaded: bootstrap band.</p>
<table><tr><th>field</th><th>R X ±2</th><th>R Y ±8</th><th>r [10⁻⁴/ns]</th><th>χ² fit / r = 0 (55 points)</th></tr>
{''.join(f"<tr><td>{E} V/cm</td><td>{d.H[l]['x']['metrics']['2']['r2400']:.3f} ± {d.H[l]['x']['metrics_err']['2']['r2400']:.3f}</td><td>{d.H[l]['y']['metrics']['8']['r2400']:.3f} ± {d.H[l]['y']['metrics_err']['8']['r2400']:.3f}</td><td>{d.F[l]['r'] * 1e4:.2f} ± {d.F[l]['r_err'] * 1e4:.2f}</td><td>{d.F[l]['chi2']:.0f} / {d.F[l]['chi2_r0']:.0f}</td></tr>" for l, E in PLAT)}</table>
<p>The fit uses X; Y is drawn on top and agrees once it is wide enough to contain its resistive-strip spread (Y ±2 reads lower,
Y ±12 the same as ±8). The residual structure at 92 and 150 V/cm in Y is a trigger-locked ~0.7 µs pickup present on
signal-free strips too (FINDINGS §20): it oscillates about zero and cannot make a monotonic loss.</p>
<p>Source: <code>results/headon_masked_k12.json</code> (<code>headon_stack.py</code>), fits in <code>fits.json</code> (<code>make_figures.py</code>).</p>''',
            short='run_71 result')


def s_time_depth(D, d):
    vel = {E: d.Gb(*d.comp, float(E))['v'] for E in (243, 150, 92)}
    P1 = sd.Plot(810, 640, x=(0, 3.1), y=(0.5, 1.12), title='against drift TIME: the three fields coincide',
                 xlabel='drift time since onset [µs]', ylabel='X ±2 / 1.08–1.26 µs level')
    P2 = sd.Plot(810, 640, x=(0, 30), y=(0.5, 1.12), title='against drift DEPTH: they separate',
                 xlabel='drift depth z = v·t [mm]')
    for P in (P1, P2):
        P.yticks([(v, f'{v:g}') for v in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)])
    P1.xticks([(v, f'{v:g}') for v in (0, 0.5, 1, 1.5, 2, 2.5, 3)])
    P2.xticks([(v, f'{v:g}') for v in (0, 5, 10, 15, 20, 25, 30)])
    r243 = d.rates[243]
    lam = vel[243] / r243
    for lab, E in PLAT:
        y, e = d.curve(lab, 'x', 2)
        t0 = d.F[lab]['t0']
        m = (d.t >= t0 + 300) & (d.t <= (t0 + d.F[lab]['T'] - 250 if E == 243 else 3840))
        ts = (d.t[m] - t0) / 1e3
        tips = [f'{E} V/cm\nt − t₀ = {a * 1e3:.0f} ns → z = {vel[E] * a:.1f} mm\n{b:.3f} ± {c:.3f}'
                for a, b, c in zip(ts, y[m], e[m])]
        P1.line(list(ts), list(y[m]), FC[E], 4, r=5, tips=tips, tip=f'{E} V/cm')
        P2.line(list(vel[E] * ts), list(y[m]), FC[E], 4, r=5, tips=tips, tip=f'{E} V/cm')
    for lab, E in PLAT[1:]:
        tt = np.linspace(0.3, 3.1, 60)
        y0 = np.exp(-(vel[E] * tt * 1e3) / lam) / np.exp(-(vel[E] * 480) / lam)
        P1.line(list(tt), list(y0), FC[E], 2.5, '4 6', markers=False,
                tip=f'If the loss were per unit depth (λ = {lam / 1e3:.0f} mm from 243 V/cm), {E} V/cm would fall like this in time.')
    P1.text(1.6, 1.06, 'dotted: a per-depth loss, predicted', 20, MUT)
    leg = sd.legend([('243 V/cm', FC[243]), ('150 V/cm', FC[150]), ('92 V/cm', FC[92])], size=22)
    span = vel[243] / vel[92]
    body = sd.title('The loss is per unit time, not per unit depth',
                    f'run_71 X ±2. Depth uses the fitted gas\'s Magboltz drift velocity; the depth reached differs ×{span:.1f} between fields.')
    body += leg + sd.row(P1.svg('time'), P2.svg('depth'), gap=44)
    D.slide('time-depth', body, f'''
<p>This is the slide-§§fingerprints§§ test between fingerprints 1 and 2. In time the three curves lie on top of each other (r = {d.rates[243] * 1e4:.2f},
{d.rates[150] * 1e4:.2f}, {d.rates[92] * 1e4:.2f}×10⁻⁴/ns); in depth they separate, the slow field losing as much over ~9 mm as the fast
one over ~30 mm. A loss with a fixed attenuation length per mm (dotted) is excluded at more than 9σ at 92 V/cm.</p>
<p>Drift velocities: {vel[243]:.1f} / {vel[150]:.2f} / {vel[92]:.2f} µm/ns from Magboltz for the fitted composition; the 243 V/cm drift end
(T = {d.F["raw700"]["T"]:.0f} ns, so {30e3 / d.F["raw700"]["T"]:.1f} µm/ns over 30 mm) and the run_63 ladder slopes check them (slide §§beam-comp§§).</p>
<p>This also rules out a field gradient or dished cathode as the source: those act per unit depth.</p>''', short='Time, not depth')


def s_undershoot(D, d):
    f = d.F['raw700']
    y, e = d.curve('raw700', 'x', 2)
    yy, ey = d.curve('raw700', 'y', 8)
    t2 = np.linspace(0, 3840, 769)
    att = box_model(t2, f['amp'], f['t0'], f['T'], f['r'])
    flat = box_model(t2, f['amp'], f['t0'], f['T'], 0.0)
    tau = 1.0 / f['r']; hp = flat.copy(); acc = 0.0; dtt = t2[1] - t2[0]
    for i in range(1, len(t2)):
        acc = acc * np.exp(-dtt / tau) + flat[i - 1] * dtt / tau
        hp[i] = flat[i] - acc
    us = d.F['undershoot']
    P = sd.Plot(1060, 640, x=(0.4, 3.84), y=(-0.38, 1.12), xlabel='time after trigger [µs]',
                ylabel='signal / 1.08–1.26 µs level')
    P.xticks([(v, f'{v:g}') for v in (0.5, 1, 1.5, 2, 2.5, 3, 3.5)]).yticks(
        [(v, f'{v:g}') for v in (-0.3, 0, 0.25, 0.5, 0.75, 1)])
    P.hline(0, MUT, None, 1.2)
    P.raw(f'<rect x="{P.X(3.0):.1f}" y="{P.y0}" width="{P.X(3.84) - P.X(3.0):.1f}" height="{P.ph}" fill="{GOLD}" '
          f'fill-opacity="0.10"{sd.tipattr("After the drift ends (≥ 3.0 µs): where a high-pass must show its undershoot.")}/>', back=True)
    P.line(list(t2 / 1e3), list(hp), GOLD, 4, '12 8', markers=False,
           tip=f'Readout high-pass with τ = 1/r = {tau / 1e3:.1f} µs: same plateau sag, then {us["highpass"]:+.2f}.')
    P.line(list(t2 / 1e3), list(att), INK, 2.5, markers=False, tip='Attachment fit.')
    k = d.t >= 400
    t, y, e, yy = d.t[k] / 1e3, y[k], e[k], yy[k]
    P.band(list(t), list(y - e), list(y + e), BLUE, 0.2)
    tips = [f'X ±2, t = {a * 1e3:.0f} ns: {b:+.3f} ± {c:.3f}' for a, b, c in zip(t, y, e)]
    P.line(list(t), list(y), BLUE, 4, r=4, tips=tips, tip='data X ±2')
    P.line(list(t), list(yy), BLUE, 3, '9 6', markers=False, tip='data Y ±8')
    P.text(3.05, -0.22, f'high-pass: {us["highpass"]:+.2f}', 22, GOLD, weight=600)
    P.text(3.12, 0.2, f'data: X {us["data_x"]:+.3f}', 22, BLUE, weight=600)
    P.text(3.12, 0.1, f'Y {us["data_y"]:+.3f}', 22, BLUE, weight=600)
    side = sd.col(
        sd.p('At 243 V/cm the drift ends inside the window (2.7 µs), so the signal\'s return to baseline is '
             'visible.', 27),
        sd.p(f'A readout high-pass that sagged the plateau this much would have to swing to '
             f'<b>{us["highpass"]:+.2f}</b> afterwards. The data settle at <b>{us["data_x"]:+.3f}</b> (X) and '
             f'<b>{us["data_y"]:+.3f}</b> (Y).', 27),
        sd.callout('The missing charge never reached the electronics: it is not the readout.', BLUE, 27),
        gap=28, w=540)
    body = sd.title('When the drift ends, nothing swings negative: not the readout',
                    'run_71 RAW, 243 V/cm. Fingerprint 3 of slide §§fingerprints§§, tested.')
    body += sd.row(P.svg('undershoot'), side, gap=60)
    D.slide('undershoot', body, f'''
<p>The high-pass curve is the no-loss signal passed through a first-order high-pass whose time constant (1/r = {tau / 1e3:.1f} µs) gives the
same sag on the plateau. Any linear readout effect that removes low-frequency charge must give it back with opposite sign after the
input stops; the measured template's own small negative lobe is already inside the attachment fit, which reaches −0.03 and matches.</p>
<p>Late level (≥ 3.0 µs): data X {us["data_x"]:+.4f}, Y {us["data_y"]:+.4f}; high-pass {us["highpass"]:+.3f}.</p>''',
            foot='Source: fits.json (undershoot), headon_masked_k12.json.', short='No undershoot')


def s_discriminators(D, d):
    # gain
    P1 = sd.Plot(540, 560, x=(0.4, 2.2), y=(0.4, 1.0), title='local gain', xlabel='gain / mean (position bins)',
                 ylabel='R (X ±2)', margin=(10, 18, 92, 96))
    P1.xticks([(0.5, '0.5'), (1, '1'), (1.5, '1.5'), (2, '2')]).yticks([(v, f'{v:g}') for v in (0.4, 0.6, 0.8, 1.0)])
    gg = [0.5, 2.1]
    P1.line(gg, [1 - 0.23 * g for g in gg], GOLD, 3, '10 7', markers=False,
            tip='Charging that made the whole loss: R = 1 − 0.23·g')
    P1.text(1.35, 0.5, 'charging would', 20, GOLD)
    P1.text(1.35, 0.465, 'follow this', 20, GOLD)
    for lab, E in PLAT:
        for key, mk in (('by_x', 'circle'), ('by_y', 'open')):
            rws = d.Gv[lab][key]['rows']
            g = np.array([r_['gain'] for r_ in rws]); g = g / g.mean()
            sl = d.Gv[lab][key]['slope_Rx_per_relgain']
            tips = [f'{E} V/cm, bins {key[3:]}: gain/mean {gi:.2f}\nR = {r_["Rx"][0]:.3f} ± {r_["Rx"][1]:.3f} (n = {r_["n"]})\n'
                    f'slope {sl[0]:+.3f} ± {sl[1]:.3f}' for gi, r_ in zip(g, rws)]
            P1.points(list(g), [r_['Rx'][0] for r_ in rws], FC[E], r=7, tips=tips, marker=mk)
    # rate / spill
    P2 = sd.Plot(540, 560, x=(-0.6, 6.6), y=(0.6, 0.95), title='beam rate · spill phase', ylabel='R (X ±2)',
                 margin=(10, 18, 92, 96))
    P2.yticks([(v, f'{v:g}') for v in (0.6, 0.7, 0.8, 0.9)])
    P2.xticks([(0, 'low'), (1, 'mid'), (2, 'high'), (4, 'early'), (5, 'mid'), (6, 'late')])
    for k, key in enumerate(('rate', 'spill')):
        for lab, E in PLAT:
            rws = d.S[lab][key]
            xs = [j + 4 * k + {243: -0.2, 150: 0, 92: 0.2}[E] for j in range(3)]
            tips = [f'{E} V/cm, {key} tercile {j + 1}: R = {r_["x"][0]:.3f} ± {r_["x"][1]:.3f} (n = {r_["n"]})'
                    for j, r_ in enumerate(rws)]
            P2.points(xs, [r_['x'][0] for r_ in rws], FC[E], r=7, tips=tips)
    P2.text(0, 0.62, 'rate', 21, MUT); P2.text(4, 0.62, 'spill', 21, MUT)
    # charge terciles vs toy
    P3 = sd.Plot(540, 560, x=(-0.4, 2.4), y=(0.6, 1.02), title='event charge vs toy', ylabel='R (X ±2)',
                 margin=(10, 18, 92, 96))
    P3.yticks([(v, f'{v:g}') for v in (0.6, 0.7, 0.8, 0.9, 1.0)]).xticks([(0, 'faint'), (1, 'mid'), (2, 'bright')])
    rws = d.S['raw700']['charge']
    P3.line([0, 1, 2], d.T['0.0']['terciles'], GREY, 3, '10 7', r=5,
            tips=[f'toy, no attachment: {v:.3f}' for v in d.T['0.0']['terciles']], tip='toy, r = 0')
    P3.line([0, 1, 2], d.T['0.00019']['terciles'], INK, 3, r=5,
            tips=[f'toy, r = 1.9×10⁻⁴/ns: {v:.3f}' for v in d.T['0.00019']['terciles']], tip='toy, r = 1.9e-4/ns')
    P3.points([0, 1, 2], [r_['x'][0] for r_ in rws], BLUE, r=8,
              tips=[f'data 243 V/cm, {nm}: {r_["x"][0]:.3f} ± {r_["x"][1]:.3f} (n = {r_["n"]})'
                    for nm, r_ in zip(('faint', 'mid', 'bright'), rws)])
    P3.text(0.1, 0.96, 'toy, no loss', 20, GREY)
    P3.text(0.9, 0.7, 'toy, r = 1.9', 20, INK)
    leg = sd.legend([('243 V/cm', FC[243]), ('150 V/cm', FC[150]), ('92 V/cm', FC[92]),
                     ('filled: x bins · open: y bins', MUT, 'dot')], size=22)
    body = sd.title('It does not follow gain, beam rate or spill phase: not charging, not space charge',
                    'run_71 RAW, X ±2, 20 000 events per field split into bins. Fingerprint 4 of slide §§fingerprints§§, tested.')
    body += leg + sd.row(P1.svg('gain'), P2.svg('rate'), P3.svg('charge'), gap=22)
    sl = [d.Gv[l][k]['slope_Rx_per_relgain'] for l, _ in PLAT for k in ('by_x', 'by_y')]
    D.slide('discriminators', body, f'''
<p><b>Gain.</b> det4's amplification has fixed stripes (×1.4–2.8 between position bins); a gain-side loss would scale with them.
Charging that produced the whole loss predicts dR/d(gain/mean) ≈ −0.23; the six fits give
{", ".join(f"{s[0]:+.3f} ± {s[1]:.3f}" for s in sl)}. No bin set shows a negative slope; at 2σ charging is at most ~20 % of the effect.
Independently, the plateau current falls ×2.0 from 243 to 92 V/cm while the loss per ns does not change.</p>
<p><b>Rate and spill.</b> Terciles of instantaneous beam rate and of position within the spill: flat within errors. Space charge in the drift
or the amplification gap would grow with both.</p>
<p><b>Event charge.</b> Attachment itself makes bright events read lower (their big clusters were preferentially early, so the reference
window is inflated). The toy (Poisson clusters with a 1/n² size tail, per-electron survival e^(−rt), gain scatter, the measured template,
random sampling phase, split exactly like the data) reproduces the pattern at r = 1.9×10⁻⁴/ns. An apparent faint-tercile excess was
a selection on missing samples (FINDINGS §19); with the corrected classification the terciles agree with the toy within 2σ.</p>''',
            short='Discriminators')


def s_all_datasets(D, d):
    rows = d.F['dataset_rows']
    names = [r_['name'].replace('$_4$', '₄').replace('$_2$', '₂') for r_ in rows]
    n = len(rows)
    P = sd.Plot(1080, 650, x=(0.3, 1.05), y=(-0.7, n - 0.3), xlabel=f'R = late / reference level (1 = no loss)',
                margin=(10, 30, 92, 20))
    P.xticks([(v, f'{v:g}') for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])
    P.vline(1.0, MUT, '4 5', 1.5)
    for i, (r_, nm) in enumerate(zip(rows, names)):
        yv = n - 1 - i
        P.raw(sd.line(P.x0, P.Y(yv), P.x0 + P.pw, P.Y(yv), RULE, 1), back=True)
        ex = r_['Rx_err'] if r_['Rx_err'] == r_['Rx_err'] else 0
        ey = r_['Ry_err'] if r_['Ry_err'] == r_['Ry_err'] else 0
        c = INK if r_['kind'] == 'RAW' else (MUT if r_['kind'] == 'ZSemu' else BLUE)
        if 'CO' in nm:
            c = ORANGE
        if ex:
            P.raw(sd.line(P.X(r_['Rx'] - ex), P.Y(yv) - 9, P.X(r_['Rx'] + ex), P.Y(yv) - 9, c, 3))
        P.points([r_['Rx']], [yv + 0.0], c, r=9,
                 tips=[f'{nm}\nX: R = {r_["Rx"]:.3f}' + (f' ± {ex:.3f}' if ex else '') +
                       ('\n(ZS bars: range between all events and the middle 60 % by charge)' if r_['kind'] == 'ZS' else '')])
        P.raw(f'<circle cx="{P.X(r_["Rx"]):.1f}" cy="{P.Y(yv) - 9:.1f}" r="0"/>')
        if r_['Ry'] == r_['Ry']:
            if ey:
                P.raw(sd.line(P.X(r_['Ry'] - ey), P.Y(yv) + 9, P.X(r_['Ry'] + ey), P.Y(yv) + 9, c, 2))
            P.points([r_['Ry']], [yv - 0.18], c, r=8, marker='open',
                     tips=[f'{nm}\nY: R = {r_["Ry"]:.3f}' + (f' ± {ey:.3f}' if ey else '')])
    labels = sd.col(*[sd.p(nm, 22, INK, extra='height:' + f'{P.ph / n:.1f}px;display:flex;align-items:center;justify-content:flex-end;text-align:right')
                      for nm in names], gap=0, w=560)
    lab_wrap = f'<div style="padding-top:{P.y0 + P.ph / n * 0.0:.0f}px">{labels}</div>'
    leg = sd.legend([('X (filled)', INK, 'dot'), ('Y (open)', MUT, 'dot'), ('RAW', INK, 'box'),
                     ('zero-suppressed, CF₄', BLUE, 'box'), ('zero-suppressed, CO₂ gas', ORANGE, 'box')], size=22)
    body = sd.title('Every beam dataset loses late charge, in both views',
                    'All det4 beam data with waveforms: two gases, flat and 25.6° mounts, RAW and zero-suppressed, 1–3 Aug.')
    body += leg + sd.row(lab_wrap, P.svg('all datasets'), gap=10)
    D.slide('all-datasets', body, '''
<p>R is the same model-free ratio on every dataset. RAW rows carry bootstrap errors. Zero-suppressed (ZS) rows carry a bar spanning the
two event selections (all events, middle 60 % by total charge): ZS censors small samples, so R depends on the selection at the ±0.05
level. ZS Y reads low because Y's wider, smaller signals are censored more (in RAW, Y = X).</p>
<p>The rotated (25.6°) run_63 blocks are a second geometry: X is head-on, Y is the drift ladder summed over all its strips, which is the
same arriving current. The CO₂ period (run_56, Ar/CO₂/iso, 1 Aug) loses more, R ≈ 0.5–0.6.</p>
<p>"run_71 → emulated ZS 4σ" is the RAW stack passed through a software ZS: it lands on run_63's real ZS value five hours earlier, which
is how the ZS distortion is calibrated (<code>zs_emulate.py</code>).</p>''',
            foot='Source: fits.json dataset_rows (make_figures.py F6), zs_timestack.json, zs_emulate.json.', short='All datasets')


def s_ladder(D, d):
    tz = 1 / math.tan(math.radians(25.64))
    P1 = sd.Plot(810, 560, x=(0, 30), y=(0.35, 1.3), title='beam: run_63 Y ladder at 25.6°',
                 xlabel='drift depth [mm]', ylabel='charge per strip / its mesh-end value')
    P2 = sd.Plot(810, 560, x=(0, 30), y=(0.35, 1.3), title='bench: inclined cosmics, Ar/iso',
                 xlabel='drift depth [mm]')
    for P in (P1, P2):
        P.xticks([(v, f'{v:g}') for v in (0, 5, 10, 15, 20, 25, 30)]).yticks([(v, f'{v:g}') for v in (0.4, 0.6, 0.8, 1.0, 1.2)])
        P.hline(1.0, MUT, '4 5', 1.2)
    cols = {'rot_d425': (FC[243], '142 V/cm'), 'rot_d325': (FC[150], '108 V/cm'), 'rot_d225': (FC[92], '75 V/cm')}
    for arm, (c, nm) in cols.items():
        rows = d.L[arm]['rows']; f = d.L[arm]['fit']
        u = np.array([r_['u'] for r_ in rows]); ma = np.array([r_['A_q70'] for r_ in rows])
        sel = (u >= f['u_range'][0] - 1e-6) & (u <= f['u_range'][1] + 1e-6)
        z = (f['u_range'][1] - u[sel]) * tz + 0.8 * tz
        yv = ma[sel] / ma[sel][-1]
        tips = [f'{nm}, z = {zi:.1f} mm: {yi:.2f}\nfit {f["per_mm"][0]:.3f} ± {f["per_mm"][1]:.3f} per mm '
                f'({d.L[arm]["n_events"]} events)' for zi, yi in zip(z, yv)]
        P1.line(list(z), list(yv), c, 4, r=5, tips=tips, tip=nm)
    zb = np.array(d.BL['z'])
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        for v, dash in (('x', None), ('y', '9 6')):
            if det == 'det4' and v == 'x':
                continue
            if v in d.BL.get(det, {}):
                q = np.array(d.BL[det][v]['Q']); q = q / np.nanmean(q[2:4])
                ds = d.BL[det][v]['deep_over_shallow_Q']
                P2.line(list(zb), list(q), BENCH_C, 2.5, dash, r=3.5,
                        tips=[f'{det} {v.upper()}, z = {zz:.1f} mm: {qq:.2f}' for zz, qq in zip(zb, q)],
                        tip=f'{det} {v.upper()} (n = {d.BL[det][v]["n"]})')
    P2.raw(f'<rect x="{P2.X(24):.1f}" y="{P2.y0}" width="{P2.X(30) - P2.X(24):.1f}" height="{P2.ph}" fill="{GREY}" '
           f'fill-opacity="0.12"{sd.tipattr("Gap end and M3 track smearing: the fall here is geometry, not loss.")}/>', back=True)
    P2.text(24.4, 0.42, 'gap end', 20, MUT)
    leg = sd.legend([('142 V/cm', FC[243]), ('108 V/cm', FC[150]), ('75 V/cm', FC[92]),
                     ('bench X (det2/3/6/7)', BENCH_C), ('bench Y (5 chambers)', BENCH_C, 'dash')], size=22)
    body = sd.title('A template-free check: on the beam, charge per strip falls with depth; on the bench it does not',
                    'Inclined tracks: each strip collects one depth slice. No electronics model, no stacking in time.')
    body += leg + sd.row(P1.svg('beam ladder'), P2.svg('bench ladder'), gap=44)
    D.slide('ladder', body, '''
<p>An inclined track spreads its depths across strips (the drift ladder), so the charge each strip collects, plotted against that strip's
depth, is a depth profile that needs no time model at all. Beam (left): run_63 with the chamber rotated 25.64°, the Y view is the ladder.
Per-strip peak amplitude, 70th percentile over <b>all</b> events with an unfired strip counted as zero, which makes it immune to ZS
censoring. It falls monotonically at every field.</p>
<p>The per-strip estimator reads ~30 % steeper in rate than the time stacks (3.2–3.3 vs ~2.4×10⁻⁴/ns on the same events): deep strips sit
closer to the ZS threshold, and transverse diffusion lowers per-strip peaks with depth. It is the depth-resolved, template-free
confirmation; the rate comes from the time stacks.</p>
<p>Bench (right): the same per-strip charge on inclined cosmics, all five chambers, normalised at 6–12 mm. Flat to ±5 % from 6 to 20 mm in
both views; det4 X is omitted (its amplification stripes run across X, so per-strip charge there is the stripe map). The rise from the mesh
and the fall past ~22 mm are geometry (first strips partial, gap end and reference-track smearing). At its modest statistics this
excludes a beam-like loss; the drift scan (slide §§bench-scan§§) is the sharp bench test.</p>''', short='Ladder')


def s_gasmodel(D, d):
    steps = [
        dict(label='Composition', sub='base gas + water % + air % (N₂/O₂/Ar replacing Ar)', color=INK,
             tip='Beam: Ar/CF₄/iso 88/10/2; CO₂ period Ar/CO₂/iso 95/3/2; bench Ar/iso 95/5. Water and air replace argon.'),
        dict(label='Magboltz', sub='v, η, D_L, D_T at each field', color=BLUE, tip=G['magboltz']),
        dict(label='Arriving current', sub='uniform ionisation, drift, diffusion, survival e^(−ηz)', color=BLUE,
             tip='predict.current_field: optional linear drift-field profile k and gap spread (geometry), shared per dataset.'),
        dict(label='Electronics', sub='effective shaper + trigger jitter', color=GREY, tip=G['shaper']),
        dict(label='Compare to every stack', sub='free per stack: amplitude, t₀ only', color=RED,
             tip='Shared across all fields and views of a dataset: composition, geometry, shaper. No free loss rate.'),
    ]
    Es = np.linspace(60, 260, 41)
    P1 = sd.Plot(810, 430, x=(60, 260), y=(0, 18), title='water sets the drift velocity',
                 xlabel='drift field [V/cm]', ylabel='v [µm/ns]', margin=(10, 24, 86, 96))
    P1.xticks([(v, f'{v}') for v in (100, 150, 200, 250)]).yticks([(v, f'{v}') for v in (0, 5, 10, 15)])
    for w, c in ((1.4, '#9bbbe0'), (1.55, BLUE), (1.7, '#163f73')):
        P1.line(list(Es), [d.Gb(w, 0.0, float(E))['v'] for E in Es], c, 3.5, markers=False,
                tip=f'{w:.2f} % water, no air')
        P1.line(list(Es), [d.Gb(w, 0.1, float(E))['v'] for E in Es], c, 2, '4 6', markers=False,
                tip=f'{w:.2f} % water + 0.10 % air: air barely moves v')
    lv = {float(k): v for k, v in d.B['ladder_v'].items()}
    T243 = d.F['raw700']['T']
    xs = list(lv) + [243.0]; ys = [v[0] for v in lv.values()] + [30e3 / T243]
    P1.points(xs, ys, RED, r=8, tips=[f'measured {E:.0f} V/cm: {v:.2f} µm/ns' + (' (run_71 drift end, 30 mm)' if E == 243 else ' (run_63 ladder)')
                                        for E, v in zip(xs, ys)])
    P1.text(70, 15.5, 'solid: 1.40 / 1.55 / 1.70 % water · dotted: +0.1 % air', 19, MUT)
    P2 = sd.Plot(810, 430, x=(60, 260), y=(0, 3.5), title='oxygen sets the loss rate',
                 xlabel='drift field [V/cm]', ylabel='η·v [10⁻⁴/ns]', margin=(10, 24, 86, 96))
    P2.xticks([(v, f'{v}') for v in (100, 150, 200, 250)]).yticks([(v, f'{v:g}') for v in (0, 1, 2, 3)])
    for a_, c in ((0.0, GREY), (0.04, '#e9a0a9'), (0.078, RED), (0.13, '#7d1f2b')):
        P2.line(list(Es), [d.Gb(1.5, a_, float(E))['etav'] * 1e4 for E in Es], c, 3.5, markers=False,
                tip=f'1.5 % water + {a_:.3f} % air = {a_ * O2_PER_AIR:.0f} ppm O₂')
        P2.text(262, d.Gb(1.5, a_, 258.0)['etav'] * 1e4, f'{a_ * O2_PER_AIR:.0f} ppm', 19, c)
    Em = [243, 150, 92]
    P2.points(Em, [d.rates[E] * 1e4 for E in Em], INK, r=8,
              tips=[f'run_71 measured r at {E} V/cm: {d.rates[E] * 1e4:.2f} ± {d.rerr[E] * 1e4:.2f}×10⁻⁴/ns' for E in Em])
    body = sd.title('From a gas composition to a waveform: water moves v, oxygen moves the loss',
                    'One physics model for every dataset. The loss rate is never fitted: it follows from the composition.')
    body += sd.flow(steps, size=22)
    body += sd.row(P1.svg('v vs water'), P2.svg('etav vs air'), gap=44)
    D.slide('gas-model', body, f'''
<p>Each composition is run through Magboltz (via Garfield++) on a high-statistics grid (3×10⁸ collisions per point; condor 4410759 and
4410787; <code>results/air_hs/</code>), and <code>gasmodel.GasGrid</code> interpolates in (water, air, E). Even at 3×10⁸ collisions the
attachment estimate scatters ±10–15 % between neighbouring fields, so η, D_L and D_T are smoothed per mixture (quadratic in log E);
v is used as computed.</p>
<p><b>Why two observables pin two unknowns.</b> Water slows the drift strongly and attaches nothing; ~0.1 % air changes v by well under
1 % but sets the attachment through its O₂. So v(E) fixes the water and the loss rate fixes the O₂, nearly independently. In this gas
η·v is flat from 75 to 150 V/cm and rises at 243 V/cm; the data's flatness at 243 V/cm is accommodated by the shared shaper and
geometry within χ² (slide §§beam-comp§§).</p>
<p>{G["o2"]}</p>''', short='Gas model')


def s_beam_comp(D, d):
    bm = d.C['beam']
    panels = []
    for i, (lab, E) in enumerate(PLAT):
        ps = d.B['per_stack']
        P = sd.Plot(548, 560, x=(0.4, 3.84), y=(-0.1, 1.15), title=f'{E} V/cm',
                    xlabel='time after trigger [µs]', ylabel='signal / 1.08–1.26 µs level' if i == 0 else '',
                    margin=(10, 18, 90, 96 if i == 0 else 40))
        P.xticks([(v, f'{v:g}') for v in (1, 2, 3)]).yticks(
            [(v, f'{v:g}' if i == 0 else '') for v in (0, 0.25, 0.5, 0.75, 1)])
        k = d.t >= 400
        t = d.t[k] / 1e3
        sx, sy = ps[f'{float(E)}_x'], ps[f'{float(E)}_y']
        P.line(list(t), list(np.array(sx['curve_noair'])[k]), NOAIR_C, 3, '10 7', markers=False,
               tip=f'Same water, no air. χ² X {sx["chi2_noair"]:.0f}, Y {sy["chi2_noair"]:.0f}')
        yx, _ = d.curve(lab, 'x', 2)
        yy, _ = d.curve(lab, 'y', 8)
        yx, yy = yx[k], yy[k]
        P.line(list(t), list(yx), FC[E], 4, r=3, tip='data X ±2',
               tips=[f'X, t = {a * 1e3:.0f} ns: {b:.3f}' for a, b in zip(t, yx)])
        P.line(list(t), list(yy), FC[E], 3, '9 6', markers=False, tip='data Y ±8')
        P.line(list(t), list(np.array(sx['curve'])[k]), INK, 2.5, markers=False,
               tip=f'Model, {bm["water"]:.2f} % water + {bm["air"]:.3f} % air. χ² X {sx["chi2"]:.0f}, Y {sy["chi2"]:.0f} (55 points each)')
        P.text(1.05, 0.3, f'χ² X {sx["chi2"]:.0f} · Y {sy["chi2"]:.0f}', 21, INK, weight=600)
        P.text(1.05, 0.18, f'no air: {sx["chi2_noair"]:.0f} · {sy["chi2_noair"]:.0f}', 20, MUT)
        panels.append(P.svg(f'beam comp {E}'))
    leg = sd.legend([('data X ±2', INK), ('data Y ±8', INK, 'dash'),
                     (f'model: {bm["water"]:.2f} % H₂O + {bm["air"]:.3f} % air', INK),
                     ('same water, no air', GREY, 'dash')], size=22)
    body = sd.title(f'The beam: one composition fits three fields and both views — {bm["o2_ppm"]:.0f} ppm O₂',
                    'run_71 RAW. Shared: composition, geometry, shaper. Free per stack: amplitude and t₀. 55 points per stack.')
    body += leg + sd.row(*panels, gap=10)
    tot = sum(bm['chi2'].values()); tot0 = sum(bm['chi2_noair'].values())
    D.slide('beam-comp', body, f'''
<p>The composition is fitted jointly to the six stacks and to the drift velocity at four fields (run_63 ladder slopes at 142/108/75 V/cm
and the run_71 drift end at 243 V/cm; slide §§gas-model§§, left). Best: <b>{bm["water"]:.2f} % water, {bm["air"]:.3f} % air = {bm["o2_ppm"]:.0f} ppm O₂</b>.
Total χ² {tot:.0f} for {55 * len(bm["chi2"])} points; the same water with no air gives {tot0:.0f}.</p>
<p>The 243 V/cm X stack carries most of the χ² (it has the sharpest feature, the drift end, and the smallest errors); 92 V/cm fits at
χ² 76 and 73 per 55 only because Magboltz's η is smoothed in field (slide §§gas-model§§).</p>
<p>Source: <code>beam_comp_fit.py --source air_hs</code> → <code>results/beam_comp_fit_air_hs.json</code> (curves stored per stack).</p>''',
            short='Beam composition')


def s_consistency(D, d):
    same = f'<span style="color:{GREEN};font-weight:600">same</span>'
    diff = f'<span style="color:{RED};font-weight:600">differs</span>'
    rows = [
        ['Chamber design', 'MX17 bulk micromegas, 30 mm drift gap, resistive strips', same],
        ['Readout', 'DREAM, register 1 0x081F 0xD023, RdClk_Div 6', same],
        ['Stacking and model code', 'cloud_basics: NaN-aware stacks, predict.current_field, Magboltz grid', same],
        ['Gas mixture', 'beam Ar/CF₄/iso 88/10/2 · bench Ar/iso 95/5', diff],
        ['Gas line', 'H4 beam line (banco P2 setup) · bench line', diff],
        ['Event timing', 'beam trigger · scintillator trigger + bundle t₀', diff],
    ]
    tips = ['If the loss came from the micromegas (charging, field, gap), the bench chambers would show it too.',
            'A readout cause (shaping, baseline, high-pass) would act identically on both.',
            'An analysis artefact would appear on both: the same code builds both stacks.',
            'The base gas changes v and the ion tail, which the model carries. Neither base gas attaches.',
            'The only ingredient that can carry O₂ to one setup and not the other.',
            'Both place events without looking at the pulse.']
    tbl = sd.table(['', 'beam (det4, H4) · bench (June)', ''], rows, size=23, widths=[290, 760, 120],
                   align=['left', 'left', 'center'], tips=tips)
    tests = [
        dict(label='1 · The bench is clean', sub='at a sensitivity well below the beam\'s O₂', color=GREEN,
             tip='Otherwise a bench null says nothing. Tested with the det3 drift scan and the same-observable overlay.'),
        dict(label='2 · The same model fits it', sub='with the bench\'s own composition, every field', color=GREEN,
             tip='One gas model, no free loss rate, for both setups: six bench fields, four bench chambers.'),
        dict(label='3 · Both compositions make sense', sub='water and O₂ from plausible sources', color=GREEN,
             tip='The water must match the bench drift velocity measured independently, and the beam O₂ must not be bulk air.'),
    ]
    body = sd.title('The consistency test: if it is the beam gas, the bench must be clean',
                    'Everything that could fake the loss is shared by the two setups. Only the gas and its line differ.')
    body += sd.row(
        sd.col(tbl, gap=12, w=1180),
        sd.col(sd.callout('A detector, readout or analysis cause predicts the <b>same loss on the bench</b>. '
                          'A gas cause predicts <b>none</b> — provided the bench could have seen it.', BLUE, 26),
               sd.p('Hover a row for why it matters.', 22, MUT), gap=20, w=440),
        gap=44)
    body += sd.p('<b>What the gas explanation has to pass</b>', 26, INK)
    body += sd.flow(tests, size=24)
    D.slide('consistency', body, '''
<p>The beam measurement alone singles out attachment among the four mechanisms of slide §§fingerprints§§. The bench is the
cross-check that the explanation is not specific to the beam data's handling: the chambers, the electronics settings and the
analysis code are the same, so any cause in them must show on the bench. The two things that differ are the gas mixture and
the gas line.</p>
<p>Three conditions, each tested on the next slides: (1) the bench shows no loss with a sensitivity below half the beam's O₂
(slides §§bench-scan§§–§§same-observable§§); (2) the same physics model, with the bench's own fitted composition, describes
all bench data (§§bench-scan§§, §§bench-chambers§§); (3) the two compositions are physically sensible (§§map§§–§§source§§),
including the bench water matching the drift velocity measured independently from micro-TPC tracks.</p>''',
            short='Consistency test')


def _bench_scan_curves(d):
    g = np.array(d.Dd['grid']); F = d.DF
    par = np.array(F['shaper']); sj = F['jitter']
    out = []
    for V, E in zip(d.Dd['volts'], d.Dd['E_Vcm']):
        r_ = d.Dd[str(V)]['x']; y = np.array(r_['stack'])
        e = np.maximum(np.nan_to_num(np.array(r_['band']), nan=1.0), 0.004)
        c, n, mc, _ = DS.fit_field(g, y, e, DS.cur(d.Gn, (F['water'], F['air']), float(E), F['k'], F['gap_spread']), par, sj)
        ca, _, ma, _ = DS.fit_field(g, y, e, DS.cur(d.Gn, (F['water'], BENCH_AIR_CONTRAST), float(E), F['k'], F['gap_spread']), par, sj)
        out.append(dict(V=V, E=E, g=g, y=y, e=e, yy=np.array(d.Dd[str(V)]['y']['stack']), mc=np.array(mc), ma=np.array(ma),
                        c=c, ca=ca, n=n, nx=r_['n']))
    return out


def s_bench_scan(D, d, scan):
    panels = []
    for i, s in enumerate(scan):
        P = sd.Plot(548, 352 if i >= 3 else 298, x=(-0.4, 1.9), y=(-0.1, 1.15), title=f'{s["V"]} V drift = {s["E"]:.0f} V/cm',
                    margin=(6, 14, 84 if i >= 3 else 30, 74 if i % 3 == 0 else 30),
                    xlabel='time after drift start [µs]' if i >= 3 else '')
        P.xticks([(v, f'{v:g}' if i >= 3 else '') for v in (0, 0.5, 1, 1.5)]).yticks(
            [(v, f'{v:g}' if i % 3 == 0 else '') for v in (0, 0.5, 1)])
        g = s['g'] / 1e3
        P.line(list(g), list(s['ma']), RED, 3, '10 7', markers=False,
               tip=f'Same gas + {BENCH_AIR_CONTRAST} % air ({BENCH_AIR_CONTRAST * O2_PER_AIR:.0f} ppm O₂, about half the beam\'s): χ² {s["ca"]:.0f}')
        P.line(list(g), list(s['yy']), ORANGE, 2, markers=False, tip=f'data Y (n = {d.Dd[str(s["V"])]["y"]["n"]})')
        P.line(list(g), list(s['y']), BENCH_C, 0, r=3, tip='data X',
               tips=[f'X, t = {a * 1e3:.0f} ns: {b:.3f}' for a, b in zip(g, s['y'])])
        P.line(list(g), list(s['mc']), INK, 2.5, markers=False,
               tip=f'Model, {d.DF["water"]:.2f} % water, no air: χ² {s["c"]:.0f} / {s["n"]}')
        P.text(0.3, 0.25, f'χ² {s["c"]:.0f} vs {s["ca"]:.0f}', 21, INK, weight=600,
               tip=f'χ² of the no-air model ({s["c"]:.0f}) against the same gas + {BENCH_AIR_CONTRAST} % air ({s["ca"]:.0f}), {s["n"]} points')
        panels.append(P.svg(f'scan {s["V"]}'))
    grid = f'<div style="display:grid;grid-template-columns:repeat(3,548px);gap:0 10px">{"".join(panels)}</div>'
    leg = sd.legend([('data X', BENCH_C, 'dot'), ('data Y', ORANGE), (f'model: {d.DF["water"]:.2f} % water, no air', INK),
                     (f'same + {BENCH_AIR_CONTRAST} % air (half the beam\'s O₂)', RED, 'dash')], size=22)
    body = sd.title('Bench: one model, six drift fields, no oxygen',
                    'det3 drift scan, 27 Jun, one gas fill. χ²: the no-air model vs the same gas with half the beam\'s air.')
    body += leg + grid
    tot = sum(s['c'] for s in scan); totn = sum(s['n'] for s in scan); tota = sum(s['ca'] for s in scan)
    D.slide('bench-scan', body, f'''
<p><b>Why a drift scan.</b> At its operating field the bench drift lasts only ~800 ns, and a single-field plateau cannot separate a small
loss from the electronics undershoot or a field gradient (all tilt it). The det3 scan on 27 Jun (drift 100–1100 V, resist 490 V) gives
six fields from one gas fill. At 35 and 104 V/cm the drift outlasts the window, so the plateau is a 1.2 µs window on pure drift — the
same observable as the beam's 92 and 150 V/cm.</p>
<p><b>Result.</b> {d.DF["water"]:.2f} % water and <b>no air</b> fit all six fields together: χ² {tot:.0f} / {totn}. With {BENCH_AIR_CONTRAST} % air
added ({BENCH_AIR_CONTRAST * O2_PER_AIR:.0f} ppm O₂, half the beam's), χ² {tota:.0f}: the low fields reject it hardest (×2.5–13 per field).
Statistical 90 % limit ≲ 2 ppm O₂; ≲ 10 ppm allowing for the model.</p>
<p>The same fit fixes the geometry: field gradient k = {d.DF["k"]:+.1f} (the edge of the allowed range) and gap spread {d.DF["gap_spread"]:.1f} mm.
The k ≈ −0.3 that single-field bench fits once wanted is excluded: a gradient acts through dv/dE, which is largest at low field.</p>
<p>Events: straight from <code>decoded_root</code> with a masked common mode, one cluster ≤ 12 mm, strips within ±3 mm summed, placed by the
det3 bundle's <code>t0_abs[ftst]</code>; 1700–2500 events per field and view. X and Y agree. Source: <code>bench_driftscan.py</code>,
<code>driftscan_fit.py --source air_hs --emin 30</code>.</p>''', short='Bench drift scan', pad='96px 128px 80px')


def s_same_observable(D, d, scan):
    """Beam and bench on one axis: the same observable, two gases; and the chi2 sensitivity."""
    N0, N1 = 0.8, 1.0                   # normalisation window after onset: past the response's overshoot
    P = sd.Plot(1000, 620, x=(0.6, 2.7), y=(0.62, 1.12), xlabel='drift time since the current starts [µs]',
                ylabel=f'signal / its level {N0:g}–{N1:g} µs after onset')
    P.xticks([(v, f'{v:g}') for v in (1, 1.5, 2, 2.5)]).yticks([(v, f'{v:g}') for v in (0.7, 0.8, 0.9, 1.0, 1.1)])
    P.raw(f'<rect x="{P.X(N0):.1f}" y="{P.y0}" width="{P.X(N1) - P.X(N0):.1f}" height="{P.ph}" fill="{BLUE}" '
          f'fill-opacity="0.10"{sd.tipattr("Normalisation window: each curve divided by its own mean here. Late enough that the electronics overshoot after the rise has settled.")}/>', back=True)
    P.hline(1.0, MUT, '4 5', 1.2)
    END0, END1 = 1.5, 1.65             # where both setups still have data
    ends = {}
    for lab, E in (('raw275', 92), ('raw450', 150)):
        y, _ = d.curve(lab, 'x', 2)
        t0 = d.F[lab]['t0']
        ts = (d.t - t0) / 1e3
        ref = win(ts, y, N0, N1)
        ends[f'beam {E}'] = win(ts, y, END0, END1) / ref
        m = (ts >= 0.7) & (ts <= 2.65)
        P.line(list(ts[m]), list(y[m] / ref), BEAM_C if E == 92 else '#e07a86', 4, r=5,
               tips=[f'beam run_71, {E} V/cm, X ±2\nt − t₀ = {a * 1e3:.0f} ns: {b:.3f}' for a, b in zip(ts[m], y[m] / ref)],
               tip=f'beam {E} V/cm')
    for s in scan[:2]:
        g = s['g'] / 1e3
        ref = win(g, s['y'], N0, N1)
        ends[f'bench {s["E"]:.0f}'] = win(g, s['y'], END0, END1) / ref
        ends[f'model {s["E"]:.0f}'] = win(g, s['mc'], END0, END1) / win(g, s['mc'], N0, N1)
        ends[f'air {s["E"]:.0f}'] = win(g, s['ma'], END0, END1) / win(g, s['ma'], N0, N1)
        m = (g >= 0.7) & (g <= 1.65)
        if s['E'] < 50:
            refa = win(g, s['ma'], N0, N1)
            P.line(list(g[m]), list(s['ma'][m] / refa), BENCH_C, 2.5, '10 7', markers=False,
                   tip=f'What the bench at {s["E"]:.0f} V/cm would show with {BENCH_AIR_CONTRAST} % air '
                       f'({BENCH_AIR_CONTRAST * O2_PER_AIR:.0f} ppm O₂, half the beam\'s): the model, normalised the same way.')
        P.line(list(g[m]), list(s['y'][m] / ref), BENCH_C if s['E'] < 50 else '#6cc79a', 4, r=5,
               tips=[f'bench det3, {s["E"]:.0f} V/cm, X\nt = {a * 1e3:.0f} ns: {b:.3f}' for a, b in zip(g[m], s['y'][m] / ref)],
               tip=f'bench {s["E"]:.0f} V/cm')
    r = np.mean(list(d.rates.values()))
    tt = np.linspace(0.75, 2.65, 30)
    P.line(list(tt), list(np.exp(-r * (tt - 0.5 * (N0 + N1)) * 1e3)), INK, 2, '4 6', markers=False,
           tip=f'e^(−rt) with the beam\'s r = {r * 1e4:.2f}×10⁻⁴/ns')
    P.text(1.68, 1.04, 'bench', 24, BENCH_C, weight=600)
    P.text(1.68, 0.885, '← bench + half the beam\'s air', 20, BENCH_C)
    P.text(2.0, 0.92, 'beam: e^(−rt)', 24, BEAM_C, weight=600)
    # chi2 sensitivity bars
    B = sd.Plot(560, 620, x=(-0.6, 5.6), y=(0, 1300), title='bench χ²: no air (dark) vs +0.04 % air', ylabel='χ² (91 points)',
                margin=(10, 16, 92, 90), xlabel='drift field [V/cm]')
    B.xticks([(i, f'{s["E"]:.0f}') for i, s in enumerate(scan)]).yticks([(v, f'{v}') for v in (0, 250, 500, 750, 1000, 1250)])
    for i, s in enumerate(scan):
        B.vbar(i - 0.18, s['c'], 26, INK, tip=f'{s["E"]:.0f} V/cm, no air: χ² {s["c"]:.0f} / {s["n"]}')
        B.vbar(i + 0.18, s['ca'], 26, RED, opacity=0.75,
               tip=f'{s["E"]:.0f} V/cm, +{BENCH_AIR_CONTRAST} % air: χ² {s["ca"]:.0f} / {s["n"]} (×{s["ca"] / s["c"]:.1f})')
    B.hline(91, MUT, '4 5', 1.5, label='91 points', anchor='end')
    leg = sd.legend([('beam 92 V/cm', BEAM_C), ('beam 150 V/cm', '#e07a86'), ('bench 35 V/cm', BENCH_C),
                     ('bench 104 V/cm', '#6cc79a'), ('bench 35 V/cm + half the beam\'s air (model)', BENCH_C, 'dash')], size=22)
    b_lo = 100 * (1 - max(ends['beam 92'], ends['beam 150']))
    b_hi = 100 * (1 - min(ends['beam 92'], ends['beam 150']))
    pairs = ', '.join(f'{E} V/cm {100 * (ends[f"bench {E}"] - 1):+.0f} % (no-air model {100 * (ends[f"model {E}"] - 1):+.0f} %)'
                      for E in ('35', '104'))
    body = sd.title(f'Over the same 0.7 µs the beam loses {b_lo:.0f}–{b_hi:.0f} %; the bench stays where its no-air model puts it',
                    f'Each stack divided by its own level 0.8–1.0 µs after onset, read at {END0:g}–{END1:g} µs. Bench: {pairs}.')
    body += leg + sd.row(P.svg('beam vs bench'), B.svg('chi2 bars'), gap=60)
    D.slide('same-observable', body, f'''
<p><b>Left.</b> The low-field stacks of both setups, where the current is still pure drift (the deepest electrons have not arrived yet),
each divided by its own level 0.8–1.0 µs after onset — late enough that the electronics overshoot after the rise has settled, so no
model enters the data curves. Time origin: the beam's fitted t₀; the bench's bundle t₀ (the drift start). The bench stays within a few
per cent to the end of its window (the small dip-and-recovery is the response's shape: compare the model curves on slide §§bench-scan§§), while
the beam falls along e^(−rt). The dashed green curve is what the bench at 35 V/cm would look like with half the beam's air. The beam at 92 V/cm drifts at {d.Gb(*d.comp, 92.0)["v"]:.1f} µm/ns; the bench at 35 V/cm at
{d.Gn(d.DF["water"], 0.0, 34.7)["v"]:.1f} µm/ns — a similar depth per ns, so this is like-for-like in depth too.</p>
<p>Level at {END0:g}–{END1:g} µs relative to 0.8–1.0 µs: beam 92 V/cm {ends["beam 92"]:.3f}, 150 V/cm {ends["beam 150"]:.3f}; bench 35 V/cm
{ends["bench 35"]:.3f}, 104 V/cm {ends["bench 104"]:.3f}. The bench's own no-air model gives {ends["model 35"]:.3f} and {ends["model 104"]:.3f} over the same
span (the response tail and diffusion are not perfectly flat, which is why the composition fits, not this ratio, carry the bench limit);
with half the beam's air it would give {ends["air 35"]:.3f} and {ends["air 104"]:.3f}.</p>
<p><b>Right.</b> χ² of the bench no-air model against the same gas with {BENCH_AIR_CONTRAST} % air. At 35–174 V/cm, where the drift
fills the window, the air is rejected by χ² ×{min(s["ca"] / s["c"] for s in scan[:3]):.1f}–{max(s["ca"] / s["c"] for s in scan[:3]):.0f}.
At the high fields the drift ends early and the test weakens (×1.5–2.5). The bench null is not an insensitivity.</p>''',
            short='Same observable')


def s_rate_vs_field(D, d):
    Es = np.linspace(30, 390, 73)
    P = sd.Plot(1060, 640, x=(20, 400), y=(0.005, 6, 'log'), xlabel='drift field [V/cm]',
                ylabel='loss rate η·v [10⁻⁴ per ns]')
    P.xticks([(v, f'{v}') for v in (50, 100, 150, 200, 250, 300, 350, 400)]).yticks(
        [(0.01, '0.01'), (0.1, '0.1'), (1, '1'), (5, '5')])
    bm = d.C['beam']
    Eb = Es[(Es >= 60) & (Es <= 275)]
    P.line(list(Eb), [max(d.Gb(*d.comp, float(E))['etav'] * 1e4, 0.005) for E in Eb], BEAM_C, 4, markers=False,
           tip=f'Beam gas model: {bm["water"]:.2f} % water + {bm["air"]:.3f} % air ({bm["o2_ppm"]:.0f} ppm O₂)')
    Ebn = Es[(Es >= 35) & (Es <= 390)]
    bench_w = d.DF['water']
    P.line(list(Ebn), [max(d.Gn(bench_w, BENCH_AIR_CONTRAST, float(E))['etav'] * 1e4, 0.005) for E in Ebn],
           BENCH_C, 3, '10 7', markers=False,
           tip=f'Bench gas with {BENCH_AIR_CONTRAST} % air ({BENCH_AIR_CONTRAST * O2_PER_AIR:.0f} ppm O₂): what the bench would show; rejected (slide §§same-observable§§)')
    lim = 10.0 / O2_PER_AIR
    up = [max(d.Gn(bench_w, lim, float(E))['etav'] * 1e4, 0.005) for E in Ebn]
    P.band(list(Ebn), [0.005] * len(Ebn), up, BENCH_C, 0.22,
           tip='Bench allowed region: ≲ 10 ppm O₂ (det3 drift scan). Below the bottom of the axis = no loss.')
    P.line(list(Ebn), up, BENCH_C, 2.5, markers=False, tip='Bench upper limit, 10 ppm O₂')
    Em = [243, 150, 92]
    P.points(Em, [d.rates[E] * 1e4 for E in Em], BEAM_C, r=10,
             tips=[f'beam run_71 measured, {E} V/cm\nr = {d.rates[E] * 1e4:.2f} ± {d.rerr[E] * 1e4:.2f}×10⁻⁴/ns\n'
                   f'model at fitted composition: {d.Gb(*d.comp, float(E))["etav"] * 1e4:.2f}' for E in Em])
    for E in Em:
        P.raw(sd.line(P.X(E), P.Y((d.rates[E] - 2 * d.rerr[E]) * 1e4), P.X(E), P.Y((d.rates[E] + 2 * d.rerr[E]) * 1e4), BEAM_C, 3))
    P.points([34.7, 104.2], [0.005, 0.005], BENCH_C, r=9, marker='diamond',
             tips=['bench 35 V/cm: no loss seen (flat to ±2 % over 1.2 µs)', 'bench 104 V/cm: no loss seen'])
    P.text(160, 3.2, 'beam: measured + model', 22, BEAM_C, weight=600)
    P.text(255, 0.62, 'bench + half the beam\'s air', 21, BENCH_C)
    P.text(250, 0.03, 'bench allowed (≲ 10 ppm O₂)', 21, BENCH_C, weight=600)
    ratios = {E: d.rates[E] / d.Gn(bench_w, lim, float(E))['etav'] for E in Em}
    rlo, rhi = min(ratios.values()), max(ratios.values())
    side = sd.col(
        sd.p('One axis for both setups: how fast drifting electrons disappear.', 27),
        sd.p(f'Beam: three measurements of r agree with Magboltz for {bm["o2_ppm"]:.0f} ppm O₂. Bench: the same '
             f'model allows at most the shaded band — <b>×{rlo:.0f}–{rhi:.0f} below</b> the beam\'s measured rate '
             'at the same fields.', 26),
        sd.callout('The beam and bench gases differ in base mixture too (CF₄ vs none). That changes v and the '
                   'shaping, which the model carries; it does not attach: the no-air beam model loses nothing.', GREY, 24),
        gap=26, w=560)
    body = sd.title(f'Loss rate against field: the beam sits ×{rlo:.0f} or more above what the bench allows',
                    'Points: measured (2σ bars). Curves: Magboltz for each setup\'s fitted gas. Log scale.')
    body += sd.row(P.svg('rate vs field'), side, gap=50)
    D.slide('rate-field', body, f'''
<p>The beam curve is the Magboltz η·v for the fitted run_71 composition, after the smoothing in field. The bench band is η·v for the fitted
bench water ({bench_w:.2f} %) with air up to 10 ppm O₂, the robust upper limit from the det3 drift scan; the dashed curve is the
contrast hypothesis (half the beam's O₂) that the scan rejects.</p>
<p>The bench points are drawn at the bottom of the axis: the fits find no air, so no loss rate. The ratio compares the beam's measured
r with the bench's 10 ppm limit at the same field: {", ".join(f"{E} V/cm ×{ratios[E]:.0f}" for E in Em)}.</p>''',
            short='Rate vs field')


def s_bench_chambers(D, d):
    S = d.BS; BFj = d.BFj; par = np.array(BFj['shaper']); g = np.array(S['grid'])
    panels = []
    for k, det in enumerate(('det2', 'det3', 'det4', 'det7')):
        f = BFj[det]; E, gap = BF.CH[det]
        y = np.array(S[det]['x']['stack']); e = np.maximum(np.array(S[det]['x']['band']), 0.004)
        q, c, n, mc = BF.fit_curve(g, y, e, BF.cur(d.Gn, (f['water'], f['air']), E, gap, f['k'], f['gap_spread']), par)
        qa, ca, _, ma = BF.fit_curve(g, y, e, BF.cur(d.Gn, (f['water'], BENCH_AIR_CONTRAST), E, gap, f['k'], f['gap_spread']), par)
        P = sd.Plot(410, 520, x=(-0.4, 1.9), y=(-0.1, 1.15), title=f'{det} · {E:.0f} V/cm',
                    xlabel='time [µs]', margin=(10, 12, 90, 70 if k == 0 else 26),
                    ylabel='signal / peak' if k == 0 else '')
        P.xticks([(v, f'{v:g}') for v in (0, 0.5, 1, 1.5)]).yticks([(v, f'{v:g}' if k == 0 else '') for v in (0, 0.5, 1)])
        gg = g / 1e3
        P.line(list(gg), list(ma), RED, 3, '10 7', markers=False, tip=f'+{BENCH_AIR_CONTRAST} % air: χ² {ca:.0f}')
        P.line(list(gg), list(y), BENCH_C, 0, r=3, tip=f'data X (n = {S[det]["x"]["n"]})',
               tips=[f'{det} X, t = {a * 1e3:.0f} ns: {b:.3f}' for a, b in zip(gg, y)])
        P.line(list(gg), list(mc), INK, 2.5, markers=False,
               tip=f'{f["water"]:.2f} % water, {f["air"]:.3f} % air: χ² {c:.0f} / {n}; gap {gap} mm')
        P.text(1.12, 0.78, f'{f["water"]:.2f} % H₂O', 20, INK, weight=600)
        P.text(1.12, 0.66, f'χ² {c:.0f} vs {ca:.0f}', 19, MUT)
        panels.append(P.svg(det))
    leg = sd.legend([('data X', BENCH_C, 'dot'), ('model, fitted (no air)', INK),
                     (f'+{BENCH_AIR_CONTRAST} % air', RED, 'dash')], size=22)
    body = sd.title('Four bench chambers at their operating field: 0.6–1.0 % water, none needs air',
                    'June bench, all-strip trigger-placed stacks (det4 X head-on only). χ² per 86 points, fitted vs +0.04 % air.')
    body += leg + sd.row(*panels, gap=4)
    D.slide('bench-chambers', body, f'''
<p>A second, weaker bench check: the other chambers at their single operating field, with the shaper shared and the geometry (gradient,
gap spread) profiled. Water: det2 {BFj["det2"]["water"]:.2f} %, det3 {BFj["det3"]["water"]:.2f} % (the same chamber as the drift scan, a
different fill), det4 {BFj["det4"]["water"]:.2f} %, det7 {BFj["det7"]["water"]:.2f} %. That agrees with the RECONSTRUCTION_BASIS drift
velocity ("v 36.6 matches Ar/iso + 0.8 % H₂O"), which was measured independently from the micro-TPC tracks.</p>
<p>Single-field fits cannot separate air from a field gradient or an undershoot on their own (slide §§method§§, trap 5), which is why the drift scan
carries the bench limit. Even so, none needs air: det2, det3 and det7 fit best with none, and det4's best 0.01 % (21 ppm O₂) improves its χ² by
only {BFj["det4"]["chi2_noair"] - BFj["det4"]["chi2"]:.1f} (68 % range up to 0.025 %).</p>
<p><b>det7 fits poorly</b> (χ² {BFj["det7"]["chi2"]:.0f}/86): its plateau rises ~8 %, which needs a field gradient beyond what det3's scan allows or a
gap different from its marginal 27.5 mm. It wants no air either way; its composition is the least certain.</p>''',
            foot='Source: bench_stack.py → bench_stack.json; bench_fit.py --source air_hs → bench_fit_x_free.json.', short='Bench chambers')


def s_ledger(D, d):
    Y = f'<span style="color:{GREEN};font-weight:600">✓</span>'
    N = f'<span style="color:{RED};font-weight:600">✗</span>'
    Pz = f'<span style="color:{GOLD};font-weight:600">~</span>'
    bm = d.C['beam']
    rows = [
        ['Beam loses late charge, both views, every dataset', 'beam', Y, Y, Y, Y, 'slides §§run71§§, §§all-datasets§§'],
        ['Same loss per unit time at 243/150/92 V/cm', 'beam', Y, Y, N, Pz, 'slide §§time-depth§§'],
        ['No undershoot when the drift ends', 'beam', Y, N, Y, Y, 'slide §§undershoot§§'],
        ['Flat in gain, rate, spill, position', 'beam', Y, Y, Y, N, 'slide §§discriminators§§'],
        ['Per-strip charge falls with depth (template-free)', 'beam', Y, N, Y, Pz, 'slide §§ladder§§'],
        [f'v(E) at four fields → {bm["water"]:.2f} % water', 'beam', Y, '—', '—', '—', 'slides §§gas-model§§–§§beam-comp§§'],
        [f'r(E) at three fields → {bm["o2_ppm"]:.0f} ppm O₂, both views', 'beam', Y, '—', '—', '—', 'slide §§beam-comp§§'],
        ['Six-field plateau flat; no air, no gradient', 'bench', Y, N, Pz, Pz, 'slides §§bench-scan§§–§§rate-field§§'],
        ['Per-strip charge flat with depth', 'bench', Y, N, Pz, Pz, 'slide §§ladder§§'],
        ['Water 0.6–1.0 % matches micro-TPC v', 'bench', Y, '—', '—', '—', 'slide §§bench-chambers§§'],
        ['Same DREAM settings, different behaviour', 'both', Y, N, Pz, Pz, 'gas differs, readout does not'],
    ]
    body = sd.title('Every observation, both setups, one explanation',
                    'Which mechanism can produce each observation. Only attachment by O₂ in the beam gas fits every row.')
    hdr = ['observation', 'setup', 'O₂ in beam gas', 'readout high-pass', 'per-depth loss', 'charging / space charge', 'evidence']
    tips = [
        'Any of the four can make a late-charge deficit on its own; this row does not discriminate. The rows below do.',
        'Per time: attachment (η·v flat in this range) and a readout effect (it acts in time). A per-depth loss would scale with v. Charging could be time-like but fails row 4.',
        'A high-pass must undershoot; attachment, per-depth and gain-side losses do not.',
        'Gain-side effects follow gain and rate; the data do not.',
        'Template-free, no time stacking: kills any electronics explanation. A per-depth loss also does it, but fails row 2.',
        'Water sets v; an air leak cannot supply this much water (slide §§source§§).',
        'Magboltz, no free loss rate. ppm scale uncertain by a factor of a few (three-body O₂ attachment).',
        'The bench rejects half the beam\'s air at its low fields: a sensitive null. The readout is the same, so a high-pass would show here too; per-depth and gain effects could in principle differ between chambers.',
        'Independent, template-free bench null, all five chambers.',
        'Independent: v from micro-TPC track slopes (RECONSTRUCTION_BASIS).',
        'Register 1 0x081F 0xD023, RdClk_Div 6 on both: a readout cause would act on both setups.',
    ]
    body += sd.table(hdr, rows, size=23, widths=[560, 110, 170, 170, 170, 210, 240],
                     align=['left', 'left', 'center', 'center', 'center', 'center', 'left'], tips=tips)
    body += sd.p(f'{Y} explains it · {N} contradicts it · {Pz} could partly · — not applicable. Hover a row for the reasoning.', 22, MUT)
    D.slide('ledger', body, '''
<p>The consistency argument in one table. Each alternative fails at least one row that the data establish, and the attachment
explanation needs nothing setup-specific except the gas composition — which is also what differs between the setups.</p>
<p>The bench rows matter as much as the beam rows: if the loss came from the readout, the micromegas or the analysis, the bench (same
DREAM registers, same chambers' design, same stacking code) would show it too. It does not, at a sensitivity well below half the beam's
oxygen.</p>
<p>Not in the table because they are not tests of the mechanism: the X-view model deficit (FINDINGS §17, §23; det4's stripes), and the
~0.7 µs trigger-locked pickup at lower drift voltages (§20), which oscillates about zero.</p>
<p><b>A third setup, weakly.</b> The July n_TOF run58 (Ar/iso 90/10, chamber A) showed a drift velocity ~12 % low, explained by ~0.2 %
water, and hit amplitude flat with depth (no attachment). That check used hit amplitudes against hit-time depth, which this project no
longer uses for depth (RECONSTRUCTION_BASIS), so it is supporting context only: a third gas line with water and no O₂.</p>''',
            short='Consistency ledger')


def s_map(D, d):
    C = d.C; zs = C['zs_arms']; co2 = C['co2_period']; bm = C['beam']
    P = sd.Plot(1100, 640, x=(1, 2000, 'log'), y=(0, 2.0), xlabel='O₂ [ppm]   (bench: upper limits)',
                ylabel='water [%]')
    P.xticks([(1, '1'), (10, '10'), (100, '100'), (1000, '1000')]).yticks([(v, f'{v:g}') for v in (0, 0.5, 1.0, 1.5, 2.0)])
    o2 = np.logspace(0, 3.3, 40)
    P.line(list(o2), list(o2 * 1e-4 * 0.05 * 100 / 100), GREY, 2.5, '10 7', markers=False,
           tip='What a bulk room-air leak brings: H₂O/O₂ ≈ 0.05 (50 % RH at 20 °C). 160 ppm O₂ would come with 0.0008 % water.')
    P.text(60, 0.08, 'room-air leak: H₂O/O₂ ≈ 0.05', 21, GREY)
    pts = [('run_63 25.6°, Aug 3 00:22', zs['r63_d425'], 'open'), ('run_63 25.6°, 00:30', zs['r63_d325'], 'open'),
           ('run_63 flat, 01:00–01:54', zs['r63_flat700'], 'circle'),
           ('run_71 RAW, 05:22–05:52', dict(o2_ppm=bm['o2_ppm'], water=bm['water'], air_68=bm['air_68'], water_68=bm['water_68']), 'circle')]
    for nm, r_, mk in pts:
        P.raw(sd.line(P.X(r_['water_68'][0] * 0 + r_['o2_ppm']), P.Y(r_['water_68'][0]), P.X(r_['o2_ppm']), P.Y(r_['water_68'][1]), BLUE, 2.5))
        P.points([r_['o2_ppm']], [r_['water']], BLUE, r=10, marker=mk,
                 tips=[f'{nm}\nO₂ {r_["o2_ppm"]:.0f} ppm, water {r_["water"]:.2f} % '
                       f'(68 %: {r_["water_68"][0]:.2f}–{r_["water_68"][1]:.2f})'])
    P.raw(sd.line(P.X(co2['o2_range'][0]), P.Y(co2['water']), P.X(co2['o2_range'][1]), P.Y(co2['water']), ORANGE, 3))
    P.points([co2['o2_ppm']], [co2['water']], ORANGE, r=10, marker='diamond',
             tips=[f'run_56, Ar/CO₂/iso, Aug 1 (ZS 5σ, rough)\nO₂ {co2["o2_ppm"]:.0f} ppm ({co2["o2_range"][0]:.0f}–{co2["o2_range"][1]:.0f}), water {co2["water"]:.2f} %'])
    lims = [('det3 drift scan (+ det2)', 10.0, d.DF['water'])] + \
           [(f'{k}', max(d.BFj[k]['air_68'][1] * O2_PER_AIR, 10.0), d.BFj[k]['water']) for k in ('det4', 'det7')]
    for nm, ul, w in lims:
        P.raw(sd.arrow(P.X(ul), P.Y(w), P.X(ul / 5), P.Y(w), BENCH_C, 3, 12))
        P.raw(sd.line(P.X(ul), P.Y(w) - 12, P.X(ul), P.Y(w) + 12, BENCH_C, 4))
        P.text(ul * 1.15, w - 0.03, f'bench {nm}', 20, BENCH_C,
               tip=f'bench {nm}: water {w:.2f} %, O₂ < {ul:.0f} ppm')
    P.text(120, 1.85, 'beam, Ar/CF₄/iso', 22, BLUE, weight=600)
    P.text(330, 1.38, 'beam, Ar/CO₂/iso', 21, ORANGE, weight=600)
    side = sd.col(
        sd.p('Two clusters, far apart on the O₂ axis: the beam at 150–290 ppm, the bench below 10.', 27),
        sd.p('Both carry water, the beam more (1.5–1.7 % against 0.6–1.0 %).', 26),
        sd.callout('Neither sits anywhere near the room-air line. Whatever brought the water did not bring the oxygen.', BLUE, 26),
        gap=26, w=500)
    body = sd.title('Where the water and the oxygen are, on every dataset',
                    'Fitted compositions with 68 % ranges (rotated beam blocks cannot pin the water: no drift end in the window).')
    body += sd.row(P.svg('composition map'), side, gap=50)
    D.slide('map', body, f'''
<p>Beam points: run_71 (RAW, joint fit) and the three zero-suppressed run_63 blocks of the night before, each with its own composition
(ZS distortion from RAW run_71 at the nearest field, additive, +1 % systematic). The run_71 composition does not describe run_63
(χ² {zs["r63_d425"]["chi2_run71comp"]:.0f}, {zs["r63_d325"]["chi2_run71comp"]:.0f}, {zs["r63_flat700"]["chi2_run71comp"]:.0f} against
{zs["r63_d425"]["chi2"]:.0f}, {zs["r63_d325"]["chi2"]:.0f}, {zs["r63_flat700"]["chi2"]:.0f} with their own fits): the gas changed over hours.</p>
<p>CO₂ period (run_56, 1 Aug): ZS 5σ only, drift field assumed 243 V/cm, water from v = 12.33 µm/ns (run_57); rough by construction.</p>
<p>Bench: upper limits on O₂ at the fitted water (det3 from the drift scan; det2/4/7 from single-field fits with 10 ppm as the floor).</p>''',
            short='Composition map')


def s_trend(D, d):
    zs = d.C['zs_arms']; bm = d.C['beam']
    pts = [(0.43, zs['r63_d425'], 'run_63 25.6°, 142 V/cm, 00:22–00:30'),
           (0.55, zs['r63_d325'], 'run_63 25.6°, 108 V/cm, 00:30–00:37'),
           (1.45, zs['r63_flat700'], 'run_63 flat, 243 V/cm, 01:00–01:54'),
           (5.62, dict(o2_ppm=bm['o2_ppm'], air_68=bm['air_68']), 'run_71 RAW, 3 fields, 05:22–05:52')]
    P = sd.Plot(1060, 600, x=(0, 6.2), y=(0, 330), xlabel='time on 3 Aug 2026 [h]', ylabel='O₂ [ppm]')
    P.xticks([(v, f'{v:02d}:00') for v in range(0, 7)]).yticks([(v, f'{v}') for v in (0, 100, 200, 300)])
    P.line([p[0] for p in pts], [p[1]['o2_ppm'] for p in pts], BLUE, 3, '6 6', r=10,
           tips=[f'{nm}\nO₂ {p["o2_ppm"]:.0f} ppm (air {p["o2_ppm"] / O2_PER_AIR:.3f} %, 68 %: '
                 f'{p["air_68"][0] * O2_PER_AIR:.0f}–{p["air_68"][1] * O2_PER_AIR:.0f})' for x, p, nm in pts],
           tip='Ar/CF₄/iso, det4')
    co2 = d.C['co2_period']
    P.hline(co2['o2_ppm'], ORANGE, '10 7', 3, label=f'Aug 1, Ar/CO₂/iso (run_56): ~{co2["o2_ppm"]:.0f} ppm',
            tip=f'run_56, two days earlier, a different base gas: {co2["o2_range"][0]:.0f}–{co2["o2_range"][1]:.0f} ppm. ZS 5σ only; rough.')
    P.band([0, 6.2], [0, 0], [10, 10], GREEN, 0.3, tip='Bench allowed region (≲ 10 ppm), June.')
    P.text(4.2, 18, 'bench (June): ≲ 10 ppm', 21, GREEN, weight=600)
    for x, p, nm in pts:
        P.text(x + 0.08, p['o2_ppm'] + 14, nm.split(',')[0], 19, MUT)
    side = sd.col(
        sd.p(f'Through the night of 2–3 Aug the O₂ fell from ~{zs["r63_d425"]["o2_ppm"]:.0f} to '
             f'~{min(zs["r63_flat700"]["o2_ppm"], bm["o2_ppm"]):.0f}–{max(zs["r63_flat700"]["o2_ppm"], bm["o2_ppm"]):.0f} ppm, '
             f'and two days earlier, in the other gas, it was ~{co2["o2_ppm"]:.0f}.', 27),
        sd.p('The water stayed at 1.5–1.7 % throughout.', 27),
        sd.callout('An O₂ level that drifts by ×1.5–2 over hours to days, while the water holds still, behaves like '
                   'air ingress on the beam gas line, not like a property of the chamber.', BLUE, 26),
        gap=26, w=540)
    body = sd.title('The beam\'s oxygen changed over hours; its water did not',
                    'O₂ fitted per data block, det4 at H4. Each block has its own composition fit.')
    body += sd.row(P.svg('trend'), side, gap=50)
    D.slide('trend', body, '''
<p>Times are the block mid-points on 3 Aug (run_63 rotated blocks at 00:22 and 00:30, flat 01:00–01:54; run_71 05:22–05:52).
The 147 and 163 ppm of the last two are within the model's resolution of each other (grid steps of 0.005–0.01 % air);
the fall from the rotated blocks to them is significant (each rotated block rejects the run_71 composition by Δχ² ≥ 95).</p>
<p>Caveat on the absolute scale: the ppm values inherit Magboltz's three-body O₂ attachment, uncertain by a factor of a few. The
<i>trend</i> is measured by one model on one chamber and does not depend on that scale.</p>''',
            short='Time trend')


def s_source(D, d):
    bm = d.C['beam']
    ratio_beam = bm['water'] / (bm['o2_ppm'] * 1e-4)
    ratio_bench = d.DF['water'] / (10e-4)
    rows = [
        ('room air (50 % RH, 20 °C)', 0.05, GREY, 'H₂O 1.2 % / O₂ 20.9 %: what a bulk leak brings', ''),
        ('beam gas (run_71)', ratio_beam, BLUE, f'{bm["water"]:.2f} % / {bm["o2_ppm"]:.0f} ppm', ''),
        ('bench gas (det3)', ratio_bench, GREEN, f'{d.DF["water"]:.2f} % / < 10 ppm: a lower limit', ''),
    ]
    bars = sd.hbars([(nm, v, c, tp) for nm, v, c, tp, _ in rows], vmax=5000, vmin=0.01, log=True, width=760, h=30,
                    label_w=420, fmt=lambda v: f'{v:,.2f}' if v < 1 else f'{v:,.0f}' + (' or more' if v > 500 else ''), size=26)
    # schematic: two gas lines
    W, H = 1664, 330
    o = []

    def line_row(y, name, supply, chamber, comp, o2):
        o.append(f'<rect x="20" y="{y - 46}" width="190" height="92" rx="14" fill="#fffdf9" stroke="{INK}" stroke-width="2"/>')
        o.append(sd.T(115, y - 6, 'supply', 21, MUT))
        o.append(sd.T(115, y + 22, supply, 21, INK, weight=600))
        o.append(sd.line(210, y, 1110, y, '#9aa3b2', 14))
        o.append(sd.T(660, y + 44, f'{name} gas line (tubing, connectors)', 20, MUT))
        for x in (330, 470, 610, 750):
            o.append(sd.arrow(x, y - 62, x, y - 12, BLUE, 3, 11))
        o.append(sd.T(540, y - 72, 'H₂O permeates in along the tubing', 20, BLUE, weight=600,
                      tip='Water permeates polymer tubing orders of magnitude faster than O₂; the dose scales as length / flow.'))
        if o2:
            o.append(sd.arrow(960, y - 62, 960, y - 12, RED, 4, 13))
            o.append(sd.T(960, y - 72, 'air ingress? (O₂)', 21, RED, weight=600,
                          tip='Where is not known: a connector, a valve, a long open section. It must bring ~0.07–0.14 % air and vary over hours.'))
        o.append(f'<rect x="1110" y="{y - 46}" width="200" height="92" rx="14" fill="#eef1f6" stroke="{INK}" stroke-width="2"/>')
        o.append(sd.T(1210, y + 8, chamber, 22, INK, weight=600))
        o.append(sd.T(1340, y - 6, comp, 22, INK, 'start', 600))
        o.append(sd.T(1340, y + 24, o2 or 'no O₂ (< 10 ppm)', 21, RED if o2 else GREEN, 'start', 600))

    line_row(105, 'bench', 'Ar/iso 95/5', 'bench det2–7', f'{d.DF["water"]:.2f} % H₂O', None)
    line_row(255, 'H4 beam', 'Ar/CF₄/iso', 'det4 at H4', f'{bm["water"]:.2f} % H₂O', f'{bm["o2_ppm"]:.0f} ppm O₂ (150–290)')
    schem = sd.svg(W, H, ''.join(o), 'two gas lines')
    body = sd.title('The water and the oxygen arrived by different routes',
                    f'H₂O / O₂ ratio, log scale. A room-air leak gives 0.05; the beam gas has ~{ratio_beam / 0.05:,.0f}× more water per O₂.')
    body += bars
    body += schem
    body += sd.callout('To check: the H4 line\'s tubing material and length, its flow, its connectors and the supply bottles — and the '
                       'bench line for comparison. None of it is in our logs.', GOLD, 24)
    D.slide('source', body, f'''
<p>If the beam's O₂ came from a bulk room-air leak, the same leak would bring ~0.05 parts of water per part O₂: 160 ppm O₂ with
~0.0008 % water. The beam gas has {bm["water"]:.2f} % water. Conversely, a leak large enough to bring 1.5 % water would bring ~30 % O₂,
which would attach every electron within millimetres. So the water has another source on both setups (permeation, or the supply), and
the O₂ is an additional, beam-line-specific ingredient.</p>
<p>Permeation fits both observations qualitatively: H₂O permeates common polymer tubing orders of magnitude faster than O₂, and both
scale as 1/flow. The O₂ level varying over hours points at an ingress that depends on conditions (flow, pressure, temperature). Neither
the H4 line's tubing nor its flow was recorded.</p>
<p>The water figures assume the nominal drift fields and the measured (bench) or nominal 30 mm (beam) gaps.</p>''',
            short='Where it comes from')


def s_consequences(D, d):
    steps = [
        dict(label='Beam (det4, H4)', sub='attachment term per run: η·v from the composition, or the measured r', color=RED,
             tip='The wft forward model currently has no attachment. Without it, late (deep) charge is under-predicted and the '
                 'fit compensates with v, diffusion or the depth profile.'),
        dict(label='Bench (June cosmics)', sub='no attachment term; water only, through v', color=GREEN,
             tip='The bench calibration bundles stand as they are on this question.'),
        dict(label='n_TOF campaign', sub='not measured here: gas line and mixture differ', color=GOLD,
             tip='The same observable can be built on n_TOF waveforms (decoded_root), but the n_TOF tracks are not head-on '
                 'beam tracks; it needs its own selection. Open.'),
    ]
    P = sd.Plot(1060, 480, x=(0, 30), y=(0, 1.08), xlabel='drift depth [mm]', ylabel='fraction of electrons surviving',
                margin=(10, 190, 86, 100))
    P.xticks([(v, f'{v}') for v in (0, 5, 10, 15, 20, 25, 30)]).yticks([(v, f'{v:g}') for v in (0, 0.25, 0.5, 0.75, 1)])
    z = np.linspace(0, 30, 61)
    for E in (243, 150, 92):
        v = d.Gb(*d.comp, float(E))['v']
        s = np.exp(-d.rates[E] * z * 1e3 / v)
        P.line(list(z), list(s), FC[E], 4, markers=False,
               tip=f'beam {E} V/cm: survival from 30 mm = {s[-1]:.2f} (v = {v:.1f} µm/ns, r = {d.rates[E] * 1e4:.2f}×10⁻⁴/ns)')
        P.text(30.3, s[-1], f'{E} V/cm: {s[-1]:.0%}', 20, FC[E])
    P.hline(1.0, BENCH_C, '10 7', 3, tip='Bench: no loss at any depth.')
    P.text(30.3, 1.0, 'bench: 100%', 20, BENCH_C)
    surv = {E: math.exp(-d.rates[E] * 30e3 / d.Gb(*d.comp, float(E))['v']) for E in (243, 150, 92)}
    side = sd.col(
        sd.p(f'In the beam gas only {min(surv.values()):.0%}–{max(surv.values()):.0%} of the electrons from the cathode arrive '
             '(92–243 V/cm). A model that assumes none is lost reads the missing charge as something else.', 26),
        sd.callout('Per run, because the O₂ moved: the compositions in this note are the per-block inputs.', RED, 24),
        gap=24, w=640)
    body = sd.title('What it changes: the beam model needs attachment, the bench model does not',
                    'Survival of drifting electrons in the run_71 beam gas, from the measured loss rates.')
    body += sd.flow(steps, size=24, arrow='')
    body += sd.row(P.svg('survival'), side, gap=40)
    D.slide('consequences', body, f'''
<p>The wft forward model (<code>wft/</code>) has no attachment term. On the bench this note confirms that is right. On the beam, deep
charge is under-predicted (only {surv[92]:.0%} survives the full gap at 92 V/cm), and a fit without attachment will absorb it in whatever parameter can tilt the depth
profile (drift velocity, diffusion, the per-depth charge profile). Two ways in: give the beam model a composition (water + air per run,
with Magboltz η·v), or a single per-run r taken from these stacks.</p>
<p>Earlier beam claims that rest on the depth profile or on a fitted v without attachment should be re-read with this in mind; the June
bench results are unaffected.</p>''', short='Consequences')


def s_open(D, d):
    items = [
        ('Which molecule.', 'The observable fixes η·v. Magboltz says water attaches nothing and O₂ does, but its three-body O₂ '
                            'attachment (H₂O or isobutane as third body) is poorly modelled: the ppm scale is uncertain by a factor of a '
                            'few. The beam/bench contrast and the trend do not depend on it.'),
        ('A gain-stage loss with no fingerprint.', 'Something inside the amplification that is time-dependent yet independent of gain, '
                                                   'rate and position, the same in both views, with no undershoot, present on the beam '
                                                   'and absent on the bench. None is known; it is the residual alternative.'),
        ('The effective shapers.', 'Fitted per setup (they include the ion tail) and different on bench and beam, although the DREAM '
                                   'settings are identical.'),
        ('Magboltz η smoothed in field.', 'With raw per-field values the 92 V/cm stacks would not fit.'),
        ('The CO₂ period and det7.', 'CO₂: ZS only, drift field assumed (~290 ppm is rough). det7: poor single-field fit.'),
    ]
    lis = ''.join(f'<li style="margin-bottom:14px"><b style="color:{DINK}">{a}</b> {b}</li>' for a, b in items)
    dec = [
        ('Add attachment to the beam model', 'per-run composition or per-run r?'),
        ('Ask for the H4 gas-line record', 'tubing, flow, supply; the bench line too'),
        ('n_TOF', 'build the same observable on its waveforms before assuming either answer'),
    ]
    dl = ''.join(f'<li style="margin-bottom:12px"><b style="color:{DINK}">{a}</b> — {b}</li>' for a, b in dec)
    body = (sd.kicker('What this does not rule out, and what is open')
            + '<div style="display:flex;gap:72px">'
            + f'<div style="flex:1.35"><ul style="font-size:25px;line-height:1.38;color:{DMUT};padding-left:28px">{lis}</ul></div>'
            + f'<div style="flex:1"><p style="font-size:30px;color:{DINK};font-weight:600;margin-bottom:18px">Decisions</p>'
            + f'<ul style="font-size:26px;line-height:1.4;color:{DMUT};padding-left:28px">{dl}</ul></div></div>')
    D.slide('open', body, '''
<p>Retracted along the way and not to be revived: the §11 field-ordered spike and the §13 fivefold O₂ spread (threshold alignment), the
§10 "no attachment" (an alignment-and-normalisation artefact in the opposite direction), the §8 bench loss (the same), and the June
hits-based λ ≈ 13–15 mm on det3 (superseded by the drift scan).</p>
<p>All numbers: <code>FINDINGS.md</code> §16–26 and the long-form report (notes/mx17-beam-attachment).</p>''',
            dark=True, short='Open')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT, 'mx17-attachment-slides.html'))
    a = ap.parse_args()
    d = Data(OUT)
    D = sd.Deck('Beam gas attachment',
                'det4 at the SPS loses late drift charge to oxygen in the beam gas; the June bench does not. '
                'The observable, the tests, and one gas model for both setups.')
    s_cover(D, d)
    s_geometry(D, d)
    s_observable(D, d)
    s_fingerprints(D, d)
    s_method(D, d)
    s_run71(D, d)
    s_time_depth(D, d)
    s_undershoot(D, d)
    s_discriminators(D, d)
    s_all_datasets(D, d)
    s_ladder(D, d)
    s_gasmodel(D, d)
    s_beam_comp(D, d)
    scan = _bench_scan_curves(d)
    s_consistency(D, d)
    s_bench_scan(D, d, scan)
    s_same_observable(D, d, scan)
    s_rate_vs_field(D, d)
    s_bench_chambers(D, d)
    s_ledger(D, d)
    s_map(D, d)
    s_trend(D, d)
    s_source(D, d)
    s_consequences(D, d)
    s_open(D, d)
    p = D.write(a.out, note_meta=dict(
        title='Beam gas attachment: the observable and one model for beam and bench',
        summary='det4 at H4 loses late drift charge at the same rate per ns at three fields, both views, no undershoot: '
                'O₂ attachment (~150–290 ppm). The June bench, same model, six fields: no O₂. Slides.',
        tags='X17, beam test, micromegas, gas, attachment',
        date='2026-10-10'))
    # cross-references are written as §§<slide id>§§ and resolved here, so inserting a slide renumbers them
    ids = [s_[0] for s_ in D.slides]
    txt = p.read_text()
    missing = sorted(set(re.findall(r'§§([a-z0-9-]+)§§', txt)) - set(ids))
    if missing:
        raise SystemExit(f'unknown slide references: {missing}')
    p.write_text(re.sub(r'§§([a-z0-9-]+)§§', lambda m: str(ids.index(m.group(1)) + 1), txt))
    print('wrote', p, f'({len(ids)} slides)')


if __name__ == '__main__':
    main()
