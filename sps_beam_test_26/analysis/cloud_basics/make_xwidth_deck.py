#!/usr/bin/env python3
"""make_xwidth_deck.py -- slide note on the bench X drift-width excess, FINDINGS §30-34 (2026-10-16/17).

Reads the analysis's own outputs, so a rerun after any of them changes updates the numbers:
results/unsat_width.json (§30-32), xw_sim.json (§33 closure toy), xw_tan.json, xw_amp.json, xw_compress.json (§34),
and the Magboltz grid (gasmodel.GasGrid 'bench', air_hs).

    make_xwidth_deck.py [--out ~/x17/cosmic_bench/cloud_basics/xwidth/x-width-excess.html]
    then: python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <out> --slug x-width-excess --force --deploy
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd                                          # noqa: E402
from slidedoc import BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT, RULE, DBLUE, DRED, DGREEN  # noqa: E402

R = os.path.join(HERE, 'results')
J = lambda n: json.load(open(os.path.join(R, n)))
UW, SIM, TAN, AMP, CMP = J('unsat_width.json'), J('xw_sim.json'), J('xw_tan.json'), J('xw_amp.json'), J('xw_compress.json')

K3 = 'sat_det3__long_run_resist_490V_drift_1000V'
K7H = 'g_det7_long__hvpool_det7_480-505V_drift_700V'
SETKEY = {'det3L': K3, 'det4': 'g_det4', 'det7R': 'g_det7_long__longer_run', 'det7H': K7H}
SETLAB = {'det3L': 'det3 long run · 347 V/cm', 'det4': 'det4 · 208 V/cm', 'det7R': 'det7 longer_run · 243 V/cm',
          'det7H': 'det7 HV pool · 243 V/cm'}
SETCOL = {'det3L': BLUE, 'det4': ORANGE, 'det7R': PURPLE, 'det7H': GREEN}
DATA_C, TOY_C, TRUTH_C, SAT_C = INK, GOLD, GREY, RED
SATS = [3500, 3000, 2500, 2000, 1500]


def g(r):
    """gap-corrected (v, D_T, err, Magboltz D_T at that v) of an unsat_width fit."""
    c = r['gapcorr']
    br = c['magboltz']['branches']
    return c['v'], c['DT'], c['DT_err'], (br[0]['DT_um_rtcm'] if br else float('nan')), (br[0]['water_pct'] if br else float('nan'))


def quad(a, b):
    return np.sign(a - b) * np.sqrt(abs(a * a - b * b))


# ----------------------------------------------------------------------------------------------- slides
def cover(D):
    v, dt, de, mb, w = g(UW[K3]['unsat'])
    ex = quad(dt, mb)
    sim = SIM['det4/base']['cuts']
    clos = [SIM[f'{s}/base']['cuts']['sat3500']['DT_bias'] for s in SETKEY]
    SK = [K3, 'g_det4', 'g_det6_long', 'o22_long_det2', 'g_det7_long__longer_run', K7H]      # operating-HV datasets
    sat = [UW[k]['frac_unsat'] for k in SK]
    body = (sd.kicker('MX17 cosmic bench · X drift width · FINDINGS §30–34 · 16–17 Oct 2026') +
            sd.title('<span style="color:#eef0f3">Most of the X-width excess was ADC saturation. '
                     'det3’s remainder is real and unexplained; det4/det7 depend on amplitude</span>') +
            sd.row(sd.bignum(f'{100 * (1 - max(sat)):.0f}–{100 * (1 - min(sat)):.0f} %', 'of head-on events saturate',
                             DRED, 'The saturated events read wide and slow. After cutting them, det4, det6 and det3 at 243 V/cm agree with Magboltz.',
                             tip='fraction of |tan_x| < 0.03 events with window max ≥ 3500 ADC (operating-HV datasets):\n'
                                 + '\n'.join(f'{k}: {100 * (1 - UW[k]["frac_unsat"]):.0f} %' for k in SK)),
                   sd.bignum(f'+{ex:.0f}', 'µm/√cm in quadrature over Magboltz on det3 at 347 V/cm', DBLUE,
                             f'{(dt - mb) / de:.0f} sd. Present at every drift field, cut, amplitude tercile and position.',
                             tip=f'det3 long run, unsat, 27.9 mm gap: D_T {dt:.0f} ± {de:.0f} vs Magboltz {mb:.0f} at {w:.2f} % water (v {v:.1f})'),
                   sd.bignum(f'{min(clos):+.0f}…{max(clos):+.0f}', 'estimator bias in a closure toy', DGREEN,
                             'Flat in the amplitude cut in every arm, so the det4/det7 cut dependence is in their data.',
                             tip='xw_sim base arm, D_T(fit) − truth at the 3500 cut, 100 replicas each:\n'
                                 + '\n'.join(f'{s}: {SIM[f"{s}/base"]["cuts"]["sat3500"]["DT_bias"]:+.1f}' for s in SETKEY)),
                   gap=56))
    D.slide('cover', body, dark=True, short='Summary', notes=(
        'Bench cosmic head-on tracks (|tan_x| &lt; 0.03 against the M3 reference). The X view crosses the resistive strips, '
        'so its width growth with drift time is drift diffusion plus whatever the readout adds. The fitted transverse '
        'diffusion D_T is compared with Magboltz for Ar/iso 95/5 at the water fraction implied by the fitted drift velocity. '
        'No new data can be taken (the chambers are gone): everything is the June bench data, simulation and Magboltz.\n\n'
        'Sections: §30 saturation, §31 the water-branch bug and the gap correction, §32 the det3 field scan and the cut '
        'scan, §33 the closure toy (lxplus condor 4474928), §34 the field tilt, amplitude terciles and footprint test. '
        'All in <code>sps_beam_test_26/analysis/cloud_basics/FINDINGS.md</code>.'))


def setup(D):
    W, H = 760, 600
    s = []
    x0, x1, yc, ym = 60, 700, 60, 470                         # cathode y, mesh y
    s.append(sd.line(x0, yc, x1, yc, INK, 5))
    s.append(sd.T(x0, yc - 14, 'drift cathode', 22, INK, 'start'))
    s.append(sd.line(x0, ym, x1, ym, INK, 2, '6 5'))
    s.append(sd.T(x1, ym - 12, 'mesh', 22, INK, 'end'))
    for i in range(13):                                       # X strips
        xs = 120 + i * 40
        s.append(f'<rect x="{xs}" y="{ym + 30}" width="30" height="14" fill="{BLUE}" fill-opacity="0.7"/>')
    s.append(sd.T(380, ym + 76, 'X strips, 0.78 mm pitch, across the resistive strips', 21, BLUE))
    s.append(sd.arrow(380, 20, 380, ym + 6, RED, 3.5, 16))
    s.append(sd.T(392, 34, 'cosmic μ, |tan_x| < 0.03', 22, RED, 'start'))
    for z, sp in ((110, 8), (190, 22), (270, 34), (350, 46), (430, 58)):   # diffusion cones
        s.append(f'<ellipse cx="380" cy="{z}" rx="{sp}" ry="7" fill="{GOLD}" fill-opacity="0.35"/>')
        s.append(sd.line(380, z, 380 - sp, ym, GOLD, 1.2, '3 4') + sd.line(380, z, 380 + sp, ym, GOLD, 1.2, '3 4'))
    s.append(sd.T(470, 270, 'charge from depth z lands', 21, MUT, 'start'))
    s.append(sd.T(470, 296, 'σ² = σ0² + D_T² z wide', 21, MUT, 'start'))
    s.append(sd.T(470, 322, 'and arrives at t = z / v', 21, MUT, 'start'))
    s.append(sd.line(30, yc, 30, ym, MUT, 2) + sd.T(22, (yc + ym) / 2, 'gap 27.5–30.6 mm', 21, MUT, rot=-90))
    fig = sd.svg(W, H, ''.join(s), 'head-on track schematic')
    steps = [dict(label='Head-on events', sub='M3 reference |tan_x| < 0.03; waveform window ±5 mm', color=RED,
                  tip='headon_cache.py: every reference event of a run with |tan| < 0.06 in a view, X and Y windows'),
             dict(label='Stack', sub='9 strips around the charge maximum, aligned on 60 ADC', color=GOLD,
                  tip='rc_diffusion.stack, unbiased: not normalised per event, aligned on the first crossing of 60 ADC by the 9-strip sum, plain mean'),
             dict(label='Fit', sub='σ0, Dd, A, t0, T free: bundle X template', color=BLUE,
                  tip='x_width.fit_x: W_o(t) = A Σ_k F_o(σ(u_k)) h(t − u_k), σ² = σ0² + 2 Dd u, u uniform over T = gap / v'),
             dict(label='Compare', sub='D_T = √(2Dd/v) vs Magboltz at the v-implied water', color=INK,
                  tip='gasmodel.GasGrid bench (Ar/iso 95/5 + water), air_hs grid; bench_water_e.invert returns every water branch')]
    txt = sd.col(sd.flow(steps[:2], size=22), sd.flow(steps[2:], size=22),
                 sd.callout('The question: does the fitted D_T match Magboltz? Its first answer, '
                            '“every chamber is ~25–60 % too wide”, would have set the detector model’s drift diffusion.', RED, 24),
                 sd.p('Hover the dotted terms, points and boxes for numbers and provenance.', 22, MUT), gap=24, w=860)
    D.slide('setup', sd.title('A head-on track turns drift diffusion into a width that grows with drift time',
                              'The measurement, and the pipeline every number in this note goes through') + sd.row(fig, txt, gap=44),
            short='Method', foot='Bench data: June 2026 cosmic bench (det2–det7), Ar/iso 95/5. Caches: ~/x17/cosmic_bench/cloud_basics/xwidth/caches/',
            notes='The fit has no RC term in X (the resistive strips run along y), so all width growth in X with time is '
                  'drift diffusion plus anything in the readout that looks like it. T (hence v) is fitted from the same stack. '
                  'Results are in the GAP = 30 frame. The comparison rescales to each chamber’s measured gap: v × gap/30, '
                  'D_T × √(30/gap).')


def saturation(D):
    rows = [('det4 · 208', 'g_det4'), ('det6 long · 243', 'g_det6_long'), ('det3 700 V · 243', 'sat_det3__drift_scan_resist_490V_drift_700V'),
            ('det7 longer · 243', 'g_det7_long__longer_run'), ('det7 HV pool · 243', K7H), ('det3 long · 347', K3),
            ('det2 o22 · 347', 'o22_long_det2')]
    P = sd.Plot(1100, 640, x=(200, 760, 'lin'), y=(-0.6, len(rows) - 0.4, 'lin'), xlabel='X D_T (µm/√cm, own gap)',
                margin=(24, 30, 92, 250))
    P.xticks([(t, str(t)) for t in range(200, 761, 100)])
    for i, (lab, k) in enumerate(rows):
        yy = len(rows) - 1 - i
        a, u = UW[k]['all'], UW[k]['unsat']
        va, da, ea, _, _ = g(a)
        vu, du, eu, mb, w = g(u)
        P.raw(sd.T(P.x0 - 16, P.Y(yy) + 8, lab, 22, INK, 'end'))
        P.raw(sd.line(P.X(du), P.Y(yy), P.X(min(da, 760)), P.Y(yy), RULE, 6))
        eu_lo, eu_hi = max(du - u['gapcorr']['DT_err'], 200), du + u['gapcorr']['DT_err']
        P.points([min(da, 760)], [yy], SAT_C, 11, tips=[f'{lab}, all events (n {a["n"]}): D_T {da:.0f} ± {ea:.0f}, v {va:.1f}'])
        P.points([du], [yy], BLUE, 11, tips=[f'{lab}, unsaturated (n {u["n"]}): D_T {du:.0f} ± {eu:.0f}, v {vu:.1f}'])
        P.raw(f'<line x1="{P.X(eu_lo):.1f}" y1="{P.Y(yy):.1f}" x2="{P.X(eu_hi):.1f}" y2="{P.Y(yy):.1f}" stroke="{BLUE}" stroke-width="3"/>')
        if mb == mb:
            P.points([mb], [yy], INK, 9, marker='open', tips=[f'Magboltz at v {vu:.1f}: {mb:.0f} ({w:.2f} % water)'])
    P.text(P.x0 + P.pw - 8, P.y0 + 20, 'all events → past 760', 20, SAT_C, 'end')
    fig = P.svg('D_T all vs unsaturated vs Magboltz')
    txt = sd.col(sd.legend([('all events', SAT_C, 'dot'), ('window max < 3500 ADC', BLUE, 'dot'), ('Magboltz at that v (open)', INK, 'dot')], 22),
                 sd.callout('Cutting the saturated events (30–85 % of the head-on sample) moves D_T down by 60–320 '
                            'and v up. At 208–243 V/cm what remains agrees with Magboltz.', BLUE, 24),
                 sd.callout('At 347 V/cm an excess survives: det3 8 sd, det2 1.5 sd. det3 at 243 V/cm is 1.3 sd high.', RED, 24),
                 sd.p('Every bench water fraction quoted before 16 Oct came from saturated samples.', 22, MUT), gap=22, w=520)
    D.slide('sat', sd.title('Saturated events read wide: the cut removes most of the excess',
                            'X D_T per dataset: all head-on events, the unsaturated ones, and Magboltz at the unsaturated v') + sd.row(fig, txt, gap=44),
            short='Saturation', foot='unsat_width.py → results/unsat_width.json; gap-corrected (det2 30.6, det3 27.9, det7 27.5, det4/det6 30 mm)',
            notes='Selection is per event on the X window maximum, which does not depend on depth, so it cannot select a '
                  'drift velocity. On the det3 amplification scan v is flat at ~39 µm/ns (GAP-30 frame) up to 485 V and falls '
                  'once saturation sets in (36.5 at 495 V, 24 at 525 V). Clipping alone does not account for the size of '
                  'the all-event widths (§33, the toy reproduces only +18…+53 of it).')


def invert_bug(D):
    from gasmodel import GasGrid
    G = GasGrid('bench', 'air_hs')
    E = UW[K3]['E_Vcm']
    ws = np.arange(0, 1.205, 0.01)
    vv = [G(w, 0, E)['v'] for w in ws]
    dd = [G(w, 0, E)['DT'] for w in ws]
    v3, dt3, de3, mb3, w3 = g(UW[K3]['unsat'])
    vbund = 36.6 * 27.9 / 30
    P = sd.Plot(900, 600, x=(0, 1.2, 'lin'), y=(32, 42, 'lin'), xlabel='water in Ar/iso 95/5 (%)', ylabel='Magboltz v (µm/ns) at 347 V/cm')
    P.xticks([(t, f'{t:.1f}') for t in np.arange(0, 1.21, 0.2)]); P.yticks([(t, str(t)) for t in range(32, 43, 2)])
    P.line(list(ws), vv, BLUE, markers=False, tip='Magboltz v(water) at 347 V/cm (air_hs grid, smoothed)')
    P.hline(v3, RED, '8 6', 2.5, label=f'det3 unsat v {v3:.1f} (27.9 mm)', tip=f'det3 long run, unsat, gap-corrected: v {v3:.2f}', anchor='start', where='below')
    P.points([w3], [v3], RED, 10, tips=[f'only solution: {w3:.2f} % water, Magboltz D_T {mb3:.0f}'])
    P.raw(sd.T(P.X(0.02), P.Y(38.8) - 14, 'dry 38.8', 20, MUT, 'start'))
    Q = sd.Plot(700, 600, x=(0, 1.2, 'lin'), y=(250, 500, 'lin'), xlabel='water (%)', ylabel='Magboltz D_T (µm/√cm)')
    Q.xticks([(t, f'{t:.1f}') for t in np.arange(0, 1.21, 0.4)]); Q.yticks([(t, str(t)) for t in range(250, 501, 50)])
    Q.line(list(ws), dd, BLUE, markers=False, tip='Magboltz D_T(water) at 347 V/cm')
    Q.hline(dt3, RED, '8 6', 2.5, label=f'det3 measured {dt3:.0f}', tip=f'det3 unsat D_T {dt3:.0f} ± {de3:.0f}')
    Q.points([w3], [mb3], RED, 10, tips=[f'Magboltz at {w3:.2f} %: {mb3:.0f}'])
    Q.points([0.14], [G(0.14, 0, E)['DT']], GREY, 9, marker='open', tips=['the wrong branch the old sort returned: 0.14 %, D_T ~450'])
    D.slide('invert', sd.title('At 347 V/cm v is not monotonic in water: the old inversion took the wrong branch',
                               'Magboltz v and D_T against water at det3’s field. A v below the dry value has only the wet solution') +
            sd.row(P.svg('v vs water'), Q.svg('DT vs water'), gap=44), short='Water branch',
            foot='bench_water_e.invert now walks a 0.005 % grid and returns every branch (§31). It also moved det3’s bundle v from 0.59 to 0.90 % water.',
            notes='Before 16 Oct the inversion sorted v and interpolated water against it. At 347 V/cm v rises from 38.8 (dry) to 41.1 '
                  'at 0.4 % and falls to 35.1 at 1 %, so a measured v had two solutions, or only the wet one below dry. The sort '
                  'returned a low-water value with a dry-like D_T, about 450, and so hid the excess at 347 V/cm. The 208 and 243 V/cm '
                  'fits were unaffected (v is monotonic there). With det3’s 27.9 mm gap applied, v sits below dry and only the wet '
                  'branch, 0.94–0.97 %, remains. That is the same as the drift-scan composition fit.')


def det3_fields(D):
    keys = [('sat_det3__drift_scan_resist_490V_drift_700V', 243), ('sat_det3__drift_scan_resist_490V_drift_900V', 312),
            (K3, 347), ('sat_det3__drift_scan_resist_490V_drift_1100V', 382)]
    P = sd.Plot(980, 600, x=(220, 400, 'lin'), y=(-2, 12, 'lin'), xlabel='drift field (V/cm)',
                ylabel='D_T² − Magboltz²  (10⁴ µm²/cm)')
    P.xticks([(t, str(t)) for t in (240, 280, 320, 360, 400)]); P.yticks([(t, str(t)) for t in range(-2, 13, 2)])
    P.hline(0, MUT, '8 6', 2, label='Magboltz')
    xs, ys, tips = [], [], []
    for k, E in keys:
        v, dt, de, mb, w = g(UW[k]['unsat'])
        ex, ee = (dt ** 2 - mb ** 2) / 1e4, 2 * dt * de / 1e4
        P.raw(sd.line(P.X(E), P.Y(ex - ee), P.X(E), P.Y(ex + ee), BLUE, 3))
        xs.append(E); ys.append(ex)
        tips.append(f'{E} V/cm, n {UW[k]["unsat"]["n"]}: D_T {dt:.0f} ± {de:.0f} vs Magboltz {mb:.0f} at {w:.2f} % water (v {v:.1f})\n'
                    f'excess {ex:.1f} ± {ee:.1f} ×10⁴, {quad(dt, mb):+.0f} in quadrature')
    P.points(xs, ys, BLUE, 11, tips=tips)
    v2, d2, e2, m2, w2 = g(UW['o22_long_det2']['unsat'])
    P.points([347 + 4], [(d2 ** 2 - m2 ** 2) / 1e4], ORANGE, 10, marker='diamond',
             tips=[f'det2 o22_long (n {UW["o22_long_det2"]["unsat"]["n"]}), 347 V/cm: D_T {d2:.0f} ± {e2:.0f} vs {m2:.0f}'])
    P.raw(sd.line(P.X(351), P.Y((d2 ** 2 - m2 ** 2 - 2 * d2 * e2) / 1e4), P.X(351), P.Y((d2 ** 2 - m2 ** 2 + 2 * d2 * e2) / 1e4), ORANGE, 3))
    wat = [g(UW[k]['unsat'])[4] for k, _ in keys]
    txt = sd.col(sd.legend([('det3, unsaturated', BLUE, 'dot'), ('det2, 347 V/cm', ORANGE, 'dot')], 22),
                 sd.callout(f'det3 is above Magboltz at every field. Within its errors the excess is the same or grows mildly with field.', BLUE, 24),
                 sd.callout(f'v implies {min(wat):.2f}–{max(wat):.2f} % water at all four fields, so the gas reading is '
                            'self-consistent. The excess is in D_T, not in v.', GREEN, 24),
                 sd.callout('det2 is the only other chamber run at 347 V/cm, and it agrees with det3 within its errors (n 86).', ORANGE, 24),
                 gap=22, w=600)
    D.slide('fields', sd.title('det3’s excess is there at every drift field from 243 to 382 V/cm',
                               'Excess of D_T² over Magboltz at the v-implied water, det3 drift scan + long run, unsaturated, 27.9 mm gap') +
            sd.row(P.svg('excess vs field'), txt, gap=44), short='det3 fields',
            foot='unsat_width.py; drift scan 900/1100 V caches built on the laptop (headon_cache.py --subrun). 500 V: v below the 1 % grid, not compared.')


def cut_scan(D):
    sets = [('det3L', K3), ('det4', 'g_det4'), ('det7R', 'g_det7_long__longer_run'), ('det7H', K7H)]
    P = sd.Plot(1060, 620, x=(1300, 3700, 'lin'), y=(100, 450, 'lin'), xlabel='cut: event window max < (ADC)',
                ylabel='X D_T (µm/√cm, GAP-30 frame)')
    P.xticks([(t, str(t)) for t in SATS]); P.yticks([(t, str(t)) for t in range(100, 451, 50)])
    for j, (s, k) in enumerate(sets):
        xs, ys, tips = [], [], []
        for c in SATS:
            kk = k if c == 3500 else f'{k}@sat{c}'
            if kk in UW and 'unsat' in UW[kk]:
                u = UW[kk]['unsat']
                x = c + (j - 1.5) * 25
                xs.append(x); ys.append(u['DT'])
                tips.append(f'{SETLAB[s]}, cut {c}: D_T {u["DT"]:.0f} ± {u["DT_err"]:.0f}, v {u["v"]:.2f}, n {u["n"]}')
                P.raw(sd.line(P.X(x), P.Y(u['DT'] - u['DT_err']), P.X(x), P.Y(u['DT'] + u['DT_err']), SETCOL[s], 2.5))
        P.line(xs, ys, SETCOL[s], tips=tips, r=8)
        tr = SIM[f'{s}/base']['truth']['DT']
        P.raw(sd.line(P.X(3620), P.Y(tr), P.X(3700), P.Y(tr), SETCOL[s], 4)
              + f'<rect class="hit" x="{P.X(3620):.1f}" y="{P.Y(tr) - 8:.1f}" width="40" height="16" fill="transparent"'
                f'{sd.tipattr(f"{SETLAB[s]}: Magboltz in the GAP-30 frame {tr:.0f}")}/>')
    P.raw(sd.T(P.X(3660), P.y0 + 22, 'MB', 20, MUT))
    txt = sd.col(sd.legend([(SETLAB[s], SETCOL[s]) for s, _ in sets], 22),
                 sd.callout('det3 holds at ~370 at every cut, and with |tan| < 0.01 as well.', BLUE, 24),
                 sd.callout('det4 and det7 fall by 90–170 as the cut tightens and end below Magboltz. Their agreement at '
                            '3500 is where a falling curve crosses it.', ORANGE, 24),
                 sd.p('v is stable against the cut on every set, so the selection does not pick a velocity.', 22, MUT),
                 gap=22, w=540)
    D.slide('cuts', sd.title('Tightening the amplitude cut: det3 stays put, det4 and det7 fall',
                             'X D_T against the cut on the event’s window maximum, unsaturated head-on events') +
            sd.row(P.svg('cut scan'), txt, gap=44), short='Cut scan',
            foot='unsat_width.py --sat N → keys <key>@sat<N>. Ticks at right: Magboltz in the GAP-30 frame (the toy’s truth).')


def toy(D):
    steps = [dict(label='Real layouts', sub='strips, noise, track x and tan of every cache event', color=INK,
                  tip='each toy event reuses one real event: pos, per-channel noise, ref_mesh_x as the true track, tan_x'),
             dict(label='Ionisation', sub='Fischle Ar clusters, 2.8 /mm; δ-ray arm', color=RED),
             dict(label='Drift', sub='Magboltz D_T, D_L at the v-implied water', color=GOLD),
             dict(label='Gain', sub='Polya θ 0.5 × per-event spread tuned to the data', color=PURPLE,
                  tip='det3/det4: lognormal; det7: lognormal + 30 % faint events (bimodal spectrum)'),
             dict(label='Readout', sub='bundle template, 60 ns, correlated noise, data ceilings', color=BLUE,
                  tip='noise lag-1 autocorrelation 0.71–0.85, common-mode 0.10–0.17, per-channel ceilings from the data’s saturated maxima'),
             dict(label='Same fit', sub='stack + fit_x at every cut', color=GREEN)]
    panels = []
    for s in ('det4', 'det7H'):
        P = sd.Plot(800, 420, x=(1300, 3700, 'lin'), y=(100, 450, 'lin'), xlabel='cut (ADC)', ylabel='D_T (GAP-30 frame)',
                    margin=(40, 20, 80, 104), title=SETLAB[s])
        P.xticks([(t, str(t)) for t in SATS]); P.yticks([(t, str(t)) for t in range(100, 451, 100)])
        tr = SIM[f'{s}/base']['truth']['DT']
        P.hline(tr, TRUTH_C, '8 6', 2, label=f'truth {tr:.0f}', tip='Magboltz D_T injected into the toy')
        c = SIM[f'{s}/base']['cuts']; d = SIM[f'{s}/base']['data']
        xs = [x for x in SATS if f'sat{x}' in c]
        lo = [c[f'sat{x}']['DT_mean'] - c[f'sat{x}']['DT_rms'] for x in xs]; hi = [c[f'sat{x}']['DT_mean'] + c[f'sat{x}']['DT_rms'] for x in xs]
        P.band(xs, lo, hi, TOY_C, 0.25, tip='toy: ± replica rms (the scatter of one data-sized sample)')
        P.line(xs, [c[f'sat{x}']['DT_mean'] for x in xs], TOY_C,
               tips=[f'toy, cut {x}: {c[f"sat{x}"]["DT_mean"]:.0f} (rms {c[f"sat{x}"]["DT_rms"]:.0f}, 100 replicas)' for x in xs])
        dx = [x for x in SATS if f'sat{x}' in d]
        for x in dx:
            P.raw(sd.line(P.X(x), P.Y(d[f'sat{x}']['DT'] - d[f'sat{x}']['DT_err']), P.X(x), P.Y(d[f'sat{x}']['DT'] + d[f'sat{x}']['DT_err']), INK, 2.5))
        P.line(dx, [d[f'sat{x}']['DT'] for x in dx], DATA_C,
               tips=[f'data, cut {x}: {d[f"sat{x}"]["DT"]:.0f} ± {d[f"sat{x}"]["DT_err"]:.0f} (n {d[f"sat{x}"]["n"]})' for x in dx])
        panels.append(P.svg(f'toy vs data {s}'))
    D.slide('toy', sd.title('Closure toy on the real layouts: the estimator is flat in the cut',
                            'Synthetic events with known D_T, pushed through the same stack and fit; 100 data-sized replicas per arm') +
            sd.flow(steps, size=21) + sd.row(*panels, gap=44) +
            sd.legend([('data', DATA_C), ('toy mean', TOY_C), ('toy ± replica rms', TOY_C, 'box'), ('truth', TRUTH_C, 'dash')], 22),
            short='Closure toy', foot='xw_sim.py (lxplus condor 4474928: 180 jobs, 9 arms × 4 datasets × 100 replicas) → results/xw_sim.json',
            notes='Tune: per dataset, the per-event gain spread and scale are fitted so the toy’s window-maximum quantiles and '
                  'saturated fraction match the data. det3: max quantiles 1286/2001/3165/3785 against the data’s 1289/2047/3024/3778, '
                  'saturated 0.43 against 0.42. det7 needed a 30 % faint population, and det7R’s faintest decile is still 2× too bright. '
                  'Truth: v and σ0 from the data’s unsat fit, D_T = Magboltz at the v-implied water in the GAP-30 frame.\n\n'
                  'Limits: the electronics template is the one the fit uses (template mismatch not tested), noise is Gaussian with '
                  'measured correlations, and saturation is a hard clip.')


def arms(D):
    order = [('noclip', 'no clipping'), ('flatgain', 'flat gain'), ('delta', 'δ-ray extent'), ('base', 'base'),
             ('share', 'neighbour copies (bundle lp kernel)'), ('refc', 'centred on the true track')]
    P = sd.Plot(1060, 640, x=(-20, 200, 'lin'), y=(-0.6, len(order) - 0.4, 'lin'),
                xlabel='D_T(fit) − truth (µm/√cm)', margin=(24, 30, 92, 370))
    P.xticks([(t, str(t)) for t in range(0, 201, 50)])
    P.vline(0, MUT, '6 5', 2)
    for i, (a, lab) in enumerate(order):
        yy = len(order) - 1 - i
        P.raw(sd.T(P.x0 - 16, P.Y(yy) + 8, lab, 22, INK, 'end'))
        for j, s in enumerate(('det3L', 'det4')):
            c = SIM[f'{s}/{a}']['cuts']
            off = (j - 0.5) * 0.28
            lo_cut = 'sat1500' if 'sat1500' in c else 'sat2000'          # flat gain on det4: too few events below 1500
            b35, b15 = c['sat3500']['DT_bias'], c[lo_cut]['DT_bias']
            P.raw(sd.line(P.X(b35), P.Y(yy + off), P.X(b15), P.Y(yy + off), SETCOL[s], 4))
            P.points([b35], [yy + off], SETCOL[s], 9, tips=[f'{SETLAB[s]}, {lab}: bias {b35:+.0f} at 3500, {b15:+.0f} at 1500, '
                                                            f'{c["all"]["DT_bias"]:+.0f} with all events'])
            P.points([b15], [yy + off], SETCOL[s], 9, marker='open', tips=[f'{SETLAB[s]}, {lab}: bias {b15:+.0f} at the {lo_cut[3:]} cut'])
    txt = sd.col(sd.legend([('det3', BLUE), ('det4', ORANGE), ('filled: 3500 cut · open: 1500', INK, 'dot')], 22),
                 sd.callout('No ingredient makes the width depend on the cut: filled and open points sit together in every arm.', GREEN, 24),
                 sd.callout('Neighbour copies (c1 ≈ 0.05, τ 130–170 ns) read as drift diffusion: +75…+90. That is det3’s '
                            'excess, but every bundle carries the same kernel.', BLUE, 24),
                 sd.callout('Centring on the true track inflates by +90…+135. That is most of the reference-centred '
                            'inflation in §32, so it was not M3 error.', RED, 24), gap=22, w=520)
    D.slide('arms', sd.title('Toy arms: clipping, gain, δ rays move nothing; crosstalk and ref-centring add a flat offset',
                             'Bias of the fitted D_T per toy arm, unsaturated cuts 3500 (filled) and 1500 (open)') +
            sd.row(P.svg('toy arms'), txt, gap=44), short='Toy arms',
            foot='Replica rms 7–24; the mean of 100 is good to ±1–2. det7 arms behave the same (results/xw_sim.json).',
            notes='Saturation in the toy: all events vs unsat moves D_T by +18…+53 and v by −0.6…−1.9 µm/ns. In the data the shift is '
                  '+62 (det3) to +325 (det7H), with v down to 26–28 on det7. Real DREAM saturation does more than clip, so the toy '
                  'cannot correct saturated events and the cut stays.')


def tilt(D):
    t = TAN['det3L']
    mids = [0.5 * (b['lo'] + b['hi']) for b in t['bins']]
    P = sd.Plot(980, 520, x=(-0.065, 0.065, 'lin'), y=(250, 550, 'lin'), xlabel='signed reference tan_x',
                ylabel='X D_T (GAP-30 frame)')
    P.xticks([(x, f'{x:+.2f}' if x else '0') for x in (-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06)])
    P.yticks([(y, str(y)) for y in range(250, 551, 50)])
    tm, ts = t['toy_mean'], t['toy_rms']
    P.band(mids, [a - b for a, b in zip(tm, ts)], [a + b for a, b in zip(tm, ts)], TOY_C, 0.25)
    P.line(mids, tm, TOY_C, tips=[f'toy (vertical field), tan {m:+.2f}: {a:.0f} (rms {b:.0f})' for m, a, b in zip(mids, tm, ts)])
    for b in t['bins']:
        P.raw(sd.line(P.X(b['tan_mean']), P.Y(b['DT'] - b['DT_err']), P.X(b['tan_mean']), P.Y(b['DT'] + b['DT_err']), INK, 2.5))
    P.points([b['tan_mean'] for b in t['bins']], [b['DT'] for b in t['bins']], DATA_C, 9,
             tips=[f'data, tan {b["lo"]:+.2f}…{b["hi"]:+.2f}: {b["DT"]:.0f} ± {b["DT_err"]:.0f} (n {b["n"]})' for b in t['bins']])
    for f, c, lab in ((t['fit_toyb'], BLUE, 'data'), (t['toy_fit'], TOY_C, 'toy')):
        xs = np.linspace(-0.06, 0.06, 61)
        P.line(list(xs), list(np.sqrt(np.maximum(f['a'] + f['b'] * (xs + f['delta']) ** 2, 0))), c, 2.5, '8 6', markers=False,
               tip=f'{lab}: D_T² = a + b (tan + δ)², δ {f["delta"]:+.4f} ± {f["delta_err"]:.4f}, floor {np.sqrt(f["a"]):.0f}')
    P.hline(t['truth_DT'], TRUTH_C, '4 4', 2, label=f'truth {t["truth_DT"]:.0f}', where='below')
    # delta by position
    Q = sd.Plot(600, 520, x=(-0.5, 5.5, 'lin'), y=(-0.04, 0.08, 'lin'), ylabel='δ (field tilt as an angle)', margin=(24, 20, 110, 104))
    Q.yticks([(y, f'{y:+.2f}' if y else '0') for y in (-0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08)])
    Q.hline(0, MUT, '6 5', 2)
    i = 0
    for ax in ('ref_mesh_x', 'ref_mesh_y'):
        for r in t[f'split_{ax}']:
            f = r['fit']
            Q.raw(sd.line(Q.X(i), Q.Y(f['delta'] - f['delta_err']), Q.X(i), Q.Y(f['delta'] + f['delta_err']), BLUE if ax.endswith('x') else PURPLE, 3))
            Q.points([i], [f['delta']], BLUE if ax.endswith('x') else PURPLE, 9,
                     tips=[f'{ax[-1]} {r["lo"]:.0f}–{r["hi"]:.0f} mm: δ {f["delta"]:+.4f} ± {f["delta_err"]:.4f}'])
            Q.raw(sd.T(Q.X(i), Q.y0 + Q.ph + 30, f'{ax[-1]}{["low", "mid", "high"][i % 3]}', 20, MUT))
            i += 1
    txt = sd.callout(f'At the minimum (δ {t["fit_toyb"]["delta"]:+.3f} ± {t["fit_toyb"]["delta_err"]:.3f}) det3 still reads '
                     f'{np.sqrt(t["fit_toyb"]["a"]):.0f}, against the toy’s {np.sqrt(t["toy_fit"]["a"]):.0f}. A spread of local tilts would '
                     f'need σ_δ ≈ 0.05 rad. The dish’s steepest step is about 0.026, and δ does not change sign across the chamber.', BLUE, 23)
    D.slide('tilt', sd.title('Not a tilted field: det3 is wide even at its narrowest angle',
                             'det3 long run, unsaturated: X D_T in signed tan_x bins (left), and the tilt δ per position tercile (right)') +
            sd.row(P.svg('tilt scan'), Q.svg('delta by position'), gap=40) + txt, short='Field tilt',
            foot='xw_tan.py → results/xw_tan.json. Cathode relief 25.7–29.2 mm: mx_june_wft/GAP_STUDY_2026-07-30.md',
            notes='A non-flat cathode tilts the drift field. In the X view the charge then walks sideways in proportion to depth, an '
                  'extra angle δ the M3 reference does not see. The stack would be narrowest at tan = −δ. det4: δ +0.003 ± 0.009 with '
                  'floor 337 against the toy’s 341. det7 is too sparse per bin (±60–130) to fit. The common δ ≈ +0.02 on det3 '
                  '(1.3–1.5σ) could equally be a residual alignment angle.')


def terciles(D):
    P = sd.Plot(1060, 620, x=(-0.5, 11.5, 'lin'), y=(150, 450, 'lin'), ylabel='X D_T (GAP-30 frame)', margin=(24, 30, 110, 104))
    P.yticks([(y, str(y)) for y in range(150, 451, 50)])
    i = 0
    for s in ('det3L', 'det4', 'det7R', 'det7H'):
        a = AMP[s]
        for j, c in enumerate(a['classes']):
            f, ty = c['free'], c.get('toy', {})
            if f['DT'] > 50 and f['DT_err'] < 150:
                P.raw(sd.line(P.X(i), P.Y(f['DT'] - f['DT_err']), P.X(i), P.Y(f['DT'] + f['DT_err']), SETCOL[s], 3))
                P.points([i], [f['DT']], SETCOL[s], 10,
                         tips=[f'{SETLAB[s]}, window max {c["lo"]:.0f}–{c["hi"]:.0f} (n {c["n"]}): D_T {f["DT"]:.0f} ± {f["DT_err"]:.0f}, '
                               f'σ0 {f["sig0"]:.2f}; median charge {c["where"]["total"]:.0f}; {100 * c["where"]["frac_edge20"]:.0f} % within 20 mm of the edge'])
            else:
                P.raw(sd.T(P.X(i), P.Y(180), 'empty', 19, SETCOL[s], tip=f'{SETLAB[s]} faint tercile: median charge {c["where"]["total"]:.0f} ADC, '
                                                                          f'{100 * c["where"]["frac_edge20"]:.0f} % within 20 mm of the edge: tracks at or past the chamber edge'))
            if ty:
                P.points([i + 0.25], [ty['DT']['mean']], TOY_C, 8, marker='diamond', tips=[f'toy, same tercile: {ty["DT"]["mean"]:.0f} (rms {ty["DT"]["rms"]:.0f})'])
            P.raw(sd.T(P.X(i), P.y0 + P.ph + 30, ['faint', 'mid', 'bright'][j], 19, MUT))
            i += 1
        P.raw(sd.T(P.X(i - 2), P.y0 + P.ph + 62, s.replace('L', '').replace('R', ' longer').replace('H', ' HV'), 22, SETCOL[s], weight=600))
    txt = sd.col(sd.legend([('data', INK, 'dot'), ('toy', TOY_C, 'dot')], 22),
                 sd.callout('det3: +57…+84 over the toy in every tercile. Its excess does not depend on amplitude.', BLUE, 24),
                 sd.callout('det4/det7: the terciles differ in both directions (det4 mid −110, det7 bright +109). It is not '
                            'simply “faint reads narrow”.', ORANGE, 24),
                 sd.p('det7’s faint tercile is reference tracks past the chamber edge (median charge 60–360 ADC).', 22, MUT), gap=22, w=520)
    D.slide('terciles', sd.title('Exclusive amplitude terciles: det3 is wide in all three; det4/det7 scatter both ways',
                                 'Free-fit X D_T per tercile of the window maximum (unsaturated head-on), data against the toy') +
            sd.row(P.svg('terciles'), txt, gap=44), short='Terciles', foot='xw_amp.py → results/xw_amp.json (fits free, T pinned and σ0+T pinned; model-free m2)',
            notes='det4 mid tercile with σ0 and T pinned at the pooled values: 315 ± 17 (free 250 ± 30, σ0 0.42 against 0.35). Its '
                  'early-time second moment is 0.57 against the toy’s 0.37 mm²: wider from the start, so the free fit trades σ0 for Dd. '
                  'Not a residual baseline: the strips’ pre-onset ADC is −3.6…+0.6 in data and toy alike.')


def footprint(D):
    panels = []
    for s in ('det3L', 'det4', 'det7H'):
        c = CMP[s]
        P = sd.Plot(530, 500, x=(0, 4200, 'lin'), y=(0, 0.6, 'lin'), xlabel='peak ADC', ylabel='neighbour / peak' if s == 'det3L' else '',
                    margin=(44, 16, 84, 96), title=SETLAB[s])
        P.xticks([(t, str(t)) for t in (0, 1000, 2000, 3000, 4000)]); P.yticks([(t, f'{t:.1f}') for t in (0, 0.2, 0.4, 0.6)])
        P.vline(3500, RED, '6 5', 2)
        for key, col_, mk in (('toy', TOY_C, 'diamond'), ('data', DATA_C, 'circle')):
            rr = [r for r in c[key] if 'med' in r]
            xs = [0.5 * (r['lo'] + r['hi']) for r in rr]
            for x, r in zip(xs, rr):
                P.raw(sd.line(P.X(x), P.Y(r['med'] - r['err']), P.X(x), P.Y(r['med'] + r['err']), col_, 2.5))
            P.line(xs, [r['med'] for r in rr], col_, 2.5, marker=mk,
                   tips=[f'{key}, peak {r["lo"]}–{r["hi"]}: r {r["med"]:.3f} ± {r["err"]:.3f} (n {r["n"]})' for r in rr])
        panels.append(P.svg(f'footprint {s}'))
    D.slide('footprint', sd.title('The footprint depends on amplitude on det4 and det7, not det3',
                                  'At the leading strip’s peak sample, r = (W[c−1] + W[c+1]) / 2W[c] against the peak ADC: data and the linear toy') +
            sd.row(*panels, gap=36) +
            sd.row(sd.legend([('data', DATA_C), ('toy (linear response)', TOY_C), ('3500 cut', RED, 'dash')], 22),
                   sd.p('det4/det7: small signals put too little on the ±1 neighbours, catching up only at 2000–3000 ADC. '
                        'No compression approaching the ceiling.', 23), gap=40),
            short='Footprint', foot='xw_compress.py → results/xw_compress.json; det7 longer_run behaves as the HV pool',
            notes='With a linear response r depends only on the cluster footprint, so it is flat in the peak amplitude. The toy shows '
                  'what flat means with Landau and Polya fluctuations included. Only the chambers whose r departs from the toy '
                  '(det4, det7) have a cut-dependent D_T. det3 follows the toy below 3000 ADC (0.34 against 0.41 at 3000–3500, just '
                  'under the cut). Above the ceiling every chamber reads r ≈ 0.50–0.52 against the toy’s 0.42–0.48: real '
                  'saturation widens more than clipping. Which part of the response is non-linear (the prompt footprint or the '
                  'neighbour copies) cannot be decided from head-on X data.')


def verdict(D):
    rows = []
    for det, k, status, syst in (('det3', K3, 'excess, unexplained', '+100 (excess)'),
                                 ('det2', 'o22_long_det2', 'like det3, n 86', 'as det3'),
                                 ('det4', 'g_det4', 'Magboltz at 3500, amplitude-dependent', '±100'),
                                 ('det6', 'g_det6_longer', 'Magboltz (gap 30 assumed)', '±100'),
                                 ('det7', 'g_det7_long__longer_run', 'amplitude-dependent', '±100')):
        v, dt, de, mb, w = g(UW[k]['unsat'])
        rows.append([det, f'{UW[k]["E_Vcm"]:.0f}', f'{v:.1f}', f'{w:.2f} %', f'{dt:.0f} ± {de:.0f}', f'{mb:.0f}', status, syst])
    tb = sd.table(['chamber', 'E (V/cm)', 'v', 'water', 'X D_T unsat', 'Magboltz', 'reading', 'systematic'], rows, 24,
                  align=['left', 'right', 'right', 'right', 'right', 'right', 'left', 'right'])
    D.slide('verdict', sd.title('What goes into the detector model', 'Unsaturated head-on X fits at the 3500 cut, each chamber at its own gap') +
            tb + sd.row(sd.callout('Freeze the kernel with Dd per chamber from the unsaturated stack at the 3500 cut (the production '
                                   'amplitude mix), and the two systematics stated beside it.', GREEN, 24),
                        sd.callout('The beam (run_71, det4) also read above Magboltz in §29, with saturation not cut there. Re-check it before '
                                   'carrying the bench kernel to the beam.', ORANGE, 24), gap=40),
            short='Verdict', foot='The ±100 µm/√cm are the size of the cut dependence (§32) and of det3’s residual; they are stated, not fitted.')


def questions(D):
    Q = [('Is the excess det3’s or 347 V/cm’s?', 'det3 at every field (243–382); det2 at 347 V/cm agrees with det3 (1.5 sd, n 86). No other chamber was run at 347.',
          'Build det2’s head-on cache from the 6-22 longer_run (desktop) for more statistics. Refit det3’s same-night o22 runs, whose v is stuck at a bound: that gives det2 vs det3 on one gas, one night.'),
         ('One gas, four water fractions?', 'v implies 0.76 % (det2, 6-22), 0.49 % (det4, 6-24), 0.72 → 0.56 % (det7, 6-26), 0.89–1.00 % (det3, 6-27).',
          'Track v through each run in time slices (det7 dries over the night). Check the bench gas log or flow history. Redo the drift-scan composition fit on unsaturated events only.'),
         ('Which gap for det4 and det6?', 'unsat_width.py uses 30 mm for both. GAP_STUDY (K = 22 basis) measured det6 27.85 and det4 25.55 (amplitude-dominated).',
          'One gap table for every script. det6 at 27.85 lowers v by 7 % and raises D_T by 4 %, and moves its inferred water. Requote det6, and treat det4 at matched amplitude.'),
         ('Two footprints disagree', 'Bundle σ_p0 0.26 (det4) / 0.42–0.43 (det3, det7) against head-on σ0 0.21–0.35. The neighbour copies add +75…+90 to D_T in the toy.',
          'Run the wft forward model per amplitude tercile: does χ² degrade on faint events? Use the lxplus avalanche simulation (mx17_response) for the footprint against avalanche size.'),
         ('Why is det4/det7’s response non-linear at low amplitude?', 'r is 0.34–0.37 against 0.45 below 2000 ADC (det4) and lower on det7; det3 is linear.',
          'Compare the same r on the Y view and against position (gain map). Check the DREAM threshold and zero-suppression settings in run_config.json for those runs.'),
         ('Saturation does more than clip', 'all events vs unsat: data +62 … +325, toy +18 … +53. Above the ceiling r is 0.50–0.52 against 0.42–0.48.',
          'Characterise saturated pulses in the existing data (plateau length, recovery, neighbour response) and add that to the toy. Needed only if saturated events are ever used.'),
         ('Bench vs beam for det4', 'Bench det4 agrees with Magboltz after the cut. Beam run_71 det4 read +118…+286 above it (§29, all events).',
          'Apply the window-max cut to run_71 in x_width_beam.py and requote.'),
         ('det3’s common δ ≈ +0.02, det7 at |tan| 0.05', 'det3 tilt 1.3–1.5σ, same everywhere; det7 reads low at |tan| = 0.05 where the toy rises (2.6σ pooled).',
          'Compare δ with the alignment’s mean TPC−M3 angle residual. Pool det7’s sets for a signed-tan m2 growth test.')]
    rows = [[f'<b>{q}</b>', e, i] for q, e, i in Q]
    D.slide('questions', sd.title('Open consistency questions across the data sets, and how to settle each from what we have',
                                  'None of these needs new data: each idea uses existing caches, simulation or Magboltz') +
            sd.table(['question', 'evidence', 'idea to resolve'], rows, 19, widths=[330, 600, 734], align=['left', 'left', 'left']),
            short='Open questions', notes='Ordered by how much each could change the kernel. The gap table (question 3) is a '
                                          'straight inconsistency between scripts and should be fixed first; it is cheap.')


def closing(D):
    body = (sd.kicker('What this does not rule out') +
            sd.row(sd.col(sd.p('<b>det3’s excess has no cause yet.</b> It is not saturation, the estimator, a field tilt, the dished '
                               'cathode or amplitude. Crosstalk of the bundle’s size would explain its magnitude but not why det3 alone.', 26, '#eef0f3'),
                          sd.p('<b>Magboltz at 347 V/cm</b> is where v(water) turns over and D_T is most sensitive to the composition. '
                               'A Magboltz or cross-section error there would look like this excess.', 26, '#eef0f3'),
                          sd.p('<b>The toy shares the fit’s template</b>, so a template mismatch is not tested, and its saturation is a hard clip.', 26, '#eef0f3'),
                          gap=24),
                   sd.col(sd.p('<b>Decisions open</b>', 28, DGREEN),
                          sd.p('1 · Freeze the kernel now with the stated systematics, or first fix the det4/det6 gaps and requote.', 26, '#eef0f3'),
                          sd.p('2 · Whether to carry the bench Dd to the beam before run_71 is requoted on unsaturated events.', 26, '#eef0f3'),
                          sd.p('3 · Whether the det4/det7 amplitude dependence belongs in the detector model (a footprint against charge) or stays a systematic.', 26, '#eef0f3'),
                          gap=24), gap=72))
    D.slide('close', body, dark=True, short='Not ruled out')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/x17/cosmic_bench/cloud_basics/xwidth/x-width-excess.html'))
    a = ap.parse_args()
    D = sd.Deck('X-width excess', 'MX17 bench: X drift-width excess after saturation, closure toy, field tilt and amplitude tests')
    for f in (cover, setup, saturation, invert_bug, det3_fields, cut_scan, toy, arms, tilt, terciles, footprint, verdict, questions, closing):
        f(D)
    D.write(a.out, note_meta=dict(
        title='MX17 bench X-width excess: saturation, closure toy, open questions',
        summary='Most of the bench X drift-width excess was ADC saturation. det3 keeps an unexplained +200–250 µm/√cm. '
                'det4/det7 have an amplitude-dependent footprint. A closure toy clears the estimator. Open consistency questions '
                'and how to settle them from existing data.',
        tags='X17, cosmic bench, micromegas, diffusion, saturation', date='2026-10-17'),
        footer='Built by sps_beam_test_26/analysis/cloud_basics/make_xwidth_deck.py from results/*.json (FINDINGS §30–34).')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
