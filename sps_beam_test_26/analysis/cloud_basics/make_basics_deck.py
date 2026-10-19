#!/usr/bin/env python3
"""make_basics_deck.py -- slide note on the detector-model basics of 19 Oct 2026: dead strips and gain maps,
the timing-free drift gap, saturation, the X footprint width (FINDINGS §35-38), and the follow-ups.

Reads the analysis's own outputs, so a rerun after any of them changes updates the numbers:
results/gain_maps.json (gainmap_export.py), strip_gain_audit.json, xw_clean.json, unsat_width.json (§35),
gap_extent.json (§36), sat_residual_det3.json (§37), chi2_split per-event files and the bench_s0 benches (§38).
Slides for the follow-up studies are added only when their result files exist.

    make_basics_deck.py [--out ~/x17/cosmic_bench/cloud_basics/basics_oct19/mx17-basics-oct19.html]
    then: python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <out> --slug mx17-basics-oct19 --force --deploy
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
import gaps                                                    # noqa: E402

R = os.path.join(HERE, 'results')
J = lambda n: json.load(open(os.path.join(R, n)))
CB = os.path.expanduser('~/x17/cosmic_bench/cloud_basics')
GM, AUD, XC, UW, GE = J('gain_maps.json'), J('strip_gain_audit.json'), J('xw_clean.json'), J('unsat_width.json'), J('gap_extent.json')
SR = J('sat_residual_det3.json')

K3 = 'sat_det3__long_run_resist_490V_drift_1000V'
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
MAPKEY = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
AUDKEY = {'det2': 'o22_long_det2', 'det3': K3, 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
GAPKEY = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det7': 'g_det7_long', 'det6': 'g_det6_long'}
DCOL = {'det2': GREEN, 'det3': BLUE, 'det4': ORANGE, 'det6': PURPLE, 'det7': RED}
TIMING_GAP = {'det3': 27.89, 'det2': 30.62, 'det4': 25.55, 'det7': 27.51, 'det6': 27.85}    # §36 table, K = 22
DEAD_C, WEAK_C = RED, GOLD


def fmt_list(v, n=12):
    v = list(v)
    return ', '.join(str(x) for x in v[:n]) + (f' … (+{len(v) - n})' if len(v) > n else '')


# ----------------------------------------------------------------------------------------------- slides
def cover(D):
    nflag = {d: len(AUD[AUDKEY[d]]['health']['x']['dead']) + len(AUD[AUDKEY[d]]['health']['x']['weak']) for d in DETS}
    ninc = sum(GE[GAPKEY[d]]['x']['int']['n'] for d in ('det2', 'det3', 'det4', 'det7'))
    nhead = sum(AUD[AUDKEY[d]]['n_headon_x'] for d in DETS)
    tip_flag = '\n'.join(f'{d}: {nflag[d]} X channels (dead + weak)' for d in DETS)
    body = (sd.kicker('MX17 cosmic bench + SPS · detector-model basics · FINDINGS §35–38 · 19 Oct 2026') +
            sd.title('<span style="color:#eef0f3">det4’s odd widths were its dead strips; the drift gaps are '
                     f'{min(gaps.GAP_GEOM.values()):.1f}–{max(gaps.GAP_GEOM.values()):.1f} mm '
                     'without any timing; the X footprint is narrower than the stacks say</span>') +
            sd.row(sd.bignum('5', 'chambers audited strip by strip', DBLUE,
                             'Gain map at strip scale, every channel’s health, per-event flags.',
                             tip='det2 (6-22 long), det3 (6-27 long run), det4 (6-24), det6 / det7 (6-26 long)'),
                   sd.bignum(f'{sum(nflag.values())}', 'X channels flagged dead or weak', DRED,
                             f'{nflag["det4"]} of them on det4.', tip=tip_flag),
                   sd.bignum(f'{ninc:,}', 'inclined tracks measure the gap', DGREEN,
                             f'and {nhead:,} head-on tracks carry the width and health studies.',
                             tip='gap_extent.py X-view tracks with 0.2 ≤ |tan| ≤ 0.8 that pass the charge-containment cut, det2/det3/det4/det7'),
                   gap=56))
    D.slide('cover', body, dark=True, short='Summary', notes=(
        'What was done on 19 Oct, all on existing data (the chambers are gone): (§35) a strip-scale gain map and a '
        'strip-health audit of every chamber, and what det4’s dead strips did to the width studies; (§36) a drift gap '
        'measured from the charge extent of inclined tracks, with no drift time involved; (§37) whether the forward '
        'model handles saturated pulses; (§38) why the physical (RC) kernel’s X χ² is worse than production’s. '
        'The later slides answer the questions raised on reading it.\n\n'
        'Source: <code>sps_beam_test_26/analysis/cloud_basics/FINDINGS.md</code> §35–38; this note is built by '
        '<code>make_basics_deck.py</code> from <code>results/*.json</code>.'))


def setup(D):
    """Readout stack schematic: where each of today's effects lives."""
    W, H = 1000, 640
    s = []
    L, Rr = 40, 960
    # drift region
    yc, ym = 50, 300
    s.append(sd.line(L, yc, Rr, yc, INK, 5) + sd.T(L, yc - 14, 'drift cathode (kapton + Cu)', 21, INK, 'start'))
    s.append(sd.line(L, ym, Rr, ym, INK, 2, '5 4') + sd.T(Rr, ym - 10, 'mesh', 21, INK, 'end'))
    s.append(sd.line(26, yc, 26, ym, MUT, 2) + sd.T(18, (yc + ym) / 2, 'drift gap', 21, MUT, rot=-90))
    # inclined track
    s.append(sd.line(260, yc - 10, 470, ym, RED, 4))
    s.append(sd.T(250, yc + 30, 'track', 21, RED, 'end'))
    for f in np.linspace(0.08, 0.92, 7):
        x = 260 + f * 210
        y = yc - 10 + f * (ym - yc + 10)
        s.append(sd.line(x, y, x, ym, GOLD, 1.5, '3 3'))
    s.append(sd.T(480, 190, 'each depth lands on its own strip →', 20, MUT, 'start'))
    s.append(sd.T(480, 214, 'charge extent across strips = gap × tan', 20, MUT, 'start'))
    s.append(sd.T(480, 238, '(§36: no drift time needed)', 20, MUT, 'start'))
    # amplification gap + ESL strips
    ye = ym + 40
    s.append(sd.T(Rr, ym + 26, '150 µm amplification gap', 20, MUT, 'end'))
    for i in range(22):
        x = L + 10 + i * 42
        s.append(f'<rect x="{x}" y="{ye}" width="29" height="10" fill="{GREY}"/>')
    s.append(sd.T(L, ye + 34, 'resistive strips along y, 550 µm on 800 µm (charge spreads along them: Y = RC)', 20, INK, 'start'))
    yk = ye + 46
    s.append(f'<rect x="{L}" y="{yk}" width="{Rr - L}" height="16" fill="{GOLD}" fill-opacity="0.35"/>')
    s.append(sd.T(Rr, yk + 13, 'kapton + glue ≈ 70 µm', 19, INK, 'end'))
    yp = yk + 24
    for i in range(23):
        x = L + 6 + i * 41
        s.append(f'<rect x="{x}" y="{yp}" width="36" height="10" fill="#b87333"/>')
    s.append(sd.T(L, yp + 34, 'pads 0.68 mm on 0.78 mm, coupled to an X and a Y line below (FR4 layers)', 20, INK, 'start'))
    yl = yp + 50
    s.append(f'<rect x="{L}" y="{yl}" width="{Rr - L}" height="10" fill="{BLUE}" fill-opacity="0.6"/>')
    s.append(f'<rect x="{L}" y="{yl + 18}" width="{Rr - L}" height="10" fill="{GREEN}" fill-opacity="0.6"/>')
    s.append(sd.T(L, yl + 54, 'readout lines → connectors → DREAM (12-bit ADC: clips at ~3550)', 20, INK, 'start'))
    fig = sd.svg(W, H, ''.join(s), 'readout stack schematic')
    steps = [dict(label='§35 gain + dead strips', sub='amplification gain at strip scale; channels that do not read', color=ORANGE,
                  tip='strip_gain_audit.py: Q = √(q_x q_y)/path from wft events at (x, y), kernel 2 mm × 15 mm; '
                      'strip health from the ±1 neighbour of each head-on cluster'),
             dict(label='§36 drift gap', sub='mesh → cathode distance from the charge extent of inclined tracks', color=GREEN,
                  tip='gap_extent.py: per-strip window integral vs depth z = (x − x_mesh)/tan_ref, half-height edges'),
             dict(label='§37 saturation', sub='samples above ~3550 ADC are clipped', color=RED,
                  tip='sat_residual.py: residuals after the clipped run vs bright unsaturated peaks'),
             dict(label='§38 X footprint', sub='how wide one landing is across the X lines', color=BLUE,
                  tip='chi2_split.py + bench_s0: per-event χ² and resolution against σ0_x')]
    txt = sd.col(*[sd.flow([st], size=22) for st in steps],
                 sd.p('Hover the dotted terms, points and boxes for numbers and provenance.', 21, MUT), gap=18, w=600)
    D.slide('setup', sd.title('Four pieces of the detector model, from the cathode to the ADC',
                              'MX17 cross-section (not to scale) and where each of today’s studies sits') + sd.row(fig, txt, gap=56),
            short='What was studied', foot='Layer numbers from docs/detector_spec (spec.py); several are from memory and flagged open there.',
            notes='The X and Y lines both read the same pads; the resistive strips run along y, so landed charge spreads along y '
                  '(the Y view’s slow RC component) but cannot spread across strips in x except over one strip’s own 550 µm width. '
                  'Every study here uses the June 2026 bench (Ar/iso 95/5) except where SPS (det4, H4) is named.')


def gain_maps(D):
    """Five gain maps, ln g at strip scale, with the dead/weak X channels marked."""
    vmin, vmax = -1.2, 1.2
    panels = []
    for d in DETS:
        g = GM[MAPKEY[d]]
        gx, gy = np.array(g['gx']), np.array(g['gy'])
        G = np.array([[np.nan if v is None else v for v in r] for r in g['lng']])     # (ny, nx)
        BX, BY = 4, 2                                                               # 4 mm x 10 mm cells
        cells = []
        for j in range(0, len(gy) - BY + 1, BY):
            for i in range(0, len(gx) - BX + 1, BX):
                blk = G[j:j + BY, i:i + BX]
                if not np.isfinite(blk).any():
                    continue
                v = float(np.nanmean(blk))
                cells.append((gx[i] - 0.5, gx[i + BX - 1] + 0.5, gy[j] - 2.5, gy[j + BY - 1] + 2.5, v,
                              f'{d}  x {gx[i]:.0f}–{gx[i + BX - 1]:.0f}, y {gy[j]:.0f}–{gy[j + BY - 1]:.0f} mm\n'
                              f'gain × {np.exp(v):.2f} of the chamber median'))
        P = sd.Plot(330, 330, x=(0, 400, 'lin'), y=(0, 400, 'lin'), margin=(44, 8, 56, 56 if d == 'det2' else 14),
                    title=None, xlabel='x (mm)', ylabel='y (mm)' if d == 'det2' else '')
        P.xticks([(t, str(t)) for t in (0, 200, 400)])
        P.yticks([(t, str(t) if d == 'det2' else '') for t in (0, 200, 400)])
        P.cells(cells, vmin, vmax, sd.DIVERGE)
        for x in g['dead_x']:
            P.raw(sd.line(P.X(x), P.y0 - 4, P.X(x), P.y0 - 18, DEAD_C, 2.5))
        for x in g['weak_x']:
            P.raw(sd.line(P.X(x), P.y0 - 4, P.X(x), P.y0 - 14, WEAK_C, 2))
        a = AUD[AUDKEY[d]]
        P.raw(sd.T(P.x0, P.y0 - 26, d, 24, INK, 'start', 600,
                   tip=f'{d} ({MAPKEY[d]}): {g["n"]} wft events in the map; X profile p90/p10 = {a["xprofile_p90_p10"]:.1f}\n'
                       f'|Δ ln g| over 3 mm at head-on events: median {a["dlng3mm_abs_quantiles"][1]:.2f}, q90 {a["dlng3mm_abs_quantiles"][3]:.2f}'))
        panels.append(P.svg(f'{d} gain map'))
    cb = sd.colorbar(330, 90, vmin, vmax, [(-1, '×0.37'), (0, '×1'), (1, '×2.7')], 'gain / chamber median', sd.DIVERGE, horizontal=True)
    leg = sd.legend([('dead X channel (tick above the map)', DEAD_C, 'line'), ('weak X channel', WEAK_C, 'line')], 21)
    q50 = {d: AUD[AUDKEY[d]]['dlng3mm_abs_quantiles'][1] for d in DETS}
    P2 = sd.Plot(1664, 220, x=(0, 400, 'lin'), y=(-2.5, 1.5, 'lin'), margin=(16, 20, 64, 104),
                 xlabel='x (mm) — det4: ln g across X (all y)', ylabel='ln g')
    P2.xticks([(t, str(t)) for t in range(0, 401, 50)]); P2.yticks([(t, str(t)) for t in (-2, -1, 0, 1)])
    for d in DETS:
        pr = AUD[AUDKEY[d]]['xprofile']
        P2.line(pr['gx'], pr['lng'], DCOL[d] if d == 'det4' else RULE, w=3 if d == 'det4' else 1.6, markers=False,
                tip=f'{d}: ln g across X averaged over y; p90/p10 = {AUD[AUDKEY[d]]["xprofile_p90_p10"]:.1f}')
    for x in GM['g_det4']['dead_x']:
        P2.raw(sd.line(P2.X(x), P2.y0 + P2.ph, P2.X(x), P2.y0 + P2.ph - 18, DEAD_C, 2.5))
    for x in GM['g_det4']['weak_x']:
        P2.raw(sd.line(P2.X(x), P2.y0 + P2.ph, P2.X(x), P2.y0 + P2.ph - 12, WEAK_C, 2))
    P2.text(6, 1.15, f'det4 swings ×{AUD["g_det4"]["xprofile_p90_p10"]:.0f} across X (p90/p10); the others ×'
                     f'{min(AUD[AUDKEY[d]]["xprofile_p90_p10"] for d in DETS if d != "det4"):.1f}–'
                     f'{max(AUD[AUDKEY[d]]["xprofile_p90_p10"] for d in DETS if d != "det4"):.1f} (grey)', 22, INK, weight=600)
    body = (sd.title('Gain at strip scale: det4 is striped, the other four vary slowly',
                     'ln(gain) per 4 × 10 mm cell, X view’s strips run vertically; dead/weak X channels ticked above each map') +
            sd.row(*panels, gap=6) + sd.row(cb, leg, gap=60, align='center') + P2.svg('det4 profile'))
    D.slide('gainmaps', body, short='Gain maps', notes=(
        'Q = √(q_x q_y)/path length, from the wft events (x_ok and y_ok, no spark) at the fitted (x_p0, y_p0), the '
        'median of ln Q in an anisotropic Gaussian kernel (2 mm across the X strips, 15 mm along them), relative to the '
        'chamber median. Cells with too few events are blank. The map is survivor-biased where nothing reconstructs '
        '(det4’s dead bands look less deep than they are); it is used to locate gradients, not as an efficiency map.\n\n'
        'Median |Δ ln g| over 3 mm at a head-on event: ' + ', '.join(f'{d} {q50[d]:.2f}' for d in DETS) + '. '
        'det4 is the chamber with 2–38 mm dead stripes along X (det4_sps_assessment); its gain changes by ~35 % within a '
        'width-stack window at a typical event.\n\n'
        'Built by <code>gainmap_export.py</code> → <code>results/gain_maps.json</code>, from <code>strip_gain_audit.gain_map</code>.'))


def strip_health(D):
    rows = []
    H = 128
    for d in DETS:
        a = AUD[AUDKEY[d]]
        h = a['health']['x']
        rel = {int(k): v / h['rel1_median'] for k, v in a['rel1']['x'].items()}
        ho = a.get('holes', {}).get('x', {})
        hr = {int(k): v / ho['median'] for k, v in ho.get('rate', {}).items()} if ho else {}
        dead = set(h['dead']) | set(ho.get('dead', []))
        weak = set(h['weak']) - dead
        P = sd.Plot(1664, H, x=(0, 512, 'lin'), y=(0, 1.6, 'lin'), margin=(10, 12, 26 if d != 'det7' else 52, 250),
                    xlabel='X channel' if d == 'det7' else '')
        P.xticks([(t, str(t) if d == 'det7' else '') for t in range(0, 513, 64)])
        P.yticks([(0, '0'), (1, '1')])
        P.hline(0.25, DEAD_C, '4 4', 1.5); P.hline(0.6, WEAK_C, '4 4', 1.5)
        oh = [c for c in sorted(hr) if c not in dead and c not in weak]
        P.points(oh, [min(hr[c], 1.55) for c in oh], BLUE, 3, tips=[f'{d} ch {c}: inclined-track peak / neighbours {hr[c]:.2f} × median' for c in oh])
        ok = [c for c in sorted(rel) if c not in dead and c not in weak]
        P.points(ok, [min(rel[c], 1.55) for c in ok], GREY, 3.5, tips=[f'{d} ch {c}: head-on ±1 peak ratio {rel[c]:.2f} × median' for c in ok])
        for grp, col, lab in ((weak, WEAK_C, 'weak'), (dead, DEAD_C, 'dead')):
            cs = sorted(grp)
            P.points(cs, [min(rel.get(c, hr.get(c, 0.0)), 1.55) for c in cs], col, 7,
                     tips=[f'{d} ch {c}: {lab}; head-on ratio ' + (f'{rel[c]:.2f}' if c in rel else 'not judged') +
                           ', inclined ' + (f'{hr[c]:.2f}' if c in hr else 'not judged') for c in cs])
        nj = len(set(rel) | set(hr))
        P.raw(sd.T(P.x0 - 44, P.y0 + 24, d, 23, INK, 'end', 600,
                   tip=f'{d}: dead {fmt_list(sorted(dead), 30)}\nweak {fmt_list(sorted(weak), 30)}'))
        P.raw(sd.T(P.x0 - 44, P.y0 + 52, f'{len(dead)} dead · {len(weak)} weak', 19, INK, 'end'))
        P.raw(sd.T(P.x0 - 44, P.y0 + 76, f'{nj} of 512 judged', 19, MUT, 'end'))
        rows.append(P.svg(f'{d} strip health'))
    leg = sd.legend([('head-on test: ±1 neighbour of a cluster', GREY, 'dot'), ('inclined test: a strip between two firing ones', BLUE, 'dot'),
                     ('dead', DEAD_C, 'dot'), ('weak', WEAK_C, 'dot')], 20)
    D.slide('health', sd.title('Strip health: det4 and det7 have 20–40 bad X channels, det6 a connector pattern',
                               'Each X channel’s peak against its neighbours, over the chamber median; two independent tests') +
            sd.col(*rows, gap=0) + '<div style="height:14px"></div>' + leg, short='Strip health', notes=(
                '<b>Head-on test</b> (this morning): the median ratio of a channel’s peak to the cluster maximum when it sits at ±1 '
                '(unsaturated events, peak 200–3500 ADC), plus the rate of clearing 5σ next to a cluster (also on saturated events). '
                'It judges only the channels the head-on sample passes over: 31–85 of 512 on det2, det6 and det7, where most head-on '
                'events saturate.\n\n'
                '<b>Inclined test</b> (added this afternoon, <code>strip_gain_audit.holes</code>): in the inclined-track caches every '
                'strip whose two spatial neighbours both fire (unsaturated) should read about as much as their mean; the median of '
                'peak / neighbours’ mean per channel, dead below 0.25 × the chamber median. It is blind to noise (a noisy channel’s '
                'higher 5σ threshold made a first firing-rate version flag ~50 live det7 channels) and judges 190–500 channels.\n\n'
                'Where both judge they agree (det3 307; det6’s alternating 36–62 / 208 / 224 / 388–392). New from the inclined test: '
                'det7 ~20 dead X channels in 67–250, det2 69 and 269, det3 39, det4 eight more. det6’s connector pattern (even '
                'channels, pairs) and Y channels 7 and 9 dead on both det6 and det7 point to the readout cabling rather than the '
                'boards. Every flagged channel (<code>strip_gain_audit.bad_set</code>) is now masked by the gap profile and by the '
                'stacks (<code>rc_diffusion.annotate_bad</code>).'))


def det4_story(D):
    x4 = XC['g_det4@g99']          # clean = no dead/weak X strip within ±4 (no gain-gradient cut)
    P = sd.Plot(810, 560, x=(0, 4200, 'lin'), y=(0.2, 0.62, 'lin'), xlabel='peak ADC of the cluster’s centre strip',
                ylabel='r = (W₋₁ + W₊₁) / 2W₀')
    P.xticks([(t, str(t)) for t in range(0, 4201, 1000)]); P.yticks([(t, f'{t:.1f}') for t in (0.2, 0.3, 0.4, 0.5, 0.6)])
    P.band([0, 4200], [0.44, 0.44], [0.47, 0.47], GOLD, 0.25, tip='linear toy (xw_sim): r 0.44–0.47 at every amplitude')
    P.text(120, 0.455, 'toy, uniform gain', 20, GOLD)
    for lab, sub, col in (('clean windows', 'clean', GREEN), ('with a dead/weak strip in ±4', 'rest', RED)):
        rr = [q for q in x4[sub]['ratio'] if 'med' in q and q['hi'] <= 3500]
        xs = [0.5 * (q['lo'] + q['hi']) for q in rr]
        P.line(xs, [q['med'] for q in rr], col, tips=[f'det4 {lab}, {q["lo"]}–{q["hi"]} ADC: r {q["med"]:.3f} ± {q["err"]:.3f} (n {q["n"]})' for q in rr])
        for x, q in zip(xs, rr):
            P.raw(sd.line(P.X(x), P.Y(q['med'] - q['err']), P.X(x), P.Y(q['med'] + q['err']), col, 2))
    P.text(200, 0.6, 'clean det4 windows sit on the toy', 22, INK, weight=600)
    cuts = [3500, 2500, 2000, 1500]
    def dt(k):
        r = UW.get(k)
        return (r['unsat']['gapcorr']['DT'], r['unsat']['gapcorr']['DT_err'], r['unsat']['n']) if r else None
    Q = sd.Plot(810, 520, x=(1300, 3700, 'lin'), y=(200, 480, 'lin'), xlabel='saturation cut (window max, ADC)',
                ylabel='det4 X D_T (µm/√cm, own gap)')
    Q.xticks([(t, str(t)) for t in cuts]); Q.yticks([(t, str(t)) for t in range(200, 481, 50)])
    for lab, suf, col in (('all events', '', GREY), ('flagged strips masked', '@mask', PURPLE)):
        pts = [(c, dt('g_det4' + ('' if c == 3500 else f'@sat{c}') + suf)) for c in cuts]
        pts = [(c, v) for c, v in pts if v]
        Q.line([c for c, _ in pts], [v[0] for _, v in pts], col,
               tips=[f'det4 {lab}, cut {c}: D_T {v[0]:.0f} ± {v[1]:.0f} (n {v[2]})' for c, v in pts])
    xf = XC.get('g_det4@g99@s00.35')            # sigma_0 pinned at 0.35 mm (and T at T0): no sigma_0-D_T trade
    sc = np.sqrt(30.0 / gaps.gap('det4'))
    for sub, col, lab in (('clean', GREEN, 'clean windows, σ0 fixed 0.35'), ('rest', RED, 'bad strip in ±4, σ0 fixed 0.35')):
        if not xf:
            break
        cl = [(c, xf[sub].get(f'unsat{c}')) for c in cuts]
        cl = [(c, m) for c, m in cl if m and 'DT' in m]
        Q.line([c for c, _ in cl], [m['DT'] * sc for _, m in cl], col,
               tips=[f'det4 {lab}, cut {c}: D_T {m["DT"] * sc:.0f} ± {m["DT_err"] * sc:.0f} (n {m["n"]})' for c, m in cl])
        for c, m in cl:
            Q.raw(sd.line(Q.X(c), Q.Y((m['DT'] - m['DT_err']) * sc), Q.X(c), Q.Y((m['DT'] + m['DT_err']) * sc), col, 2))
    Q.text(1380, 465, 'fixed σ0: flat in the cut, split by strip health', 22, INK, weight=600)
    leg = sd.legend([('all windows, σ0 free', GREY, 'line'), ('flagged strips masked, σ0 free', PURPLE, 'line'),
                     ('clean, σ0 fixed', GREEN, 'line'), ('bad strip in ±4, σ0 fixed', RED, 'line')], 21)
    D.slide('det4', sd.title('On det4 the amplitude cut was a position cut onto dead strips',
                             'Left: neighbour/centre ratio by amplitude, clean vs touched windows. Right: D_T against the saturation cut') +
            sd.row(P.svg('det4 r by amplitude'), sd.col(Q.svg('det4 DT vs cut'), leg, gap=8), gap=44), short='det4 dead strips',
            foot='xw_clean.py → results/xw_clean.json; unsat_width.py --mask → unsat_width.json (keys …@mask)', notes=(
                'det4 head-on events with window maximum below 1000 ADC sit at local gain e^−0.66 and 71 % of them have a dead or '
                'weak strip within ±4; above 3500 they sit at e^+1.1 (48 %). Tightening the saturation cut keeps the faint ones, '
                'which are the ones next to dead strips, so the cut walked the sample onto the bad strips.\n\n'
                'The model-free ratio r is 0.43–0.48 at every amplitude on clean windows (the uniform-gain toy gives 0.44–0.47), '
                'and 0.27–0.37 when a dead/weak strip sits in ±2. With σ0 fixed at 0.35 mm the clean-window D_T is flat in the cut '
                '(413 / 409 / 396 / 420 at 3500 / 2500 / 2000 / 1500 ADC, GAP-30 frame; reproduced and stored this afternoon as '
                '<code>g_det4@g99@s00.35</code>, XW_MIN_EV=60), and so is the D_T of the windows with a bad strip, ~130 lower '
                '(282 / 277 / 266 / 278). The all-event cut dependence is the mix of the two changing with the cut. With σ0 free, '
                'σ0 rises 0.43 → 0.51 as D_T falls 354 → 280 (clean): the two trade within errors.\n\n'
                'Masking only the flagged strips (NaN in the stack) is not enough: det4 widens to 378 ± 23 but stays cut-dependent, '
                'because its gain falls over the band edges around the dead strips. det4 widths need the clean-window selection.\n\n'
                '<b>What had to be done with the dead strips</b>: (1) flag them per chamber from the data themselves '
                '(<code>strip_gain_audit.py</code>, per-event flags in <code>gainflags_&lt;key&gt;.pkl</code>); (2) mask them in every '
                'stack (<code>rc_diffusion.annotate_bad</code>, <code>unsat_width.py --mask</code>) and out of the gap profile’s charge '
                'and exposure; (3) for det4, keep only windows with no flagged strip in ±4 and a small gain gradient. Still to do: '
                'the closure toy with the same masks, and the SPS run_71 det4 stacks (beam channel numbering).'))


def gap_method(D):
    zg = GE['sat_det3']['zgrid']
    z = zg[0] + zg[1] * np.arange(zg[2])
    P = sd.Plot(1000, 600, x=(-6, 36, 'lin'), y=(-0.05, 1.2, 'lin'), xlabel='depth z = (x − x_mesh) / tan_ref (mm)',
                ylabel='charge per mm of depth (normalised)')
    P.xticks([(t, str(t)) for t in range(-5, 36, 5)]); P.yticks([(t, f'{t:.1f}') for t in (0, 0.5, 1.0)])
    P.hline(0, RULE, None, 1)
    for d in ('det3', 'det2', 'det4', 'det7'):
        r = GE[GAPKEY[d]]['x']['int']
        h = r['half']
        prof = [None if v is None else v / h['plateau'] for v in r['profile']]
        m = [(zz, v) for zz, v in zip(z, prof) if v is not None and -6 <= zz <= 36]
        P.line([q[0] for q in m], [q[1] for q in m], DCOL[d], w=3, markers=False,
               tip=f'{d}: {r["n"]} tracks; half-height edges {h["z0"]:.2f} and {h["z1"]:.2f} mm → gap {h["gap"]:.2f} ± {h["gap_err"]:.2f}')
        for e in (h['z0'], h['z1']):
            P.points([e], [0.5], DCOL[d], 7, marker='open', tips=[f'{d} half-height edge at {e:.2f} mm'])
    P.hline(0.5, MUT, '6 6', 1.5, label='half height', anchor='start')
    P.text(-4.5, 1.12, 'one step per chamber; its width is the gap', 22, INK, weight=600)
    leg = sd.legend([(f'{d}  {GE[GAPKEY[d]]["x"]["int"]["half"]["gap"]:.2f} mm', DCOL[d], 'line') for d in ('det3', 'det2', 'det4', 'det7')], 22)
    steps = [dict(label='Inclined tracks', sub='0.2 ≤ |tan| ≤ 0.8 from the M3 reference', color=RED),
             dict(label='Each strip = a depth', sub='window integral spread over its z interval (pitch / |tan|)', color=GOLD),
             dict(label='Stack in z', sub='dead/weak strips out; saturated events out; ≥ 85 % of the charge in the column', color=GREEN),
             dict(label='Half-height edges', sub='blur moves neither; x_mesh offset moves both', color=BLUE)]
    txt = sd.col(leg, *[sd.flow([st], size=21) for st in steps], gap=14, w=600)
    D.slide('gapmethod', sd.title('The drift gap without a clock: where the charge of an inclined track starts and stops',
                                  'Per-strip charge against depth, stacked over every inclined bench track (X view, integral, normalised to plateau)') +
            sd.row(P.svg('gap profiles'), txt, gap=56), short='Gap: method',
            foot='gap_extent.py → results/gap_extent.json; closure toy gap_extent_toy.py', notes=(
                'Every earlier gap was a drift time times a velocity (T_end × v), carrying a ±1 mm kernel systematic and a strong '
                'amplitude dependence (det4 +9 mm). Here no time enters: the X strip at x collects the ionisation from depth '
                'z = (x − x_mesh)/tan_ref over a slice pitch/|tan| thick. The half-height points of a smeared step stay where the '
                'step is for any symmetric blur (footprint, diffusion, reference error, the strip’s own z interval, and a mixture of '
                'them over the tan range). An x_mesh error moves both edges together and leaves the gap.\n\n'
                '<b>Closure</b> (toy with Fischle clusters, Polya gain, Magboltz-size diffusion, the bundle template at the data’s '
                'peak sample, 60 ns sampling, noise, reference smear): G = 30 returns 30.0–30.1 at v = 26–38 µm/ns; G = 27.9 returns '
                '27.75. The peak-sample variant reads +0.6 to +2.3 mm high (blur-dependent) and is not used. Normalising each event '
                'to its own plateau instead of its total let a few mis-referenced events dominate (an early det3 pass read 30.2; '
                'superseded).\n\nThe M3 slope scale is not included: the gap scales as 1/scale.'))


def gap_result(D):
    order = ['det3', 'det2', 'det4', 'det7', 'det6']
    P = sd.Plot(1000, 600, x=(-0.6, 4.6, 'lin'), y=(24, 34, 'lin'), ylabel='drift gap (mm)', xlabel='')
    P.xticks([(i, d) for i, d in enumerate(order)]); P.yticks([(t, str(t)) for t in range(24, 35, 2)])
    P.band([-0.6, 4.6], [28, 28], [32, 32], GREY, 0.12, tip='mechanical 30 mm with “a couple of mm” of build tolerance (Dylan, 19 Oct)')
    P.hline(30, MUT, '8 6', 2, label='mechanical 30', anchor='start')
    for i, d in enumerate(order):
        if d == 'det6':
            lo = GE['g_det6_long']['x']['int']['half']['gap']
            P.raw(sd.arrow(P.X(i), P.Y(lo), P.X(i), P.Y(lo + 2.5), PURPLE, 3))
            P.points([i], [lo], PURPLE, 9, tips=[f'det6: lower limit ≥ {lo:.1f} mm — its deep charge runs past the 32-sample window'])
            P.text(i + 0.12, lo - 0.6, 'lower limit', 19, PURPLE)
        else:
            g, e = gaps.GAP_GEOM[d], gaps.GAP_GEOM_ERR[d]
            P.raw(sd.line(P.X(i), P.Y(g - e), P.X(i), P.Y(g + e), DCOL[d], 3))
            P.points([i], [g], DCOL[d], 11, tips=[f'{d}: {g:.2f} ± {e:.2f} mm (gaps.py: mean of the two selections)'])
            for suf, dx, lab in (('', -0.13, 'unsaturated events only'), ('@satmask', 0.13 - 0.36, 'all events, clipped strips masked')):
                ks = [k + suf for k in (['g_det7_long', 'g_det7_longer'] if d == 'det7' else [GAPKEY[d]])]
                vs = [(GE[k]['x']['int']['half']['gap'], GE[k]['x']['int']['half']['gap_err'], GE[k]['x']['int']['n']) for k in ks if k in GE]
                for j, (v, ve, n) in enumerate(vs):
                    xx = i + (dx if suf == '' else 0.13) + 0.06 * j
                    P.raw(sd.line(P.X(xx), P.Y(v - ve), P.X(xx), P.Y(v + ve), DCOL[d], 1.5))
                    P.points([xx], [v], DCOL[d], 6, marker='open' if suf else 'circle',
                             tips=[f'{d} {ks[j].split("@")[0]}, {lab}: {v:.2f} ± {ve:.2f} mm ({n} tracks)'])
        P.points([i + 0.3], [TIMING_GAP[d]], GREY, 8, marker='open',
                 tips=[f'{d}: old timing-based gap (T_end × v, K = 22) {TIMING_GAP[d]:.2f} mm'])
    for hv, key, dx in ((900, 'sat_det3__drift_scan_resist_490V_drift_900V', -0.22), (1100, 'sat_det3__drift_scan_resist_490V_drift_1100V', -0.32)):
        r = GE[key]['x']['int']['half'] if hv == 1100 else None
        val, err = (28.04, 0.71) if hv == 1100 else (27.5, 1.1)
        P.points([dx], [val], BLUE, 7, marker='open', tips=[f'det3 drift scan {hv} V: {val} ± {err} mm (≈200 tracks; {"half-height" if hv == 1100 else "erf fit"})'])
    P.text(1.4, 33.3, 'all within ~2 mm of the mechanical 30', 22, INK, weight=600)
    leg = sd.legend([('timing-free gap (big: adopted)', INK, 'dot'), ('small: unsaturated only / open: all, clipped strips masked', INK, 'open'),
                     ('old timing gap (T_end × v)', GREY, 'open'), ('det3 at 900 / 1100 V drift', BLUE, 'open')], 21)
    rows = [[d, f'{gaps.GAP_GEOM[d]:.2f} ± {gaps.GAP_GEOM_ERR[d]:.2f}', f'{TIMING_GAP[d]:.2f}'] for d in ('det3', 'det2', 'det4', 'det7')]
    tab = sd.table(['chamber', 'gap (mm)', 'timing'], rows + [['det6', '≥ 27.6', '27.85']], 22)
    G = gaps.GAP_GEOM
    D.slide('gapresult', sd.title(f'det3 {G["det3"]:.1f}, det2 {G["det2"]:.1f}, det4 {G["det4"]:.1f}, det7 {G["det7"]:.1f} mm: the short det3 gap is real',
                                  'Timing-free gaps per chamber against the mechanical 30 mm and the old timing-based values') +
            sd.row(P.svg('gaps'), sd.col(leg, tab, gap=28, w=600), gap=56), short='Gap: result',
            foot='Errors: bootstrap ⊕ half the selection difference; the M3 slope scale (gap × (1 + scale error)) is not included. Table: cloud_basics/gaps.py',
            notes=(
                '<b>Two selections</b> (afternoon, §39): the morning’s numbers dropped every event with a saturated sample, which '
                'left det2 148 and det7 139 tracks. <code>gap_extent.py --satmask</code> keeps those events and masks only their clipped '
                'strips (out of charge and exposure, normalisation scaled by the good fraction of the column). det3 reads the same '
                '(28.10 → 28.06); det2 and det4 read 0.6 and 0.4 mm shorter, det7 1.2 mm longer. The adopted value is the mean, with '
                'half the difference added to the error. The morning’s det7 30.1 also lost 20 newly flagged dead strips (28.2 on its '
                'long run with them masked).\n\n'
                'det3’s short gap agrees with its timing value (28.1 ± 0.2 against 27.9) now that one of the two does not depend on '
                'any kernel. det2 − det3 = +3.4 mm, as the timing contrast said (+2.8 to +3.4). det4’s timing value (25.6) was the '
                'amplitude/dead-strip problem of §35.\n\n'
                'det2’s +tan and −tan halves differ by 2.4 mm, which a drift-field or reference tilt of tan ≈ 0.012 does; the mean '
                'is first-order free of it. det3’s halves differ by 0.9 mm (δ ≈ 0.005).\n\n'
                'det6 cannot be measured this way: its deep charge runs past the 32-sample window (the deep strips’ last sample is at '
                '40–86 % of their peak), so its value is a lower limit and gaps.py keeps 30 ± 2. det3 on 6-22 (8.9 µm/ns that night) '
                'is excluded for the same reason, and so is det3 at 700 V drift.\n\n'
                'Effect on the width comparison (no refit): det3 unchanged (+230 µm/√cm at 347 V/cm, 7.8 sd); det7 now sits on '
                'Magboltz (+88, 0.3 sd); det2 is two-valued (0.09 % water −228 or 0.65 % +188, ~1 sd each); det4 +62 (0.3 sd).'))


def saturation(D):
    K = SR['K']
    P = sd.Plot(810, 540, x=(-6.5, 12.5, 'lin'), y=(-0.04, 0.09, 'lin'), xlabel='samples after the clipped run (sat) / after the peak (bright)',
                ylabel='(data − model) / model peak')
    P.xticks([(t, str(t)) for t in range(-6, 13, 2)]); P.yticks([(t, f'{t:+.2f}' if t else '0') for t in (-0.04, 0, 0.04, 0.08)])
    P.hline(0, MUT, '6 6', 1.5)
    for lab, cls, col in (('saturated strip', 'sat', RED), ('bright unsaturated peak strip (> 1500 ADC)', 'unsat_bright', BLUE)):
        pts = [(q['k'], q['rel']) for q in SR['res']['x'][cls]['0'] if q['rel'] is not None]
        P.line([p[0] for p in pts], [p[1] for p in pts], col,
               tips=[f'{lab}, k = {q["k"]}: {q["rel"]:+.3f} of the model peak (n {q["n"]})' for q in SR['res']['x'][cls]['0'] if q['rel'] is not None])
    P.vline(0, GREY, '4 4', 1.5)
    P.text(1.6, 0.083, 'same shape with or without clipping', 22, INK, weight=600)
    Q = sd.Plot(810, 540, x=(-6.5, 12.5, 'lin'), y=(-0.04, 0.09, 'lin'), xlabel='samples (same alignment)', ylabel='')
    Q.xticks([(t, str(t)) for t in range(-6, 13, 2)]); Q.yticks([(t, f'{t:+.2f}' if t else '0') for t in (-0.04, 0, 0.04, 0.08)])
    Q.hline(0, MUT, '6 6', 1.5)
    for cls, col, dash in (('sat', RED, None), ('unsat_bright', BLUE, '8 5')):
        for o in ('-1', '1'):
            pts = [(q['k'], q['rel']) for q in SR['res']['x'][cls][o] if q['rel'] is not None]
            Q.line([p[0] for p in pts], [p[1] for p in pts], col, w=2.5, dash=dash, markers=False,
                   tip=f'{"saturated" if cls == "sat" else "bright unsaturated"} event, neighbour at {int(o):+d}')
    Q.text(-6, 0.083, '±1 neighbours: no pick-up after the clip', 22, INK, weight=600)
    leg = sd.legend([('saturated', RED, 'line'), ('bright, unsaturated', BLUE, 'line')], 21)
    D.slide('sat', sd.title('Saturated pulses: the censored fit is as good as on bright unclipped ones',
                            'det3 X residuals of the rcms25 forward model, at the clipped strip (left) and its ±1 neighbours (right)') +
            sd.row(P.svg('sat residual'), sd.col(Q.svg('neighbours'), leg, gap=6), gap=44), short='Saturation',
            foot='sat_residual.py → results/sat_residual_det3.json; production censors samples ≥ 3550 ADC with a one-sided penalty',
            notes=(
                'On det3 saturated events carry 4–6× the χ²/dof of unsaturated ones in every kernel (X production 186 vs 32). '
                'That is brightness, not clipping: the residual just before and after the clipped run (+0.073, +0.059 at k = +1, '
                '+0.019 at +2, −0.014 at +5) is the same as around the peak of bright unsaturated strips (+0.061 / +0.055 / +0.022 / '
                '−0.029): a template mismatch that grows with amplitude (MODEL_FRAC is the existing handle). No recovery tail, '
                'baseline shift or neighbour pick-up shows after the clip.\n\n'
                'So per event the forward model uses saturated pulses as well as the data allow. The stack studies are what cannot: '
                'a stack has no per-sample censoring, so they drop saturated events — 70–85 % of det2/det6/det7 — and on a chamber '
                'with gain structure that selects low-gain regions (§35). The answer is a per-event censored width fit, below.'))


def residual_by_offset(n_boot=300):
    """Per-sample mean residual (noise units) by strip offset, unsaturated head-on det3 events, with event bootstrap."""
    p = os.path.join(CB, 'chi2_split', 'chi2_split_det3_rcm_rcms25_rcms30_rcms35_events.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    rng = np.random.default_rng(1)
    for arm in ('production', 'rcm', 'rcms25'):
        S, N = [], []
        for e, o in d[arm].items():
            q = o.get('x')
            if not q or q['nsat'] > 0 or abs(q['tr']) >= 0.05:
                continue
            s = np.array([q['by'].get(str(k), [0, 0, 0])[1] for k in range(-4, 5)], float)
            n = np.array([q['by'].get(str(k), [0, 0, 0])[2] for k in range(-4, 5)], float)
            S.append(s); N.append(n)
        S, N = np.array(S), np.array(N)
        m = S.sum(0) / np.maximum(N.sum(0), 1)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(S), len(S))
            bs.append(S[i].sum(0) / np.maximum(N[i].sum(0), 1))
        out[arm] = dict(m=m.tolist(), e=np.std(bs, 0).tolist(), n=len(S))
    return out


def footprint(D):
    rb = residual_by_offset()
    ss = [0.20, 0.25, 0.30]
    P = sd.Plot(810, 560, x=(0.17, 0.62, 'lin'), y=(0.9, 1.9, 'lin'), xlabel='σ0_x in the RC model (mm)',
                ylabel='X χ²/dof ÷ production')
    P.xticks([(t, f'{t:.1f}') for t in (0.2, 0.3, 0.4, 0.5, 0.6)]); P.yticks([(t, f'{t:.1f}') for t in (1.0, 1.2, 1.4, 1.6, 1.8)])
    P.hline(1.0, INK, '8 6', 2, label='production kernel', anchor='end')
    for d in DETS:
        f = os.path.join(CB, 'bench_s0', f'bench_{d}_s0.json')
        if not os.path.exists(f):
            continue
        B = json.load(open(f))
        c0 = B['production']['blind']['x']['chi2dof']
        h = B[f'{d}_rcm']['hyper']
        s_rcm = float(h.get('sigma_p0_x', h['sigma_p0']))
        xs = [s_rcm] + ss
        ys = [B[f'{d}_rcm']['blind']['x']['chi2dof'] / c0] + [B[f'{d}_rcms{int(s * 100)}']['blind']['x']['chi2dof'] / c0 for s in ss]
        o = np.argsort(xs)
        P.line([xs[i] for i in o], [ys[i] for i in o], DCOL[d],
               tips=[f'{d}, σ0_x {xs[i]:.2f}: χ²/dof {ys[i] * c0:.0f} vs production {c0:.0f}' + (' (rcm: σ0 from the RC fit)' if i == 0 else '') for i in o])
    P.text(0.18, 1.83, 'narrower X footprint → production’s χ²', 22, INK, weight=600)
    leg = sd.legend([(d, DCOL[d], 'line') for d in DETS], 21)
    Q = sd.Plot(810, 560, x=(-4.6, 4.6, 'lin'), y=(-5.5, 5.5, 'lin'), xlabel='strip offset from the cluster centre',
                ylabel='mean residual per sample (σ_noise)')
    Q.xticks([(t, f'{t:+d}' if t else '0') for t in range(-4, 5)]); Q.yticks([(t, f'{t:+d}' if t else '0') for t in (-4, -2, 0, 2, 4)])
    Q.hline(0, MUT, '6 6', 1.5)
    if rb:
        offs = list(range(-4, 5))
        for arm, col, lab, dx in (('rcm', GREY, 'RC model, σ0_x 0.41', -0.12), ('rcms25', BLUE, 'RC model, σ0_x 0.25', 0.0), ('production', ORANGE, 'production', 0.12)):
            r = rb[arm]
            Q.line([o + dx for o in offs], r['m'], col,
                   tips=[f'{lab}, offset {o:+d}: {m:+.2f} ± {e:.2f} (n {r["n"]} unsaturated head-on events)' for o, m, e in zip(offs, r['m'], r['e'])])
            for o, m, e in zip(offs, r['m'], r['e']):
                Q.raw(sd.line(Q.X(o + dx), Q.Y(m - e), Q.X(o + dx), Q.Y(m + e), col, 2))
        Q.text(-4.4, 4.9, 'wide footprint: centre under, ±1 over', 22, INK, weight=600)
    leg2 = sd.legend([('RC model as fitted (σ0_x 0.41)', GREY, 'line'), ('σ0_x 0.25', BLUE, 'line'), ('production', ORANGE, 'line')], 21)
    D.slide('footprint', sd.title('The RC model’s worse X χ² is a footprint that is too wide',
                                  'Left: held-out X χ²/dof against σ0_x, five chambers. Right: det3 residual by strip offset (head-on, unsaturated)') +
            sd.row(sd.col(P.svg('chi2 vs s0'), leg, gap=4), sd.col(Q.svg('residual by offset'), leg2, gap=4), gap=44), short='X footprint',
            foot='chi2_split.py, run_bench_s0.sh → ~/x17/cosmic_bench/cloud_basics/bench_s0/; blind start, 2000 held-out events per chamber',
            notes=(
                'The RC kernel’s X χ²/dof is worse than production’s on every chamber (det3 58 → 79). The deficit is uniform '
                '(×1.2–1.4) on unsaturated, saturated, head-on, inclined and faint events, so it is not saturation. By strip offset '
                'the RC model under-predicts the centre and over-predicts ±1: its footprint is too wide. Lorentzian tails make it worse.\n\n'
                'Resolution (scale-corrected s68 against production, blind start): narrowing σ0_x never costs overall X resolution '
                '(det3 +0.005 → +0.013 ± 0.019°), moves every angle slope toward 1 by 0.007–0.030, and removes most of det4’s X deficit '
                '(+0.164 → +0.061 ± 0.044°). Head-on, the wide footprint is still worth something on det7 (−0.16 vs −0.06) and det3 '
                '(+0.01 vs +0.04).\n\n'
                'The per-event fit wants σ0_x ≈ 0.20–0.25 mm where the head-on stacks measure 0.35–0.43. The next slides take up '
                'whether that makes physical sense, and the left-over asymmetry.'))


def closing(D):
    items = [
        ('det3’s D_T excess', '+160 to +260 µm/√cm at every field, at the gap both timing and charge extent agree on. '
         'Not the gap, not the field, not saturation. Unexplained.'),
        ('det6', 'its gap is not measurable (window), and its v implies a different water from det7’s in the same run.'),
        ('The beam gap of det4', 'not testable: the ladder v is estimator-dependent by 25 %, and the beam water’s low-field constraint inherits it.'),
        ('The X asymmetry', 'real in the data; the mechanism is a hypothesis. Electronics crosstalk is not excluded by these tests.'),
        ('Charge of saturated events', '6–12 % low until the pulse template depends on amplitude.'),
        ('The M3 slope scale', 'every timing-free gap scales as 1/scale; not in the errors.')]
    body = (sd.kicker('What this does not rule out') + sd.title('<span style="color:#eef0f3">Open after 19 Oct</span>') +
            sd.col(*[sd.p(f'<b style="color:#eef0f3">{a}.</b> <span style="color:#a3abb9">{b}</span>', 25) for a, b in items], gap=18))
    D.slide('closing', body, dark=True, short='Open')


# ---------------------------------------------------------------------------------------- follow-up slides
def Jopt(n):
    p = os.path.join(R, n)
    return json.load(open(p)) if os.path.exists(p) else None


def vdrift_scan(D):
    V = Jopt('vdrift_consistency.json')
    if not V:
        return
    from gasmodel import GasGrid
    G = GasGrid('bench', 'air_hs')
    f = V['scan_fixed']
    P = sd.Plot(900, 600, x=(220, 410, 'lin'), y=(20, 44, 'lin'), xlabel='drift field E = V / gap (V/cm)',
                ylabel='drift velocity v = gap / T (µm/ns)')
    P.xticks([(t, str(t)) for t in (240, 280, 320, 360, 400)]); P.yticks([(t, str(t)) for t in range(20, 45, 4)])
    Es = list(np.arange(220, 411, 5.0))
    for w, dash in ((0.6, '6 5'), (0.8, '6 5'), (1.1, '6 5')):
        P.line(Es, [G(w, 0, e)['v'] for e in Es], GREY, w=1.6, dash=dash, markers=False, tip=f'Magboltz Ar/iso 95/5 + {w} % water')
        P.text(226, G(w, 0, 226)['v'] + 0.4, f'{w} %', 18, GREY, 'start')
    P.line(Es, [G(f['water'], 0, e)['v'] for e in Es], BLUE, w=3, markers=False,
           tip=f'best single water {f["water"]:.3f} % [{f["water_lo"]:.3f}, {f["water_hi"]:.3f}], χ² {f["chi2"]:.1f} / {f["ndf"]}')
    for p in f['points']:
        P.raw(sd.line(P.X(p['E']), P.Y(p['v'] - p['v_err']), P.X(p['E']), P.Y(p['v'] + p['v_err']), INK, 2))
    P.points([p['E'] for p in f['points']], [p['v'] for p in f['points']], INK, 8,
             tips=[f'{p["label"]}: E {p["E"]:.0f} V/cm, v {p["v"]:.2f} ± {p["v_err"]:.2f} (Magboltz {p["v_mb"]:.2f})' for p in f['points']])
    P.text(226, 42.6, f'one water ({f["water"]:.2f} %) fits all six: χ² {f["chi2"]:.1f}/{f["ndf"]}', 22, INK, weight=600)
    pr = V['scan_profile']; sg = V['scan_gap']
    Q = sd.Plot(700, 600, x=(25, 33, 'lin'), y=(0, 40, 'lin'), xlabel='gap assumed for det3 (mm)', ylabel='χ² of the six v points')
    Q.xticks([(t, str(t)) for t in range(25, 34)]); Q.yticks([(t, str(t)) for t in range(0, 41, 10)])
    g3, e3 = gaps.GAP_GEOM['det3'], gaps.GAP_GEOM_ERR['det3']
    Q.band([g3 - e3, g3 + e3], [0, 0], [40, 40], GREEN, 0.25, tip=f'timing-free gap {g3:.2f} ± {e3:.2f} mm (gap_extent.py)')
    Q.line([q['gap'] for q in pr], [min(q['chi2'], 40) for q in pr], BLUE, markers=False,
           tips=None, tip='χ²(gap), water refitted at each gap')
    Q.points([sg['best']], [sg['chi2']], BLUE, 9, tips=[f'minimum {sg["best"]:.2f} mm, Δχ² = 1 at [{sg["lo"]:.2f}, {sg["hi"]:.2f}]'])
    Q.text(25.3, 37, 'timing + Magboltz alone:', 22, INK, weight=600)
    Q.text(25.3, 33.5, f'{sg["best"]:.2f} [{sg["lo"]:.2f}, {sg["hi"]:.2f}] mm', 22, BLUE, weight=600)
    Q.text(g3 + 0.3, 3, f'timing-free {g3:.2f}', 20, GREEN)
    D.slide('vscan', sd.title('The det3 drift scan meshes: one water fits every field, at the gap the charge extent gives',
                              'det3 6-27, v = gap / T of the unsaturated head-on stacks against Magboltz; right: the gap the scan alone prefers') +
            sd.row(P.svg('det3 v(E)'), Q.svg('gap profile'), gap=44), short='Drift scan',
            foot='vdrift_consistency.py → results/vdrift_consistency.json; Magboltz air_hs grid (Ar/iso 95/5 + water); E = V / gap (was V / 2.88 cm for every chamber)',
            notes=(
                'T is the charge column’s arrival length from the stack fit (x_width.fit_x, T free), with its bootstrap error and '
                f'a {100 * V["v_sys"]:.1f} % systematic per stack. The six points are the 700 / 900 / 1100 V scan sub-runs, the 1000 V '
                'long run and the two 1000 V amplification-scan points (mesh 445 / 465 V), which must and do agree with each other. '
                'The 500 V and lower sub-runs are window-limited (no drift end inside 1.9 µs) and carry no v.\n\n'
                'Water and gap are not degenerate here because v(E) in Ar/iso changes shape with water: at ~1 % it still rises '
                'steeply at 250 V/cm and flattens by 390. Profiling the gap (water refitted at each) gives the minimum at '
                f'{sg["best"]:.2f} mm, against the timing-free {g3:.2f} ± {e3:.2f}: two estimates with nothing in common but the data.\n\n'
                '<b>The beam cannot do the same for det4.</b> The run_63 ladder v (7.77 / 5.83 / 4.04 µm/ns) is a ratio of loss rates '
                'per ns and per mm on strip peak times; the peak-time ladder itself reads 10.0 µm/ns at the same field '
                '(angled_kernel), the 20–30 % compression of RECONSTRUCTION_BASIS. With either, ladder + Magboltz + the run_71 drift end '
                'returns a gap independent of the one assumed (28.6 mm with 7.77), so the beam carries no gap information at the 1 mm '
                'level. It also means the beam composition fit’s low-field v constraint is weaker than its quoted 3 %.'))


def vdrift_single(D):
    V = Jopt('vdrift_consistency.json')
    if not V:
        return
    keep = [('det2', 'o22_long_det2'), ('det4', 'g_det4'), ('det7', 'g_det7_long'), ('det3', 'sat_det3__long_run_resist_490V_drift_1000V')]
    rows = {r['key']: r for r in V['single']}
    panels = []
    for det, key in keep:
        r = rows.get(key)
        if not r:
            continue
        sc = r['scan']
        P = sd.Plot(405, 470, x=(26, 34, 'lin'), y=(250, 480, 'lin'), margin=(40, 10, 70, 70 if det == 'det2' else 30),
                    xlabel='assumed gap (mm)', ylabel='D_T (µm/√cm)' if det == 'det2' else '')
        P.xticks([(t, str(t)) for t in (26, 28, 30, 32, 34)]); P.yticks([(t, str(t) if det == 'det2' else '') for t in range(250, 481, 50)])
        g, e = gaps.GAP_GEOM[det], gaps.GAP_GEOM_ERR[det]
        P.band([g - e, g + e], [250, 250], [480, 480], GREEN, 0.25, tip=f'{det} timing-free gap {g:.2f} ± {e:.2f}')
        P.line([q['gap'] for q in sc], [q['DT'] for q in sc], INK, markers=False, tip=f'{det}: measured D_T rescaled to the assumed gap (× √(30/gap))')
        de = r['DT_err']
        P.band([q['gap'] for q in sc], [q['DT'] - de for q in sc], [q['DT'] + de for q in sc], INK, 0.1)
        pts = [(q['gap'], dt, w) for q in sc for w, dt in q['branches'] if 250 <= dt <= 480]
        P.points([p[0] for p in pts], [p[1] for p in pts], BLUE, 4,
                 tips=[f'{det}, gap {p[0]:.2f}: water {p[2]:.2f} % from v, Magboltz D_T {p[1]:.0f}' for p in pts])
        for x in r['dt_gap']:
            P.points([x], [float(np.interp(x, [q['gap'] for q in sc], [q['DT'] for q in sc]))], BLUE, 8, marker='open',
                     tips=[f'{det}: D_T matches Magboltz at a gap of {x:.1f} mm'])
        P.raw(sd.T(P.x0 + 8, P.y0 + 24, det, 24, INK, 'start', 600))
        panels.append(P.svg(f'{det} DT vs gap'))
    leg = sd.legend([('measured D_T (±1σ band)', INK, 'line'), ('Magboltz at the v-implied water (every branch)', BLUE, 'dot'),
                     ('timing-free gap', GREEN, 'box'), ('crossing', BLUE, 'open')], 21)
    D.slide('vsingle', sd.title('Every other chamber: the gap where D_T matches Magboltz is the timing-free gap, except det3',
                                'One field each: v = gap / T gives the water, Magboltz gives D_T at that water; both move with the assumed gap') +
            sd.row(*panels, gap=6) + leg, short='Single fields', foot='vdrift_consistency.py; D_T from unsat_width.py (unsaturated head-on stacks), rescaled × √(30/gap)',
            notes=(
                'A chamber with one field has one free gas parameter (water) per run, so v alone cannot test the gap. Diffusion can: '
                'the measured D_T falls as √(30/gap) with the assumed gap, while the Magboltz D_T at the water that v implies rises '
                '(a longer gap means a faster v, so a drier gas, so more diffusion). Where they cross is a gap from timing and '
                'diffusion together. det2 crosses at ' + ', '.join(f'{x:.1f}' for x in rows['o22_long_det2']['dt_gap']) +
                ' mm (timing-free 31.1), det4 at ' + ', '.join(f'{x:.1f}' for x in rows['g_det4']['dt_gap']) +
                ' (30.5), det7’s long run at ' + ', '.join(f'{x:.1f}' for x in rows['g_det7_long']['dt_gap']) +
                ' (29.4; its longer run 32.3 and HV pool 27.2 scatter by ±2.5). The crossing is shallow, ±1–2 mm per chamber.\n\n'
                '<b>det3 is the exception</b>: its D_T would need a 31 mm gap, where the scan and the charge extent both say 28.1. '
                'Its excess is +160 to +260 µm/√cm in quadrature at every field of the scan (249–392 V/cm) at the scan’s own water: '
                'a constant extra transverse spread per unit drift length on that chamber, not a gap or a field effect.\n\n'
                '<b>det6 and det7 ran side by side</b> (6-26, same run, same field) and their v implies 0.8–0.9 % and 0.4–0.6 % water. '
                'The gas was not the same in the two chambers, or det6’s stack T is biased by its window-limited deep charge. '
                'Per-chamber water is consistent with the 10-08 synthesis (water ingress local to each chamber, x·Q ≈ const). '
                'det7’s long and longer runs differ by 4σ in v (water drifting over the night).'))


def sat_closure(D):
    C = {d: Jopt(f'sat_closure_{d}.json') for d in ('det3', 'det2')}
    C = {d: c for d, c in C.items() if c}
    if not C:
        return
    arms = [('cens', 'censored (production)', BLUE), ('cens2', 'bound in solve', GREEN), ('naive', 'clip as real', GOLD),
            ('drop', 'strips dropped', RED)]
    P = sd.Plot(810, 520, x=(-0.5, 3.5, 'lin'), y=(0, 0.04, 'lin'), xlabel='', ylabel='angle resolution vs M3 (s68 of Δtan)')
    P.xticks([(i, lab.split(' (')[0]) for i, (_, lab, _) in enumerate(arms)]); P.yticks([(t, f'{t:.2f}') for t in (0, 0.01, 0.02, 0.03, 0.04)])
    Q = sd.Plot(810, 520, x=(-0.5, 3.5, 'lin'), y=(-0.3, 0.1, 'lin'), xlabel='', ylabel='fitted charge vs unclipped (median − 1)')
    Q.xticks([(i, lab.split(' (')[0]) for i, (_, lab, _) in enumerate(arms)]); Q.yticks([(t, f'{t:+.1f}' if t else '0') for t in (-0.3, -0.2, -0.1, 0, 0.1)])
    Q.hline(0, MUT, '6 6', 1.5)
    for di, (d, c) in enumerate(C.items()):
        for fi, f in enumerate(c['fracs']):
            r = c['rows'][f'{f:g}']
            dx = (di * 2 + fi - 1.5) * 0.17
            for i, (arm, lab, col) in enumerate(arms):
                a = r['arms'][arm]
                P.vbar(i + dx, a['res_s68'], 22, col, opacity=0.55 + 0.35 * fi,
                       tip=f'{d}, clip at {f:g} × max ({r["run_len_median"]:.0f}-sample runs): {lab}: s68 {a["res_s68"]:.4f} '
                           f'(unclipped {r["ref_res_s68"]:.4f}); Δtan vs unclipped s68 {a["dtan_s68"]:.4f}, n {a["n"]}')
                Q.vbar(i + dx, min(max(a['dq_median'], -0.3), 0.095), 22, col, opacity=0.55 + 0.35 * fi, base=0,
                       tip=f'{d}, clip at {f:g} × max: {lab}: charge {100 * a["dq_median"]:+.1f} % (s68 {100 * a["dq_s68"]:.0f} %)')
        P.hline(r['ref_res_s68'], INK, '8 6', 1.5)
    P.text(-0.4, 0.037, 'censoring costs no angle resolution', 22, INK, weight=600)
    P.text(3.45, list(C.values())[0]['rows'][f'{list(C.values())[0]["fracs"][0]:g}']['ref_res_s68'] + 0.001, 'unclipped', 19, INK, 'end')
    Q.text(-0.4, 0.075, 'but reads the charge 5–12 % low', 22, INK, weight=600)
    Q.text(3.0, -0.05, 'dropped: up to +64 %,', 18, RED, 'middle')
    Q.text(3.0, -0.075, 'off scale', 18, RED, 'middle')
    D.slide('satclosure', sd.title('Clipping unsaturated pulses on purpose: censoring keeps the angle, loses some charge',
                                   'det3 and det2, 540–740 bright unclipped events clipped at 0.4 / 0.6 × their maximum (4–6-sample runs, like real ones)') +
            sd.row(P.svg('sat closure angle'), Q.svg('sat closure charge'), gap=44), short='Saturation test',
            foot='sat_closure.py → results/sat_closure_<det>.json; bars: det3 then det2, darker = clip at 0.6; production bundle, reference start',
            notes=(
                '<b>The question</b>: should saturated pulses be reconstructed from the slopes on either side, with a larger '
                'uncertainty on the interpolated samples? <b>What production does</b> is the limit of that idea: the clipped samples '
                'are left out of the charge solve, so the pulse template fitted to the unclipped rising and falling edges fills the '
                'peak, and a one-sided penalty is added wherever the model falls below the clip. Nothing is interpolated by hand, and '
                'no error has to be invented for interpolated samples.\n\n'
                '<b>The test</b> takes events with no saturation (window max 1500–3400 ADC) and clips them at a fraction of their own '
                'maximum, so the truth is the same event unclipped. Four treatments: production censoring; the bound model ≥ clip '
                'enforced inside the solve (new, <code>wft.model.SAT_SOLVE</code>, opt-in; an active set as in a Tobit fit); clipped '
                'samples taken at face value; strips with a clipped sample dropped.\n\n'
                '<b>Result</b>: censoring moves the angle by s68 0.003–0.005 in tan against a 0.018–0.021 resolution, and the '
                'resolution against M3 is unchanged. Taking clipped samples at face value doubles the position scatter and loses '
                '10–27 % of the charge; dropping the strips is far worse. The charge of censored fits reads 6–12 % low, and the '
                'in-solve bound only recovers 1–3 points of that: the deficit is the template under-filling bright pulses (the '
                'same +6–7 % peak residual seen on bright unclipped strips, §37), not the censoring.\n\n'
                '<b>Next for saturation</b>: an amplitude-dependent pulse template, measured from bright unclipped pulses; then the '
                'charge (dE/dx, gain maps) of saturated events is usable. Angles and positions already are.'))


def footprint_physics(D):
    W, H = 1000, 640
    s = []
    yp, yk, ye = 520, 488, 456                                   # pads, kapton, resistive strips (top edges)
    for i in range(5):
        x = 80 + i * 180
        s.append(f'<rect x="{x}" y="{yp}" width="157" height="26" fill="#b87333"/>')
        s.append(sd.T(x + 78, yp + 56, ['−2', '−1', '0', '+1', '+2'][i], 22, INK))
    s.append(sd.T(80, yp + 96, 'pads 0.68 mm on 0.78 mm, each read by one X line', 20, INK, 'start'))
    s.append(f'<rect x="60" y="{yk}" width="900" height="24" fill="{GOLD}" fill-opacity="0.35"/>')
    s.append(sd.T(955, yk + 17, 'kapton + glue ≈ 70 µm', 18, INK, 'end'))
    off = -40                                                    # resistive strips shifted toward -x
    for i in range(5):
        x = 80 + 78 - 63 + i * 180 + off
        s.append(f'<rect x="{x}" y="{ye}" width="127" height="20" fill="{GREY}"/>')
    s.append(sd.T(60, ye - 12, 'resistive strips (550 µm), offset δ from the pad columns', 20, INK, 'start'))
    cx = 80 + 2 * 180 + 78 + off + 20
    s.append(sd.arrow(cx, 300, cx, ye - 4, RED, 3, 14))
    s.append(sd.T(40, 60, '① the electron cloud lands: prompt width ~0.1 mm (avalanche + induction)', 20, RED, 'start'))
    s.append(sd.line(cx, ye + 10, cx - 85, ye + 10, BLUE, 5) + sd.line(cx, ye + 10, cx + 40, ye + 10, BLUE, 5))
    s.append(sd.T(40, 120, '② it spreads across its resistive strip in τ ≈ (w/2)² / D_rc ≈ 100–300 ns,', 20, BLUE, 'start'))
    s.append(sd.T(40, 146, '    then stops at the strip edges: bounded in X (σ ≈ 0.16 mm); along y it keeps going', 20, BLUE, 'start'))
    s.append(sd.arrow(cx - 80, ye + 22, cx - 120, yp - 4, PURPLE, 3, 12))
    s.append(sd.T(40, 206, '③ with an offset δ the strip overlaps pad −1 only: a one-sided, late copy', 20, PURPLE, 'start'))
    fig = sd.svg(W, H, ''.join(s), 'footprint physics')
    rows = [['avalanche spread (Magboltz, 150 µm gap)', '≈ 0.025 mm', 'both'],
            ['induction through 70 µm onto 0.68 mm pads', '≈ 0.05–0.1 mm', 'both'],
            ['spread over the resistive strip’s 550 µm width', '→ box, σ ≈ 0.16 mm, in ~100–300 ns', 'X only'],
            ['RC spread along the resistive strip', '2·D_rc·t, unbounded', 'Y only'],
            ['strip integration over the 0.78 mm pitch', 'already in the model (erf)', 'both']]
    tab = sd.table(['piece', 'size', 'view'], rows, 21)
    txt = sd.col(tab, sd.callout('Expected prompt X footprint beyond the strip integration: √(0.025² + 0.08² + 0.16²) ≈ 0.18–0.2 mm. '
                                 'The per-event fit’s 0.20–0.25 is the physical one; the stacks’ 0.35–0.43 add the one-sided copy '
                                 'and ~0.1 mm of track walk.', BLUE, 22), gap=22, w=620)
    D.slide('s0phys', sd.title('σ0_x ≈ 0.2–0.25 mm is what the stack-up predicts, and X differs from Y for a reason',
                               'Where the prompt footprint comes from, piece by piece (schematic: one resistive strip over the pads, not to scale)') +
            sd.row(fig, txt, gap=44), short='σ0 physics', foot='Layer numbers from docs/detector_spec (several from memory, flagged open there); D_rc per chamber from the RC arms (2.8–7.9e-4 mm²/ns)',
            notes=(
                '<b>Does narrowing σ0_x make physical sense?</b> Yes. The model already integrates each landing over the 0.78 mm pitch '
                '(that alone is σ = 0.225 mm), so σ0 is only what the charge adds beyond it. The avalanche is tiny (≈ 25 µm over the '
                '150 µm gap, Magboltz). Induction through the ≈ 70 µm kapton + glue onto 0.68 mm pads adds of order the insulator '
                'thickness. The one large term is the resistive strip itself: charge landed on a 550 µm strip spreads across its width '
                'until it fills it, a box of σ = 0.55/√12 = 0.16 mm. In quadrature, ≈ 0.18–0.2 mm.\n\n'
                '<b>Is it different from Y?</b> It should be. Along y the landed charge keeps spreading along the strip (RC diffusion, '
                'width² += 2 D_rc t, the slow component the RC kernel models); across it, in x, the spread stops at the strip edges. '
                'So Y’s effective width is dominated by RC from the first tens of ns, X’s by one bounded box.\n\n'
                '<b>How fast is the spread across the strip?</b> The same RC constant governs it: τ ≈ (w/2)² / D_rc = 0.0756 mm² / '
                '(2.8–7.9e-4 mm²/ns) ≈ 95–270 ns per chamber. That is inside the shaping time, which is why X has a small slow part '
                'too, and it is the time scale of the one-sided copy on the next slide (measured 105–120 ns).\n\n'
                '<b>Why the stacks read wider</b>: a head-on stack’s ±1 charge includes that one-sided delayed copy and, for '
                '|tan| < 0.03, ~0.1 mm of track walk (§30); a symmetric Gaussian fitted to it widens to cover both.'))


def asym_evidence(D):
    X, B, S = Jopt('xasym_tests.json'), Jopt('xasym_boundary.json'), Jopt('xasym_samechannel.json')
    if not (X and S):
        return
    sh = X['det3']['shape']
    t = [(i - 6) * 60 for i in range(len(sh['centre']))]
    P = sd.Plot(810, 520, x=(-360, 820, 'lin'), y=(-0.005, 0.13, 'lin'), xlabel='time from the centre strip’s peak (ns)',
                ylabel='per-sample charge / centre integral')
    P.xticks([(v, str(v)) for v in (-300, 0, 300, 600)]); P.yticks([(v, f'{v:.2f}') for v in (0, 0.04, 0.08, 0.12)])
    P.line(t, sh['centre'], GREY, w=2, markers=False, dash='6 4', tip='centre strip')
    P.line(t, sh['m1'], RED, w=3, markers=False, tip=f'−1 neighbour (lower x): integral {sum(sh["m1"]):.3f}')
    P.line(t, sh['p1'], BLUE, w=3, markers=False, tip=f'+1 neighbour (higher x): integral {sum(sh["p1"]):.3f}')
    P.text(-340, 0.122, f'det3: −1 carries a copy delayed {sh["delay_ns"]:.0f} ns', 22, INK, weight=600)
    leg = sd.legend([('centre', GREY, 'dash'), ('−1 neighbour', RED, 'line'), ('+1 neighbour', BLUE, 'line')], 21)
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'det4_sps_assessment')))
    from det4_sps_map import POSITION_MM
    b = S['det4_beam']
    pa = sorted((float(POSITION_MM[int(c)]), int(c), v) for c, v in b['A'].items())
    Q = sd.Plot(810, 520, x=(100, 255, 'lin'), y=(-0.8, 0.9, 'lin'), xlabel='x on det4 (mm), H4 beam spot',
                ylabel='A_j: channel j as −1 vs as +1 neighbour')
    Q.xticks([(v, str(v)) for v in range(100, 256, 25)]); Q.yticks([(v, f'{v:+.1f}' if v else '0') for v in (-0.8, -0.4, 0, 0.4, 0.8)])
    Q.hline(0, INK, '6 6', 1.5)
    for xe in (149.76, 199.68, 249.6):
        Q.vline(xe, GREY, '3 4', 1.2, tip=f'DREAM chip boundary at {xe} mm')
    Q.points([p[0] for p in pa], [p[2] for p in pa], PURPLE, 6,
             tips=[f'ch {c} (chip {c // 64}, local {c % 64}) at {x:.2f} mm: A = {v:+.2f}' for x, c, v in pa])
    Q.text(103, 0.8, 'det4 beam: A depends on the live band', 22, INK, weight=600)
    Q.text(150.5, -0.55, '≈ 0 here', 20, PURPLE)
    Q.text(186, 0.18, '+0.4…+0.75', 20, PURPLE)
    D.slide('asym', sd.title('det3’s raw waveforms carry a one-sided, delayed X neighbour copy; det4’s beam is band-dependent',
                             'Left: det3 bench, stacked head-on ±1 pulses at the time-symmetric angle. Right: det4 at H4, each channel against itself') +
            sd.row(sd.col(P.svg('asym shapes'), leg, gap=4), Q.svg('A per channel'), gap=44), short='X asymmetry: data',
            foot='xasym_tests.py, xasym_samechannel.py, xasym_boundary.py → results/xasym_*.json; raw windows, no template or kernel',
            notes=(
                '<b>Model-free throughout</b>: raw waveform windows (rows in position order), unsaturated, no flagged strip within ±2.\n\n'
                '<b>Not a walk.</b> A drift walk (a tilt between the field, the chamber and the reference) makes the two neighbours '
                'differ in timing. The ±1 mean-time difference crosses zero at tan_ref ≈ −0.005 on det3, the same 0.005 offset its '
                '±tan gap halves gave. At that angle the timing is symmetric, yet Q₋₁ exceeds Q₊₁ by '
                f'{X["det3"]["head_aQ"][0]:+.3f} ± {X["det3"]["head_aQ"][1]:.3f} (head-on, aQ = (Q₋₁ − Q₊₁)/(Q₋₁ + Q₊₁)). '
                f'The −1 minus +1 difference is {sh["a"]:.2f} × the centre pulse delayed by ≈ {sh["delay_ns"]:.0f} ns '
                f'({100 * sh["explained"]:.0f} % of it explained); det4 bench {X["det4"]["shape"]["delay_ns"]:.0f} ns, det7 '
                f'{X["det7"]["shape"]["delay_ns"]:.0f} ns (n {X["det7"]["shape"]["n"]}). H4 had “−1 delayed 112 ns, +1 prompt”.\n\n'
                '<b>Not the strip gains.</b> On the beam, aQ follows position (−0.27 at x ≈ 105, 0.00 at 155, +0.31…+0.39 at '
                '185–215 mm): det4’s strip-scale gain swings dominate a cluster-level ratio there. The gain-free test compares each '
                'channel with itself: its share of the cluster when it is the −1 neighbour against when it is the +1 neighbour. '
                f'Median over {b["n_channels"]} channels {b["median"]:+.2f} ± {b["err"]:.2f}; a toy with symmetric sharing and a ×3 '
                'random (uncorrelated) gain spread analysed the same way gives −0.05…0.00. But A is not one number: it is ≈ 0 over '
                'the whole 149–161 mm band (chip 1, local 48–63) and +0.4…+0.75 over 182–214 mm (chip 1 local 6–22 and chip 2). '
                'Same chip, different band, different answer, so it is not a chip property; but det4’s gain changes ×20 across its '
                '12 mm live bands, and a gradient that size enters the cluster normalisation, so the beam test is not clean. '
                'It is inconclusive, not evidence either way. On the bench, with uniform illumination, the cluster-level '
                'aQ is already gain-averaged: det3 ' + f'{X["det3"]["head_aQ"][0]:+.2f} ± {X["det3"]["head_aQ"][1]:.2f}, det2 '
                f'{X["det2"]["head_aQ"][0]:+.2f} ± {X["det2"]["head_aQ"][1]:.2f}, det4 {X["det4"]["head_aQ"][0]:+.2f} ± '
                f'{X["det4"]["head_aQ"][1]:.2f}, det7 {X["det7"]["head_aQ"][0]:+.2f} ± {X["det7"]["head_aQ"][1]:.2f}, det6 '
                f'{X["det6"]["head_aQ"][0]:+.2f} ± {X["det6"]["head_aQ"][1]:.2f}. Same sign on four of five; size per chamber.\n\n'
                '<b>Not a pitch beat.</b> An 800 µm resistive pitch over 780 µm pads would modulate the asymmetry with a 31.2 mm '
                'period; the periodogram has no peak there (power 0.4–7 against a null mean of 2) or anywhere else significant.\n\n'
                '<b>Electronics crosstalk: not settled by these data.</b> Channel numbers fall with +x in every 64-channel block on '
                'all chambers, so position and channel order never separate except at chip edges, where spatial neighbours sit on '
                'different chips. On the beam, clusters centred on the four edge strips give aQ −0.31, +0.39, +0.27, −0.02: dominated '
                'by those strips’ gains, so not decisive. A separate study of quiet events in raw data (nb_asym_raw, 19 Oct) is '
                'looking at channel-to-channel pick-up directly.'))


def asym_mechanism(D):
    items = [
        ('Established', 'det3: a one-sided X neighbour copy, delayed ~110 ns, in raw waveforms, 8σ; same sign on det2/4/7, '
         'none on det6; Y symmetric.', INK),
        ('Candidate mechanism', 'the resistive strips offset from the pad columns (slide 13, ③): Y’s RC spread, seen in X '
         'where a strip overlaps one neighbouring pad.', BLUE),
        ('Tests that need no new data', 'delay ∝ 1/D_rc chamber by chamber; a stack-up simulation with a plausible δ; quiet-event '
         'crosstalk (nb_asym_raw) of the same size would kill it.', PURPLE),
        ('Not claimed', 'as a model term. Production’s c1_asym stays an effective parameter until those agree.', GREY)]
    body = (sd.title('A candidate explanation for the X asymmetry, offered to the simulation side',
                     'A hypothesis with tests, not a result') +
            sd.col(*[sd.callout(f'<b>{a}.</b> {b}', c, 27) for a, b, c in items], gap=26))
    D.slide('asymmech', body, short='X asymmetry: mechanism?', notes=(
        'Delays measured (det3 114, det4 105, det7 120 ns) against τ = (0.275 mm)² / D_rc from each chamber’s RC arm '
        '(det3 272, det4 270, det7 132, det2 180, det6 95 ns): the same order, shorter by up to ×2.5 on det3/det4, as expected '
        'if the overlap begins inside the strip half-width. Not a fit; D_rc is itself a bench-fitted effective constant.\n\n'
        'Predictions: a fixed sign per board; size set by the screen-print registration δ (tens of µm), so chamber-dependent '
        'and possibly varying across a board (det3’s aQ ranges 0.05–0.34 along x); no effect in Y (its strips run along the '
        'resistive strips); a 31.2 mm beat if the resistive pitch were 800 µm against the 780 µm pads. No beat is seen, which is '
        'weak evidence that the pitches match (docs/detector_spec q_esl_geom: 800 µm was from memory).\n\n'
        'Electronics crosstalk is not excluded: channel order follows position everywhere except chip edges, and the chip-edge '
        'test on det4 is dominated by the gain structure.'))


FOLLOWUPS = [vdrift_scan, vdrift_single, sat_closure, footprint_physics, asym_evidence, asym_mechanism]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(CB, 'basics_oct19', 'mx17-basics-oct19.html'))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    D = sd.Deck('Detector basics, 19 Oct', 'MX17: dead strips and gain maps, timing-free drift gaps, saturation, the X footprint')
    for f in [cover, setup, gain_maps, strip_health, det4_story, gap_method, gap_result, saturation, footprint] + FOLLOWUPS + [closing]:
        f(D)
    D.write(a.out, note_meta=dict(
        title='MX17 detector basics: dead strips, gain maps, drift gaps, saturation, X footprint',
        summary='det4’s amplitude-dependent X width was its dead strips; strip-scale gain maps and strip health for five chambers; '
                'drift gaps 28.1–31.5 mm measured without timing; saturation handled per event; the RC model’s X χ² deficit is '
                'its footprint width. Plus the drift-velocity consistency, saturation and asymmetry follow-ups.',
        tags='X17, cosmic bench, micromegas, detector model', date='2026-10-19'),
        footer='Built by sps_beam_test_26/analysis/cloud_basics/make_basics_deck.py from results/*.json (FINDINGS §35 onward).')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
