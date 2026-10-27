"""Slide note mx17-angle-floor: the micro-TPC angle floor, why 60 ns depth bins break at steep angles, and the realism
ladder from the floor up to the measured resolution (model-audit follow-up FD, item 10).

Reads this directory's results/*.json (scan_a, onset, ladder_<det>[_alone], measured_chi2, data_split, tilt_map_xval,
budget), so rerunning the analysis and this script updates every number and chart together.

    .venv/bin/python rerun/model_audit/followups/fd_floor_realism/make_floor_deck.py
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        rerun/model_audit/followups/fd_floor_realism/deck/mx17-angle-floor.html --slug mx17-angle-floor --force --deploy
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT,  # noqa: E402
                      DBLUE, DRED, DGREEN, DGREY, term)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagram  # noqa: E402

OUT = HERE / 'deck' / 'mx17-angle-floor.html'
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
DCOL = dict(det2=BLUE, det3=RED, det4=GOLD, det6=PURPLE, det7=GREEN)
BCOL = {20.0: GREEN, 30.0: BLUE, 60.0: RED, 90.0: PURPLE}
R = HERE / 'results'


def J(name):
    return json.loads((R / name).read_text())


SC, ON, ME, DS, TM, BU = (J('scan_a.json'), J('onset.json'), J('measured_chi2.json'), J('data_split.json'),
                          J('tilt_map_xval.json'), J('budget.json'))
LAD = {d: J(f'ladder_{d}.json') for d in DETS}
LADA = {d: J(f'ladder_{d}_alone.json') for d in DETS}
RUNGS = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
RNAME = dict(L0='floor', L1='production fit', L2='footprint', L3='X copy', L4='gain map', L5='pulse shape',
             L6='extra diffusion', L7='dead + CM', L8='v drift')
RTIP = dict(
    L0='Ideal events (Fischle clusters, delta rays, Magboltz diffusion, Polya gain, X template, noise, ADC ceiling) '
       'fitted with the toy\'s own exact model, started at the truth: the floor (ws_floor).',
    L1='The same ideal events fitted by the production reconstruction itself: wft.reco.fit_plane with the chamber\'s '
       'promoted bundle (physical kernel, pivot start, per-event sigma0 choice, t0 prior, dead mask).',
    L2='Prompt X footprint 0.25 mm (SPS head-on 0.24-0.29, cloud_basics 38/42) + pseudo-Voigt tails (eta 0.10-0.36, '
       'gamma 0.53-0.91 mm, cloud_basics 23 footprint_test.json).',
    L3='One-sided delayed copy onto the lower neighbour: det3 0.13 x the strip pulse, 114 ns late (cloud_basics 42); '
       'det2/det4 0.063, det7 0.15, det6 0 (scaled by head-on aQ).',
    L4='Strip-to-strip gain spread, not known to the fit (bundle gain map = unit): sigma ln g det2/3 0.06, det6/7 0.09, '
       'det4 0.31 (cloud_basics 35).',
    L5='Pulse narrower than the plane template: falling edge compressed so an isolated pulse leaves +6 % at the peak '
       '(cloud_basics 37). An upper estimate: the depth profile absorbs most of it on a real column.',
    L6='det3 only: extra transverse spread +235 um/sqrt(cm) over Magboltz in quadrature (cloud_basics 40). 0 elsewhere.',
    L7='Dead strips the mask misses (cloud_basics 39, ws_dead_charge) + raw common mode 135 ADC per chip, removed by the '
       'production plain 64-channel median, per-channel slope spread 3.3 % (rerun/cm_study).',
    L8='v changes over the run, fit uses the run mean: rms det2/3 0.6 %, det4 2.9 %, det6 3.5 %, det7 1.2 % '
       '(FOLLOWUPS 1, uniform drift over the run).')


def f2(x):
    return f'{x:.2f}'


# --------------------------------------------------------------------------- #
def s_cover(D):
    b = BU
    fl = [b[d]['floor'] for d in DETS]
    me = [b[d]['measured'] for d in DETS]
    ty = [b[d]['toy_all'] for d in DETS]
    body = (sd.kicker('MX17 · June cosmic bench, X view · model-audit follow-up FD · 27 Oct 2026') +
            sd.title('<span style="color:#eef0f3">The floor is 0.3°. Every measured imperfection we can simulate '
                     'brings it only to 0.4–0.8°; the rest of the 1.1–1.8° is a tilt that varies across each chamber '
                     'and a term the waveforms do not see</span>') +
            sd.row(sd.bignum(f'{min(fl):.2f}–{max(fl):.2f}°', 'the floor', DGREEN,
                             'perfect model, all five chambers, 0–20°', tip='L0 of the realism ladder, s68'),
                   sd.bignum(f'{min(ty):.2f}–{max(ty):.2f}°', 'with every measured effect', DBLUE,
                             'simulated, fitted by the production reconstruction'),
                   sd.bignum(f'{min(me):.2f}–{max(me):.2f}°', 'measured against M3', DRED,
                             'June golden rows, X view, all angles'), gap=64))
    D.slide('cover', body, notes='Questions (FOLLOWUPS item 10): (a) why the perfect-model resolution with 60 ns depth '
            'bins jumps above ~25°; (b) why the measured resolution does not approach the floor. Code and numbers: '
            '<code>rerun/model_audit/followups/fd_floor_realism/</code> (REPORT.md).', dark=True)


def s_setup(D):
    steps = [dict(label='ionisation', sub='clusters + δ rays', color=BLUE,
                  tip='Fischle et al. 1991 cluster sizes, 2.8 clusters/mm, delta-ray ranges'),
             dict(label='drift', sub='Magboltz diffusion', color=BLUE, tip='bundle Dp, D_L'),
             dict(label='avalanche', sub='Polya gain', color=BLUE, tip='theta = 1'),
             dict(label='readout', sub='footprint, strips, pulse, noise', color=BLUE,
                  tip='0.78 mm strips, measured X template, 60 ns sampling, 8 ADC noise, 3550 ADC ceiling'),
             dict(label='+ one real effect', sub='the realism ladder', color=ORANGE,
                  tip='each rung adds one measured imperfection to the GENERATION only'),
             dict(label='fit', sub='production wft', color=RED,
                  tip='wft.reco.fit_plane with the chamber\'s promoted bundle; the floor uses the toy\'s exact model')]
    body = (sd.title('An electron-by-electron toy, fitted like the data',
                     'Each simulated track goes through the full chain; what we vary is what the fit does not know') +
            sd.flow(steps, size=26) +
            sd.row(sd.card(sd.p('<b>(a)</b> With a perfect model and 60 ns depth bins, steep tracks lose resolution. '
                                'Which length sets where it starts?', 28)),
                   sd.card(sd.p('<b>(b)</b> Add the imperfections we have measured one at a time. Do they reach the '
                                'measured resolution and fit quality?', 28)), gap=44) +
            sd.p('Hover the dotted terms and the data points for values, counts and sources.', 22, MUT))
    D.slide('setup', body, notes='The toy is ws_floor/floor_toy.py (unchanged). Rung L0 uses its exact mean model and '
            'starts at the truth; every other rung is fitted by <code>wft.reco.fit_plane</code> on the promoted bundle '
            'in <code>~/x17/cosmic_bench/model_audit/&lt;det&gt;/calib_bundle</code>, i.e. exactly the production '
            'estimator. 150 events per angle (3°, 9°, 16°: the medians of the measured angle bins), 5 chambers, condor '
            'cluster 4583195.', foot='Toy: rerun/model_audit/ws_floor/floor_toy.py; ladder: ladder.py')


def s_floor_curve(D):
    P = sd.Plot(1040, 640, x=(0, 46), y=(0, 2.1), xlabel='track angle θ (°)', ylabel='angle resolution s68 (°)')
    P.xticks([(a, str(a)) for a in range(0, 46, 5)]).yticks([(v / 2, f'{v / 2:.1f}') for v in range(0, 5)])
    for b in (20, 30, 60, 90):
        r = SC[f'det3_full_bin{b}_p78']['by_angle']
        xs = [float(a) for a in r]
        ys = [v['s68'] for v in r.values()]
        tips = [f'{b} ns bins, θ {a}°: s68 {v["s68"]:.2f} ± {v["s68_err"]:.2f}°, one bin spans '
                f'{v["x"]:.2f} strip' for a, v in r.items()]
        P.line(xs, ys, BCOL[float(b)], tips=tips, w=4 if b == 60 else 3)
    m = ME['det3_x']['bins']
    P.points([r['mid'] for r in m], [r['s68'] for r in m], INK, r=9, marker='diamond',
             tips=[f'measured det3 X, |θ| {r["lo"]}–{r["hi"]}°: s68 {r["s68"]:.2f}°, n = {r["n"]}' for r in m])
    P.text(1, 1.95, 'Perfect model: flat at 0.3° with fine bins,', 24, INK, weight=600)
    P.text(1, 1.83, 'steep tracks degrade with 60–90 ns bins', 24, INK, weight=600)
    leg = sd.legend([('20 ns bins', GREEN), ('30 ns', BLUE), ('60 ns (production)', RED), ('90 ns', PURPLE),
                     ('measured det3 X', INK, 'dot')], 24)
    side = sd.col(sd.callout('Below ~20° every bin width gives the same 0.3°: diffusion sets it.', GREEN),
                  sd.callout('Above it the coarse bins rise, the 60 ns curve ×4 by 45°.', RED),
                  sd.callout('The measured 1.0–1.1° is already 3× the floor where bins do not matter.', INK), w=560)
    body = sd.title('The floor is 0.3°; coarse depth bins break it for steep tracks',
                    'det3, perfect-model toy at four depth-bin widths, 100 tracks per point') + \
        sd.row(sd.col(P.svg('floor vs angle'), leg, w=1040), side, gap=56)
    D.slide('floor', body, notes='Perfect model = the toy fitted by its own exact mean model, started at the truth '
            '(scan_a.py, condor). s68 = 68th percentile of |error − median| with |error| > 5° set aside (fleet '
            'convention). Measured: finalize tables, M3-matched single tracks, w on the promoted w0/kw '
            '(measured_chi2.py); the M3 reference (≤ 0.15°) is not subtracted.', foot='scan_a.py, measured_chi2.py')


def s_mechanism(D):
    body = (sd.title('Why: the model draws each 60 ns slice as a uniform bar',
                     'det3, one bin = v·Δt = 2.15 mm of drift; across the strips to scale, depth compressed. Blue: what the model '
                     'puts in a bin. Red: the real charge, a few lumps.') +
            diagram.mechanism_svg(sd) +
            sd.legend([('model: charge spread uniformly along the bin\'s segment', BLUE, 'box'),
                       ('real: ionisation clusters × avalanche gain', RED, 'dot'), ('readout strips', GREY, 'box')],
                      24))
    D.slide('mechanism', body, notes='The fit has a free charge per depth bin and assumes that charge is spread '
            'uniformly along the bin\'s piece of track. A shallow track\'s piece is shorter than the footprint of one '
            'electron on the strips (σ_eff ≈ 0.55 mm on det3: 0.24 mm prompt footprint, 0.4 mm diffusion at mid-gap, '
            'the strip width), so whether the charge is lumpy inside it cannot be seen. A steep track\'s piece is '
            'longer: the strips see 1–3 separate lumps where the model expects a smooth bar. The model cannot move '
            'charge within a bin, so the fit bends the slope and the per-bin profile to make up for it: a random '
            'angle error plus a bias toward shallow (shorter bars look more like lumps). Finer bins remove it '
            'because each bin then holds one lump or less.')


def s_collapse(D):
    P = sd.Plot(800, 540, x=(0, 5), y=(0, 2.2), xlabel='bin segment / footprint width  L/σ_eff',
                ylabel='excess over the floor (°)', title='every chamber, bin width and footprint')
    P.xticks([(v, str(v)) for v in range(0, 6)]).yticks([(v / 2, f'{v / 2:.1f}') for v in range(0, 5)])
    for tag, o in ON.items():
        if o['config'] != 'full':
            continue
        xs = [L / o['sigma_eff'] for L in o['L']]
        tips = [f'{tag}: θ {a:.0f}°, L {L:.2f} mm, L/σ_eff {L / o["sigma_eff"]:.2f}, excess {e:.2f}°'
                for a, L, e in zip(o['angles'], o['L'], o['excess'])]
        keep = [i for i, x in enumerate(xs) if x <= 5]
        P.line([xs[i] for i in keep], [o['excess'][i] for i in keep], BCOL[o['dt_bin']], w=2, r=4,
               tips=[tips[i] for i in keep])
    P.hline(0.30, MUT, label='onset threshold 0.3°', anchor='end', where='below')
    m_full = [o['L_over_sig'] for o in ON.values() if o['config'] == 'full' and o['L_over_sig'] and o['dt_bin'] >= 30]
    m_nl = [o['L_over_sig'] for o in ON.values() if o['config'] == 'no_landau' and o['L_over_sig']]
    P.vline(sum(m_full) / len(m_full), RED, label=f'onset L ≈ {sum(m_full) / len(m_full):.2f} σ_eff')
    # right: onset in two units for the det3 variants
    Q = sd.Plot(800, 540, x=(0.3, 2.5), y=(0.6, 2.6), xlabel='onset L / strip pitch', ylabel='onset L / σ_eff',
                title='which length sets the onset? (det3 variants)')
    Q.xticks([(v / 2, f'{v / 2:.1f}') for v in range(1, 6)]).yticks([(v / 2, f'{v / 2:.1f}') for v in range(2, 6)])
    for tag, o in ON.items():
        if not o['L_over_sig'] or o['dt_bin'] < 30:
            continue
        var = o['det'] == 'det3' and (o['pitch'] != 0.78 or abs(o['s0'] - 0.238) > 0.01)
        c = ORANGE if var else (RED if o['config'] == 'full' else GREY)
        Q.points([o['L_over_pitch']], [o['L_over_sig']], c, r=9 if var else 7,
                 tips=[f'{tag}: pitch {o["pitch"]} mm, σ0 {o["s0"]:.2f} mm, σ_eff {o["sigma_eff"]:.2f} mm, '
                       f'onset {o["onset_deg"]:.1f}°, L/pitch {o["L_over_pitch"]:.2f}, L/σ_eff {o["L_over_sig"]:.2f}'])
    Q.text(0.4, 2.45, 'spread: L/pitch ±16 %, L/σ_eff ±12 %', 22, INK, weight=600)
    body = (sd.title('The jump starts once a bin\'s segment exceeds ~1.5 footprint widths',
                     'Perfect-model excess √(s68² − floor²) against L = v·Δt·tanθ; 5 chambers × 20/30/60/90 ns, '
                     'det3 with pitch 0.5/1.2 mm and footprint 0.10/0.45 mm') +
            sd.row(P.svg('collapse'), Q.svg('onset units'), gap=44) +
            sd.legend([('20 ns', GREEN), ('30 ns', BLUE), ('60 ns', RED), ('90 ns', PURPLE),
                       ('right: full ionisation', RED, 'dot'), ('without clustering', GREY, 'dot'),
                       ('pitch / footprint changed', ORANGE, 'dot')], 22))
    nl = sum(m_nl) / len(m_nl)
    D.slide('collapse', body, notes=f'Onset = the angle where the excess first exceeds 0.3° (onset.py). Mean over the '
            f'30/60/90 ns runs with full ionisation: L = {sum(m_full) / len(m_full):.2f} σ_eff; without clustering '
            f'{nl:.2f} σ_eff (later). σ_eff² = σ0² + Dp²·u_mid + pitch²/12 ≈ 0.52–0.68 mm. On the MX17 chambers σ_eff ≈ '
            '0.7 pitch, so "one strip" (the first answer) is numerically right, but the pitch variants show the scale '
            'is the footprint width: pitch 0.5 mm moves the onset in L/pitch from 1.14 to 1.66 but not in L/σ_eff '
            '(1.58 → 1.56); a wider footprint (0.45 mm) delays it. Without cluster-size fluctuations the excess does '
            'NOT vanish: it halves (det3 60 ns at 45°: 0.68° against 1.61°), because single electrons with Polya '
            'gain are lumpy too. So the first answer is confirmed for the mechanism, corrected on two points: the '
            'scale is the footprint, and clustering doubles the effect rather than creating it.',
            foot=None)


def s_ladder_setup(D):
    steps = []
    for r in RUNGS:
        steps.append(dict(label=RNAME[r], sub=r, color=GREEN if r == 'L0' else (RED if r == 'L1' else ORANGE),
                          tip=RTIP[r]))
    body = (sd.title('(b) The realism ladder: one measured imperfection per rung',
                     'Every size is a measurement from the data; nothing is tuned to the resolution') +
            sd.flow(steps[:5], size=24) + sd.flow(steps[5:], size=24) +
            sd.row(sd.card(sd.p('<b>Measured outside the toy:</b> ' +
                                term('position-dependent tilt', 'cross-validated 2D map of the median angle residual in '
                                     '40 × 40 mm cells (tilt_map_xval.py); includes the bow') + ' and the ' +
                                term('M3 reference', '≤ 0.15°: slope error 0.02° + multiple scattering'), 26)),
                   sd.card(sd.p('<b>Each rung is run twice:</b> cumulative (the walk) and alone on top of the '
                                'production fit (its own size).', 26)), gap=44))
    D.slide('ladder_setup', body, notes='<ol>' + ''.join(f'<li><b>{r} {RNAME[r]}</b>: {RTIP[r]}</li>' for r in RUNGS)
            + '</ol>', foot='ladder.py (REALISM dict carries every input with its source)')


def s_walk(D):
    P = sd.Plot(1000, 600, x=(-0.5, 10.4), y=(0, 2.0), ylabel='angle resolution s68 (°), 3–16°',
                title='resolution')
    SHORT = dict(L0='floor', L1='fit', L2='foot', L3='copy', L4='gain', L5='pulse', L6='diff', L7='dead', L8='v')
    P.xticks([(i, SHORT[r]) for i, r in enumerate(RUNGS)] + [(9.8, 'data')])
    P.yticks([(v / 2, f'{v / 2:.1f}') for v in range(0, 5)])
    for d in DETS:
        ys = [LAD[d]['rungs'][r]['all']['s68'] for r in RUNGS]
        tips = [f'{d} {r} {RNAME[r]}: s68 {LAD[d]["rungs"][r]["all"]["s68"]:.2f} ± '
                f'{LAD[d]["rungs"][r]["all"]["s68_err"]:.2f}°, χ²/dof {LAD[d]["rungs"][r]["all"]["chi2dof"]:.0f}'
                for r in RUNGS]
        P.line(list(range(len(RUNGS))), ys, DCOL[d], tips=tips, w=3)
        P.points([9.8], [BU[d]['measured']], DCOL[d], r=10, marker='diamond',
                 tips=[f'{d} X measured: s68 {BU[d]["measured"]:.2f}°, χ²/dof {BU[d]["chi2_data"]:.0f}'])
    Q = sd.Plot(640, 600, x=(-0.5, 10.4), y=(1, 1000, 'log'), ylabel='median χ²/dof', title='fit quality')
    Q.xticks([(0, 'floor'), (8, 'v drift'), (9.8, 'data')]).yticks([(1, '1'), (10, '10'), (100, '100'), (1000, '1000')])
    for d in DETS:
        ys = [max(LAD[d]['rungs'][r]['all']['chi2dof'], 1) for r in RUNGS]
        Q.line(list(range(len(RUNGS))), ys, DCOL[d], w=3, r=4)
        Q.points([9.8], [BU[d]['chi2_data']], DCOL[d], r=10, marker='diamond',
                 tips=[f'{d} X data χ²/dof {BU[d]["chi2_data"]:.0f}; toy at the top of the ladder '
                       f'{BU[d]["chi2_toy_L8"]:.0f}'])
    ratio = [BU[d]['chi2_data'] / BU[d]['chi2_toy_L8'] for d in DETS]
    Q.text(0, 700, f'data χ² {min(ratio):.0f}–{max(ratio):.0f}× the toy', 24, INK, weight=600)
    fr = [BU[d]['toy_all'] / BU[d]['measured'] for d in DETS]
    P.text(0, 1.85, f'the ladder ends at {min(fr) * 100:.0f}–{max(fr) * 100:.0f} % of the measured value', 24, INK, weight=600)
    body = (sd.title('Every measured imperfection together reaches neither the resolution nor the χ²',
                     'Cumulative ladder, X view, 150 tracks at each of 3°, 9°, 16°, fitted by production wft') +
            sd.row(P.svg('ladder walk'), Q.svg('chi2 walk'), gap=24) +
            sd.legend([(d, DCOL[d]) for d in DETS] + [('measured (diamond)', INK, 'dot')], 24))
    D.slide('walk', body, notes='The production fit on ideal events is at the floor (L1 ≈ L0, det4/det7 +0.1°: their '
            'per-event σ0 choice and low charge). The biggest single steps: det4 gain map (+0.4°), det6 v drift, '
            'det7 footprint tails and dead strips. Statistical error on each toy s68 ≈ ±0.02–0.03°. The pulse-shape '
            'rung adds almost nothing because the free depth profile absorbs a sharper pulse; so does the det3 '
            'diffusion excess. χ²/dof: toy medians with the same noise weighting as production.',
            foot=None)


def s_budget(D):
    comps = [('floor', 'floor', GREEN), ('production fit', 'prodfit_term', GREY),
             ('measured imperfections (toy)', 'measured_terms', ORANGE), ('position-dependent tilt', 'position', BLUE),
             ('M3 reference', 'm3', PURPLE), ('unexplained', 'unexplained', RED)]
    P = sd.Plot(1100, 560, x=(0, 6), y=(0, 3.4), ylabel='variance (deg²), stacked', xlabel='')
    P.xticks([(i + 1, d) for i, d in enumerate(DETS)]).yticks([(v / 2, f'{v / 2:.1f}') for v in range(0, 7)])
    for i, d in enumerate(DETS):
        base = 0.0
        for name, key, c in comps:
            v = BU[d][key] ** 2
            P.vbar(i + 1, base + v, 110, c, base=base, tip=f'{d} {name}: {BU[d][key]:.2f}° ({v:.2f} deg²)')
            base += v
        P.text(i + 1, base + 0.08, f'{BU[d]["measured"]:.2f}°', 22, INK, 'middle', 600)
    rows = []
    for d in DETS:
        b = BU[d]
        top = sorted(b['terms'].items(), key=lambda kv: -kv[1])[:2]
        rows.append([d, f2(b['floor']), f2(b['measured_terms']), ', '.join(f'{RNAME[k]} {v:.2f}' for k, v in top),
                     f2(b['position']), f2(b['unexplained']), f2(b['measured'])])
    fr = [BU[d]['unexplained'] ** 2 / BU[d]['measured'] ** 2 for d in DETS]
    body = (sd.title(f'{min(fr) * 100:.0f}–{max(fr) * 100:.0f} % of the measured variance is in nothing we simulated',
                     'X view, quadrature budget per chamber; numbers on top = measured s68') +
            sd.row(sd.col(P.svg('budget'), sd.legend([(n, c, 'box') for n, _, c in comps], 22), w=1100),
                   sd.col(sd.table(['', 'floor', 'toy terms', 'largest', 'tilt', 'rest'],
                                   [r[:6] for r in rows], size=20), w=520), gap=44))
    D.slide('budget', body, notes='Quadrature: toy terms = √(L8² − L0²); each term alone = √(alone² − L1²); '
            'position = √(s68² − s68 after the cross-validated 2D 40 mm tilt map²) (tilt_map_xval.py, which also '
            'subtracts the linear angle response first); M3 0.15°; unexplained = what remains. The tilt estimate is '
            'conservative: 40 mm cells cannot follow shorter-range tilts, and the cell medians add their own noise '
            '(≈0.3° in quadrature), so part of the "unexplained" bar may be tilt at finer scale.',
            foot=None)


def s_not_waveform(D):
    P = sd.Plot(800, 560, x=(0.5, 5.5), y=(0, 2.4), xlabel='quintile (low → high)', ylabel='measured s68 (°)',
                title='X resolution vs χ²/dof (solid) and charge (dashed)')
    P.xticks([(i, str(i)) for i in range(1, 6)]).yticks([(v / 2, f'{v / 2:.1f}') for v in range(0, 5)])
    for d in DETS:
        for key, dash in (('chi2dof', None), ('q', '8 6')):
            rows = DS[f'{d}_x'][key]
            P.line(list(range(1, 6)), [r['s68'] for r in rows], DCOL[d], dash=dash, w=3, r=5,
                   tips=[f'{d} X, {key} {r["lo"]:.0f}–{r["hi"]:.0f} (median {r["med"]:.0f}): s68 {r["s68"]:.2f}°, '
                         f'n {r["n"]}' for r in rows])
    Q = sd.Plot(800, 560, x=(0.5, 5.5), y=(0, 2.0), ylabel='s68 (°)', title='after a cross-validated tilt map')
    Q.xticks([(i + 1, d) for i, d in enumerate(DETS)]).yticks([(v / 2, f'{v / 2:.1f}') for v in range(0, 5)])
    for i, d in enumerate(DETS):
        t = TM[f'{d}_x']
        for j, (k, c, lab) in enumerate((('s68', GREY, 'none'), ('1d_40', BLUE, '1D (along x)'),
                                         ('2d_40', GREEN, '2D (x, y)'))):
            Q.vbar(i + 1 + (j - 1) * 0.25, t[k], 40, c, tip=f'{d} X {lab}: {t[k]:.2f}°')
    body = (sd.title('The rest is χ²-blind: good fits are no better, and part of it is a tilt map over (x, y)',
                     'Measured X view, |θ| < 12°: left split by the fit\'s χ²/dof and charge; right corrected by the '
                     'median residual of its 40 mm cell, measured on the other half of the events') +
            sd.row(sd.col(P.svg('splits'), sd.legend([(d, DCOL[d]) for d in DETS], 22)),
                   sd.col(Q.svg('tilt'), sd.legend([('as measured', GREY, 'box'), ('1D map (along x)', BLUE, 'box'),
                                                    ('2D map (x, y)', GREEN, 'box')], 22)), gap=44))
    hyp = BU['det3']['hyp']
    hs = ', '.join(f'{k} {v:.2f}°' for k, v in hyp.items())
    D.slide('not_waveform', body, notes='If the excess were waveform model error it would grow with χ² and with '
            'charge (bright samples dominate both). It does neither except in the top χ² quintile. Waveform-level '
            f'candidates we could not measure, put on det3 at generous sizes, alone on top of the production fit '
            f'(quadrature added): {hs} (H1 10 ns per-channel time skew, H2 ±5 % per-channel pulse width, H3 per-event '
            'footprint 0.20–0.45 mm, H4 the FC footprint 0.30 mm with tails). None comes near the 0.7° needed, and '
            'none raises the toy χ²/dof above ~15. The 2D tilt map, in contrast, removes 0.6° (det3) to 1.1° (det6) '
            'in quadrature, more than the 1D map along the measured coordinate (ws_widthtilt\'s angle_map), and '
            'det6/det7 (read in the same triggers) share part of their residual (correlation 0.15 X, 0.31 Y), which '
            'falls in the chamber centre: the reference or the mounting, as well as each chamber\'s own drift field.',
            foot='data_split.py, tilt_map_xval.py; hypotheses: ladder.py H1–H4 (det3)')


def s_improve(D):
    pos = [BU[d]['position'] for d in DETS]
    ex60 = ON['det3_full_bin60_p78']['excess_at_45']
    t = {d: BU[d]['terms'] for d in DETS}
    rows = [('2D tilt map per chamber', f'measured from M3, cross-validated: removes {min(pos):.2f}–{max(pos):.2f}° '
             'in quadrature', GREEN, 'tilt_map_xval.py: 40 mm cells, even/odd halves'),
            ('20 ns (or interpolated) depth grid', f'removes the steep-track jump ({ex60:.1f}° at 45° on det3) and '
             'the shallow bias', GREEN, 'scan_a: perfect model flat to 45° at 20 ns'),
            ('gain map in the bundle (det4)', f'det4 strip gains cost {t["det4"]["L4"]:.2f}° in quadrature', ORANGE,
             'ladder L4 alone; det4 sigma ln g 0.31'),
            ('v(t) in the bundle (det4, det6)', f'v drift costs det4 {t["det4"]["L8"]:.2f}°, det6 '
             f'{t["det6"]["L8"]:.2f}°', ORANGE, 'ladder L8 alone'),
            ('dead masks + footprint tails (det7)', f'det7: dead strips + CM {t["det7"]["L7"]:.2f}°, tails '
             f'{t["det7"]["L2"]:.2f}°', ORANGE, 'ladder L7, L2 alone')]
    cards = ''.join(sd.card(sd.p(f'<b>{a}</b>', 28) + sd.p(b, 24, MUT), tip=t) for a, b, c, t in rows[:2])
    cards2 = ''.join(sd.card(sd.p(f'<b>{a}</b>', 26) + sd.p(b, 22, MUT), tip=t) for a, b, c, t in rows[2:])
    body = (sd.title('What would improve the measured resolution (reconstruction, not new data)',
                     'Ranked by what each removes in quadrature on the bench X view') +
            sd.row(cards, gap=32) + sd.row(cards2, gap=32) +
            sd.callout(f'The remaining {min(BU[d]["unexplained"] for d in DETS):.1f}–{max(BU[d]["unexplained"] for d in DETS):.1f}° is not reproduced by anything we simulated. Before more modelling: '
                       'check the tilt map at finer cells and with the Y-view track, and whether it transfers between '
                       'runs of the same chamber.', RED, 26))
    D.slide('improve', body, notes='The FC decision (one σ0 = 0.30 mm for both views) is resolution-neutral in the toy '
            '(H4 on det3/det7). The per-event σ0 choice costs det4/det7 ≈ 0.1° on ideal events (L1 vs L0). '
            'The tilt map is geometry (mounting/sag/field), so like w0 it must never be transferred from the bench '
            'to n_TOF without an in-situ check.')


def s_close(D):
    body = (sd.title('<span style="color:#eef0f3">What this does not rule out</span>') +
            sd.col(sd.p('• A waveform effect we did not think of, large enough to make 0.7° and χ² 5–10× the toy '
                        'while staying blind to χ² event by event.', 28, DGREY),
                   sd.p('• The tilt at scales finer than 40 mm (cells too coarse) — the "unexplained" bar may be '
                        'partly tilt.', 28, DGREY),
                   sd.p('• A larger M3 / multiple-scattering term than the 0.15° bound for low-momentum cosmics.',
                        28, DGREY),
                   sd.p('• Y view: the toy is X only (the Y RC spread along the strips is not simulated).', 28, DGREY),
                   sd.p('• n_TOF gas (half the diffusion) and the n_TOF angle mix (30–45°): the floor there is lower '
                        'and the 60 ns jump is the main term.', 28, DGREY), gap=18))
    D.slide('close', body, dark=True)


def main():
    D = sd.Deck('MX17 angle floor', 'The micro-TPC angle floor, the 60 ns jump, and the realism ladder')
    for f in (s_cover, s_setup, s_floor_curve, s_mechanism, s_collapse, s_ladder_setup, s_walk, s_budget,
              s_not_waveform, s_improve, s_close):
        f(D)
    OUT.parent.mkdir(exist_ok=True)
    D.write(OUT, note_meta=dict(
        title='MX17 angle floor: why we do not reach it',
        summary='Floor 0.3°; 60 ns bins break steep tracks once a bin spans ~1.5 footprint widths; every measured '
                'imperfection together gives 0.4–0.8° against 1.1–1.8° measured; the rest is a 2D tilt map and a '
                'χ²-blind term.',
        tags='Cosmic bench, Micromegas, Reconstruction', date='2026-10-27'))
    print(OUT, f'{OUT.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
