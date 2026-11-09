#!/usr/bin/env python3
"""make_deck.py -- slide note on the single-time angle bias: is it unified across TPCs, and is there a paper in it.

Reads the result files of papers/paper2/hits_daq (results/*.json) and $WORK_DATA/paper2_hits/{depth,phasecl,remedies}, so
titles, charts and verdict numbers follow the analysis. The only typed-in data are literature values from sec. 4 of the draft
(the families table and Scholer 2021), which have no machine-readable source.

    ../../.venv/bin/python make_deck.py [--out note.html]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py note.html --slug tpc-single-time-bias --force --deploy
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HD = os.path.join(HERE, '..', 'paper2', 'hits_daq')
WD = os.environ.get('WORK_DATA', '/media/ucla/work') + '/paper2_hits'
L, PITCH, V = 28.08, 0.78, 0.0358
K = (L / PITCH) ** 2
XC, YC, MC, GC, PC, SC, RC = sd.BLUE, sd.PURPLE, sd.GREY, sd.GREEN, sd.ORANGE, sd.GOLD, sd.RED


def res(name):
    return json.load(open(os.path.join(HD, 'results', name)))


U, TA, TP, TD, CH = (res(f) for f in ('universal.json', 'toy_arms.json', 'toy_pitch.json', 'toy_detectors.json', 'chambers_C.json'))
H2 = {v: res('ana1106_h2c_sat_det3.json')[f'sat_det3/{v}'] for v in 'xy'}
DP = json.load(open(f'{WD}/depth/full/summary.json'))
PH = json.load(open(f'{WD}/phasecl/full/summary.json'))
RM = json.load(open(f'{WD}/remedies/run1.json'))
fits, curves = U['fits'], U['curves']


def C_of(key):
    return fits[key]['a'] * K


def errbars(pl, xs, lo, hi, color, w=2.5):
    for x, a, b in zip(xs, lo, hi):
        if a is None or b is None or not all(map(math.isfinite, (a, b))):
            continue
        pl.raw(sd.line(pl.X(x), pl.Y(a), pl.X(x), pl.Y(b), color, w))


def curve(key, tmax=0.45, nmin=60):
    return [r for r in curves.get(key, []) if r['n'] >= nmin and r['tan'] < tmax]


# ------------------------------------------------------------------ headline numbers
Cx1, Cx2, Cy1, Cy2 = (C_of(f'data|run{r}|{v}|u_dream') for v in 'xy' for r in (1, 2))
Cx, Cy = Cx1, Cy1
geom, foot_, nodiff_, metal, base = C_of('resim|x|point|u_dream'), None, C_of('resim|x|dp0|u_dream'), C_of('resim|x|metal|u_dream'), C_of('resim|x|base|u_dream')
foot_inc = nodiff_ - geom
diff_inc = base - nodiff_
ybase, yrc0 = C_of('resim|y|base|u_dream'), C_of('resim|y|rc0|u_dream')
rc_inc = ybase - yrc0
chi = {k: (fits[k]['chi2'], fits[k]['ndf']) for k in fits if k.startswith('data|run') and k.endswith('u_dream')}

# toy law
xs_law, ys_law, names_law = [], [], []
for T in (TA, TP):
    for n, c in T['configs'].items():
        f = c['fits'].get('dream_plain')
        if not f:
            continue
        pr = c['params']
        s2 = pr['sigma0'] ** 2 + pr['Dp'] ** 2 * (pr['L'] / pr['v']) / 2
        xs_law.append(s2 / pr['pitch'] ** 2), ys_law.append(f['C']), names_law.append(f'{n}, p={pr["pitch"]}, σ0={pr["sigma0"]}, Dp={pr["Dp"]}')
law_b, law_a = np.polyfit(xs_law, ys_law, 1)
law_res = float(np.std(np.array(ys_law) - (law_a + law_b * np.array(xs_law))))
n_law = len(xs_law)
s2A = 0.30 ** 2 + 0.021 ** 2 * (L / V) / 2


def toy_ratio(cfg, key, tan):
    cv = TD['configs'][cfg]['curves']
    ts = sorted(float(t) for t in cv)
    pts = [(t, cv[str(t)][key]['ratio']) for t in ts if key in cv.get(str(t), {}) and cv[str(t)][key]['ratio'] > 1]
    return 1 + math.exp(np.interp(math.log(tan), [math.log(a) for a, _ in pts], [math.log(b - 1) for _, b in pts]))


CFG = {'Bortfeldt': 'FSM-like 6mm/0.5', 'Flierl': 'GEM-like 8mm/0.4', 'Iodice': 'NSW-like 5mm/0.4', 'Ntekas': 'NSW-like 5mm/0.4'}
par = []
for l in U['lit']:
    if '40°' in l['label']:          # beyond the toy's tan range
        continue
    meas = 'LTspice' not in l['label'] and 'MC' not in l['label']
    tr = toy_ratio(CFG[l['label'].split()[0]], 'fd_plain', l['tan_true'])
    par.append(dict(l, meas=meas, toy=tr))
meas_par = [q for q in par if q['meas']]
worst = max(max((q['toy'] - 1) / (q['ratio'] - 1), (q['ratio'] - 1) / (q['toy'] - 1)) for q in meas_par)
# Scholer 2021 (NSW VMM, sec. 8.3 and 11.3.3): test-beam peak at the true 28 deg; simulation with sharing off peaks 7 deg high
sch_tan = math.tan(math.radians(28))
sch_N = 5 * sch_tan / 0.425
sch_toy = toy_ratio('NSW-like 5mm/0.4', 'peak_plain', sch_tan)
sch_sim = math.tan(math.radians(35)) / sch_tan

# chambers
ORDER = ['det2', 'det3', 'det4', 'det7']
cmp_rows = []
for d in ORDER:
    c = CH['chambers'][d]
    for v in 'xy':
        A = c['views'][v].get('arms', {})
        if 'data' in A and 'C_err' in A['data']:
            cmp_rows.append((c['name'], v.upper(), A['data']['C'], A['data'].get('C_err', 0), A['model']['C']))
n_within = sum(abs(m / d - 1) <= 0.115 for _, _, d, _, m in cmp_rows)
out_row = max(cmp_rows, key=lambda r: abs(r[4] / r[2] - 1))

# phase floor: outer bins (> quarter pitch)
far = lambda g: [b['far'] if b else None for b in PH['groups'][g]['bins']]  # noqa: E731
floor_x = np.mean(far('x|data|pass')[2:])
floor_y = far('y|data|pass')[-1]

# remedies
Vx, Vy = RM['views']['x'], RM['views']['y']
TIERS = [('dream', 'Leading edge (no correction)', GC and MC), ('dream_rob', '+ robust line', MC), ('dream_e50', '+ edge cut 1/2', MC),
         ('dream_sub15_rob', '+ 15 % subtraction, robust', PC), ('dream_sub30', '+ 30 % subtraction', PC), ('cw_rob', 'Charge-weighted, robust', SC),
         ('cw_sub15_rob', 'Charge-weighted + 15 %', SC)]
best = {v: min((RM['views'][v]['tiers'][k]['H0']['all']['s68'] for k, _, _ in TIERS)) for v in 'xy'}
sW = {v: RM['views'][v]['W']['all']['s68'] for v in 'xy'}

# ------------------------------------------------------------------ deck
D = sd.Deck('Single-time angle bias across TPCs', 'Is the micro-TPC angle bias one effect across TPC families, and is there a paper in it?')

# 1 cover
n_proj = 12366 + 38364
body = (sd.kicker('MX17 · Paper II follow-up · 9 November 2026')
        + '<h1 style="font-size:70px;font-weight:600;line-height:1.1;letter-spacing:-1px">One time per strip reads tracks too steep because of the two ends of the column. '
          'Strip µTPCs fit one picture; other TPCs are predicted, not tested.</h1>'
        + sd.row(sd.bignum(f'{n_proj:,}', 'track projections, chamber A', sd.DBLUE, 'each with a per-track silicon reference', size=64),
                 sd.bignum(f'{len(ORDER)} of 5', 'chambers measured', sd.DGREEN, 'chamber C is 92 % saturated: not extracted', size=64),
                 sd.bignum(f'{len(meas_par)}', 'published measured angle points', sd.DRED, 'from 3 chambers; the whole external test', size=64),
                 sd.bignum('Yes', 'there is a paper', sd.DGREY, 'a research paper, after the fixes on slide 17', size=64), gap=44)
        + sd.row(sd.p('<b>Unified?</b> One mechanism and one scaling explain where the effect appears and how big it is. It is measured only in strip µTPCs.', 30, sd.DINK), sd.p('<b>Paper?</b> Yes, as a measurement with a survey: per-track reference, decomposition, coupling bound, remedies on the same tracks.', 30, sd.DINK), gap=64))
D.slide('cover', body, f'''<p><b>Question 1: did we understand the effect in a unified way in all TPCs?</b> As one mechanism and one scaling, yes. As a test, only for
strip µTPCs. Pad TPCs and LArTPCs are explained by a long column and so predicted to show nothing, directional detectors have no reference, GridPix has nothing
to aggregate. One published result does not fit (Scholer's test beam, slide 8).</p><p><b>Question 2: can we add to the literature?</b> Yes. The new content is the
per-track measurement, the decomposition, the depth and position-in-strip tests of the coupling, and the remedies scored on the same tracks.</p>''', dark=True, short='Answer')

# 2 setup
W_, H_ = 780, 520
x0, zT, zB, dx = 150, 50, 400, 76
inner = []
sp = lambda z: 20 + 34 * math.sqrt(z)  # noqa: E731
zz = np.linspace(0, 1, 40)
tx = [x0 + 90 + 340 * z for z in zz]
ty = [zB - (zB - zT) * z for z in zz]
inner.append(sd.poly([x - sp(z) for x, z in zip(tx, zz)] + [x + sp(z) for x, z in zip(tx[::-1], zz[::-1])], ty + ty[::-1], '#b9cbe6', 0, fill='#c7d6ec'))
inner.append(sd.poly(tx, ty, sd.INK, 4))
inner += [sd.line(40, zB, W_ - 20, zB, sd.INK, 3), sd.line(40, zT, W_ - 20, zT, sd.MUT, 2, '8 6'), sd.T(46, zT - 12, 'cathode', 22, sd.MUT, 'start'), sd.T(46, zB - 12, 'mesh', 22, sd.MUT, 'start')]
for i in range(-1, 8):
    a = x0 + i * dx - 10
    b = a + dx - 8
    xa, xb = x0 + 90, x0 + 430
    c = RC if (b <= xa or a >= xb) else (PC if (a < xa or b > xb) else GC)
    inner.append(f'<rect x="{a}" y="{zB + 12}" width="{b - a}" height="30" rx="3" fill="{c}"/>')
inner.append(sd.T(W_ / 2, zB + 86, 'strips: one time each', 24, sd.MUT))
schem = sd.svg(W_, H_, ''.join(inner), 'Track crossing a drift gap, with the strips coloured by what they collect')
N_ = 8
pl = sd.Plot(800, 520, x=(-1.6, 8.6, 'lin'), y=(-1.2, 8.6, 'lin'), xlabel='strip number (cartoon)', ylabel='hit time (cartoon)')
pl.xticks([]), pl.yticks([])
pl.line([-1.5, 8.5], [-1.5, 8.5], sd.INK, w=3, dash='3 7', markers=False)
pts_x = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
pts_y = [0.15, 0.45, 1, 2, 3, 4, 5, 6, 6.7, 7.35]
cf = np.polyfit(pts_x, pts_y, 1)
pl.line([-1.5, 8.5], list(np.polyval(cf, [-1.5, 8.5])), RC, markers=False)
pl.points(pts_x[2:8], pts_y[2:8], GC, r=10), pl.points([0, 7], [0.45, 6.7], PC, r=10), pl.points([-1, 8], [0.15, 7.35], RC, r=10, marker='diamond')
pl.text(-1.2, 7.4, 'The ends rotate the fit: too steep a track', 26, sd.INK, weight=600)
D.slide('setup', sd.title('A strip gets charge from many depths; only the end strips move the slope', 'Interior strips shift together. Partial and ghost strips shift in opposite directions at the two ends.')
        + sd.row(schem, pl.svg('Cartoon of the time ladder'), gap=44)
        + sd.legend([('interior strip', GC, 'box'), ('partial end strip', PC, 'box'), ('ghost: only spread reaches it', RC, 'box'), ('true ladder', sd.INK, 'dash'), ('fitted ladder', RC)]),
        f'''<p>Displacing the two ends of a least-squares line through N strips rotates it by an amount that falls as 1/N². So
ratio − 1 = tan_rec/tan_true − 1 ≈ C/N² + b, with N = L tanθ/p the strips in the drift column. C holds the end displacements; b is any interior compression.</p>
<p>The published "capacitive coupling" is one way of feeding the ghost strips. Diffusion, the induction footprint and resistive spreading feed them too.</p>
<p>Hover the data points and dotted terms for values. Details hold the method and caveats. Draft: <code>papers/tpc_mixing_review/main.pdf</code>.</p>''',
        foot='Cartoon, not data. Hover any data point in later slides for its value and sample.', short='Mechanism')

# 3 ratio vs tan
def rcurve(pl, key, color, marker='circle', dash=None, line=False):
    r = curve(key)
    t, y = [q['tan'] for q in r], [q['ratio'] for q in r]
    tips = [f'|tan| {q["tan"]:.3f}, N = {q["N"]:.1f}\nratio {q["ratio"]:.3f} ± {q["ratio_err"]:.3f}\nn = {q["n"]}' for q in r]
    if line:
        pl.line(t, y, color, dash=dash, markers=False, tip=key)
    else:
        errbars(pl, t, [q['ratio'] - q['ratio_err'] for q in r], [q['ratio'] + q['ratio_err'] for q in r], color)
        pl.line(t, y, color, w=0 if marker == 'open' else 2, r=8, tips=tips, marker=marker)


r10 = {v: next(q for q in curves[f'data|run1|{v}|u_dream'] if abs(q['tan'] - 0.10) < 0.011)['ratio'] for v in 'xy'}
pl = sd.Plot(1100, 590, x=(0.03, 0.40, 'lin'), y=(0.8, 7, 'log'), xlabel='|tan θ| from the silicon reference', ylabel='tan θ(hits) / tan θ(reference)')
pl.xticks([(t, f'{t:.2f}') for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35)]), pl.yticks([(v, f'{v:g}') for v in (1, 1.5, 2, 3, 5, 7)])
pl.hline(1, sd.INK, label='unbiased')
rcurve(pl, 'data|run1|y|u_dream', YC, 'open'), rcurve(pl, 'data|run1|x|u_dream', XC), rcurve(pl, 'data|run1|x|W', GC, 'diamond')
pl.text(0.17, 4.6, f'At tan θ = 0.1, hits read ×{r10["x"]:.2f} (X) and ×{r10["y"]:.2f} (Y)', 28, sd.INK, weight=600)
pl.text(0.17, 3.7, 'The waveform fit stays on 1', 28, GC, weight=600)
D.slide('ratio', sd.title('Against a per-track reference, the hit angle is far too steep at small angle', 'Chamber A, cosmic muons, Run 1: leading-edge hits against the waveform fit')
        + sd.row(pl.svg('Ratio of hit angle to reference angle'),
                 sd.col(sd.legend([('X view, hits', XC, 'dot')]), sd.legend([('Y view, hits', YC, 'open')]), sd.legend([('X view, waveform fit', GC, 'dot')]),
                        sd.p('No earlier µTPC paper we found compares to a reference <i>per track</i>; they use the chamber&#39;s set angle.', 26, sd.MUT), gap=24, w=500), gap=48),
        '''<p>Hits: a five-sample boxcar above 5σ, 30 % leading-edge time, amplitude-weighted straight line. Reference: the four-plane M3 telescope. Run 2 agrees within 0.02 at every angle.</p>
<p>"Not found" comes from a limited literature search (<code>hits_daq/literature_survey/strip_microtpcs.md</code>): internal ATLAS and BESIII notes are not public.</p>''',
        foot='Median per |tan| bin, 12,366 projections. Source: hits_daq/results/universal.json', short='The bias')

# 4 collapse test
pa, pb = (sd.Plot(810, 540, x=(1.2, 16, 'log'), y=yr, xlabel='N = L tan θ / p (strips in the column)', ylabel=yl)
          for yr, yl in (((0.008, 6, 'log'), 'ratio − 1'), ((0, 16, 'lin'), '(ratio − 1) · N²')))
nt = [(n, str(n)) for n in (2, 3, 5, 8, 12)]
pa.xticks(nt), pb.xticks(nt), pa.yticks([(v, f'{v:g}') for v in (0.01, 0.03, 0.1, 0.3, 1, 3)]), pb.yticks([(v, str(v)) for v in (0, 4, 8, 12, 16)])
flat = {}
for key, col, mk in (('data|run1|x|u_dream', XC, 'circle'), ('data|run2|x|u_dream', XC, 'open'), ('data|run1|y|u_dream', YC, 'circle'), ('data|run2|y|u_dream', YC, 'open')):
    r = [q for q in curve(key) if q['ratio'] > 1]
    Ns = [q['N'] for q in r]
    tips = [f'{key.split("|")[1]} {key.split("|")[2].upper()}: N = {q["N"]:.1f}, tan {q["tan"]:.3f}\nratio {q["ratio"]:.3f} ± {q["ratio_err"]:.3f}, n = {q["n"]}\n(ratio−1)N² = {(q["ratio"] - 1) * q["N"] ** 2:.1f}' for q in r]
    pa.line(Ns, [q['ratio'] - 1 for q in r], col, w=0, r=7, tips=tips, marker=mk)
    errbars(pb, Ns, [(q['ratio'] - 1 - q['ratio_err']) * q['N'] ** 2 for q in r], [(q['ratio'] - 1 + q['ratio_err']) * q['N'] ** 2 for q in r], col)
    pb.line(Ns, [(q['ratio'] - 1) * q['N'] ** 2 for q in r], col, w=0, r=7, tips=tips, marker=mk)
    flat[key] = [(q['ratio'] - 1) * q['N'] ** 2 for q in r if 2 <= q['N'] <= 9]
for C, c in ((Cx, XC), (Cy, YC)):
    pb.hline(C, c, label=f'C = {C:.1f}')
    nn = np.logspace(math.log10(1.25), math.log10(15), 30)
    pa.line(list(nn), list(C / nn ** 2), c, w=2, dash='6 6', markers=False)
fx = flat['data|run1|x|u_dream'] + flat['data|run2|x|u_dream']
fy = flat['data|run1|y|u_dream'] + flat['data|run2|y|u_dream']
span = lambda v: (min(v), max(v))  # noqa: E731
pa.text(1.45, 0.018, 'Falls as 1/N² over a decade', 26, sd.INK, weight=600)
chis = ', '.join(f'{k.split("|")[2].upper()} Run {k.split("|")[1][-1]} {v[0]:.0f}/{v[1]}' for k, v in sorted(chi.items()))
D.slide('collapse', sd.title(f'One coefficient per view: C = {Cx:.1f} (X) and {Cy:.1f} (Y), both runs', f'N = 2–9: X {span(fx)[0]:.1f}–{span(fx)[1]:.1f}, Y {span(fy)[0]:.1f}–{span(fy)[1]:.1f}. X is not statistically a pure 1/N² (right panel).')
        + sd.row(pa.svg('ratio minus one against N'), pb.svg('ratio minus one times N squared against N'), gap=44)
        + sd.legend([('X view, Run 1', XC, 'dot'), ('X view, Run 2', XC, 'open'), ('Y view, Run 1', YC, 'dot'), ('Y view, Run 2', YC, 'open'), ('fit C/N²', sd.MUT, 'dash')]),
        f'''<p>The two-parameter fit of Δtan = a/tan + b·tan gives C = a(L/p)². χ²/ndf: {chis}. So on X the form is a good description but not a statistically complete one. The
paper must state this; quoting C alone hides it. Above N ≈ 9 the quantity is a small difference, so any 1 % angle-scale error moves (ratio − 1)N² strongly.</p>
<p>The data ratios use the calibrated angle scale (w_ref = kw·v·tan + w0). The timing velocity inflates every ratio by kw − 1 and looks like an interior term.</p>''',
        foot='Chamber A, leading-edge hits. Bars are statistical. Source: hits_daq/results/universal.json', short='1/N² test')

# 5 decomposition
pl = sd.Plot(1660, 620, x=(0, 10, 'lin'), y=(0, 16, 'lin'), ylabel='C (ratio − 1 = C / N²)', xlabel='')
pl.xticks([]), pl.yticks([(v, str(v)) for v in (0, 2, 4, 6, 8, 10, 12, 14, 16)])
bw = 118
yfoot, yd = yrc0 - geom, 0
steps = [(0.8, 'geometry', 0, geom, GC), (1.8, '+ footprint', geom, nodiff_, PC), (2.8, '+ diffusion', nodiff_, base, SC), (3.8, 'model, X', 0, base, MC), (4.8, 'data, X', 0, Cx, XC),
         (6.2, 'model, Y\nno RC', 0, yrc0, MC), (7.2, '+ RC', yrc0, ybase, RC), (8.2, 'model, Y', 0, ybase, MC), (9.2, 'data, Y', 0, Cy, YC)]
for x, lab, lo, hi, c in steps:
    pl.vbar(x, hi, bw, c, base=lo, label=f'{"+" if lo else ""}{hi - lo:.1f}', tip=f'{lab}: C from {lo:.1f} to {hi:.1f}'.replace('\n', ' '))
    for j, part in enumerate(lab.split('\n')):
        pl.text(x, -1.0 - 0.75 * j, part, 23, sd.INK, 'middle')
pl.text(0.3, 14.6, f'Four physical terms, each measured or switched off: {geom:.1f} + {foot_inc:.1f} + {diff_inc:.1f}, and +{rc_inc:.1f} on Y', 28, sd.INK, weight=600)
D.slide('decomp', sd.title(f'The bias splits by arm: strip geometry is only {100 * geom / Cx:.0f} % of it on X', 'Model arms on chamber A, each ingredient added in turn, with exact truth and the real residual added back')
        + pl.svg('Waterfall of the end coefficient'),
        f'''<p>This replaces the "share of coupling vs footprint", which is a parametrisation of one variance and has no meaning. Moving charge between a Gaussian footprint and a nearest-neighbour coupling
of equal variance leaves the compression unchanged.</p><p>The arms carry the real data residual, so they include whatever the model does not describe. The clean white-noise toy gives the same increments about one unit lower (C = 6.0 for the full X model, against the data's {Cx:.1f}).
"Reproduced to 2 %" therefore means the arms-with-residual; the pure model is about 15 % below on X.</p>
<p>Draft correction: the text quotes RC as +4.6, which is model Y − model X ({ybase:.1f} − {base:.1f}). Switching RC off on Y gives +{rc_inc:.1f}.</p>''',
        foot='Source: hits_daq/results/universal.json (resim arms). Chamber A, Run 1 data.', short='Decomposition')

# 6 toy law
pl = sd.Plot(1100, 590, x=(0, 1.85, 'lin'), y=(0, 23, 'lin'), xlabel='(σ_rms / p)²   with σ_rms² = σ₀² + D²·T/2', ylabel='C')
pl.xticks([(v, f'{v:.1f}') for v in (0, .4, .8, 1.2, 1.6)]), pl.yticks([(v, str(v)) for v in (0, 5, 10, 15, 20)])
pl.line([0, 1.85], [law_a, law_a + law_b * 1.85], sd.INK, w=2.5, dash='8 6', markers=False)
pl.points(xs_law, ys_law, MC, r=9, tips=[f'{n}\nx = {x:.3f}, C = {y:.2f}' for n, x, y in zip(names_law, xs_law, ys_law)])
dx_, dy_, dt_ = [], [], []
for n, c in TD['configs'].items():
    pr = c['params']
    f = c['fits'].get('fd_plain')
    if f:
        x = (pr['sigma0'] ** 2 + pr['Dp'] ** 2 * (pr['L'] / pr['v']) / 2) / pr['pitch'] ** 2
        dx_.append(x), dy_.append(f['C']), dt_.append(f'{n}: Fermi-Dirac rise, 25 ns sampling\nx = {x:.3f}, C = {f["C"]:.2f}')
pl.points(dx_, dy_, PC, r=9, tips=dt_, marker='diamond')
gx, gy, gt = [], [], []
for k in (1, 2, 3, 4, 6):
    f = fits.get(f'gang|x|data|k{k}')
    if f:
        x, y = s2A / (k * PITCH) ** 2, f['a'] * (L / (k * PITCH)) ** 2
        gx.append(x), gy.append(y), gt.append(f'Chamber A data, {k} strips summed (p = {k * PITCH:.2f} mm)\nx = {x:.3f}, C = {y:.2f}')
pl.points(gx, gy, XC, r=10, tips=gt)
pl.text(0.45, 20, f'C = {law_a:.1f} + {law_b:.1f} (σ_rms/p)²', 34, sd.INK, weight=600)
pl.text(0.45, 18, f'{n_law} toy configurations, rms residual {law_res:.2f}', 26, sd.MUT)
D.slide('law', sd.title('A simple toy with no coupling gives a closed-form C, and real data follow it', 'Toy: strips, Gaussian spread, clustered ionisation, real pulse shape. Blue: chamber A with strips summed in software')
        + sd.row(pl.svg('C against spread over pitch squared'),
                 sd.col(sd.legend([('toy, chamber A shaper', MC, 'dot')]), sd.legend([('toy, other geometries', PC, 'dot')]), sd.legend([('data, strips summed', XC, 'dot')]),
                        sd.callout(f'Multiply out: ratio − 1 ≈ ({law_a:.1f} p² + {law_b:.0f} σ²) / (L tan θ)². The spread term does not depend on the pitch.', sd.INK, 27), gap=24, w=500), gap=48),
        f'''<p>C is the leading-edge end coefficient; σ_rms is the lateral spread averaged over the column. The constant is the strip geometry and the quadratic term the ghost strips. The orange points use a different shaper
(50 ns CR–RC², 25 ns sampling, Fermi-Dirac time) and fall on the same line; that is a real check, since they were not in the fit. Other estimators move C by about ±20 %.</p>
<p>Chamber A data at k = 1 sits about one unit above the line, which the draft traces to the unmodelled same-time floor (slide 12). The line is dominated at the top by one point, p = 0.39 mm.</p>
<p>The law holds for X-type spreading only. The RC term on Y is not in it.</p>''',
        foot='Source: hits_daq/results/toy_{arms,pitch,detectors}.json, universal.json (gang|x|data). The fit is computed here.', short='Toy law')

# 7 literature on N
one = math.radians(1)
pl = sd.Plot(1100, 590, x=(1.4, 18, 'log'), y=(0.005, 3, 'log'), xlabel='N = L tan θ / p', ylabel='ratio − 1')
pl.xticks(nt), pl.yticks([(v, f'{v:g}') for v in (0.01, 0.03, 0.1, 0.3, 1, 3)])
for C, lab, c in ((geom, f'C = {geom:.1f} geometry', GC), (Cx, f'C = {Cx:.1f} chamber A, X', XC), (Cy, f'C = {Cy:.1f} chamber A, Y', YC)):
    nn = np.logspace(math.log10(1.45), math.log10(17), 30)
    pl.line(list(nn), list(C / nn ** 2), c, w=2.5, dash='6 6', markers=False)
    pl.text(13.5, C / 13.5 ** 2 * 1.25, lab.split()[2], 22, c, 'middle')
for q in U['lit']:
    meas = 'LTspice' not in q['label'] and 'MC' not in q['label']
    rec = math.atan(q['ratio'] * q['tan_true'])
    th = math.atan(q['tan_true'])
    if meas:
        hi = math.tan(rec) / math.tan(th - one) - 1
        lo = math.tan(rec) / math.tan(th + one) - 1
        errbars(pl, [q['N']], [max(lo, 0.0051)], [hi], PC)
    tip = f'{q["label"]}\nN = {q["N"]:.1f}, ratio {q["ratio"]:.3f}, (ratio−1)N² = {(q["ratio"] - 1) * q["N"] ** 2:.1f}\n{q["note"]}'
    pl.points([q['N']], [q['ratio'] - 1], PC if meas else SC, r=11 if meas else 8, marker='diamond' if meas else 'open', tips=[tip])
pl.points([sch_N], [sch_sim - 1], RC, r=10, marker='open', tips=[f'Scholer 2021 simulation with sharing off: peak 7° high (tracks above 25°)\nDrawn at 28°: ratio {sch_sim:.2f}, N = {sch_N:.1f}. The true-angle distribution is not given.'])
pl.raw(sd.T(pl.X(sch_N), pl.Y(0.0058), '▼', 32, RC, tip=f'Scholer 2021 test beam: peak at the true 28°, no shift. Resolution 7.1°.\nNo-coupling toy predicts ratio {sch_toy:.2f} here. Drawn at the bottom of the axis.'))
pl.text(5.4, 0.0062, 'Scholer test beam: no shift', 24, RC, 'end', weight=600)
pl.text(17, 1.7, 'Measured points order with the spread, within large errors', 25, sd.INK, 'end', weight=600)
D.slide('lit', sd.title('Published strip µTPCs sit in the same band, with ±1° set-angle errors', 'Tan ratios converted from quoted set and reconstructed angles, using each chamber&#39;s gap and pitch')
        + sd.row(pl.svg('Published points against N'),
                 sd.col(sd.legend([('published, measured (bar: ±1° on set angle)', PC, 'dot')], 25), sd.legend([('published, simulation', SC, 'open')], 25), sd.legend([('Scholer 2021: the two points that do not fit', RC, 'open')], 25),
                        sd.p('(ratio − 1)·N²: Bortfeldt 2.3, Iodice 3.4, Flierl 8–11, in the order of σ/p. Three chambers, D<sub>T</sub> assumed for two.', 25, sd.MUT), gap=24, w=500), gap=48),
        f'''<p>Measured points quote a most probable or mean angle of a selected distribution, against a set angle known to about 1° (Bortfeldt gives 10 ± 1°). At 10° that is about 10 % in tan, as large as the effect.</p>
<p><b>Not in the draft's figure.</b> Scholer 2021 (NSW, VMM): the test-beam peak equals the true 28°, while the simulation with sharing off peaks 7° high. Klitzner's MSc reports most layers slightly <i>small</i> in Ne:CF4. Ntekas's gas-only simulation (0.2° at 30°) and Scholer's sharing-off simulation (7° above 25°) differ by more than an order of magnitude, and neither source explains why. The ±1° bars are illustrative, not each paper's own error.</p>
<p>Sources: <code>hits_daq/literature_survey/strip_microtpcs.md</code>, <code>hits_daq/reports/Capacitive coupling in strip microTPCs.md</code>.</p>''',
        foot='Draft sec. 4 and universal.json; Scholer values typed from the survey notes.', short='Literature')

# 8 parity
pl = sd.Plot(760, 600, x=(0.005, 1.5, 'log'), y=(0.005, 1.5, 'log'), xlabel='ratio − 1, toy with no coupling', ylabel='ratio − 1, published')
tk = [(v, f'{v:g}') for v in (0.01, 0.03, 0.1, 0.3, 1)]
pl.xticks(tk), pl.yticks(tk)
pl.line([0.005, 1.5], [0.005, 1.5], sd.INK, w=2.5, markers=False)
pl.line([0.005, 1.5], [0.01, 3], sd.MUT, w=2, dash='6 6', markers=False)
pl.line([0.01, 1.5], [0.005, 0.75], sd.MUT, w=2, dash='6 6', markers=False)
for q in par:
    x, y = q['toy'] - 1, q['ratio'] - 1
    if not (0.005 < y):
        continue
    if q['meas']:
        rec, th = math.atan(q['ratio'] * q['tan_true']), math.atan(q['tan_true'])
        errbars(pl, [x], [max(math.tan(rec) / math.tan(th + one) - 1, 0.0051)], [math.tan(rec) / math.tan(th - one) - 1], PC)
    pl.points([x], [y], PC if q['meas'] else SC, r=11 if q['meas'] else 8, marker='diamond' if q['meas'] else 'open',
              tips=[f'{q["label"]}\npublished {q["ratio"]:.3f}, toy {q["toy"]:.3f}\n{q["note"]}'])
pl.points([sch_toy - 1], [sch_sim - 1], RC, r=10, marker='open', tips=['Scholer sharing-off simulation against the toy at 28° (peak time)'])
pl.raw(sd.T(pl.X(sch_toy - 1), pl.Y(0.006), '▼', 32, RC, tip=f'Scholer test beam: 0 shift; toy {sch_toy:.2f}'))
rows = []
for q in meas_par:
    rows.append([q['label'].replace(' (meas.)', ''), f'{q["ratio"]:.2f}', f'{q["toy"]:.2f}', f'{(q["toy"] - 1) / (q["ratio"] - 1):.1f}'])
rows.append(['Scholer 2021 NSW 28° test beam', '1.00', f'{sch_toy:.2f}', '<b style="color:%s">no shift seen</b>' % RC])
D.slide('parity', sd.title(f'No coupling needed: toy within ×{worst:.1f} of each measured point, with one exception', 'Toy on each published geometry; footprints assumed, not tuned; dashed lines are ±×2')
        + sd.row(pl.svg('Published against toy'),
                 sd.col(sd.table(['Measured point', 'published', 'toy', 'toy/pub (ratio−1)'], rows, 25, widths=[430, 120, 90, 150]),
                        sd.legend([('measured', PC, 'dot'), ('simulation', SC, 'open'), ('Scholer', RC, 'open')]),
                        sd.p('This shows coupling is <i>not required</i>. It does not show coupling is absent in those chambers.', 28, sd.INK, 600), gap=28, w=840), gap=44),
        f'''<p>The toy uses D_T = 365 µm/√cm for all, a footprint of 0.10 mm (0.15 for the GEM), a 50 ns CR–RC² shaper sampled at 25 ns, and the Fermi-Dirac rise time of the LMU analyses. The VMM points use the peak time.</p>
<p>Agreement to a factor {worst:.1f} on ratio − 1, with four points and an assumed diffusion, is a consistency check. The draft's phrase "within the uncertainty of their assumed footprints" should say this number.</p>
<p>Against the simulations the toy is less close: Ntekas's gas-only value at 30° is 1.008 where the toy gives 1.105. The circuit model of Bortfeldt, which has coupling but no diffusion, also lands near these values: the angle alone cannot tell the two apart. Scholer's test beam is the one measured result the picture misses; its 7.1° resolution, peak-based statistic and angle cuts may explain it, but nobody has shown that.</p>''',
        foot='Source: toy_detectors.json interpolated in tan; published values from universal.json and the survey notes', short='Toy vs published')

# 9 family map
FAM = [('MX17 chamber A', 10.8, 10.8, 0.75, 0.75, XC, 'measured per track'),
       ('ATLAS NSW', 3.4, 3.4, 0.6, 0.6, PC, 'published, consistent'), ('LMU floating strip', 3.6, 3.6, 0.56, 0.56, PC, ''), ('LMU GEM', 6, 6, 0.82, 0.82, PC, ''), ('BESIII CGEM', 2.3, 2.3, 0.38, 0.38, PC, ''),
       ('T2K ERAM', 27, 27, 0.3, 0.3, MC, 'predicted invisible'), ('ALICE, STAR', 110, 110, 0.6, 0.9, MC, ''), ('LArTPCs', 230, 230, 0.3, 0.8, MC, ''),
       ('Directional, short-track', 1.5, 6, 0.5, 8, SC, 'no angle reference'), ('GridPix', 110, 110, 8, 8, GC, 'nothing to aggregate')]
pl = sd.Plot(1660, 590, x=(1, 500, 'log'), y=(0.2, 12, 'log'), xlabel='N at tan θ = 0.3 (strips or pads in a full-drift column)', ylabel='σ_T(L) / p  (spread over pitch)')
pl.xticks([(v, str(v)) for v in (1, 3, 10, 30, 100, 300)]), pl.yticks([(v, f'{v:g}') for v in (0.3, 1, 3, 10)])
for f, c in ((0.1, RC), (0.01, PC), (0.001, MC)):
    ss = np.logspace(math.log10(0.2), math.log10(12), 40)
    NN = np.sqrt((law_a + law_b * ss ** 2 / 2) / f)
    keep = NN <= 500
    pl.line(list(NN[keep]), list(ss[keep]), c, w=2.5, dash='8 6', markers=False)
    i0 = int(np.argmin(np.abs(ss - 0.24)))
    pl.text(float(NN[i0]) * 1.05, float(ss[i0]), f'bias {f * 100:g} %', 23, c, weight=600)
LP = {'MX17 chamber A': (1.08, 1.28), 'ATLAS NSW': (0.86, 1.18), 'LMU floating strip': (1.1, 0.74), 'LMU GEM': (0.9, 1.12), 'BESIII CGEM': (0.93, 0.95), 'T2K ERAM': (1.1, 1.05), 'ALICE, STAR': (0.9, 0.78), 'LArTPCs': (0.55, 0.62), 'Directional, short-track': (1.18, 1.05), 'GridPix': (1.1, 1.05)}
for n, x1, x2, y1, y2, c, _ in FAM:
    xm, ym = math.sqrt(x1 * x2), math.sqrt(y1 * y2)
    if x2 > x1:
        pl.raw(sd.line(pl.X(x1), pl.Y(ym), pl.X(x2), pl.Y(ym), c, 3))
    if y2 > y1:
        pl.raw(sd.line(pl.X(xm), pl.Y(y1), pl.X(xm), pl.Y(y2), c, 3))
    pl.points([xm], [ym], c, r=12, tips=[f'{n}\nN(0.3) = {x1:g}–{x2:g}, σ_T/p = {y1:g}–{y2:g}'])
    a, b = LP[n]
    pl.text(xm * a, ym * b, n, 24, sd.INK, 'end' if a < 1 else 'start', weight=600 if c == XC else 400)
D.slide('map', sd.title('The same two numbers place every TPC family, and only strip µTPCs land in the visible region', 'Dashed lines: the toy law at tan θ = 0.3, charge-integrated spread σ_rms² ≈ σ_T²/2')
        + pl.svg('TPC families on the N, sigma over pitch plane')
        + sd.legend([('measured per track', XC, 'dot'), ('published, consistent', PC, 'dot'), ('predicted invisible, not tested', MC, 'dot'), ('no angle reference', SC, 'dot'), ('single electrons, nothing to aggregate', GC, 'dot')]),
        '''<p>Pad TPCs and LArTPCs have the same spread in pitch units as MX17, but columns 10–100 times longer. They also read a charge-weighted or deconvolved time. Their drift scale is fixed to 0.1 % by lasers and silicon, so no compression is expected or seen.</p>
<p>The contours extrapolate the toy law to N and σ/p outside its fit (fit range N ≲ 20, σ_rms/p ≲ 1.3) and to estimators it was not fit to. Read them as order-of-magnitude. Values come from the draft's Table 1 and are derived from published geometry; the LMU diffusion is assumed.</p>
<p>The GridPix point is at N = 110, where the law would apply only if many electrons shared a time; each electron has its own, so nothing is aggregated.</p>''',
        foot='Draft Table 1 (sec. 4). The line "bias 0.1 %" is about the precision of the large-TPC drift calibration.', short='All TPCs')

# 10 verdict table
ch = lambda t, c: f'<b style="color:{c}">{t}</b>'  # noqa: E731
rows = [['Strip µTPC, MX17', 'Per-track reference, 4 chambers, full decomposition', ch('Tested', sd.GREEN)],
        ['Strip µTPC, published', f'{len(meas_par)} measured points, toy within ×{worst:.1f}; Scholer test beam misses', ch('Consistent, 1 exception', sd.ORANGE)],
        ['Pad TPC (ALICE, STAR, T2K)', 'Column 10–100× longer, charge-weighted time; bias &lt; 0.1 % predicted', ch('Explained, untested', sd.GREY)],
        ['LArTPC', 'Hundreds of wires, 2-D deconvolution; short segments never tested', ch('Explained, untested', sd.GREY)],
        ['Directional, short track', 'Has the column; describes end artefacts, but no reference to reveal a slope bias', ch('Cannot be tested', sd.GOLD)],
        ['GridPix', 'One time per electron', ch('Effect absent by design', sd.GREEN)],
        ['Drift tubes', 'Angle-dependent space-time relation: same idea, already calibrated in', ch('Analogy only', sd.GREY)]]
D.slide('verdict', sd.title('Unified as one mechanism; tested only where the column is short', 'What the evidence actually is, family by family')
        + sd.table(['Family', 'Evidence', 'Status'], rows, 29, widths=[430, 950, 290], align=['left', 'left', 'left'])
        + sd.callout('So: a single account of why the effect appears, in which detectors, and how big. Not a measurement in every TPC, and the dataset cannot make it one.', sd.INK, 30),
        '''<p>"Unified" is justified in this sense: the formula needs only N and σ/p, and the absence of the effect in long-column detectors follows from it. The paper should not claim that the long-column and directional families have been shown to obey it.</p>''', short='Family verdict')

# 11 depth
g = np.array(DP['grid'])
mk = (g >= 0) & (g <= 800)
panels = []
for v in 'xy':
    C = DP['curves'][f'{v}|tight|phall']
    pl = sd.Plot(810, 520, x=(0, 800, 'lin'), y=(0, 2.2 if v == 'y' else 1.4, 'lin'), xlabel='time after the mesh charge (ns)', ylabel='first-neighbour share', title=f'{v.upper()} view, n = {C["n"]} planes')
    pl.xticks([(t, str(t)) for t in (0, 200, 400, 600, 800)]), pl.yticks([(t, f'{t:.1f}') for t in np.arange(0, (2.2 if v == 'y' else 1.4), 0.4)])
    pl.band(list(g[mk]), list(np.array(C['data']['r1'])[mk] - np.array(C['data']['r1_err'])[mk]), list(np.array(C['data']['r1'])[mk] + np.array(C['data']['r1_err'])[mk]), sd.INK, alpha=0.2)
    for arm, c, dash in (('pass', XC, None), ('metal', PC, None), ('nodiff', GC, None), ('pureK', RC, '6 6')):
        pl.line(list(g[mk]), list(np.array(C[arm]['r1'])[mk]), c, dash=dash, markers=False, tip=arm)
    pl.line(list(g[mk]), list(np.array(C['data']['r1'])[mk]), sd.INK, w=4, markers=False, tip='data')
    panels.append(pl.svg(f'Neighbour share against depth, {v}'))
sl = 3.5
D.slide('depth', sd.title('Neighbour charge grows with drift depth: diffusion does that, a coupling cannot', 'Waveform stacks on near-normal tracks (|tan θ| &lt; 0.008), chamber A, Runs 1+2')
        + sd.row(*panels, gap=44)
        + sd.legend([('data (band: bootstrap)', sd.INK), ('full model', XC), ('metallic: floor footprint, no RC', PC), ('no diffusion', GC), ('footprint replaced by 13 % coupling', RC, 'dash')]),
        '''<p>Plotted to 800 ns: beyond the column end the share is a ratio of tails and diverges on Y. The first-neighbour share is (W₋₁ + W₊₁)/W₀ in the same drift-time slice. Without diffusion the model stays flat. The rate of growth prefers D_p ≈ 0.021 mm/√ns (D_T ≈ 350 µm/√cm), 8 % below the pass value, a recorded tension in <code>docs/PARAMETERS.md</code>.</p>
<p>On Y the resistive spreading adds most of the growth. The mesh-end value says a same-time share is required: the metallic arm starts at half the data's value.</p>
<p>At depth the second-neighbour share exceeds every arm, so the lateral tails are heavier than Gaussian. Its origin is not identified.</p>''',
        foot='Source: $WORK_DATA/paper2_hits/depth/full/summary.json (coupling/depth_ratio.py)', short='Depth')

# 12 phase
pb_ = PH['phase_bins']
mid = [(pb_[i] + pb_[i + 1]) / 2 for i in range(len(pb_) - 1)]
panels = []
for v in 'xy':
    pl = sd.Plot(810, 500, x=(0, 0.5, 'lin'), y=(0, 0.25, 'lin'), xlabel='|track position within the strip| (pitch)', ylabel='far-neighbour share at t₀', title=f'{v.upper()} view')
    pl.xticks([(t, f'{t:.1f}') for t in (0, .1, .2, .3, .4, .5)]), pl.yticks([(t, f'{t:.2f}') for t in (0, .05, .1, .15, .2, .25)])
    for grp, c, dash in ((f'{v}|noK_white|refit', XC, None), (f'{v}|pureK_white|refit', RC, None)):
        pl.line(mid, far(grp), c, dash=dash, r=7, tip=grp)
    B = PH['groups'][f'{v}|data|pass']['bins']
    errbars(pl, mid, [b['far'] - b['far_err'] for b in B], [b['far'] + b['far_err'] for b in B], sd.INK)
    pl.line(mid, [b['far'] for b in B], sd.INK, w=4, r=9, tips=[f'|phase| {m:.2f}: far share {b["far"]:.3f} ± {b["far_err"]:.3f}, n = {b["n"]}' for m, b in zip(mid, B)])
    panels.append(pl.svg(f'Far-neighbour share against position in strip, {v}'))
D.slide('phase', sd.title(f'A fixed coupling is bounded at about {100 * floor_x:.0f} % (X) and {100 * floor_y:.0f} % (Y) per neighbour', 'A Gaussian footprint falls off across the strip; a fixed coupling stays flat. Published coupling values are 15–30 %.')
        + sd.row(*panels, gap=44)
        + sd.legend([('data', sd.INK, 'dot'), ('closure: Gaussian footprint, no coupling', XC, 'dot'), ('closure: 13 % coupling, 33 µm footprint', RC, 'dot')])
        + sd.p('The floor is real but unexplained, and its label does not change the angle bias.', 28, sd.INK, 600),
        f'''<p>The closure curves are windows regenerated with known truth and fresh white noise, then refitted with the production fitter exactly like the data. Refit minus truth is 0.007 mm (Gaussian) to 0.07 mm (coupling), so the test is not an artefact of the fitted position.</p>
<p>X: far share flat at 0.060–0.067 from a quarter pitch out, between Gaussian (0.040) and 13 % coupling (0.127). Y: 0.10 at the edge, with the centre bin below the Gaussian: the Y core is narrower than the fitted 0.28 mm.</p>
<p><b>Draft inconsistency:</b> the first paragraph of sec. 5.4 says no same-time floor is required on Y; the closure paragraph finds one at 0.10. The closure paragraph is the one that stands. The bound is on this chamber only.</p>''',
        foot='Source: $WORK_DATA/paper2_hits/phasecl/full/summary.json (coupling/phase_closure.py). ~2,900 planes per view.', short='Coupling bound')

# 13 chambers
names = [CH['chambers'][d]['name'] for d in ORDER]
panels = []
for v, ymax in (('x', 10), ('y', 13)):
    pl = sd.Plot(540, 500, x=(-0.6, 3.6, 'lin'), y=(0, ymax, 'lin'), xlabel='chamber', ylabel='C', title=f'{v.upper()} view')
    pl.xticks([(i, n) for i, n in enumerate(names)]), pl.yticks([(t, str(t)) for t in range(0, ymax + 1, 2)])
    for i, d in enumerate(ORDER):
        A = CH['chambers'][d]['views'][v].get('arms', {})
        if 'data' not in A or 'C_err' not in A['data']:
            pl.text(i, 0.5, 'too few', 20, sd.MUT, 'middle')
            continue
        for arm, c, r_, mkk in (('point', GC, 8, 'circle'), ('metallic', PC, 8, 'circle'), ('model', XC, 10, 'diamond')):
            pl.points([i + 0.16], [A[arm]['C']], c, r=r_, marker=mkk, tips=[f'{names[i]} {v.upper()} {arm}: C = {A[arm]["C"]:.2f}'])
        errbars(pl, [i - 0.12], [A['data']['C'] - A['data']['C_err']], [A['data']['C'] + A['data']['C_err']], sd.INK)
        pl.points([i - 0.12], [A['data']['C']], sd.INK, r=10, tips=[f'{names[i]} {v.upper()} data: C = {A["data"]["C"]:.2f} ± {A["data"]["C_err"]:.2f}, n = {A["data"]["n"]}'])
    panels.append(pl.svg(f'C on four chambers, {v}'))
pl = sd.Plot(520, 500, x=(1.5, 6, 'lin'), y=(0, 7, 'lin'), xlabel='D_RC (10⁻⁴ mm²/ns)', ylabel='C increment from RC, Y', title='RC term')
pl.xticks([(t, str(t)) for t in (2, 3, 4, 5, 6)]), pl.yticks([(t, str(t)) for t in range(0, 8)])
xr, ymo, yda, lb = [], [], [], []
for d in ORDER[:3]:
    c = CH['chambers'][d]
    A = c['views']['y']['arms']
    xr.append(c['hyper']['rc_D_y'] * 1e4), ymo.append(A['model']['C'] - A['rc0']['C']), yda.append(A['data']['C'] - A['rc0']['C']), lb.append(c['name'])
ix = np.argsort(xr)
pl.line([xr[i] for i in ix], [ymo[i] for i in ix], XC, r=9, tips=[f'{lb[i]}: model increment {ymo[i]:.1f}' for i in ix])
pl.points([xr[i] for i in ix], [yda[i] for i in ix], sd.INK, r=9, tips=[f'{lb[i]}: data C − model RC-off = {yda[i]:.1f}' for i in ix])
for i in ix:
    pl.text(xr[i], max(ymo[i], yda[i]) + 0.55, lb[i], 22, sd.INK, 'middle')
panels.append(pl.svg('RC increment against spreading constant'))
nm, nv, dv, _, mv = out_row[0], out_row[1], out_row[2], out_row[3], out_row[4]
D.slide('chambers', sd.title(f'Across chambers: {n_within} of {len(cmp_rows)} views agree within 11 %; chamber {nm} {nv} is {100 * abs(mv / dv - 1):.0f} % off', 'Clean calibration planes per chamber. The RC term orders with the spreading constant, on the three chambers that can test it.')
        + sd.row(*panels, gap=24)
        + sd.legend([('data', sd.INK, 'dot'), ('model', XC, 'dot'), ('metallic arm', PC, 'dot'), ('geometry only', GC, 'dot')]),
        f'''<p>The truth here is the fit's own slope on the calibrated scale, not the silicon reference, on the clean bootstrap planes. Absolute C on chamber A is 10–15 % lower than on the full sample ({Cx:.1f}/{Cy:.1f}), so only chamber-to-chamber comparison means anything.</p>
<p>Chamber E's X value is the outlier against the toy law (8.0 against 5.8); it has dead and low-gain bands and a short column. Chamber D's X has 177 planes and Y too few to fit. Chamber C is excluded (92 % saturated).</p>
<p><b>Draft correction:</b> the conclusions say the three further chambers reproduce the model within 10 % on both views. Chamber D's X does not ({dv:.1f} against {mv:.1f}, 177 planes).</p>''',
        foot='Source: hits_daq/results/chambers_C.json. Chambers A = det3, B = det2, D = det7, E = det4.', short='Chambers')

# 14 remedies
blocks = []
for v, title_ in (('x', 'X view'), ('y', 'Y view')):
    rows = [(lab, RM['views'][v]['tiers'][k]['H0']['all']['s68'], c, f'{k}: calibrated s68, steepness bias {RM["views"][v]["tiers"][k]["naive"]["all"]["steep"]:+.2f}° raw') for k, lab, c in TIERS]
    rows.append(('Forward model on hits (H2c)', H2[v]['H2c']['s68'], XC, 'one time + one amplitude per strip, same physics as the waveform fit'))
    rows.append(('Forward fit of waveforms', sW[v], GC, 'calibrated on the reference'))
    blocks.append(sd.col(sd.p(title_, 30, sd.INK, 600), sd.hbars(rows, 3.0, width=330, h=30, label_w=380, fmt=lambda x: f'{x:.2f}°', size=24), w=800))
D.slide('remedies', sd.title('No hit-level correction closes the gap to the waveform fit', 'Angle resolution s68 after a global slope calibration, chamber A Run 1, every correction on the same 6,190 projections per view')
        + sd.row(*blocks, gap=44)
        + sd.p(f'Best hit recipe {best["x"]:.2f}° (X) and {best["y"]:.2f}° (Y), against {sW["x"]:.2f}° and {sW["y"]:.2f}° for the waveform fit. The best 30 % factor on X is not a capacitance: the floor allows about 6 %.', 28, sd.INK, 600),
        '''<p>Edge cut and shift lower the raw bias by 10–15 % at every angle on X (the toy predicts 24 %). <b>Draft correction:</b> the section heading and the conclusions say "a quarter". The 15 % subtraction reduces the resolution gap to the waveform fit by about 40 %; the text says "halves".</p>
<p>30 % subtraction helps X and hurts Y, where the neighbours' signals are real delayed RC charge. The charge-weighted time alone does nothing, since there is no interior term to remove. The forward model on hit lists is close to the waveform fit: most of the remaining gap is information that hits do not keep.</p>''',
        foot='Source: $WORK_DATA/paper2_hits/remedies/run1.json; H2c from results/ana1106_h2c_sat_det3.json. Grey: edge, orange: subtraction, gold: charge-weighted.', short='Remedies')

# 15 what the review said
rows = [['Q1 Coupling replaces footprint?', 'Rejected: position-in-strip test survives closure', ch('Holds', sd.GREEN)],
        ['Q2 Coupling vs footprint shares?', 'Meaningless; replaced by the C decomposition', ch('Replaced', sd.BLUE)],
        ['Q3 Fitted k1 above closure?', 'Means nothing at χ²/dof ≈ 66', ch('Dropped', sd.GREY)],
        ['Q4 Chip boundary isolates electronics?', 'Suggestive for −2 strip only; 33+33 planes', ch('Weakened', sd.ORANGE)],
        ['Q5 Origin of the same-time floor?', 'Not found; seen again as mesh-end ghosts on X', ch('Open', sd.RED)],
        ['Q6 Unexplained items', '−1 trend: maybe a baseline offset (untested). Below model at centre: Y only. Interior slope: open', ch('Partly open', sd.ORANGE)]]
D.slide('review', sd.title('The second read confirmed the core and removed the weak claims', 'The six questions of the morning handoff, answered on existing data')
        + sd.table(['Question', 'Outcome', 'Status'], rows, 29, widths=[560, 880, 220], align=['left', 'left', 'left'])
        + sd.callout('Limit of that check: the session that reviewed the coupling study also wrote the toy, the universal fit and the draft. Those have had no second read.', RC, 29),
        '''<p>Review memo: <code>papers/paper2/hits_daq/REVIEW_COUPLING_2026-11-09.md</code>. The numbers on the draft-corrections slide came from this deck's reading of the draft against its own tables. They are not a full check of the 24 pages.</p>''', short='Second read')

# 16 novelty
rows = [['Hits read too steep, ∝ 1/tan²', 'Known (Bortfeldt, ATLAS, BESIII, LMU)', 'Quantified as C/N²'],
        ['Blamed on capacitive coupling', 'The standard statement', ch('New: bounded at 6–10 % on A; not required by the angle', sd.GREEN)],
        ['Gas-only simulations still show a bias', 'Known, left unexplained', ch('New: explained by the end term; closed-form C', sd.GREEN)],
        ['Per-track external reference', 'Not found in a limited search', ch('New measurement', sd.GREEN)],
        ['Decomposition by physical term', 'Not found', ch('New', sd.GREEN)],
        ['Depth and position-in-strip separation', 'Not found for strip µTPCs', ch('New method', sd.GREEN)],
        ['Remedies compared on the same tracks', 'Each paper tests only its own', ch('New', sd.GREEN)],
        ['Waveform forward fit', 'Relatives at T2K and in LArTPC', 'Belongs to Paper II']]
D.slide('novelty', sd.title('Yes, there is a paper, but it is a measurement with a survey, not a review', 'What is already in print, and what we would add')
        + sd.table(['Claim', 'In the literature', 'Our contribution'], rows, 26, widths=[590, 540, 530], align=['left', 'left', 'left'])
        + sd.callout('Recommended framing: "Origin and scaling of the track-angle bias in single-time micro-TPC readout". Measurement first, family survey as the discussion.', sd.BLUE, 29),
        '''<p>A review journal would expect balanced coverage; this has one new dataset at its centre and takes a position. NIM A or JINST fit it. The venue and author list are Dylan's decision (<code>\\tbw</code> marks in the draft).</p>
<p>Overlap: the forward fit and H2c should stay in Paper II and be cited. The draft leans on Papers I and II, so its submission order matters. The novelty rows are stated as "not found" because the search was limited: Alexopoulos 2019 (NIM A 937) is closed-access and unread, and internal ATLAS/BESIII notes are not public.</p>''', short='Is it a paper?')

# 17 fixes
rows = [['Text ≠ own tables', f'RC +4.6 → +{rc_inc:.1f}; "within 10 %" (D X: 7.2 vs 5.1); edge cuts "a quarter" → 10–15 %; "halves" → ~40 %; Y floor stated both ways', 'Hours'],
        ['Fit quality', f'Report χ²/ndf of the C/N² + b form ({chi["data|run1|x|u_dream"][0]:.0f}/{chi["data|run1|x|u_dream"][1]} and {chi["data|run2|x|u_dream"][0]:.0f}/{chi["data|run2|x|u_dream"][1]} on X)', 'Hours'],
        ['"Reproduced to 2 %"', 'Say it is the arms carrying the real residual; the clean toy is about one unit of C lower', 'Hours'],
        ['Scholer test beam', 'Add it to Figure 2 and discuss; also Klitzner and the Ntekas–Scholer simulation mismatch', 'A day'],
        ['Literature points', 'Add set-angle errors; state the toy–data agreement as a factor', 'A day'],
        ['Independent second read', 'Toy, universal fit and the full draft; citation check of every quote', 'A session'],
        ['Known-angle cross-check', 'SPS rows from DAGs 4646000/4647206: the one test on another chamber against a known angle', 'When scored'],
        ['Open physics', 'Floor origin, E X outlier: state as limitations', 'Wording']]
D.slide('fixes', sd.title('Before submission: mostly wording, plus one published result to confront', 'No new data needed; every item uses what we have')
        + sd.table(['Item', 'What to do', 'Effort'], rows, 27, widths=[370, 1090, 170], align=['left', 'left', 'left']),
        '''<p>The draft-vs-table discrepancies are from reading sections 5, 6 and 8 against tables 2–5 and the result files; the corrected numbers are computed on the matching slides. "Effort" is my estimate.</p>
<p>The DAGs were not scored for this deck. See the handoff for the scoring steps.</p>''', short='To do')

# 18 close
ul = lambda items: '<ul style="font-size:29px;line-height:1.4;padding-left:34px;display:flex;flex-direction:column;gap:14px">' + ''.join(f'<li>{i}</li>' for i in items) + '</ul>'  # noqa: E731
body = (sd.kicker('What this does not rule out, and what is yours to decide')
        + sd.row(sd.col(sd.p('Not ruled out', 36, sd.DRED, 600),
                        ul(['Real coupling of 15–30 % in the <i>other</i> chambers: the bound is on chamber A.',
                            'A different cause of Scholer&#39;s unshifted test beam, or a limit of the picture on VMM chambers.',
                            'A same-time process we have not named: the 6–10 % floor has no origin.',
                            'Short segments in long-drift TPCs: predicted a few per cent, never measured.',
                            'Everything measured here is one gas, one drift field, 60 ns DREAM sampling.']), w=800),
                 sd.col(sd.p('Your decisions', 36, sd.DGREEN, 600),
                        ul(['Keep it as a third paper, and call it a measurement rather than a review.',
                            'Whether Paper II keeps its own sections 3 and 6 or points here.',
                            'Authors and venue.',
                            'Is chamber C worth a window extraction, given 92 % saturation?']), w=740), gap=64))
D.slide('close', body, '''<p>The dataset is final, so "other gases and shaping" stay stated limitations. The toy reaches other shapers and geometries only as simulation.</p>
<p>Draft: <code>papers/tpc_mixing_review/main.pdf</code>; every number: <code>papers/tpc_mixing_review/NUMBERS.md</code>; this deck: <code>papers/tpc_mixing_review/make_deck.py</code>.</p>''', dark=True, short='Limits')

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.environ.get('CLAUDE_JOB_DIR', HERE), 'tmp' if os.environ.get('CLAUDE_JOB_DIR') else '', 'tpc-single-time-bias.html'))
a = ap.parse_args()
print(D.write(a.out, note_meta=dict(title='Single-time angle bias across TPCs', summary='Is the micro-TPC angle bias one effect across TPC families, and is there a paper in it? Study, literature placement, draft status.', tags='X17,TPC,reconstruction,paper', date='2026-11-09')))
print(f'C {Cx:.2f}/{Cy:.2f}; geom {geom:.2f} foot {foot_inc:.2f} diff {diff_inc:.2f} rc {rc_inc:.2f}; law {law_a:.2f}+{law_b:.2f} n={n_law} res={law_res:.2f}; worst {worst:.2f}; floor {floor_x:.3f}/{floor_y:.3f}; within {n_within}/{len(cmp_rows)} out {out_row}; best {best} W {sW}; sch_toy {sch_toy:.3f}')
print('N2-9', span(fx), span(fy), 'par', [(q['label'], round(q['ratio'], 3), round(q['toy'], 3)) for q in par])
