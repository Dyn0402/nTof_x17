#!/usr/bin/env python3
"""make_deck.py [--out note.html] -- slide note: what every TPC family does about depth mixing, and what MX17 adds.

Reads survey_data.py (the literature, from the notes in this directory) and ../results/ana1106_*.json (our numbers), so
numbers in titles and charts are computed.  Scan results (gang_scan, resim_shape, H2c on variants) appear when their
JSON files exist in ../results/, and say 'running' otherwise.
Publish: python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py note.html --slug mx17-tpc-literature --force --deploy
"""
import argparse
import html
import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import survey_data as S  # noqa: E402

RES = os.path.join(HERE, '..', 'results')
FC = {'mx': sd.INK, S.STRIP: sd.RED, S.PAD: sd.BLUE, S.LAR: sd.GREEN, S.SHORT: sd.GOLD}
FN = {'mx': 'MX17', S.STRIP: 'Strip micro-TPC', S.PAD: 'Pad TPC', S.LAR: 'Liquid argon', S.SHORT: 'Short-track / pixel'}
BINS = [('0.05-0.1', 0.075), ('0.1-0.2', 0.15), ('0.2-0.3', 0.25), ('0.3-0.5', 0.37)]


def tbl(cols, rows, **k):
    return sd.table(cols, rows, align=['left'] * len(cols), **k)


def J(n):
    return json.load(open(os.path.join(RES, n)))


def lam_mid(g):
    a, b = S.lam(g)
    return math.sqrt(a * b)


def fig_num(v):
    if v >= 0.01:
        return f'{v:.2g}'
    m, e = f'{v:.1e}'.split('e-0')
    return m + '×10⁻' + e.translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))


def column_bar(g, w=600):
    """to scale: the projected column at tan 0.3 is the full bar; ticks are readout channels; the bell is sigma_T at full drift"""
    a, b = S.lam(g)
    L = g['L'] * S.TAN_REF
    n = L / g['pitch'] if g.get('pitch') else None
    x0, xw, y = 20, w - 40, 100
    o = [sd.T(x0, 20, 'Projected column at tan θ = 0.3, same width for every detector', 19, sd.MUT, 'start')]
    o.append(f'<rect x="{x0}" y="{y}" width="{xw}" height="20" fill="{sd.BG2}" stroke="{sd.RULE}"/>')
    if n and n < 90:
        for i in range(int(n) + 1):
            xx = x0 + xw * i / n
            o.append(sd.line(xx, y, xx, y + 20, sd.MUT, 1))
    s_px = max(xw * math.sqrt(a * b), 1.5)
    cx = x0 + xw * 0.72
    pts = [(cx + k / 60 * 3 * s_px, y - 3 - 50 * math.exp(-0.5 * (k / 60 * 3) ** 2)) for k in range(-60, 61)]
    pts = [q for q in pts if x0 <= q[0] <= x0 + xw]
    o.append(sd.poly([q[0] for q in pts], [q[1] for q in pts], sd.RED, 2.5))
    o.append(sd.T(x0, y - 22, 'red: σ_T at full drift', 20, sd.RED, 'start'))
    nt = f'{n:.0f} channels' if n else 'channel count not read'
    lab = fig_num(a) if abs(a - b) / a < 0.02 else f'{fig_num(a)}-{fig_num(b)}'
    star = ' (D_T assumed)' if g.get('assumed') else ''
    o.append(sd.T(x0, y + 50, f'{(f"{g['L'] * 0.3:.0f}" if g['L'] * 0.3 >= 100 else f"{g['L'] * 0.3:.3g}")} mm, {nt}. Λ = {lab}{star}', 23, sd.INK, 'start', 600))
    return sd.svg(w, 160, ''.join(o), 'column to scale')


def li(items, size=25):
    return ''.join(f'<p style="font-size:{size}px;line-height:1.28;padding-left:18px;text-indent:-18px;margin-bottom:9px">• {x}</p>' for x in items)


def head(t, color):
    return f'<p style="font-size:21px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:{color};margin-bottom:10px">{t}</p>'


def specs_tbl(rows):
    return ''.join(f'<div style="display:flex;gap:14px;padding:5px 0;border-bottom:1px solid {sd.RULE}">'
                   f'<p style="width:150px;font-size:21px;color:{sd.MUT};flex:none">{k}</p><p style="font-size:21px">{v}</p></div>' for k, v in rows)


def quotes_html(e):
    q = ''.join(f'<p>“{html.escape(t)}” — <a href="{u}">{html.escape(w)}</a></p>' for t, w, u in e['quotes'])
    return f'<p><b>{e["name"]}. Sources, verbatim.</b> Quotes were string-matched against the extracted paper text; (derived) numbers are ours.</p>{q}<p><b>Not covered:</b> {html.escape(e["gaps"])}</p>'


def exp_slide(D, e):
    c = FC[e['fam']]
    left = []
    if e.get('geo'):
        left.append(column_bar(e['geo'], 620))
    left.append(sd.card(head('Used for', c) + sd.p(e['use'], 22) + '<div style="height:12px"></div>' + specs_tbl(e['specs']), w=640, pad=22))
    right = sd.col(
        sd.row(sd.card(head('What they did', c) + li(e['did']), w=490, pad=22),
               sd.card(head('What they concluded', c) + li(e['concl']), w=490, pad=22), gap=20),
        sd.callout('<b>For us:</b> ' + e['us'], color=c, size=24), gap=18, w=1000)
    body = sd.title(e['title'], e['name'] + ' · ' + FN[e['fam']]) + sd.row(sd.col(*left, gap=12, w=640), right, gap=24)
    D.slide(e['id'], body, quotes_html(e), short=e['short'], pad='64px 128px 72px')


def ratio(steep, c):
    if steep is None:
        return None
    return math.tan(math.atan(c) + math.radians(steep)) / c


def effect_diagram():
    w, h = 1000, 540
    o = []
    gx0, gx1, gy0, gy1 = 110, 560, 70, 400
    o.append(f'<rect x="{gx0}" y="{gy0}" width="{gx1 - gx0}" height="{gy1 - gy0}" fill="{sd.CARD}" stroke="{sd.RULE}"/>')
    o.append(sd.T(gx1 - 8, gy0 + 28, 'cathode', 21, sd.MUT, 'end'))
    o.append(sd.T(gx0 + 8, gy1 - 12, 'mesh', 21, sd.MUT, 'start'))
    o.append(sd.T(gx0 - 12, (gy0 + gy1) / 2, 'drift depth', 22, sd.MUT, 'middle', rot=-90))
    n = 9
    sw = (gx1 - gx0) / n
    for i in range(n):
        o.append(f'<rect x="{gx0 + i * sw + 3:.0f}" y="{gy1 + 6}" width="{sw - 6:.0f}" height="16" fill="{sd.ORANGE if i == 4 else sd.GREY}"/>')
    o.append(sd.T((gx0 + gx1) / 2, gy1 + 56, 'strips (one time each)', 22, sd.MUT))
    x_t, x_b = gx0 + 15, gx1 - 15
    o.append(sd.line(x_t, gy0 + 2, x_b, gy1 - 2, sd.INK, 3.5))
    o.append(sd.T(x_t + 50, gy0 + 70, 'track', 22, sd.INK, 'start', 700))
    sx = gx0 + 4.5 * sw
    for f, col in ((0.3, sd.BLUE), (0.5, sd.ORANGE), (0.7, sd.PURPLE)):
        px = x_t + f * (x_b - x_t)
        py = gy0 + f * (gy1 - gy0)
        o.append(sd.arrow(px, py, sx, gy1, col, 2.5))
        o.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="7" fill="{col}"/>')
    x0, x1, y0, y1 = 660, 970, 80, 400
    o.append(sd.line(x0, y1, x1, y1, sd.MUT, 2) + sd.line(x0, y0, x0, y1, sd.MUT, 2))
    o.append(sd.T((x0 + x1) / 2, y1 + 34, 'strip', 22, sd.MUT))
    o.append(sd.T(x0 - 14, (y0 + y1) / 2, 'time', 22, sd.MUT, rot=-90))
    o.append(sd.line(x0 + 10, y0 + 20, x1 - 10, y1 - 20, sd.INK, 3, '8 6'))
    ys = [y0 + 60, y0 + 74, y0 + 100, y0 + 136, y0 + 160, y0 + 188, y0 + 222, y0 + 244, y0 + 258]
    xs = [x0 + 10 + i * (x1 - x0 - 20) / 8 for i in range(9)]
    o.append(sd.poly(xs, ys, sd.RED, 3.5))
    o.append(sd.T(x1 - 10, y0 + 30, 'true', 22, sd.INK, 'end', 700))
    o.append(sd.T(x0 + 20, y1 - 56, 'one time per strip:', 22, sd.RED, 'start', 700))
    o.append(sd.T(x0 + 20, y1 - 28, 'ends pulled in', 22, sd.RED, 'start', 700))
    return sd.svg(w, h, ''.join(o), 'one strip collects charge from several depths')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'note', 'mx17-tpc-literature.html'))
    a = ap.parse_args()
    T = J('ana1106_tiers_sat_det3.json')['views']
    Rr = J('ana1106_resim_score.json')
    H = J('ana1106_h2c_sat_det3.json')
    SPS = J('ana1106_sps_score.json')['rows']
    D = sd.Deck('How every TPC treats depth mixing', 'Literature survey for Paper II')

    nexp = len(S.E) + len(S.NEIGHBOURS) + len(S.OTHER_PAD)
    nsrc = sum(len(e['quotes']) for e in S.E)
    mx = lam_mid(S.MX17)
    big = [S.lam(g) for lab, f, g, _ in S.MAP if f in (S.PAD, S.LAR)]
    fa, fb = mx / max(q[1] for q in big), float(f'{mx / min(q[0] for q in big):.2g}')

    # ---- cover
    body = (sd.kicker('Paper II · literature survey · 2026-11-08') +
            sd.title('Other strip micro-TPCs do see our bias. Long TPCs cannot. Our contribution is the measurement and the fit') +
            sd.row(sd.bignum(str(nexp), 'experiments and detector groups surveyed', sd.DBLUE, 'four detector families'),
                   sd.bignum(f'×{fa:.0f}-{fb:.0f}', 'smaller mixing relative to column length in pad and LAr TPCs', sd.DRED, 'derived: Λ at tan θ = 0.3'),
                   sd.bignum('0', 'per-track slope tests against a telescope found elsewhere', sd.DGREEN, 'negative result of a limited search'), gap=64))
    D.slide('cover', body, 'Survey and method: papers/paper2/hits_daq/literature_survey/README.md. Report: LITERATURE_TPC_SURVEY.md.', dark=True, short='Cover')

    # ---- the effect
    body = sd.title('The effect: one strip collects charge from several depths, and gets one time', 'Mixing from diffusion, the prompt footprint and (Y only) resistive spreading') + sd.row(
        effect_diagram(),
        sd.col(sd.p(sd.term('Λ', 'σ_T at full drift / (L tan θ), diffusion only, tan θ = 0.3 from the drift axis. Derived by us; no source states it.') + ' measures the mixing width against the projected column.', 26),
               sd.p('Bench gas: Ar/iC₄H₁₀ 95/5 with about 1 % water.', 24, sd.MUT),
               sd.callout('Hover dotted terms and data points. Each experiment’s Details holds verbatim quotes and links.', sd.BLUE, 24), gap=26, w=560), gap=60)
    D.slide('effect', body, 'Mechanism and measured values: RECONSTRUCTION_BASIS.md and P0_XBIAS_2026-11-05.md. Slope ratios det3 vs M3: X 1.19, Y 1.37.', short='The effect')

    # ---- map
    rowsL, rowsS = [], []
    for lab, f, g, note in sorted(S.MAP, key=lambda r: -lam_mid(r[2])):
        a0, b0 = S.lam(g)
        tip = f'{FN[f]}. L = {g["L"]:.0f} mm, sigma_T {S.sigma(g)[0]:.2f}-{S.sigma(g)[1]:.2f} mm. {note}. Derived.'
        rowsL.append((lab, lam_mid(g), FC[f], tip))
        if g.get('pitch'):
            s0 = math.sqrt(S.sigma(g)[0] * S.sigma(g)[1]) / g['pitch']
            rowsS.append((lab, s0, FC[f], f'sigma_T {S.sigma(g)[0]:.2f}-{S.sigma(g)[1]:.2f} mm over pitch {g["pitch"]} mm. Derived.'))
    leg = sd.legend([(FN[k], FC[k]) for k in FC], size=22)
    left = sd.col(sd.p('Λ: mixing width / column length (log)', 25, weight=700), sd.hbars(rowsL, 0.3, width=440, h=22, log=True, vmin=4e-4, label_w=240, fmt=fig_num, size=21), w=860)
    right = sd.col(sd.p('σ_T / pitch at full drift: no separation', 25, weight=700), sd.hbars(rowsS[:], 8.0, width=300, h=22, log=True, vmin=0.1, label_w=240, fmt=lambda v: f'{v:.2f}', size=21), w=780)
    body = sd.title('Diffusion over pitch is similar everywhere; diffusion over column length is what differs', 'At tan θ = 0.3, full drift, diffusion only. All values derived from cited inputs') + leg + sd.row(left, right, gap=20)
    D.slide('map', body, '<p>Λ = D_T √L / (L tan θ). The earlier notes quoted 0.04-0.08, 0.07 and 9.5 % for MX17: same formula, different inputs. 0.076 uses det3 fitted Dp = 0.023 mm/√ns, v = 36.8 µm/ns. Detectors without a D_T in the sources read (MIMAC, NEWAGE, DRIFT, BEAST, ACTAR, MAIKo) cannot be placed. No paper states this argument: it is an inference.</p>', short='Λ map', pad='64px 128px 60px')

    # ---- conditions matrix
    rows = [
        ('Strip µTPC', '✔', '✔', '~ set angle ±1°', 'Saw it, under other names'),
        ('Pad TPC', 'charge-weighted', f'✘ {fa:.0f}-{fb:.0f}× longer', '✘ track calibration absorbs', 'Could not see it'),
        ('T2K ERAM', '✔ peak time', '✘ no time ladder', '✘ trigger only', 'Silent'),
        ('LArTPC', '✔ after deconvolution', '✘ for long tracks', '✘ fixes t₀ and x, not slope', 'Width and charge only'),
        ('Directional', '✔ mostly', '✔', '✘ resolution 10-58°', 'Hidden'),
        ('Active target', '✔', 'varies', '✘ self-consistency', 'Not tested'),
        ('GridPix', '✘ single electrons', '✔', '✘ calibrates z', 'No mechanism'),
        ('MX17', '✔', '✔', '✔ per-track telescope', 'Measured'),
    ]
    body = sd.title('To see the bias you need three things; only strip micro-TPCs had them', 'One aggregated time per channel, a short column, and a reference that does not absorb the result') + tbl(
        ['Family', 'One aggregated time', 'Short column (Λ large)', 'Slope reference', 'Outcome'], rows, size=27, widths=[260, 340, 340, 400, 320])
    D.slide('who', body, 'Each cell is summarised from the per-experiment slides; every absence is a negative result of a limited search.', short='Who could see it')

    def section(sid, kick, ttl, sub, nums=()):
        D.slide(sid, sd.kicker(kick) + sd.title(ttl, sub) + sd.row(*[sd.bignum(v, l, c, n) for v, l, c, n in nums], gap=64), dark=True, short=kick)

    section('s_strip', 'Strip micro-TPCs', 'Same geometry class: they reported it, and named it differently', 'ATLAS NSW, LMU, BESIII and neighbours', [('3-8 mm', 'drift gaps', sd.DRED, 'against 28 mm for MX17'), ('1.08-1.52', 'published tan ratios', sd.DGOLD if hasattr(sd,'DGOLD') else sd.GOLD, 'derived from quoted angles'), ('Steep', 'sign in nearly all reports', sd.DBLUE, 'exceptions: one NSW peak, Ne:CF₄ layers')])
    for i in ('nsw', 'fsm', 'flierl', 'besiii'):
        exp_slide(D, next(e for e in S.E if e['id'] == i))
    D.slide('neighbours', sd.title('Other strip chambers use centroids, so they neither confirm nor refute it', 'No per-strip drift fit') + tbl(['Detector', 'Layout', 'Estimator', 'For this question'], S.NEIGHBOURS, size=26, widths=[240, 640, 440, 340]), 'From strip_microtpcs.md. CLAS12 BMT uses the same DREAM ASIC family as MX17.', short='Other strip chambers')
    D.slide('fixes', sd.title('Every published fix is a filter or a calibration; deconvolution removes everything but diffusion', 'What has been done about it, across families') + tbl(['Treatment', 'Who', 'What it does'], S.CORRECTIONS, size=25, widths=[620, 420, 620]), 'Sources per experiment slide. The last row is ours and is “not found elsewhere” in a limited search.', short='Published fixes')

    # ---- curve
    def ser(view, variant=None):
        o = []
        for b, c in BINS:
            if variant is None:
                st = T[view]['tiers']['V0_dream']['naive'][b]['steep']
            else:
                st = Rr[view]['variants'][variant]['fams']['V0_dream']['score'][b]['steep']
            o.append(ratio(st, c))
        return o
    xs = [S.lam(S.MX17)[0] * S.TAN_REF / c for _, c in BINS]
    P = sd.Plot(1660, 640, x=(0.04, 0.4, 'log'), y=(0.9, 2.1), xlabel='Λ = σ_T / (L tan θ)  (derived)', ylabel='tan θ read / tan θ true')
    P.xticks([(0.05, '0.05'), (0.1, '0.1'), (0.2, '0.2'), (0.4, '0.4')])
    P.yticks([(1.0, '1.0'), (1.25, '1.25'), (1.5, '1.5'), (1.75, '1.75'), (2.0, '2.0')])
    P.hline(1.0, sd.MUT)
    sx = [(x, y) for x, y in zip(xs, ser('x')) if y]
    sy = [(x, y) for x, y in zip(xs, ser('y')) if y]
    P.line([q[0] for q in sx], [q[1] for q in sx], sd.INK, tips=[f'det3 X data, ratio {q[1]:.2f}, Lambda {q[0]:.2f}' for q in sx])
    P.line([q[0] for q in sy], [q[1] for q in sy], sd.GREY, tips=[f'det3 Y data, ratio {q[1]:.2f}, Lambda {q[0]:.2f}' for q in sy])
    lx, ly, lt = [], [], []
    for lab, ang, rec, g, st in S.LIT_POINTS:
        rt = math.tan(math.radians(rec)) / math.tan(math.radians(ang))
        lm = math.sqrt(S.sigma(g)[0] * S.sigma(g)[1]) / (g['L'] * math.tan(math.radians(ang)))
        lx.append(lm)
        ly.append(rt)
        lt.append(f'{lab}: ratio {rt:.2f}, Lambda {lm:.2f}. {st}')
    P.points(lx, ly, sd.RED, r=11, tips=lt)
    P.text(0.05, 1.85, 'Same trend as LMU chambers', 28, sd.INK, weight=700)
    leg = sd.legend([('MX17 det3 X, data vs telescope', sd.INK), ('MX17 det3 Y, data vs telescope', sd.GREY), ('Published strip µTPC (LMU)', sd.RED)], size=23)
    body = sd.title('Binned by angle, MX17 falls close to the published strip-chamber trend', 'Ratio from the median angle error at each bin centre; published points use a set angle') + leg + P.svg('curve')
    D.slide('curve', body, '<p>Suggestive, not a measurement: LMU D_T is assumed (365 µm/√cm), those points are peak or mean angles at a set angle, and ours are medians per track. Scholer’s NSW 28° peak sits at the true angle and is not plotted because the same thesis contains a +7° simulation; Klitzner’s Ne:CF4 layers read slightly small. Within one detector Λ and pitch/(L tan θ) are collinear, so this curve cannot separate them.</p>', short='Same curve')

    section('s_pad', 'Large pad TPCs', 'Similar diffusion per pad, far longer columns, and a charge-weighted time', 'ALICE, STAR, T2K, LCTPC and others', [('1-2.5 m', 'typical drift', sd.DBLUE, 'T2K 97 cm'), ('0.3-1', 'σ_T / pad', sd.DRED, 'same as ours'), ('None', 'drift-coordinate bias found', sd.DGREEN, 'negative result of a limited search')])
    for i in ('alice', 'star', 't2k', 'lctpc'):
        exp_slide(D, next(e for e in S.E if e['id'] == i))
    D.slide('otherpad', sd.title('The rest of the pad-TPC record is variance terms, and thin in what we could read', 'Older detectors at abstract or slide level only') + tbl(['Detector', 'Layout', 'Statement', 'What was read'], S.OTHER_PAD, size=26, widths=[200, 520, 620, 320]), 'From pad_tpcs.md. ALEPH and PEP-4 NIM papers and Blum-Rolandi-Riegler were not retrievable.', short='Other pad TPCs')

    section('s_lar', 'Liquid argon TPCs', 'The convolution is theirs; the slope bias and the depth fit are not', 'MicroBooNE, ProtoDUNE-SP, ICARUS', [('2018', '2D convolution in channel and time', sd.DGREEN, 'Adams et al.'), ('0.14-0.6 %', 'Λ on full-drift tracks', sd.DBLUE, 'derived'), ('None', 'hit-time slope bias found', sd.DRED, 'negative result; short segments untested')])
    exp_slide(D, next(e for e in S.E if e['id'] == 'uboone'))
    cmp_rows = [('Model', 'M(wire, t) = S ⊗ R(Δwire, Δt)', 'same class'), ('Unknowns', 'free charge per wire and tick', 'track (p₀, w, t₀) + charge per depth bin'),
                ('Kernel', 'field ⊗ electronics, computed', 'diffusion, footprint, RC, electronics; calibrated per detector'), ('Diffusion', 'stays in the output', 'inside the kernel'),
                ('Stationary?', 'yes, needed for FFT', 'no: diffusion grows with depth'), ('Solver', 'linear, Fourier, Wiener filter', 'per-event constrained fit'), ('Physical spread', 'induced current, bipolar', 'real charge on several strips')]
    D.slide('deconv', sd.title('2D deconvolution and our forward fit solve the same equation in different ways', 'Where the difference actually lies, and it must be stated as a difference') + tbl(['', 'LArTPC 2D deconvolution', 'MX17 forward model'], cmp_rows, size=28, widths=[260, 660, 740]), 'From Adams et al. 2018 (1802.08709) Sec. 2-3; ProtoDUNE-SP (2007.06722) Eq. 4.5. The MX17 column follows wft/model.py.', short='Deconvolution vs fit')
    for i in ('protodune', 'icarus'):
        exp_slide(D, next(e for e in S.E if e['id'] == i))

    section('s_short', 'Short-track and pixel TPCs', 'Short tracks should be the worst case, but almost nobody can check', 'Directional, active-target, pixel and polarimetry', [('1-58°', 'angle resolutions', sd.DRED, 'a few-degree bias is hidden'), ('1', 'quantified external-truth test', sd.DBLUE, 'MIMAC, with opposite sign'), ('Canoe', 'BEAST’s name for our family', sd.DGREEN, 'unquantified, in a caption')])
    for i in ('mimac', 'directional', 'pixtpc', 'active', 'migdal', 'gridpix', 'ixpe'):
        exp_slide(D, next(e for e in S.E if e['id'] == i))

    # ---- hypotheses
    cards = [sd.card(sd.p(h[0], 28, weight=700) + sd.p(h[1], 26, getattr(sd, h[3].upper()), 700) + '<div style="height:10px"></div>' + sd.p(h[2], 23), w=800, pad=26) for h in S.HYPOTHESES]
    D.slide('hyp', sd.title('Why others don’t show it: four explanations, each right for a different family') + sd.col(sd.row(cards[0], cards[1], gap=24), sd.row(cards[2], cards[3], gap=24), gap=24), 'Full reasoning: LITERATURE_TPC_SURVEY.md, “The four hypotheses”.', short='Hypotheses')

    # ---- what we bring
    section('s_new', 'What we bring', 'Known: that it happens. Ours: how big each cause is, and a fit that removes it', 'Compared with what has already been done', [('1', 'telescope reference per track', sd.DBLUE, 'M3, against set angles ±1°'), ('4', 'ingredients separated', sd.DGREEN, 'diffusion, footprint, strip ends, RC'), ('0.99-1.00', 'fit slope ratio on re-simulation', sd.DRED, 'up to 10× RC')])
    D.slide('knownnew', sd.title('Already understood against what MX17 adds') + tbl(['Already in the literature', 'Who', 'What MX17 adds', 'Basis'], S.KNOWN_NEW, size=23, widths=[500, 340, 620, 200]), 'Every “not found” is a limited search. Basis: MEASURED = data vs telescope; MODEL = re-simulation validated on the data’s compression; METHOD = the reconstruction; RUNNING = condor scans in progress.', short='Known vs new', pad='64px 128px 72px')

    # decomposition
    arms = [('data', 'Data vs telescope', sd.INK, None), ('base', 'Full model', sd.BLUE, 'base'), ('metal', 'Metallic readout', sd.GREEN, 'metal'), ('dp0', 'No diffusion', sd.ORANGE, 'dp0'), ('point', 'Strip geometry only', sd.GREY, 'point')]
    P = sd.Plot(1660, 640, x=(0.05, 0.42), y=(0.95, 2.0), xlabel='|tan θ| (bin centre)', ylabel='slope ratio (X view)')
    P.xticks([(0.1, '0.1'), (0.2, '0.2'), (0.3, '0.3'), (0.4, '0.4')])
    P.yticks([(1.0, '1.0'), (1.25, '1.25'), (1.5, '1.5'), (1.75, '1.75'), (2.0, '2.0')])
    P.hline(1.0, sd.MUT)
    cs = [c for _, c in BINS]
    val = {}
    for k, lab, col, v in arms:
        y = ser('x', v)
        val[k] = y
        P.line(cs, y, col, dash=None if k == 'data' else '10 6' if k == 'point' else None, tips=[f'{lab}: {q:.2f} at |tan| {c}' if q else 'n/a' for q, c in zip(y, cs)])
    ex = lambda k: val[k][1] - 1  # noqa: E731
    P.text(0.2, 1.75, f'At |tan θ| 0.15: diffusion is {1 - ex("dp0") / ex("base"):.0%} of the excess', 26, sd.INK, weight=700)
    leg = sd.legend([(l, c) for _, l, c, _ in arms], size=23)
    D.slide('decomp', sd.title('We can say what the bias is made of, one ingredient at a time', 'det3 X: each physics ingredient switched off in the model validated on the data') + leg + P.svg('decomp'), f'<p>Excess over 1 at |tan θ| = 0.15: data {ex("data"):.2f}, full model {ex("base"):.2f}, metallic {ex("metal"):.2f}, no diffusion {ex("dp0"):.2f}, geometry only {ex("point"):.2f}. Derived by converting each bin’s median angle error at the bin centre, so approximate. The nearest literature statement is Scholer’s simulation with sharing off (+7°, cause left open). Source: results/ana1106_tiers_sat_det3.json, ana1106_resim_score.json.</p>', short='Decomposition')

    # RC scan
    order = ['rc0', 'rc025', 'rc05', 'base', 'rc2', 'rc5', 'rc10']
    fx = {'rc0': 0.1, 'rc025': 0.25, 'rc05': 0.5, 'base': 1, 'rc2': 2, 'rc5': 5, 'rc10': 10}
    P = sd.Plot(1660, 620, x=(0.08, 13, 'log'), y=(0.8, 2.7), xlabel='RC spreading on Y relative to measured (leftmost point: no RC)', ylabel='slope ratio against exact truth')
    P.xticks([(0.1, '0'), (0.25, '¼'), (0.5, '½'), (1, '1'), (2, '2'), (5, '5'), (10, '10')])
    P.yticks([(1.0, '1.0'), (1.5, '1.5'), (2.0, '2.0'), (2.5, '2.5')])
    P.hline(1.0, sd.MUT)
    V = Rr['y']['variants']
    h0 = [V[k]['fams']['V0_dream']['ratio'] for k in order]
    w0 = [V[k]['W']['ratio'] for k in order]
    xx = [fx[k] for k in order]
    P.line(xx, h0, sd.RED, tips=[f'hits V0: {q:.2f}' for q in h0])
    P.line(xx, w0, sd.GREEN, tips=[f'waveform fit: {q:.3f}' for q in w0])
    P.text(0.14, 2.3, f'Waveform fit {min(w0):.2f}-{max(w0):.2f} over the whole scan', 28, sd.GREEN, weight=700)
    D.slide('rc', sd.title('The forward fit stays unbiased as the resistive spreading grows', 'Re-simulation of real det3 events, Y view, exact truth, |tan θ| > 0.1') + sd.legend([('Hits (DREAM-style, one time per strip)', sd.RED), ('Forward waveform fit', sd.GREEN)], size=23) + P.svg('rc'), 'On the model with exact truth, and the fit knows the generating kernel: this shows the method, not an absolute resolution. Model validated on data compression: X 1.18 / Y 1.34 simulated against 1.19 / 1.37 measured. T2K has flagged a global waveform fit as under study.', short='Fit vs resistivity')

    # resolution
    def s(view, tier, kind='naive'):
        return T[view]['tiers']['V0_dream'][tier if kind is None else kind]['all']['s68']
    groups = [('Naive hits', 'naive'), ('Global remap (H0)', 'H0'), ('Binned remap (H2a)', 'H2a')]
    vals = {v: [T[v]['tiers']['V0_dream'][k]['all']['s68'] for _, k in groups] + [H[f'sat_det3/{v}']['H2c']['s68'], T[v]['W']['score']['all']['s68']] for v in 'xy'}
    labs = [g[0] for g in groups] + ['Model on hits (H2c)', 'Waveform fit']
    P = sd.Plot(1660, 620, x=(-0.5, 4.5), y=(0, 4.2), xlabel='', ylabel='angle resolution s68 [deg]')
    P.yticks([(0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4')])
    P.xticks([(i, x) for i, x in enumerate(labs)])
    for i in range(5):
        P.vbar(i - 0.14, vals['x'][i], 110, sd.BLUE, tip=f'X {labs[i]}: {vals["x"][i]:.2f} deg')
        P.vbar(i + 0.14, vals['y'][i], 110, sd.ORANGE, tip=f'Y {labs[i]}: {vals["y"][i]:.2f} deg')
    D.slide('res', sd.title(f'Same events, same telescope: hits {vals["x"][0]:.1f}°, waveform fit {vals["x"][4]:.2f}°', 'det3 Run 1, X and Y, vs M3 (within ±5°)') + sd.legend([('X view', sd.BLUE), ('Y view', sd.ORANGE)], size=23) + P.svg('res'), 'H2a resolution is partly the cosmic angle prior: it is a regression toward the training distribution and reads shallow on steep tracks (h2a_edge.py). The published comparison is usually a peak or mean angle, not a per-track resolution, so these numbers have no direct counterpart in the strip µTPC literature.', short='Resolution')

    # SPS
    ks = [k for k in SPS if SPS[k]['inclined']]
    def rr(k, t):
        return SPS[k]['views']['y']['tiers'][t].get('ratio')
    fields = {'run57_rot25_co2__d550': 'd550', 'run57_rot25_co2__d600': 'd600', 'run57_rot25_co2__d650': 'd650', 'run57_rot25_co2__d700': 'd700', 'run59_co2__co2': 'run59'}
    P = sd.Plot(1660, 600, x=(-0.5, len(ks) - 0.5), y=(0.98, 1.16), xlabel='row (drift field rising to the right)', ylabel='hit slope / waveform-fit slope')
    P.xticks([(i, fields.get(k, k)) for i, k in enumerate(ks)])
    P.yticks([(1.0, '1.00'), (1.05, '1.05'), (1.10, '1.10'), (1.15, '1.15')])
    P.hline(1.0, sd.MUT)
    y1 = [rr(k, 'V0_dream') for k in ks]
    y2 = [rr(k, 'V0r_dream') for k in ks]
    P.line(list(range(len(ks))), y1, sd.RED, tips=[f'V0 {q:.3f}' for q in y1])
    P.line(list(range(len(ks))), y2, sd.PURPLE, tips=[f'V0 robust {q:.3f}' for q in y2])
    D.slide('sps', sd.title('A second, reference-free test: the same compression on a chamber at a known 25.4°', 'det4 at SPS H4, inclined view, CO₂ rows. Relative to the waveform fit') + sd.legend([('Hits (DREAM-style)', sd.RED), ('Hits, robust line', sd.PURPLE)], size=23) + P.svg('sps'), 'The stage angle sets v per row through the waveform fit, so this is relative to the fit on the same planes and cannot join the bench curve without caveat. 25.4° is the yaw measured in the run-57 alignment. H2c on these rows is still being tuned.', short='SPS known angle')

    exists = {n: os.path.exists(os.path.join(RES, n)) for n in ('gang_scan.json', 'shape_scan.json', 'h2c_resim.json', 'charge_weighted.json')}
    # scan results
    G = J('gang_scan.json') if exists['gang_scan.json'] else None
    if G:
        ks = [1, 2, 3, 4, 6]
        P = sd.Plot(1660, 600, x=(0.5, 6.5), y=(1.0, 1.8), xlabel='strips summed into one readout element (pitch × 0.78 mm)', ylabel='slope ratio, X view, |tan θ| 0.2-0.3')
        P.xticks([(k, str(k)) for k in ks])
        P.yticks([(1.0, '1.0'), (1.2, '1.2'), (1.4, '1.4'), (1.6, '1.6'), (1.8, '1.8')])
        P.hline(1.0, sd.MUT)
        leg = []
        for arm, lab, col in (('data', 'Data vs telescope', sd.INK), ('base', 'Full model', sd.BLUE), ('metal', 'Metallic readout', sd.GREEN), ('point', 'Strip geometry only', sd.GREY)):
            y = [G['x'][arm][str(k)]['dream']['0.2-0.3'] for k in ks]
            cov = [G['x'][arm][str(k)]['dream']['coverage'] for k in ks]
            P.line(ks, y, col, tips=[f'{lab}, {k} strips: {q:.2f}, coverage {c:.2f} (|tan|>0.1)' for k, q, c in zip(ks, y, cov)])
            leg.append((lab, col))
        P.text(1.3, 1.65, 'Wider strips make it worse, even with no diffusion', 28, sd.INK, weight=700)
        D.slide('pitch', sd.title('Fine strips are not the problem: summing them increases the compression', 'det3, Runs 1 and 2: neighbouring strips summed on the data and on the model') + sd.legend(leg, size=23) + P.svg('pitch'), 'Noise adds in quadrature; extra strip capacitance not modelled. Coverage falls with width (below 50 % at 6 strips for |tan θ| > 0.1), so wide-strip points contain only tracks spanning at least 3 elements. Most of the growth is present in the geometry-only arm, so it is the strip-end term. No literature comparison can separate this from the lever arm. papers/paper2/hits_daq/gang_scan.py, results/gang_scan.json.', short='Strip pitch')
    if exists['shape_scan.json']:
        Z = J('shape_scan.json')
        names = ['tp25', 'tp50', 'tp100', 'tp200']
        xv = [25, 50, 100, 200]
        P = sd.Plot(1660, 600, x=(20, 250, 'log'), y=(1.0, 1.6), xlabel='front-end peaking time [ns] (CR-RC², signal-to-noise held at DREAM’s)', ylabel='slope ratio (|tan θ| > 0.1), time at peak')
        P.xticks([(v, str(v)) for v in xv])
        P.yticks([(1.0, '1.0'), (1.2, '1.2'), (1.4, '1.4'), (1.6, '1.6')])
        P.hline(1.0, sd.MUT)
        leg = []
        for v_, det, lab, col in (('y', 'base', 'Y, resistive', sd.ORANGE), ('x', 'base', 'X', sd.BLUE), ('y', 'metal', 'Y, metallic', sd.GREEN)):
            y = [Z[v_]['variants'][f'{det}_{n}']['fams']['V1_peak']['ratio'] for n in names]
            P.line(xv, y, col, tips=[f'{lab}, {t} ns: {q:.3f}' for t, q in zip(xv, y)])
            leg.append((lab, col))
        P.text(30, 1.55, 'Faster shaping does not remove it', 28, sd.INK, weight=700)
        D.slide('shaping', sd.title('A faster front end does not fix it: shaping from 25 to 200 ns barely matters', 'Re-simulated det3 events, VMM-style time at peak, exact truth') + sd.legend(leg, size=23) + P.svg('shaping'), 'Noise is white noise through the same shaper at each strip’s measured rms; real ENC changes with shaping and is not modelled. Pulse: CR-RC² convolved with a 10 % prompt / 90 % 100 ns ion current. resim_shape.py, results/shape_scan.json.', short='Shaping')
    if exists['h2c_resim.json']:
        Y = J('h2c_resim.json')['y']['variants']
        names = ['rc0', 'rc05', 'base', 'rc2', 'rc5']
        xv = [0.1, 0.5, 1, 2, 5]
        P = sd.Plot(1660, 600, x=(0.08, 7, 'log'), y=(0.8, 2.2), xlabel='RC spreading on Y relative to measured (leftmost: no RC)', ylabel='slope ratio against exact truth')
        P.xticks([(0.1, '0'), (0.5, '½'), (1, '1'), (2, '2'), (5, '5')])
        P.yticks([(1.0, '1.0'), (1.5, '1.5'), (2.0, '2.0')])
        P.hline(1.0, sd.MUT)
        a0 = [Y[k]['fams']['V0_dream']['ratio'] for k in names]
        a1 = [Y[k]['fams']['V0/H2c_dream']['ratio'] for k in names]
        P.line(xv, a0, sd.RED, tips=[f'hits: {q:.2f}' for q in a0])
        P.line(xv, a1, sd.PURPLE, tips=[f'model on hits: {q:.3f}' for q in a1])
        P.text(0.12, 2.0, 'The model recovers the slope from hit lists alone', 28, sd.PURPLE, weight=700)
        D.slide('h2csim', sd.title('On simulation, the model fitted to hit lists also recovers the slope', f'Y view, {Y["base"]["n"]} re-simulated planes so far (partial), the fit knows the generating kernel') + sd.legend([('Plain hits line', sd.RED), ('Forward model fitted to hits (H2c)', sd.PURPLE)], size=23) + P.svg('h2csim'), 'On the data H2c gives 1.20 / 1.28° against 1.04 / 1.01° for the waveform fit, with the gap near normal incidence (far-strip timing), so the model is what matters, and waveforms still help. h2c_resim.json; 39 of 48 nodes at the time of this build.', short='Model on hits')

    # running
    rows = [('Strip pitch: sum neighbouring strips, data and model, by angle', 'done (slide above)', 'Wider strips are worse; growth is mostly the strip-end term'),
            ('Front-end shaping 25-200 ns', 'done (slide above)', 'Shaping barely matters'),
            ('Forward model fitted to hit lists, resistivity scan', 'partial', 'The model recovers the slope on simulation; data gap stays near normal'),
            ('Charge-weighted time on our data', 'to add' if not exists['charge_weighted.json'] else 'done', 'Tests the pad-TPC estimator; the MAIKo midpoint too')]
    D.slide('running', sd.title('Four more results would sharpen the claim, all from existing data') + tbl(['Study', 'Status', 'What it adds'], rows, size=27, widths=[760, 160, 740]), 'Jobs: condor DAG 4637865. SPS H2c tuning (spstune) is still running.', short='Still to come')

    q = ('Single-time-per-strip µTPC reconstruction is known to read tracks too steep, an effect attributed to capacitive coupling and edge strips. '
         'We measure it per track against an independent telescope, show that it is dominated by same-time mixing from diffusion and the prompt footprint with a further strip-end term, '
         'that it would persist on a metallic readout, and that a per-event forward fit of the depth-resolved waveform removes it.')
    dont = ['“First” for the effect or the convolution model', '“Only MX17 sees this”', '“Whatever the estimator” (charge-weighted untested)',
            'Any resistive-vs-metallic statement about NSW', 'A direct numerical comparison with Scholer’s test beam']
    body = sd.title('What Paper II can safely write') + sd.row(
        sd.card(head('Safe wording', sd.GREEN) + sd.p('“' + q + '”', 28) + sd.p('Add: we did not find this decomposition or such a fit in a limited search.', 23, sd.MUT), w=980, pad=30),
        sd.card(head('Do not write', sd.RED) + li(dont, 25) + head('Must cite', sd.BLUE) + li(['Bortfeldt, Flierl, Scholer, Vanadia, Sessa, Alexeev', 'Adams 2018; MicroBooNE 2021; ProtoDUNE-SP', 'T2K ERAM (RC, waveform model)'], 25), w=640, pad=30), gap=44)
    D.slide('wording', body, 'Current text to change: “delayed copies on resistive strips” (OUTLINE_PLAN), “whatever the estimator”, “~4° too steep” (true only near normal incidence; sample median 2.5° X, 3.1° Y), “20-30 %” (16 % X, 27 % Y).', short='Safe wording')

    body = sd.title('What this does not rule out') + sd.row(
        sd.col(*[sd.p('• ' + x, 28, sd.DINK) for x in ['A search is not a proof: internal ATLAS and BESIII notes, DUNE/SBN notes, some theses are unread', 'Short LAr segments have a Λ as large as ours and are simply untested', 'Scholer’s data-simulation tension and MIMAC’s opposite sign are unresolved', 'Lever arm is our inference: no paper states it, and LMU points assume a D_T']], gap=20, w=950),
        sd.col(sd.p('Open for you', 30, sd.DRED, 700), *[sd.p('• ' + x, 27, sd.DINK) for x in ['Which claim leads the paper: decomposition or forward fit?', 'Add charge-weighted and midpoint estimators?', 'Contact T2K about their global waveform fit?']], gap=18, w=620), gap=64)
    D.slide('end', body, 'Not read: Lösel, Klitzner PhD, Alexopoulos 2019, Iodice, Bradt, Mauss, Giovinazzo 2020, ALEPH and PEP-4 NIM papers, Blum-Rolandi-Riegler.', dark=True, short='Limits')

    D.write(a.out, note_meta=dict(title='How every TPC family treats depth mixing', summary='Literature survey for MX17 Paper II: what each experiment did and concluded about charge mixing in the time signal, and what MX17 adds.', tags='MX17, TPC, literature', date='2026-11-08'))
    print('wrote', a.out, len(D.slides), 'slides')


if __name__ == '__main__':
    main()
