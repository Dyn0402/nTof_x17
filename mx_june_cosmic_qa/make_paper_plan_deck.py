#!/usr/bin/env python3
"""
make_paper_plan_deck.py -- the two-paper plan for MX17, as a slide note.

Since 2026-10-08 the detector paper is two companion papers: I, the chambers
and their performance (June cosmic bench + det4 in SPS H4); II, the
reconstruction (the waveform forward model, and how much a hit-based readout
loses). This deck shows the split, both outlines, the hits-vs-waveforms
comparison design, the work still to do, and the evidence already in hand.

Reads, so that rerunning after any of them moves the slides:
  * mx_june_cosmic_qa/PAPER_PLAN.md             -- outlines, work list, n_TOF tripwires (edit THIS)
  * mx_june_wft/FLEET_DIGEST.md                 -- the r06 fleet numbers (wft vs hits)
  * mx_june_wft/RETIRE_C2GTC1_2026-08-21.md     -- the retired c2/c1 ratios
  * mx_june_cosmic_qa/waveform_first_threading/WAVEFORM_FIRST_THREADING.md
                                                -- §3 compression table, §10 tier benchmark
  * sps_beam_test_26/analysis/sharing_kernel/{fit_kernel,bench_kernel_y}.json
  * sps_beam_test_26/analysis/spatial_resolution/results.json
  * sps_beam_test_26/analysis/angled_kernel/results.json

    python mx_june_cosmic_qa/make_paper_plan_deck.py [--out PATH]
    python ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug mx17-detector-paper-status --force --deploy
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))

import slidedoc as sd                                                  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,     # noqa: E402
                      INK, MUT, RULE, DBLUE, DRED, DGREEN, DMUT, DINK)

PLAN = REPO / 'mx_june_cosmic_qa' / 'PAPER_PLAN.md'
FLEET = REPO / 'mx_june_wft' / 'FLEET_DIGEST.md'
RETIRE = REPO / 'mx_june_wft' / 'RETIRE_C2GTC1_2026-08-21.md'
WFT_DOC = REPO / 'mx_june_cosmic_qa' / 'waveform_first_threading' / 'WAVEFORM_FIRST_THREADING.md'
SPS = REPO / 'sps_beam_test_26' / 'analysis'

DETS = ['sat_det3', 'o22_long_det2', 'g_det4', 'g_det6_long', 'g_det7_long']
DLAB = {'sat_det3': 'det3', 'o22_long_det2': 'det2', 'g_det4': 'det4',
        'g_det6_long': 'det6', 'g_det7_long': 'det7'}

# Status colours, used on every slide that carries a status.
READY, REQUOTE, OPEN, RETIRED = GREEN, BLUE, GOLD, GREY
STATE_COL = {'ready': GREEN, 'requote': BLUE, 'partial': PURPLE, 'open': GOLD,
             'decision': RED, 'deferred': GREY}
STATE_LAB = {'ready': 'ready', 'requote': 're-quote', 'partial': 'partial', 'open': 'open',
             'decision': 'decision', 'deferred': 'deferred'}
HITS, WFT = GREY, GREEN
# One colour per reconstruction tier, on every slide that shows tiers.
TIER = dict(H0=GREY, H1=ORANGE, H2a=GOLD, H2b=PURPLE, H2c=BLUE, W=GREEN, V=GREY)
P1C, P2C = BLUE, GREEN          # paper I / paper II accents

# Numbers the SPS report states in prose (spatial_resolution/make_report.py),
# not in its results.json. The scattering figure is arithmetic, not a measurement.
M3_POINT_MM = (0.21, 0.24)
MS_MRAD, MS_LEVER_MM = 1.1, 558
# The bench residual the SPS report compares with: det3 σ68 at θ < 5°, x / y
# (PAPER_STATUS topic 9 on the χ² < 1 recipe; the wft core σ|r| is a narrower core fit).
BENCH_S68 = (0.61, 0.72)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def md_table(txt, heading):
    """The first markdown table under '## <heading>' -> list of {column: cell}."""
    lines = txt.splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip() == f'## {heading}')
    i = next(k for k in range(i, len(lines)) if lines[k].startswith('|'))
    cols = [c.strip() for c in lines[i].strip().strip('|').split('|')]
    out = []
    for l in lines[i + 2:]:
        if not l.startswith('|'):
            break
        cells = [c.strip() for c in l.strip().strip('|').split('|')]
        out.append(dict(zip(cols, cells)))
    return out


def load_plan():
    txt = PLAN.read_text()
    work = md_table(txt, 'Work list')
    for w in work:
        w['needs'] = [] if w['needs'] in ('–', '-', '') else w['needs'].split()
    return dict(p1=md_table(txt, 'Paper I — outline'), p2=md_table(txt, 'Paper II — outline'),
                work=work, trip=md_table(txt, 'n_TOF tripwires'))


def load_july():
    """WAVEFORM_FIRST_THREADING.md: §3 compression rows and §10 tier benchmark (det3, R&D model)."""
    txt = WFT_DOC.read_text()
    comp = {}
    for l in txt.splitlines():
        if l.startswith('| u since mesh'):
            comp['u'] = [float(c) for c in l.strip('|').split('|')[1:]]
        m = re.match(r'\| (matched-filter t50|leading-edge 20 %)(?: residual)? \(x\) \[ns\] \|(.*)\|', l)
        if m:
            comp[m.group(1)] = [float(c.replace('−', '-').replace('+', '')) for c in m.group(2).split('|')]
    bench = {}
    sec = txt[txt.index('## 10. Benchmark'):txt.index('## 11.')]
    for l in sec.splitlines():
        m = re.match(r'\| \**([^|*]+?)\** \| \**([\d.]+) / ([\d.]+)\** \|', l)
        if m:
            bench[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return comp, bench


def load_fleet():
    """FLEET_DIGEST.md first table -> {metric: {key: (new, old|None)}}, plus bundles."""
    txt = FLEET.read_text()
    lines = txt.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('| quantity'))
    keys = [c.strip() for c in lines[i].strip('|').split('|')][1:]
    out = {}
    for l in lines[i + 2:]:
        if not l.startswith('|'):
            break
        cells = [c.strip() for c in l.strip('|').split('|')]
        row = {}
        for k, c in zip(keys, cells[1:]):
            m = re.match(r'([+-]?[\d.]+)\s*(?:\(was ([+-]?[\d.]+))?', c)
            if m:
                row[k] = (float(m.group(1)), float(m.group(2)) if m.group(2) else None)
        out[cells[0]] = row
    bund = dict(re.findall(r'^\| (\w+) \| `(calib_bundle\w*)` \|', txt, flags=re.M))
    return out, bund


def load_retired():
    """The 'was' column of RETIRE §2: [(label, ratio)] for every inverted bundle."""
    rows = []
    for l in RETIRE.read_text().splitlines():
        m = re.match(r'\| (mx17_\d)(?: \(([\w]+)\))? \| .*?\| \*\*([\d.]+)\*\* \|', l)
        if m:
            rows.append((m.group(1).replace('mx17_', 'det') + (f' {m.group(2)}' if m.group(2) else ''),
                         float(m.group(3))))
    return rows


def load_sps():
    fk = json.loads((SPS / 'sharing_kernel' / 'fit_kernel.json').read_text())
    beam = []
    for run, r in fk['y'].items():
        par, err = r['delay']['par'], r['delay']['err']
        q = par['c2'] / par['c1']
        e = q * ((err['c2'] / par['c2']) ** 2 + (err['c1'] / par['c1']) ** 2) ** 0.5
        beam.append((run, r['field_Vcm'], r['n_events'], q, e))
    bk = json.loads((SPS / 'sharing_kernel' / 'bench_kernel_y.json').read_text())
    bp, be = bk['delay']['par'], bk['delay']['err']
    bq = bp['c2'] / bp['c1']
    bench = (bq, bq * ((be['c2'] / bp['c2']) ** 2 + (be['c1'] / bp['c1']) ** 2) ** 0.5, bk['n_events'])
    res = json.loads((SPS / 'spatial_resolution' / 'results.json').read_text())
    ang = json.loads((SPS / 'angled_kernel' / 'results.json').read_text())
    return dict(beam=beam, bench=bench, res=res, ang=ang)


# --------------------------------------------------------------------------- #
# glossary
# --------------------------------------------------------------------------- #
G = dict(
    wft=('Waveform-first reconstruction (wft/): a forward model of every strip waveform, including the '
         'resistive sharing, fitted per track. The basis for any position, angle or depth since 2026-07-28.'),
    hits=('The original chain: geometry from per-strip combined_hits times. Each hit time mixes in delayed '
          'neighbour charge, which compresses the drift ladder 20–30 % and reads ~4° too steep.'),
    hit=('A hit: one time and one amplitude per strip above threshold. That is all a VMM-style digitiser '
         'records, so hits taken from the DREAM waveforms stand in for any hit readout.'),
    r06=('calib_bundle_r06: the shipped calibration since 2026-08-19, with c2 pinned to 0.6 × c1. '
         'Every earlier bundle had c2 > c1, which no resistive film can produce.'),
    kernel=('The sharing kernel: how much of a strip’s charge appears on its ±1 and ±2 neighbours (c1, c2), '
            'and how late. c2 must be < c1: the ±2 strip is reached only through the ±1.'),
    sigt='σ_θ: angular resolution, core fit of reconstructed − M3-reference track angle, per readout plane.',
    w5='Fraction of M3 reference rays with a reconstructed track within 5 mm (|r| < 5 mm).',
    core='Core σ of the radial residual |r| to the M3 reference ray. Includes the reference’s own error.',
    pitch='Strip pitch 0.78 mm. A one-strip (binary) readout would give pitch/√12 = 225 µm.',
    rd=('The July R&D forward model (forward_model2.py, hyper_v2: c2/c1 = 0.17, v = 36.6 µm/ns), on det3 '
        'sat_det3, 2,000-event disjoint test set, |tan| 0.10–0.40. Physical kernel ordering, but not the '
        'packaged wft/ code and not the r06 bundle.'),
)


def chip(state, size=21):
    c = STATE_COL[state]
    return f'<span style="color:{c};font-weight:600;font-size:{size}px;white-space:nowrap">● {STATE_LAB[state]}</span>'


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D, P, F, S):
    work = [w for w in P['work'] if w['state'] not in ('deferred',)]
    todo = [w for w in work if w['state'] not in ('ready',)]
    n2 = sum(1 for w in todo if w['paper'] == 'II')
    n1 = sum(1 for w in todo if w['paper'] == 'I')
    nb = sum(1 for w in todo if w['paper'] == 'both')
    st = F['sigma_theta X']
    lo = min(min(F['sigma_theta X'][k][0], F['sigma_theta Y'][k][0]) for k in DETS)
    hi = max(max(F['sigma_theta X'][k][0], F['sigma_theta Y'][k][0]) for k in DETS)
    hlo = min(min(F['sigma_theta X'][k][1], F['sigma_theta Y'][k][1]) for k in DETS)
    hhi = max(max(F['sigma_theta X'][k][1], F['sigma_theta Y'][k][1]) for k in DETS)
    sig = S['res']['fit']
    todo_tip = '\n'.join(f'{w["id"]} ({w["paper"]}): {w["short"]} [{w["state"]}]' for w in todo)
    nums = ''.join([
        sd.bignum('2', 'companion papers', DBLUE,
                  'I: the chambers and their performance. II: the reconstruction, and what a hit readout loses.',
                  tip='Same journal, same arXiv day. I cites II for every geometric number.'),
        sd.bignum(f'{hhi:.1f} → {hi:.1f}°', 'worst plane σ_θ, hits → waveforms', DGREEN,
                  f'Five chambers, r06. Hits {hlo:.1f}–{hhi:.1f}°, waveforms {lo:.2f}–{hi:.2f}°: II’s headline, '
                  'to be extended to corrected hits.',
                  tip='mx_june_wft/FLEET_DIGEST.md, r06 promotion 2026-08-19. Hits = production combined_hits chain.'),
        sd.bignum(f'{len(todo)}', 'work items still open', DRED,
                  f'{nb} block both papers, {n2} for II, {n1} for I. The first is reproducing the r06 numbers.',
                  tip=todo_tip)])
    body = (sd.kicker('MX17 papers · cosmic bench + SPS H4 · plan of 8 Oct 2026')
            + '<h1 style="font-size:78px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1660px">'
              'Two papers: the chambers, and how to reconstruct them. '
              'The reconstruction paper goes first</h1>'
            + '<div style="flex:1"></div>'
            + f'<p style="font-size:28px;color:{DMUT}">Scope of both: the June cosmic bench (det2, 3, 4, 6, 7) and '
              'det4 in the SPS H4 beam. n_TOF is a separate physics paper.</p>'
            + f'<div style="display:flex;gap:64px">{nums}</div>')
    D.slide('cover', body, '''
<p>On 2026-10-08 the planned MX17 detector paper was split in two. <b>Paper I</b> is the detector: design, operation, gas, timing, efficiency, and tracking performance on the June cosmic bench and with det4 in SPS H4. <b>Paper II</b> is the reconstruction: why per-strip hit times fail on resistive strips, the waveform forward model that fixes it, and how much a hit-based readout loses.</p>
<p>The hit comparison uses hits taken from the same DREAM waveforms, with and without a sharing correction. A hit is one time and one amplitude per strip, which is what any hit digitiser such as the VMM records. A full VMM emulation (shaper, threshold and neighbour logic, dead time) is left open for later.</p>
<p>This note replaces the 2026-10-02 status note at the same address. The plan itself is <code>mx_june_cosmic_qa/PAPER_PLAN.md</code>. Its tables (outlines, work list, n_TOF tripwires) are parsed by the generator, so edit them there and rebuild.</p>''',
            dark=True, short='Answer')


def s_split(D):
    W, H = 1664, 650
    o = []

    def box(x, y, w, h, c, head, lines, tip, fill='#fffdf9'):
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{c}" '
                 f'stroke-width="3"{sd.tipattr(tip)}/>')
        o.append(sd.T(x + 28, y + 50, head, 32, c, 'start', 600, tip=tip))
        for i, l in enumerate(lines):
            o.append(sd.T(x + 28, y + 96 + 36 * i, l, 23, INK, 'start'))

    box(0, 0, 720, 330, P1C, 'Paper I · the detector',
        ['design, resistive layer, charge balance', 'HV, gain, sparks, efficiency, edges',
         'gas: v(E), attachment, gap; timing', 'tracking performance on cosmics',
         'det4 intrinsic resolution in SPS H4'],
        'Paper I quotes position, angle and v_drift through II’s reconstruction, and the kernel from II.')
    box(944, 0, 720, 330, P2C, 'Paper II · the reconstruction',
        ['sharing kernel, measured head-on in SPS', 'why hit times compress the drift ladder',
         'the waveform forward model + calibration', 'hits vs waveforms, same events',
         'what a hit readout must record'],
        'Paper II stands alone with a short detector description; it cites I for the rest.')
    # dependency arrow II -> I (I cites II)
    o.append(sd.arrow(940, 130, 726, 130, INK, 4, 18))
    o.append(sd.T(833, 112, 'cited by', 24, INK, weight=600))
    o.append(sd.T(833, 172, 'every position, angle,', 20, MUT))
    o.append(sd.T(833, 198, 'v_drift and the kernel', 20, MUT))
    o.append(sd.arrow(726, 260, 940, 260, MUT, 2, 12))
    o.append(sd.T(833, 248, 'detector description', 19, MUT))
    # datasets
    ds = [(0, 'June cosmic bench', 'det2, 3, 4, 6, 7 · M3 telescope', GREEN,
           'Five chambers, golden run each, M3 reference (χ² < 1, NClus = 4). In both papers.'),
          (575, 'SPS H4 beam', 'det4 · uRWELL telescope · 0° and 25.6°', GREEN,
           'Kernel head-on, intrinsic resolution, known-angle rotation. In both papers.'),
          (1150, 'n_TOF campaign', 'chambers A–D · separate physics paper', GREY,
           'Not in either paper. Its reconstruction work feeds II only through the tripwires (slide 11).')]
    for x, h, sub, c, tip in ds:
        o.append(f'<rect x="{x}" y="470" width="514" height="130" rx="14" fill="#f5f3ee" stroke="{c}" '
                 f'stroke-width="2" stroke-dasharray="{"8 6" if c == GREY else "none"}"{sd.tipattr(tip)}/>')
        o.append(sd.T(x + 24, 516, h, 27, c if c != GREY else MUT, 'start', 600, tip=tip))
        o.append(sd.T(x + 24, 560, sub, 22, MUT, 'start'))
    o.append(sd.T(0, 440, 'data', 22, MUT, 'start', 600))
    for x in (257, 832):
        for tx in (360, 1304):
            o.append(sd.line(x, 470, tx, 330, RULE, 2, '6 6'))
    body = sd.title('Paper I leans on paper II, never the reverse',
                    'What goes where, and which way the citations run. Hover the boxes, the dotted terms and the '
                    'data points throughout this note.')
    body += sd.svg(W, H, ''.join(o), 'two-paper split')
    D.slide('split', body, '''
<p><b>Why one-way.</b> Companion papers that cite each other are common (NIM A and JINST both take them), but referees object when a key number in one can only be understood with the other, and vice versa. Here the dependency is naturally one-directional. Every geometric quantity in I (position, angle, drift depth, v_drift) comes out of II's reconstruction, while II needs only a page of detector description. So II should be at least as advanced as I. Ideally it is submitted first, or the same day.</p>
<p><b>The kernel lives in II.</b> The sharing kernel is a property of the resistive layer, so it could go in either paper. It is the input the forward model needs and the reason hit times fail, so it opens II. I quotes it.</p>
<p><b>Mechanics.</b> Post both to arXiv on the same day and cite each other by arXiv number. Submit both to the same journal with a cover letter naming them as companions, so the editor can send them to the same referees.</p>''',
            short='The split')


def _outline_cards(rows, accent):
    ncol = 4 if len(rows) <= 8 else 5
    gap = 37 if ncol == 4 else 24
    w = (1664 - gap * (ncol - 1)) // ncol
    h = 300
    cards = []
    for r in rows:
        c = STATE_COL[r['state']]
        cards.append(
            f'<div{sd.tipattr(r["backed by"])} style="width:{w}px;height:{h}px;'
            f'background:#fffdf9;border:1px solid {RULE};border-top:8px solid {c};border-radius:14px;'
            f'padding:18px 22px;display:flex;flex-direction:column;gap:8px">'
            f'<p style="font-size:20px;color:{MUT};font-weight:600">§{r["§"]} · {chip(r["state"], 20)}</p>'
            f'<p style="font-size:26px;font-weight:600;line-height:1.15">{r["section"]}</p>'
            f'<p style="font-size:19px;color:{MUT};line-height:1.3">{r["backed by"]}</p></div>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:20px {gap}px;width:1664px">{"".join(cards)}</div>'


def _state_legend(states):
    return sd.legend([(STATE_LAB[s], STATE_COL[s], 'box') for s in states], size=22)


def s_outline1(D, P):
    rows = P['p1']
    n_ready = sum(r['state'] == 'ready' for r in rows)
    body = sd.title(f'Paper I: {len(rows)} sections, {n_ready} ready to write',
                    'Detector & performance. Colour is the state of the numbers each section needs.')
    body += _state_legend(['ready', 'requote', 'partial', 'open']) + _outline_cards(rows, P1C)
    D.slide('paper1', body, '''
<p>This is the 2026-10-02 storyline with the reconstruction sections (the sharing kernel and the forward model) moved to Paper II. What remains is what a detector paper reviewer expects: the chamber, how it runs, the gas, the timing, and the performance it delivers.</p>
<p>§7 is the one place I depends on II. The numbers come from II's reconstruction on the r06 calibration, and the table has to say which bundle each chamber used (det4 lp_t0p, det6 lp, the rest r06). The text needs two context notes: det6's X plane is <i>not</i> dead (wft σ_θ 2.2°, contrary to the July table), and det4's bench numbers reflect its non-amplifying stripes (§9).</p>
<p>The start point for figures is the MPGD26 deck (<code>mpgd26/</code>). Its <code>make_*.py</code> scripts already build the setup renders, the efficiency loss budget and the angle-resolution figures.</p>''',
            short='Paper I outline')


def s_outline2(D, P):
    rows = P['p2']
    n_ready = sum(r['state'] == 'ready' for r in rows)
    body = sd.title(f'Paper II: {len(rows)} sections; the hits comparison is the new work',
                    f'Reconstruction. {n_ready} sections are ready; §5–7 carry the comparison with a '
                    f'{sd.term("hit", G["hit"])}-based readout.')
    body += _state_legend(['ready', 'partial', 'open']) + _outline_cards(rows, P2C)
    D.slide('paper2', body, '''
<p>The arc: the resistive layer shares charge with a measurable delay (§1). That makes any per-strip time an aggregate, so hit-time ladders are compressed whatever estimator is used (§2). A forward model of the waveforms handles the sharing without inverting it (§3), and reaches the diffusion floor (§4). Then comes the question a reader with a hit-based readout will ask: how much of that can I get without waveforms (§5–7)?</p>
<p>The hits come from the same DREAM waveforms and are compared on the same events against the same reference. That isolates the information content of the readout from everything else. Not modelled: a VMM's own shaping time, its neighbour-trigger logic and dead time. Those are the deferred VMM emulation (work item V1).</p>
<p>Most of §1–4 is written already in <code>WAVEFORM_FIRST_THREADING.md</code> and the SPS kernel reports. The July numbers there come from the R&amp;D model, though, so they need re-running on the packaged <code>wft/</code> code and r06 before they are quoted next to the fleet numbers.</p>''',
            short='Paper II outline')


def s_compression(D, comp):
    u = comp['u']
    mf, le = comp['matched-filter t50'], comp['leading-edge 20 %']
    P = sd.Plot(1040, 600, x=(0, 1000), y=(-150, 75), xlabel='drift time since the mesh, u [ns]',
                ylabel='hit time − true ladder [ns]')
    P.xticks([(v, str(v)) for v in range(0, 1001, 200)])
    P.yticks([(v, f'{v:+d}' if v else '0') for v in range(-150, 76, 50)])
    P.hline(0, MUT, label='no bias', anchor='end', where='above')
    P.line(u, mf, TIER['H1'], tips=[f'matched filter, u = {a:.0f} ns: {b:+.0f} ns' for a, b in zip(u, mf)])
    P.line(u, le, TIER['H0'], dash='10 6',
           tips=[f'leading edge 20 %, u = {a:.0f} ns: {b:+.0f} ns' for a, b in zip(u, le)])
    side = sd.col(
        sd.legend([('matched filter (t50)', TIER['H1']), ('leading edge 20 %', TIER['H0'], 'dash')]),
        sd.p(f'Each strip’s waveform is its own charge plus delayed copies of its neighbours’. Any time read off '
             f'it is pulled toward the cluster mean: late at the mesh, early at the cathode.'),
        sd.p('Ladders shrink by 20–30 %, so tracks read ~4° too steep. Changing the estimator or the threshold '
             'does not help. This is the problem paper II exists to solve.'),
        sd.callout('det3, R&amp;D model, July. R1 repeats it on r06 for all five chambers.', GOLD, 24),
        gap=22, w=560)
    body = sd.title('Every hit-time estimator compresses the drift ladder',
                    'Per-strip time minus the reference-implied ladder, core strips, x plane, det3 cosmics '
                    '(|tan| > 0.08, t0 floated per event).')
    body += sd.row(P.svg('compression'), side, gap=60)
    D.slide('compression', body, '''
<p>Source: <code>WAVEFORM_FIRST_THREADING.md</code> §3 (scripts <code>04_mf_ladder.py</code>, <code>05_estimator_compare.py</code>), parsed directly. The y plane has the same shape, slightly larger. CFD and the production rising-edge hits show it too. Implied v from the ladder slope: matched filter 42/44 µm/ns (x/y), production hits 47/50, against 36.6 from the forward model.</p>
<p>For paper II, this is §2's figure. It needs re-making on r06 for every chamber (work item R1), and adding the hit estimators a VMM would use (peak time, threshold crossing) to the family.</p>''',
            foot='Source: mx_june_cosmic_qa/waveform_first_threading/WAVEFORM_FIRST_THREADING.md §3.',
            short='The problem')


def s_tiers(D, bench):
    steps = [
        dict(label='H0 · production hits', sub='combined_hits: rising-edge time + amplitude per strip',
             color=TIER['H0'], tip='What the DAQ hit-finder writes today. One time and one amplitude per strip.'),
        dict(label='H1 · best hit time', sub='CFD, leading edge, matched filter, peak time',
             color=TIER['H1'], tip='Still one number per strip, but the best estimator we can make from a waveform.'),
        dict(label='H2 · hits + correction', sub='a: slope remap · b: unsharing hybrid · c: forward fit on hits',
             color=TIER['H2c'], tip='H2c is new: the forward model fed hit lists instead of waveforms (work item R3).'),
        dict(label='W · waveform forward model', sub='wft/ on r06: the reference', color=TIER['W'],
             tip='Full waveform matrix, sharing modelled forward.'),
        dict(label='V · VMM emulation', sub='deferred: shaper, neighbour logic, dead time', color=GREY,
             fill='#f5f3ee', tip='Work item V1. g4_digi already injects simulated tracks into real ADC; a VMM '
                                 'digitiser would plug in there.'),
    ]
    keymap = [('H0', 'production raw ladder', 'production raw ladder'),
              ('H2a', 'mf ladder + slope remap (cheap)', 'slope remap (H2a)'),
              ('H2b', 'SOTA hybrid (unshared+cal)', 'unsharing hybrid (H2b)'),
              ('W', 'forward v2', 'forward model (W)')]
    P = sd.Plot(1040, 430, x=(-0.6, 3.6), y=(0, 6.5), ylabel='σ_θ [deg]', margin=(20, 30, 70, 104))
    P.yticks([(v, str(v)) for v in range(0, 7, 2)])
    P.xticks([(i, lab) for i, (_t, _k, lab) in enumerate(keymap)])
    for i, (t, k, lab) in enumerate(keymap):
        if k not in bench:
            continue
        x, y = bench[k]
        P.vbar(i - 0.17, x, 52, TIER[t], tip=f'{lab}, x plane: σ_θ {x:.2f}°', label=f'{x:.1f}')
        P.vbar(i + 0.17, y, 52, TIER[t], opacity=0.5, tip=f'{lab}, y plane: σ_θ {y:.2f}°', label=f'{y:.1f}')
    side = sd.col(
        sd.p(f'July prototype, det3, same 2,000 events ({sd.term("R&amp;D model", G["rd"])}). Solid x, pale y.',
             22, MUT),
        sd.p('A correction gets hits most of the way in σ. What it cannot fix is the angle-dependent bias: '
             'implied v still drifts ~9 µm/ns across angle bins (forward model: under 1).'),
        sd.callout('H1 and H2c have never been measured. R1–R5 fill in every bar on r06, all chambers.', BLUE, 24),
        gap=18, w=560)
    body = sd.title('Four hit tiers against the waveform fit, on the same events',
                    'The comparison paper II is built around. Every tier is computed from the same DREAM waveforms.')
    body += sd.flow(steps, size=23) + sd.row(P.svg('tier benchmark'), side, gap=60)
    D.slide('tiers', body, '''
<p><b>Why hits from DREAM stand in for a hit readout.</b> A hit digitiser such as the VMM records, per strip, one time (peak or threshold crossing) and one amplitude. Reducing the DREAM waveform to exactly that throws away the same information. What it does not reproduce is the VMM's own shaping time (25–200 ns against DREAM's), its per-channel threshold and neighbour-enable logic, and its dead time. Those set second-order differences and belong to the deferred emulation (V1). R4's threshold and neighbour scan bounds the most important of them.</p>
<p><b>The tiers.</b> H0 is the production hit-finder. H1 is the best single time per strip, to show the problem is not the estimator. H2a is the cheap correction from July (matched-filter re-time plus one trained slope remap, α ≈ 0.86–0.89). H2b is the unsharing-plus-calibration hybrid of scripts 26–34; its constants were fitted on hits and must be refitted on r06. H2c is new: give the forward model a hit list (time and amplitude per strip) instead of a waveform matrix, and let it fit the same kernel. It tells a hit-readout user what a sharing-aware fit can recover.</p>
<p><b>Metrics</b> for every tier: σ_θ in angle bins, angle bias, implied-v flatness (the compression signature), full-depth threading &lt; 1 mm, mesh position, efficiency, and near-vertical behaviour. July table (§10 of the threading report): production 4.97/5.74°, hybrid 1.58/1.53°, slope remap 1.51/1.86°, forward v2 1.06/1.10°; threading &lt; 1 mm 24/16 %, 54/52 %, 74/70 %. The fleet r06 numbers for H0 and W are on slide 12.</p>''',
            foot='Bars: WAVEFORM_FIRST_THREADING.md §10 (det3, |tan| 0.10–0.40, R&D model v2; c2/c1 = 0.17).',
            short='Hits vs waveforms')


def _work_table(items):
    rows, tips = [], []
    for w in items:
        needs = ', '.join(w['needs']) if w['needs'] else '–'
        parts = w['item'].split(': ', 1)
        sub = parts[1] if len(parts) > 1 else (w['notes'] or parts[0])
        rows.append([f'<b>{w["id"]}</b>', f'<b>{w["short"]}</b><br><span style="color:{MUT};font-size:18px;'
                     f'line-height:1.25">{sub}</span>', chip(w['state']), w['size'], needs])
        tips.append(w['item'] + ('\n' + w['notes'] if w['notes'] else ''))
    return sd.table(['', 'work item', 'state', 'size', 'needs'], rows, size=22,
                    widths=[70, 1120, 170, 80, 160], align=['left', 'left', 'left', 'center', 'left'], tips=tips)


def _work_notes(items):
    trs = ''.join(f'<tr><td>{w["id"]}</td><td>{w["item"]}</td><td>{w["state"]}</td><td>{w["notes"]}</td></tr>'
                  for w in items)
    return f'<table><tr><th>id</th><th>item</th><th>state</th><th>notes</th></tr>{trs}</table>'


def s_work(D, P):
    W = P['work']
    a = [w for w in W if w['paper'] == 'both'] + [w for w in W if w['id'] in ('R1', 'R2', 'R3', 'R4', 'R5')]
    b = [w for w in W if w['paper'] == 'II' and w not in a]
    c = [w for w in W if w['paper'] == 'I']
    body = sd.title('Work left: one reproduction blocks both papers',
                    'Shared items and the core of paper II. Size: S ≈ days, M ≈ a week, L ≈ weeks. '
                    'Hover a row for the full item.')
    body += _work_table(a)
    D.slide('work-core', body, '<p>Full items, from <code>PAPER_PLAN.md</code>:</p>' + _work_notes(a),
            short='Work: core')
    body = sd.title('Work left: the rest of paper II',
                    'Supporting sections and the deferred VMM emulation. Hover a row for the full item.')
    body += _work_table(b)
    D.slide('work-ii', body, '<p>Full items, from <code>PAPER_PLAN.md</code>:</p>' + _work_notes(b),
            short='Work: II')
    body = sd.title('Work left: paper I is gas, timing and a re-quote',
                    'Detector & performance. Hover a row for the full item.')
    body += _work_table(c)
    D.slide('work-i', body, '<p>Full items, from <code>PAPER_PLAN.md</code>:</p>' + _work_notes(c),
            short='Work: I')


def s_order(D, P):
    W = {w['id']: w for w in P['work'] if w['state'] not in ('deferred', 'decision')}
    depth = {}

    def dep(k):
        if k not in depth:
            depth[k] = 0 if not W[k]['needs'] else 1 + max(dep(n) for n in W[k]['needs'] if n in W)
        return depth[k]
    for k in W:
        dep(k)
    ncol = max(depth.values()) + 1
    lanes = [('both', 'both'), ('II', 'paper II'), ('I', 'paper I')]
    colw, gap, bh, top = 288, 26, 52, 50
    lane_h = {ln: max(sum(1 for k in W if W[k]['paper'] == ln and depth[k] == c) for c in range(ncol))
              for ln, _ in lanes}
    o = []
    phases = ['no blockers: start now'] + [f'after step {c}' for c in range(1, ncol)]
    x0 = 128
    for c in range(ncol):
        o.append(sd.T(x0 + c * (colw + gap) + colw / 2, 24, phases[c] if c else phases[0], 22, MUT, weight=600))
    y = top
    pos = {}
    for ln, lab in lanes:
        h = lane_h[ln] * (bh + 12) + 12
        o.append(f'<rect x="0" y="{y}" width="1664" height="{h}" rx="10" fill="#ebe8e1" fill-opacity="0.5"/>')
        o.append(sd.T(16, y + 38, lab, 23, P1C if ln == 'I' else P2C if ln == 'II' else INK, 'start', 600))
        for c in range(ncol):
            ks = [k for k in W if W[k]['paper'] == ln and depth[k] == c]
            for i, k in enumerate(ks):
                w = W[k]
                bx, by = x0 + c * (colw + gap), y + 12 + i * (bh + 12)
                col = STATE_COL[w['state']]
                tip = f'{k}: {w["item"]}\nstate: {w["state"]}, size {w["size"]}'
                o.append(f'<rect x="{bx}" y="{by}" width="{colw}" height="{bh}" rx="10" fill="#fffdf9" '
                         f'stroke="{col}" stroke-width="3"{sd.tipattr(tip)}/>')
                o.append(sd.T(bx + 12, by + 34, f'{k} · {w["short"]}', 20, INK, 'start', 600, tip=tip))
                pos[k] = (bx, by)
        y += h + 14
    for k, w in W.items():
        for n in w['needs']:
            if n in pos and k in pos:
                (ax, ay), (bx, by) = pos[n], pos[k]
                x1, y1, x2, y2 = ax + colw, ay + bh / 2, bx, by + bh / 2
                xm = (x1 + x2) / 2
                o.append(f'<path d="M{x1:.0f},{y1:.0f} C{xm:.0f},{y1:.0f} {xm:.0f},{y2:.0f} {x2:.0f},{y2:.0f}" '
                         f'fill="none" stroke="{MUT}" stroke-width="1.6" stroke-dasharray="5 5" opacity="0.7"/>')
    H = y + 10
    wx = x0 + ncol * (colw + gap)
    o.append(f'<rect x="{wx}" y="{top}" width="{1664 - wx}" height="{y - top - 14}" rx="12" fill="#fffdf9" '
             f'stroke="{INK}" stroke-width="2"/>')
    for i, l in enumerate(['draft II', 'draft I', 'C3 tripwire check', 'arXiv + submit', 'together']):
        o.append(sd.T(wx + (1664 - wx) / 2, top + 70 + 46 * i, l, 24, INK, weight=600 if i != 4 else 400))
    dec = [w for w in P['work'] if w['state'] == 'decision']
    dtxt = ' · '.join(f'{w["id"]} {w["short"]}' for w in dec)
    body = sd.title('Order of work: most of it can start today',
                    'Each column waits only on the one before it (dependencies from PAPER_PLAN.md, dashed). '
                    f'Decisions to take now: {dtxt}.')
    body += sd.svg(1664, H, ''.join(o), 'order of work')
    D.slide('order', body, '''
<p>Columns are computed from the <code>needs</code> column of the work list. An item sits one column to the right of the latest thing it waits for. Deferred items (the VMM emulation) and the decisions (C2 calibration freeze, C4 journal and sign-off) are left out of the grid.</p>
<p><b>Critical path:</b> C1 (reproduce r06) → R1 (hit tiers) → R2/R3/R4 → R5 (the benchmark). R3, the forward fit on hit lists, is the only large new piece of code. Everything in the first column can run in parallel with C1, and D7 (writing the ready sections of I) can start immediately.</p>
<p><b>C2 should be decided before C1 finishes.</b> If the per-plane c2/c1 ratio is built, every number moves again, and R1–R5 should not start on r06 only to be redone.</p>''',
            short='Order of work')


def s_trip(D, P):
    rows = []
    for t in P['trip']:
        eff = t['effect on II']
        watch = 'watch' in eff
        c = GOLD if watch else GREEN
        lab = 'watch' if watch else 'no change'
        rows.append([t['finding'], f'<span style="color:{c};font-weight:600">● {lab}</span>',
                     f'<span style="color:{MUT}">{eff.replace("**watch**: ", "").replace("**", "")}</span>'])
    nw = sum('watch' in t['effect on II'] for t in P['trip'])
    body = sd.title(f'n_TOF so far changes the calibration recipe, not the model; {nw} items to watch',
                    'What the ongoing n_TOF reconstruction work has found, and what it would mean for paper II.')
    body += sd.table(['n_TOF finding', 'for II', 'why'], rows, size=22, widths=[640, 150, 860],
                     align=['left', 'left', 'left'], tips=[t['source'] for t in P['trip']])
    D.slide('tripwires', body, '''
<p>The forward model was built and validated on the bench and in SPS. The n_TOF chambers ran it under different conditions: S/N 7–10× lower, a different gas state, and calibration bundles borrowed from the bench. So far, everything the n_TOF work found falls into three groups. Some are beam-setting choices (the seeder minimum). Some are calibration-recipe traps that belong in II's protocol section (v taken from a prior, cuts in raw-tan units). One is physics outside the reconstruction: the beam/cosmic angle gap is electron scattering, shown in Geant4 with an ideal reconstruction.</p>
<p>The two <b>watch</b> items are the ones that could change II. If chamber C's angle response stays non-linear with its own in-situ bundle, the model's angle linearity becomes a question II has to answer. On chamber A the in-situ bundle is flat (y 1.00, x ±3 %), which points to calibration transfer, not the model.</p>
<p>C3 in the work list is a last check of this table before submission. Sources: <code>ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md</code> §7–14 and <code>sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md</code>.</p>''',
            short='n_TOF tripwires')
def s_angles(D, F):
    P = sd.Plot(1060, 620, x=(-0.6, 4.6), y=(0, 4), ylabel='σ_θ [deg]')
    P.yticks([(v, str(v)) for v in range(0, 5)])
    P.xticks([(i, DLAB[k]) for i, k in enumerate(DETS)])
    off = {('X', 'hits'): -0.3, ('X', 'wft'): -0.1, ('Y', 'hits'): 0.1, ('Y', 'wft'): 0.3}
    for i, k in enumerate(DETS):
        for pl in ('X', 'Y'):
            new, old = F[f'sigma_theta {pl}'][k]
            b = F[f'bias {pl} deg'][k][0]
            P.vbar(i + off[(pl, 'hits')], old, 36, HITS, opacity=0.55 if pl == 'Y' else 1,
                   tip=f'{DLAB[k]} {pl}, hits chain: σ_θ {old:.2f}°')
            P.vbar(i + off[(pl, 'wft')], new, 36, WFT, opacity=0.55 if pl == 'Y' else 1,
                   tip=f'{DLAB[k]} {pl}, waveform-first (r06): σ_θ {new:.2f}°, bias {b:+.2f}°',
                   label=f'{new:.1f}')
    ratios = [F[f'sigma_theta {pl}'][k][1] / F[f'sigma_theta {pl}'][k][0] for k in DETS for pl in 'XY']
    side = sd.col(
        sd.legend([('hits chain', HITS, 'box'), ('waveform-first, r06', WFT, 'box')]),
        sd.p('Solid bars are the X plane, pale bars Y.', 22, MUT),
        sd.p(f'Every plane improves, by ×{min(ratios):.2f} to ×{max(ratios):.1f}: halved on det3, least on '
             f'det6 Y and det4 Y. The bias is consistent with zero everywhere.'),
        sd.p(f'The improvement is not a tuning gain: the {sd.term("hits chain", G["hits"])} is biased by '
             'construction, and the forward model removes the bias.'),
        sd.callout('det3 and det2 are the headline chambers; det6 and det7 carry board-C/D sharing; det4 is '
                   'gain-limited and better shown through SPS.', GREY, 24),
        gap=22, w=520)
    body = sd.title('H0 → W on the fleet: σ_θ improves on every plane',
                    f'{sd.term("σ_θ", G["sigt"])} per plane against the M3 reference, golden run per detector, '
                    f'{sd.term("r06", G["r06"])} calibration (det4: lp_t0p, det6: lp).')
    body += sd.row(P.svg('sigma theta per detector'), side, gap=72)
    D.slide('angles', body, '''
<p>Source: <code>mx_june_wft/FLEET_DIGEST.md</code> (written by <code>mx_june_wft/digest.py</code>, last committed 2026-08-19 with the r06 promotion), which compares both chains on the same events with the same M3 recipe (χ² &lt; 1, NClus = 4).</p>
<p>The bundles differ by detector: r06 (c2 = 0.6 c1) on det2, det3 and det7; <code>calib_bundle_lp_t0p</code> on det4 and <code>calib_bundle_lp</code> on det6, whose ratios were already below 1. Columns built on different bundles are not interchangeable; the paper table has to say which bundle each row used.</p>
<p>The r06 cost, from the paired bootstrap in <code>R06_GATE_2026-08-19.md</code>: det3 Y +0.06°, det7 X +0.08°, det7 Y +0.13° relative to the inverted kernel. The data prefer the inverted kernel <i>inside this model</i>; the proposed per-plane ratio is the identified fix and has not been built.</p>''',
            foot='Source: mx_june_wft/FLEET_DIGEST.md (r06 promotion, 2026-08-19).', short='Angles')


def s_position(D, F):
    panels = []
    P = sd.Plot(810, 520, x=(-0.6, 4.6), y=(0, 100), ylabel='rays with a track within 5 mm [%]',
                title='efficiency within 5 mm')
    P.yticks([(v, f'{v}') for v in (0, 25, 50, 75, 100)])
    P.xticks([(i, DLAB[k]) for i, k in enumerate(DETS)])
    for i, k in enumerate(DETS):
        new, old = F['within 5 mm %'][k]
        n = F['rays'][k][0]
        P.vbar(i - 0.17, old, 44, HITS, tip=f'{DLAB[k]} hits chain: {old:.1f} % of {n:.0f} rays')
        P.vbar(i + 0.17, new, 44, WFT, tip=f'{DLAB[k]} waveform-first: {new:.1f} % of {n:.0f} rays',
               label=f'{new:.0f}')
    panels.append(P.svg('within 5 mm'))
    Q = sd.Plot(810, 520, x=(-0.6, 4.6), y=(0, 1.2), ylabel='|r| to the M3 ray [mm]',
                title='position residual (includes the reference)')
    Q.yticks([(v / 10, f'{v / 10:.1f}') for v in range(0, 13, 2)])
    Q.xticks([(i, DLAB[k]) for i, k in enumerate(DETS)])
    Q.hline(0.225, PURPLE, label='binary, pitch/√12', tip=G['pitch'], anchor='end', where='below')
    for i, k in enumerate(DETS):
        c = F['core sigma r mm'][k][0]
        m = F['median r mm'][k][0]
        Q.vbar(i - 0.17, c, 44, WFT, tip=f'{DLAB[k]} core σ|r| {c:.2f} mm (wft)', label=f'{c:.2f}')
        Q.vbar(i + 0.17, m, 44, WFT, opacity=0.45, tip=f'{DLAB[k]} median |r| {m:.2f} mm (wft)')
    panels.append(Q.svg('core sigma'))
    body = sd.title('Position holds on det2/3 and recovers det4/6/7',
                    f'{sd.term("Within 5 mm", G["w5"])} (left) and the {sd.term("core σ", G["core"])} and median of |r| '
                    '(right, solid and pale). Same events as the angle slide.')
    body += sd.legend([('hits chain', HITS, 'box'), ('waveform-first', WFT, 'box')]) + sd.row(*panels, gap=44)
    gains = ', '.join(f'{DLAB[k]} +{F["within 5 mm %"][k][0] - F["within 5 mm %"][k][1]:.0f}'
                      for k in ('g_det4', 'g_det6_long', 'g_det7_long'))
    D.slide('position', body, f'''
<p>Gains in points of within-5 mm: {gains}. det3 and det2 were already at ~92–93 % and stay there. det4's low value is its geometry: 62 % of its active area does not amplify, in fixed stripes (<code>DET4_SPS_ASSESSMENT.md</code>).</p>
<p>All residuals here are detector ⊕ reference. The next slide is why the bench cannot measure the chamber's intrinsic position resolution, and what does.</p>
<p>Spark fractions in the same digest (det3 8 %, det2 10 %, det4 10 %, det6 22 %, det7 37 %) are identical between chains, as they should be: detection is still hits-defined.</p>''',
            short='Position')


def s_sps_res(D, S):
    res = S['res']
    fit = res['fit']
    P = sd.Plot(1000, 700, x=(-0.6, 5.6), y=(0, 0.8), ylabel='σ [mm]')
    P.yticks([(v / 10, f'{v / 10:.1f}') for v in range(0, 9, 2)])
    zl = []
    i = 0
    for coord, zs in res['zones'].items():
        for z in zs:
            zl.append((i, f'{coord[-1]} {z["pitch"]:g}'))
            P.vbar(i, z['sigma_res'], 34, GREY, opacity=0.55,
                   tip=f'{coord}, uRWELL pitch {z["pitch"]} mm: residual σ {1000 * z["sigma_res"]:.0f} µm, n = {z["n"]:,}')
            P.points([i], [z['sigma_det4']], BLUE, r=10,
                     tips=[f'{coord} pitch {z["pitch"]}: det4 after removing the pointing term '
                           f'({1000 * z["pointing"]:.0f} µm): {1000 * z["sigma_det4"]:.0f} µm'])
            i += 1
    bench = (sum(x * x for x in M3_POINT_MM) / 2) ** 0.5
    ms = MS_MRAD * MS_LEVER_MM / 1000
    tot = (fit['sigma_det4'] ** 2 + bench ** 2 + ms ** 2) ** 0.5
    P.xticks([(a, b) for a, b in zl] + [(5, 'bench')])
    P.vbar(5, tot, 34, ORANGE, opacity=0.35,
           tip=f'176 µm ⊕ M3 pointing {1000 * bench:.0f} µm ⊕ scattering {1000 * ms:.0f} µm = {1000 * tot:.0f} µm '
               f'(arithmetic, not a fit)')
    P.points([5, 5], list(BENCH_S68), ORANGE, r=10,
             tips=[f'det3 bench σ68 {v:.2f} mm, θ < 5° ({c}), the residual the SPS report compares with'
                   for v, c in zip(BENCH_S68, 'xy')])
    P.hline(fit['sigma_det4'], BLUE, label=f'det4 fit {1000 * fit["sigma_det4"]:.0f} ± '
                                            f'{1000 * fit["sigma_det4_err"]:.0f} µm', anchor='end', where='below')
    P.hline(0.225, PURPLE, label='binary 225 µm', tip=G['pitch'], anchor='end')
    side = sd.col(
        sd.legend([('residual to the uRWELL', GREY, 'box'), ('det4, reference removed', BLUE, 'dot'),
                   ('bench', ORANGE, 'dot')], size=22),
        sd.p(f'The uRWELL back plane has three pitches, so the reference’s own term is fitted out, not modelled: '
             f'det4 = {1000 * fit["sigma_det4"]:.0f} µm = {fit["f_back"]:.2f} × '
             f'{sd.term("pitch", G["pitch"])}.'),
        sd.p(f'The bench reference points with a core σ of {M3_POINT_MM[0]:.2f}/{M3_POINT_MM[1]:.2f} mm, without '
             f'scattering. ~{MS_MRAD} mrad over {MS_LEVER_MM} mm adds {1000 * ms:.0f} µm and closes the bench residual.'),
        sd.callout('PLAN_37 asked for this deconvolution. The beam answers it, but the scattering term is '
                   'inferred, not measured.', GOLD, 24),
        gap=20, w=580)
    body = sd.title(f'Paper I §8: det4 resolves {1000 * fit["sigma_det4"]:.0f} µm; the bench sees its reference',
                    f'det4 in SPS H4, run 53, {res["run53"]["n"]:,} tracks at normal incidence, by uRWELL zone '
                    '(view, pitch in mm).')
    body += sd.row(P.svg('sps resolution'), side, gap=64)
    D.slide('sps-resolution', body, '''
<p>Source: <code>sps_beam_test_26/analysis/spatial_resolution/</code> (<code>resolution.py</code> → <code>results.json</code>; report <code>report.html</code>). The five zones are independent: three uRWELL pitches on X, two on Y. The joint fit over the back plane's pitches gives σ_det4 = 176 ± 10 µm with χ² = 2.8.</p>
<p>The bench bar is arithmetic: 176 µm ⊕ the M3 pointing core (quadrature mean of 0.21 and 0.24 mm, from the M3 self-resolution study, which excludes multiple scattering by construction) ⊕ 1.1 mrad × 558 mm. The SPS report's own caveat: <i>the scattering explanation is arithmetic, not a measurement</i>. A paper would want either a scattering simulation of the bench stack or the M3 inter-doublet kink (2.6 mrad measured) used directly.</p>
<p>Caveat for the paper: this is det4 at normal incidence in its live bands, in Ar/CF₄/iso. It establishes what the chamber design can do, not det3's number at cosmic angles.</p>''',
            short='SPS resolution')


def s_kernel(D, S, retired):
    rows = [(lab, q, 0.0, RED, f'{lab}: retired 2026-08-21, c2/c1 = {q:.2f}') for lab, q in retired]
    rows.append(('det6 (lp)', 0.82, 0.0, GREY, 'det6 calib_bundle_lp: c2/c1 = 0.82, physical, kept'))
    rows.append(('shipped r06', 0.60, 0.0, GREEN, 'calib_bundle_r06: c2 pinned to 0.6 c1 on det2, det3, det7'))
    bq, be, bn = S['bench']
    rows.append(('det3 bench, measured', bq, be, ORANGE,
                 f'near-vertical det3 cosmics, delay form: c2/c1 = {bq:.2f} ± {be:.2f}, n = {bn}'))
    for run, fld, n, q, e in S['beam']:
        rows.append((f'SPS {fld:.0f} V/cm', q, e, BLUE,
                     f'SPS H4 det4 head-on, {run}, Y view, delay form: c2/c1 = {q:.3f} ± {e:.3f}, n = {n:,}'))
    n = len(rows)
    P = sd.Plot(1040, 640, x=(0, 2.4), y=(n - 0.4, -0.6), xlabel='c2 / c1', margin=(24, 30, 92, 290))
    P.xticks([(v / 2, f'{v / 2:g}') for v in range(0, 5)] + [(2.4, '')])
    P.yticks([(i, r[0]) for i, r in enumerate(rows)])
    P.raw(f'<rect x="{P.X(1):.1f}" y="{P.y0}" width="{P.X(2.4) - P.X(1):.1f}" height="{P.ph}" '
          f'fill="{RED}" fill-opacity="0.06"/>', back=True)
    P.vline(1, RED, tip='The ±2 strip is reached only through the ±1, so it cannot receive more.')
    P.text(1.06, n - 1.6, 'c2 > c1:', 22, RED, weight=600)
    P.text(1.06, n - 1.0, 'not a resistive film', 22, RED)
    for i, (lab, q, e, c, tip) in enumerate(rows):
        if e:
            P.raw(sd.line(P.X(q - e), P.Y(i), P.X(q + e), P.Y(i), c, 3))
        P.points([q], [i], c, r=10, tips=[tip])
    beams = [q for _r, _f, _n, q, _e in S['beam']]
    fl = S['ang']['flat700']['x']
    side = sd.col(
        sd.p(f'Head-on in the beam, the cross-relation n₀∗W_d = n_d∗W₀ cancels the unknown drive signal, so the '
             f'{sd.term("kernel", G["kernel"])} is measured with no deconvolution.'),
        sd.p(f'c2/c1 = {min(beams):.2f}–{max(beams):.2f} at three drift fields; det3 bench cosmics agree within '
             f'errors. Every shipped bundle before 8-19 sat on the wrong side of 1.'),
        sd.p(f'The ±1 delay is {fl["pm1_shift_ns"]:.0f} ns on the flat mount and survives a 25.6° rotation.', 24, MUT),
        gap=20, w=560)
    body = sd.title(f'Paper II §1 · beam: c2/c1 = {sum(beams) / len(beams):.2f}; every retired bundle had c2 > c1',
                    'Sharing ratio c2/c1: the retired bundles, the shipped one, and the two direct measurements.')
    body += sd.row(P.svg('kernel ratio'), side, gap=56)
    D.slide('kernel', body, '''
<p>Sources: <code>sps_beam_test_26/analysis/sharing_kernel/fit_kernel.json</code> (beam, Y view, delay form fitted on RAW run 71 at 243/156/95 V/cm) and <code>bench_kernel_y.json</code> (det3 near-vertical cosmics, tan θ &lt; 0.05). Retired ratios from <code>mx_june_wft/RETIRE_C2GTC1_2026-08-21.md</code> §2. Errors are the fits' bootstrap errors.</p>
<p>What the beam did <i>not</i> pin: the absolute time constant walks 629 → 1040 ns as the fit window grows, because 44–52 % of the central-strip amplitude is still present at the last sample. τ and c2 are bounds. The cascade (lp) form fits the cross-relation at 2.1 % residual against the shipped delay form's 4.2 %, but transplanted onto det3 it costs σ_θ(Y) 1.14 → 1.54°, so it is not adopted.</p>
<p>The cosmic χ² prefers c2 ≥ c1 inside the current model (R06_GATE). The paper should present that as evidence about the model, with the beam as the arbiter.</p>''',
            short='Kernel')


def s_where(D, F, bund):
    items = [
        (GREEN, 'In git', 'mx_june_wft/FLEET_DIGEST.md',
         'The only written record of the r06 golden numbers: every value on slides 12–13. Committed with the '
         'r06 promotion, 2026-08-19.'),
        (BLUE, 'On lxplus', '~/wft_campaign_r06/',
         '152 condor result tarballs of 2026-08-21 (events.parquet per sub-run × detector), 165 MB. The full '
         'June campaign on r06. Not yet pulled back.'),
        (GOLD, 'On this machine', '~/x17/cosmic_bench/Analysis',
         'r06 calibration bundles (8-19) are here, but the reco products, angles and efficiency JSONs are '
         'from 8-05, on the earlier bundles.'),
        (RED, 'Not found anywhere checked', 'golden r06 products',
         'The per-key angles/efficiency JSONs behind FLEET_DIGEST and the SUPERSEDED manifest. They were '
         'made on another machine.'),
    ]
    cards = ''.join(
        f'<div style="flex:1;background:#fffdf9;border:1px solid {RULE};border-top:8px solid {c};'
        f'border-radius:14px;padding:26px;display:flex;flex-direction:column;gap:14px">'
        f'<p style="font-size:30px;font-weight:600;color:{c}">{h}</p>'
        f'<p style="font-size:23px;font-family:IBM Plex Mono,monospace">{path}</p>'
        f'<p style="font-size:24px;color:{MUT};line-height:1.35">{txt}</p></div>' for c, h, path, txt in items)
    steps = [
        dict(label='pull', sub='rsync wft_campaign_r06 back; collect_results.py --promote', color=BLUE),
        dict(label='re-run 01–04', sub='alignment, efficiency, angles, maps on the five golden keys', color=BLUE),
        dict(label='digest', sub='digest.py must reproduce FLEET_DIGEST.md', color=GREEN,
             tip='If it does not, the paper numbers have no reproducible source. Settle this before writing.'),
        dict(label='tier-B rows', sub='six det3 drift-scan v-refits, locally on r06', color=GOLD),
    ]
    bl = ', '.join(f'{DLAB[k]} {bund.get(k, "?").replace("calib_bundle_", "")}' for k in DETS)
    body = sd.title('Cosmic numbers have a record, not local products',
                    'Where the r06 reconstruction lives: checked 2 Oct, unchanged on 8 Oct. This is work item C1.')
    body += f'<div style="display:flex;gap:28px">{cards}</div>'
    body += sd.p('Before quoting anything: make the record reproducible on this machine.', 28, INK, 600)
    body += sd.flow(steps, size=24)
    D.slide('where', body, f'''
<p>Checked on 2026-10-02: <code>find /media/dylan/data/x17/cosmic_bench/Analysis -path '*wft*' -newermt 2026-08-18</code> finds no angle, efficiency or events metadata files; sat_det3's <code>events_lp.meta.json</code> points at <code>calib_bundle_lp_sp0free</code> (8-05). <code>~/x17</code> is the same disk. On lxplus, <code>~/wft_campaign</code> holds the pre-fix 8-13 tarballs and <code>~/wft_campaign_r06</code> the 8-21 re-run; <code>/eos/user/d/dneff/x17</code> has no wft campaign output.</p>
<p>Bundles named in FLEET_DIGEST: {bl}.</p>
<p>The campaign packager has two known traps (<code>RETIRE_C2GTC1_2026-08-21.md</code> §6a–6b): it ran det4 without the t0 prior once ("t0p-adopted dets: none yet" is the tell), and a v-refit seeded from an r06 bundle used to write c2 = 0 with no ratio (fixed, tested). Expect bit-identity on det2/3/4/6; det7 differed on 3 of 8,082 events.</p>''',
            short='Where the numbers are')




def s_close(D):
    items = [
        ('Hits from DREAM are not a VMM', 'Same information per strip (one time, one amplitude), but not the VMM’s '
         'shaping time, neighbour logic or dead time. II must say so, and V1 stays open.'),
        ('The July tier numbers are a prototype', 'R&amp;D model on det3 only. Nothing from it is quoted until '
         'R5 re-runs it on the packaged code and r06.'),
        ('One calibration, one model', 'r06 is better physics and a worse fit (Y up to +0.13°). The per-plane '
         'ratio and the lp-vs-delay branch are unresolved; either would re-open the gates.'),
        ('The kernel tail is outside every window', 'τ and c2 are bounds. No beam for three years.'),
        ('n_TOF is still moving', 'Two tripwires on watch. Chamber C’s angle linearity could add a section to II.'),
    ]
    rows_ = ''.join(f'<div style="display:flex;gap:28px;padding:12px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:26px;font-weight:600;width:430px">{a}</p>'
                    f'<p style="font-size:22px;color:{DMUT};flex:1;line-height:1.35">{b}</p></div>' for a, b in items)
    dec = [('C2 · freeze r06?', 'Or build the per-plane c2/c1 first. Decide before C1 finishes, so R1–R5 run once. '
            'Recommendation: freeze, unless the ratio is a few days’ work.'),
           ('Order', 'II first or the same day. I cites II for every geometric number.'),
           ('C4 · journal', 'NIM A or JINST, the same for both, as companions.'),
           ('Chambers in I’s table', 'det3 + det2 headline; det6/7 as the board-C/D contrast; det4 via SPS.')]
    drows = ''.join(f'<div style="display:flex;flex-direction:column;gap:6px;padding:12px 0;border-top:1px solid #333b4a">'
                    f'<p style="font-size:26px;font-weight:600;color:{DBLUE}">{a}</p>'
                    f'<p style="font-size:22px;color:{DMUT};line-height:1.35">{b}</p></div>' for a, b in dec)
    body = (f'<div style="display:flex;gap:80px">'
            f'<div style="flex:1.25;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:50px;font-weight:600">What this does not rule out</h2>{rows_}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px">'
            f'<h2 style="font-size:50px;font-weight:600">Decisions</h2>{drows}</div></div>')
    D.slide('close', body, '''
<p>Record of this plan: <code>mx_june_cosmic_qa/PAPER_PLAN.md</code> (the tables this deck parses). <code>PAPER_STATUS.md</code> keeps the 2026-10-02 re-audit and the July audit beneath it.</p>
<p>Key inputs for writing: <code>RECONSTRUCTION_BASIS.md</code> (migration table), <code>waveform_first_threading/WAVEFORM_FIRST_THREADING.md</code>, <code>mx_june_wft/FLEET_DIGEST.md</code>, <code>mx_june_wft/RETIRE_C2GTC1_2026-08-21.md</code>, <code>sps_beam_test_26/analysis/README.md</code> with its <code>sharing_kernel</code>, <code>angled_kernel</code> and <code>spatial_resolution</code> reports, <code>mpgd26/slides/RUNNING_ORDER.md</code>, and for the VMM follow-up <code>ntof_cosmics/g4_digi/</code>.</p>''',
            dark=True, short='Caveats & decisions')


# --------------------------------------------------------------------------- #
def build(out: Path) -> Path:
    P = load_plan()
    F, bund = load_fleet()
    S = load_sps()
    comp, bench = load_july()
    retired = load_retired()
    D = sd.Deck('MX17 Paper Plan',
                'The MX17 detector work as two companion papers (detector & performance; reconstruction, with '
                'a hits-vs-waveforms comparison): outlines, the work left, and the evidence in hand.')
    s_cover(D, P, F, S)
    s_split(D)
    s_outline1(D, P)
    s_outline2(D, P)
    s_compression(D, comp)
    s_tiers(D, bench)
    s_work(D, P)
    s_order(D, P)
    s_trip(D, P)
    s_angles(D, F)
    s_position(D, F)
    s_sps_res(D, S)
    s_kernel(D, S, retired)
    s_where(D, F, bund)
    s_close(D)
    meta = dict(title='MX17 papers: the two-paper plan',
                summary='The detector paper split into two companions (detector & performance; reconstruction '
                        'with a hits-vs-waveforms comparison): outlines, work left, order of work, n_TOF tripwires.',
                tags='X17,detector,paper', date=dt.date.today().isoformat())
    return D.write(out, meta, footer=f'Built {dt.datetime.now():%Y-%m-%d %H:%M} by '
                                     'nTof_x17/mx_june_cosmic_qa/make_paper_plan_deck.py with slidedoc.py.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path,
                    default=Path(os.path.expanduser('~/x17/paper_status/mx17-detector-paper-status.html')))
    a = ap.parse_args()
    print('wrote', build(a.out))


if __name__ == '__main__':
    main()
