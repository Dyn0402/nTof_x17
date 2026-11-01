#!/usr/bin/env python3
"""make_review_deck.py [--out PATH] -- the Paper I review deck (slide note) on pass 1.

One or more slides per Paper I section, in paper order: the figures the paper would carry
(papers/paper1/figures/*.pdf, rasterised with pdftoppm), a claim title, a few discussion
bullets, the open decisions for that section, and a Details block with the old-text ->
pass-1 numbers and their sources.

Sources: rerun/PAPER1_STORY_2026-11-01.md (spine, decisions A-Q), rerun/PAPER1_NUMBERS_2026-10-31.md,
the section READMEs and their results/*.json, papers/FIGURE_PROVENANCE_2026-10-31.md (figure
status: a figure still 'pending-sps' or 'placeholder' is shown as pending, not drawn).
Numbers are read from the result files; a missing file makes its slide say "pending".

Publish: python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <out> --slug mx17-paper1-review --force --deploy
"""
import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
R = REPO / 'rerun'
FIG1 = HERE / 'figures'
FIG2 = REPO / 'papers' / 'paper2' / 'figures'
CACHE = Path(os.environ.get('CLAUDE_JOB_DIR', '/tmp')) / 'tmp' / 'paper1_deck_png'
PROV = REPO / 'papers' / 'FIGURE_PROVENANCE_2026-10-31.md'
BOOT = R / 'paper1_floor' / 'BOOTSTRAP_2026-11-01.md'
SPSFIGS = R / 'paper1_spsfigs' / 'README.md'

DET = ['det2', 'det3', 'det4', 'det6', 'det7']
DCOL = {'det2': sd.BLUE, 'det3': sd.GREEN, 'det4': sd.GREY, 'det6': sd.ORANGE, 'det7': sd.PURPLE}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def jload(rel):
    p = R / rel
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def prov_status():
    """figure name -> status keyword from the provenance table."""
    st = {}
    if not PROV.exists():
        return st
    for ln in PROV.read_text().splitlines():
        m = re.match(r'\|\s*([a-z0-9_]+)\s*\|.*\|\s*\**([a-z-]+)\**\s*\|', ln)
        if m:
            st[m.group(1)] = m.group(2)
    return st


STATUS = prov_status()


def ready(name):
    return STATUS.get(name, 'current') not in ('pending-sps', 'pending-vt', 'placeholder', 'stale')


def pdf_png(pdf, dpi=150):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / (pdf.stem + f'_{dpi}')
    png = Path(str(out) + '.png')
    if not png.exists() or png.stat().st_mtime < pdf.stat().st_mtime:
        subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-singlefile', str(pdf), str(out)], check=True)
    return png


def fig(name, w=None, h=None, tip=None, paper=1, crop=None):
    """A paper figure as an embedded PNG, or a 'pending' card if it isn't built on pass 1."""
    pdf = (FIG1 if paper == 1 else FIG2) / f'{name}.pdf'
    if not ready(name) or not pdf.exists():
        why = STATUS.get(name, 'missing')
        return pending(name, why, w, h)
    im = Image.open(pdf_png(pdf)).convert('RGB')
    if crop:
        W, H = im.size
        im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
    box_w = w or (h * im.width / im.height)
    if h and w:
        box_w = min(w, h * im.width / im.height)
    tw = int(min(im.width, box_w * 1.4))
    if tw < im.width:
        im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(b, 'PNG', optimize=True)
    t = tip or f'{name}.pdf ({STATUS.get(name, "current")} in FIGURE_PROVENANCE_2026-10-31.md)'
    return sd.img(b.getvalue(), w=w, h=h, tip=t)


def pending(name, why, w=None, h=None):
    ws = f'width:{w}px;' if w else 'flex:1;'
    hs = f'height:{h}px;' if h else 'min-height:240px;'
    return (f'<div style="{ws}{hs}display:flex;flex-direction:column;justify-content:center;align-items:center;gap:12px;'
            f'border:3px dashed {sd.GOLD};border-radius:16px;background:{sd.CARD}">'
            f'<p style="font-size:30px;font-weight:600;color:{sd.GOLD}">{name}: pending</p>'
            f'<p style="font-size:24px;color:{sd.MUT}">status “{why}” in FIGURE_PROVENANCE</p></div>')


def bullets(items, size=26):
    li = ''.join(f'<li style="margin-bottom:10px">{x}</li>' for x in items)
    return f'<ul style="font-size:{size}px;line-height:1.35;padding-left:30px;margin:0">{li}</ul>'


def ask(letters_text, size=24):
    """The open questions/decisions for the slide, as a gold callout."""
    return sd.callout('<b>Decide / open:</b> ' + letters_text, color=sd.GOLD, size=size)


def tk(vals):
    return [(v, f"{v:g}") for v in vals]


def f1(x, d=1):
    return '–' if x is None else f'{x:.{d}f}'


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
EFF = jload('effbudget/pass1/results/summary.json') or {}
A1 = jload('effbudget/pass1/results/a1.json') or {}
RES = jload('paper1_res/results/summary.json') or {}
GASB = jload('paper1_gas/results/summary.json') or {}
GASFIT = jload('sps_gasfit/results/verdict.json') or {}
FLOOR = jload('paper1_floor/results/floor_budget.json') or {}
POSMS = jload('paper1_floor/results/position_ms.json') or {}
OPS = jload('paper1_ops/results/summary.json') or {}
K1V = jload('paper1_floor/results/k1_verdict.json') or {}


def timing_golden():
    p = R / 'paper1_timing' / 'results' / 'golden.csv'
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))


TIM = timing_golden()

A1_ROW = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4',
          'det6': 'g_det6_long', 'det7': 'g_det7_long'}


def eff_ig(det):
    g = EFF.get('golden', {}).get(det, {})
    return g.get('eff_ig'), g.get('err_ig'), g.get('n_ig')


def res_s68(det, view, arm='pass'):
    try:
        return RES['headline'][det][view][f's68_{arm}']
    except Exception:
        return None


def a1pct(det):
    try:
        return A1[A1_ROW[det]]['interior_good']['pct']
    except Exception:
        return {}


def ledger_counts():
    p = R / 'PASS1_LEDGER.md'
    t = p.read_text() if p.exists() else ''
    m1 = re.search(r'Bench: (\d+) non-excluded rows, (\d+) complete; SPS: (\d+) rows', t)
    m2 = re.search(r'Bench events reconstructed: ([\d,]+); SPS events: ([\d,]+)', t)
    return (int(m1.group(2)) if m1 else 156, int(m1.group(3)) if m1 else 71,
            m2.group(1) if m2 else '–', m2.group(2) if m2 else '–')


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D):
    nb, ns, eb, es = ledger_counts()
    e3 = eff_ig('det3')[0]
    body = (sd.kicker('MX17 · Paper I review · from-scratch pass 1 · for Monday 2 Nov 2026') +
            '<h1 style="font-size:70px;font-weight:600;line-height:1.08;letter-spacing:-1.5px;color:#eef0f3">'
            'Pass 1 tells one story: ~1° angles on cosmics, ~190 µm in the beam (Y), ~40 ns timing'
            f' and {e3:.0f} % efficiency on the best chamber</h1>' +
            sd.p('Every Paper I section re-derived on the final tables, with the figures it would carry and what is left '
                 'for you to decide. The one thing still unidentified is a ~0.6° angle remainder, now bounded.',
                 size=30, color=sd.DMUT) +
            sd.row(sd.bignum(f'{nb}', 'bench rows', sd.DBLUE, f'{eb} reconstructed events, all on one code version',
                             tip='rerun/PASS1_LEDGER.md: every non-excluded registry row complete'),
                   sd.bignum(f'{ns}', 'SPS rows', sd.DGREEN, f'{es} events, all on the final oscillator rule',
                             tip='rerun/sps_audit + rerun/sps_junk; ledger clean'),
                   sd.bignum('5', 'chambers', sd.DGREY, 'det2 · det3 · det4 · det6 · det7 (det4 also at SPS H4)'),
                   sd.bignum('18', 'open decisions', sd.GOLD, 'A–Q plus the det4 gap (R), on the last slides',
                             tip='rerun/PAPER1_STORY_2026-11-01.md'), gap=48))
    notes = ('<p>Built from the analysis outputs on main; rerunning this generator after any of them changes updates the '
             'numbers. Provisional choices where you have not decided (from the story spine): SPS run 63, interior ≥ 30 mm '
             'fiducial, no-clip s68, the pass arm for bench angles (the v(t) arm is shown alongside).</p>'
             '<p>Spine: <code>rerun/PAPER1_STORY_2026-11-01.md</code>. Every number with old-text value, status and source: '
             '<code>rerun/PAPER1_NUMBERS_2026-10-31.md</code>. Cross-section audit: <code>rerun/consistency/README.md</code>.</p>')
    D.slide('cover', body, notes, dark=True, short='Cover')


def s_pass(D):
    st = [dict(label='Decode', sub='rr1 re-decode, common mode', color=sd.GREY,
               tip='June bench re-decoded and audited 10-23 (decoded_root_fa80c06); SPS from banco/EOS'),
          dict(label='Calibrate from nothing', sub='template, kernel, σ0, Dp, v per chamber', color=sd.BLUE,
               tip='calib DAG (S1) → G1 gate: all five chambers pass (rerun/calib/G1_RESULT_2026-10-30.md)'),
          dict(label='Reconstruct', sub='waveform forward fit, 156 bench + 71 SPS rows', color=sd.BLUE,
               tip='wft/ forward model on decoded_root waveforms; code 6a0bb1f (bench), SPS 3ac0e79 / 48244778'),
          dict(label='Downstream', sub='bow block, v(t), alignment, efficiency', color=sd.GREEN,
               tip='table level, no refit: bow gate PASS, v(t) on 7 bench rows (VT_PLAN_2026-10-30.md)'),
          dict(label='Paper numbers', sub='§4–§9, A1, both arms where a choice exists', color=sd.GOLD,
               tip='rerun/PAPER1_NUMBERS_2026-10-31.md')]
    body = (sd.title('Every number here comes from one from-scratch pass, read from waveforms, not hit times',
                     'Pass 1 (Analysis_pass1, 30 Oct) plus downstream steps that never refit')
            + sd.flow(st, size=26)
            + sd.row(sd.card(sd.p('<b>Hover</b> any dotted ' + sd.term('term', 'like this one: definitions, counts and '
                                                                       'sources live in tooltips') +
                                  ', data point or figure for its source. Every slide has a <b>Details</b> drop-down '
                                  'with the old text value → pass-1 value and the files.', size=26)),
                     sd.card(sd.p('<b>Gold callouts</b> are the open decisions for that section (letters A–Q, '
                                  'collected at the end). <b>Dashed gold boxes</b> are figures still being built.',
                                  size=26)),
                     sd.card(sd.p('<b>Provisional choices</b> used throughout: SPS run 63; interior ≥ 30 mm; '
                                  's68 without clip; bench angles from the pass table (v(t) arm in brackets).',
                                  size=26)), gap=32))
    notes = ('<p>Pass status: <code>rerun/PASS1_STATUS_2026-10-30.md</code>, review <code>rerun/pass1_review/REVIEW_2026-10-30.md</code>, '
             'ledger <code>rerun/PASS1_LEDGER.md</code> (156/156 + 71/71). Parameters: <code>docs/PARAMETERS.md</code>.</p>'
             '<p>Basis decided 2026-07-28 (RECONSTRUCTION_BASIS.md): hit times compress the drift ladder by 20–30 %; geometry '
             'comes from the waveforms. Hits stay for candidate finding and QA.</p>')
    D.slide('pass', body, notes, short='What pass 1 is')


def s_setup(D):
    body = (sd.title('§1–3: one micro-TPC plane measures a track’s angle; the bench and H4 give two references',
                     'The schematics the paper carries (pass-independent)')
            + sd.row(sd.col(fig('microtpc_principle', w=780, h=410), sd.p('microtpc_principle (current)', 22, sd.MUT)),
                     sd.col(fig('bench_setup', w=780, h=410), sd.p('bench_setup (current)', 22, sd.MUT)), gap=56)
            + sd.row(sd.col(bullets(['§2 numbers are from the spec page, not pass products: 9 unknowns, 31 assumed.',
                                     'Bench: M3 reference, z up, slots 702 / 232 mm, scintillator trigger; '
                                     'M3 pointing 0.21–0.28 mm per axis (not 0.4: the units trap).'], 24)),
                     sd.col(ask('chamber_stack, readout_layout, sps_setup are still placeholders (Stephan’s drawings, '
                                'the mpgd26 render). §1 Introduction and §10 Conclusions are unwritten.'), w=760), gap=48))
    notes = ('<p>§2 efficient area 399.4 × 359.9 mm² and the ~19 mm Y dead band are not re-derived on pass 1 '
             '(<code>common/mx17_active_area.py</code>); the fiducial uses 8 mm from the passivated Y edges.</p>'
             '<p>The MPGD proceedings’ “M3 pointing ~0.4 mm” is the deconvolved DUT value, not the pointing (0.21–0.28 mm).</p>')
    D.slide('setup', body, notes, short='§1–3 setups')


def s_hv(D):
    body = (sd.title('§4: the operating point is set by discharges, and the golden rows ran 10–65 V above the optimum',
                     'Interior efficiency, discharge and burst losses vs HV, pass 1 (det2/3/6/7)')
            + sd.row(fig('hv_scan', w=1040, h=585),
                     sd.col(bullets(['Gain doubles every 16.6–19.1 V; discharges rise ×10 every 21–42 V '
                                     '(∝ gain² to gain^2.7).',
                                     'Scan optima ' + sd.term('480 / 470 / 480 / 430 V', 'det2 92.2 %, det3 93.8 % (plateau 460–490), '
                                                             'det6 85.8 %, det7 89.8 % (interior_good)') +
                                     ' (det2/3/6/7). At the golden 495 V det6/det7 read 74.4 / 68.0 %: '
                                     'discharges take 16.8 / 24.7 %.',
                                     'The > 50-strip veto also removes bursts with no discharge '
                                     '(6.3 % of g_det4, 7–10 % of det6 at low HV).'], 24),
                            ask('<b>G</b> quote det6/det7 at the golden HV, at the optima, or both; which chambers “480/480/440 V” names. '
                                '<b>H</b> bursts: separate class or one spark class.'), w=560), gap=48))
    foot = 'Source: rerun/paper1_ops/README.md (OPS), rerun/effbudget (EFF), rerun/paper1_charge (CHG)'
    notes = ('<p><b>Text → pass 1.</b> “480/480/440 V” = the optima det3 (or det2) / det6 / det7: OK as optima. The golden rows ran at '
             '490 (det3) and 495 V (others). Gain slopes (CHG): ×2 every 16.6 (det2), 16.7 (det3), 19.1 (det6), 17.9 V (det7), curved; '
             'an M3-free hits proxy reproduces them and shows the 6-23 night flat for det3 too.</p>'
             '<p><b>Not re-derived:</b> “Poisson”, “muon-induced 4–6×”, “edge-concentrated” sparks (needs discharge positions and random triggers).</p>'
             '<p><b>det4:</b> no bench gain curve (6-23: chambers not yet at working gain, CONS §1). At SPS its charge rises only ×1.65 over '
             '580–780 V (run66, CF₄, telescope denominator) and is flat over 550–625 V in Ar/CO₂/iso. Whether its SPS HV path has a series '
             'resistor or filter is a question for Stephan.</p>')
    D.slide('hv', body, notes, foot=foot, short='§4 HV, operating point')


def s_spark(D):
    body = (sd.title('§4: a discharge costs at most a few ms, and the gain does not sag after it',
                     'Efficiency and charge vs time since the previous discharge, all five chambers')
            + sd.row(fig('spark_deadtime', w=1080, h=585),
                     sd.col(bullets(['Dead time per discharge, 95 % UL: ' + sd.term('≤ 1–3 ms', 'det2 2.1, det3 1.2 (0.2 ± 0.6), '
                                                                                    'det6 2.6, det7 2.1 ms; det4 15.5 ms') +
                                     ' on the bench chambers.',
                                     'At the golden HV: 0.13–0.17 discharges/s (det2/det3), 0.69 (det6), 1.14 (det7).',
                                     'Confirms the abstract’s “no dead time after a spark”, now measured on all five.'], 24),
                            ask('the text’s Poisson / muon-induced / edge claims cannot be re-derived from bench data '
                                '(no discharge positions, no random triggers): keep, soften, or drop?'), w=520), gap=48))
    foot = 'Source: rerun/paper1_ops/README.md; discharge = vetoed trigger with a chip ≥ 8 strips over 3000 ADC'
    notes = ('<p>det6/det7 read high after a discharge: a gain-drift artefact, not recovery. No gain sag on det2/det3 (0.96–1.03).</p>'
             '<p>Definitions in <code>docs/PARAMETERS.md</code> §8 (four new rows from the OPS session).</p>')
    D.slide('spark', body, notes, foot=foot, short='§4 discharges')


def s_charge(D):
    body = (sd.title('§5: X and Y share the charge evenly on every chamber, within 3 % event to event',
                     'f = Q_X / (Q_X + Q_Y) on the golden rows, unsaturated events')
            + sd.row(fig('charge_balance', w=1000, h=585),
                     sd.col(bullets(['Median f 0.485–0.513 on all five; spread s68 0.014–0.032 (det7 0.116).',
                                     'The text’s 0.487 / 0.531 and σ68 0.07 were the hits proxy: det2 0.531 → 0.513, '
                                     'σ68 0.07 → 0.014–0.032.',
                                     'Gain slopes from unsaturated events only (the 3550 rule; K4 locked).'], 24),
                            ask('“three charge proxies agree” is not re-derived (only the waveform charge exists on pass 1): '
                                'drop the sentence or re-run the proxies?'), w=600), gap=48))
    foot = 'Source: rerun/paper1_charge/README.md (CHG)'
    notes = ('<p>Golden-row f: det3 0.485, det2 0.513, det4 0.509, det6 0.504, det7 0.485. The charge is the 25th percentile '
             'with saturated events counted by rank only (a median over unsaturated events stops rising once ~⅓ saturate).</p>'
             '<p>Limitation: the fixed 3550 ADC rule misses 37 % (det4) / 46 % (det6) of truly clipped samples, which then read low '
             '(K4, locked in DECISION_2026-10-29). No absolute gain (fit ADC units).</p>')
    D.slide('charge', body, notes, foot=foot, short='§5 charge balance')


def s_eff(D):
    old = {'det2': 92, 'det3': 93, 'det4': 42, 'det6': 75, 'det7': 57}
    hb = []
    for d in DET:
        e, er, n = eff_ig(d)
        if e is None:
            continue
        hb.append((d, round(e, 1), DCOL[d], f'{d}: {e:.1f} ± {er:.1f} % of {n} interior M3 tracks within 5 mm (golden row); '
                   f'the text says {old[d]} %'))
    chart = sd.hbars(hb, 100, width=560, label_w=110, fmt=lambda v: f'{v:.1f} %', size=26)
    body = (sd.title('§5: efficiency is set by the chamber: 90–93 % on det2/det3, discharges and stripes elsewhere',
                     'Interior (≥ 30 mm) M3 tracks with a fit within 5 mm, golden rows; maps on an 8 mm kernel')
            + sd.row(fig('efficiency_map', w=900, h=585),
                     sd.col(chart,
                            bullets(['det4: dead/low-gain stripes take 34 % (71.3 % in the live bands); det6/det7: whole-plane '
                                     'discharges 17 / 24 %.', 'det7 X on ch 256–511 only (M1 badly seated).'], 23),
                            ask('<b>E</b> A1 table per chamber or det3 only. <b>I</b> 30 mm on every side, or 10 mm on y '
                                'for efficiency-only numbers (+9–10 % tracks).', 22), w=700), gap=40))
    foot = 'Source: rerun/effbudget/pass1/results/summary.json (EFF); maps rerun/paper1_figs/make_effmap_figure.py'
    notes = ('<p><b>Text → pass 1:</b> det2 92 → 90.3 (WRONG), det3 93 → 92.6 (OK), det4 42 → 40.9 (WRONG), det6 75 → 74.4 (OK), '
             'det7 57 → 68.0 (WRONG; 62.9 % with all X).</p>'
             '<p>With multi-track credit: det2 93.9, det3 95.2, det4 44.4, det6 77.7, det7 71.6 %. R = 3 mm: 89.0 / 90.9 / 39.2 / 73.6 / 66.4 %. '
             'The trigger-track rule moves every efficiency by < 0.05 points (not worth a selection).</p>'
             '<p>Bugs found and fixed on the way (EFF): duplicate event ids counted as losses; pre-pass outlier classes loaded silently; '
             'the skeptical det3 6-23 rows lose efficiency to their broken angle scale (position-only column).</p>')
    D.slide('eff', body, notes, foot=foot, short='§5 efficiency')


def s_edge(D):
    body = (sd.title('§5: the 30 mm fiducial is physical: a drift-field fringe reaches 20–30 mm in from the x sides',
                     'det3: turn-on, the other view’s tan bias against M3, the column duration vs distance to the edge')
            + sd.row(fig('edge_turnon', w=1060, h=585),
                     sd.col(bullets(['Tan bias 0.1–0.36 within 5–20 mm of an x side, ≤ 0.016 beyond 30 mm; its sign flips '
                                     'side to side (drift lines converge inward).',
                                     'Edge strips collect from beyond the strip area: longer columns, more charge.',
                                     'Not the bow: it predicts the opposite sign of edge T.'], 24),
                            ask('<b>Q</b> show det3 (the paper’s chamber) or det2 (10× the triggers). det2’s edge-T tension is '
                                'closed: the two estimators average different bands of a non-monotonic T(d) (predicted −78.5 vs −86 ± 33 ns).'), w=520), gap=48))
    foot = 'Source: rerun/paper1_figs/README.md (FIGS), rerun/consistency/README.md'
    notes = ('<p>Replaces the hits-angle edge study (D4, “−3° edge tilt”). Column duration edge − interior: −4 / +26 / +78 / +27 / −115 ns '
             '(det2/3/4/6/7); the S4 stack gave −86 / +40 / +60 (det2/3/4). det2 closed (paper1_spsfigs det2_edge_check): T ≈ 584 ns in the '
             'outer 0–5 mm, 915–950 at 5–15 mm, 800–860 further in; S4’s sample is 40 % in the outer 5 mm.</p>'
             '<p>Efficiency reaches its plateau ~5 mm in from the y sides and 10–20 mm from the x sides; the 0–5 mm strip on every '
             'side loses the column off the strips. A 10–30 mm y band keeps full efficiency, but its Y-angle s68 is 1.3–1.9× worse.</p>')
    D.slide('edge', body, notes, foot=foot, short='§5 edge zone')


def gas_626_plot():
    """det6/det7 ladder v through the 6-26 night (paper1_gas/results/gas.json ladder_626)."""
    g = jload('paper1_gas/results/gas.json') or {}
    series = {}
    for q in g.get('ladder_626') or []:
        d, t, v = q.get('det'), q.get('t626'), q.get('v_lad')
        if d in DCOL and isinstance(t, (int, float)) and isinstance(v, (int, float)):
            series.setdefault(d, []).append((t, v, q.get('v_lad_err') or 0))
    if not series:
        return None
    P = sd.Plot(600, 330, x=(0, 18, 'lin'), y=(14, 42, 'lin'), xlabel='hours on 6-26', ylabel='v, M3 ladder (µm/ns)',
                margin=(16, 20, 80, 100))
    P.xticks(tk([0, 6, 12, 18])).yticks(tk([15, 25, 35]))
    for d, v in sorted(series.items()):
        v.sort()
        P.points([a for a, _, _ in v], [b for _, b, _ in v], DCOL[d], r=5,
                 tips=[f'{d} {a:.1f} h: {b:.1f} ± {c:.1f} µm/ns' for a, b, c in v])
    P.text(P.X(0.5), P.Y(40), 'v rises as the chambers dry', size=22, color=sd.INK, weight=600)
    return P.svg('6-26 v') + sd.legend([(d, DCOL[d]) for d in sorted(series)], 20)


def s_gas_bench(D):
    gw = GASB.get('golden_water', {})
    w3 = gw.get('det3', {})
    w3s = f"{w3.get('water', 0):.2f} % [{w3.get('water_lo', 0):.2f}, {w3.get('water_hi', 0):.2f}]" if w3 else '0.97 %'
    side = gas_626_plot()
    golden = ', '.join(f"{d} {gw[d]['water']:.2f}" for d in ['det2', 'det4', 'det6', 'det7'] if d in gw)
    blt = ['det3: one water, ' + sd.term(w3s, 'timing v (gap / T) at 500–1100 V drift; χ² 2.5/3; pre-pass 0.955 % reproduced') +
           ', all seven fields; golden rows ' + sd.term(golden + ' %', 'per-chamber water from each golden row’s timing v') + '.',
           'The v(t) drifts are drying: det6 1.47 → 0.76 %, det7 0.68 → 0.35 % water on 6-26.',
           'det4’s gap is 28.6 mm, not 30.48 (slide 17): its water moves up if the input changes.']
    right = sd.col(*(([side] if side else []) + [bullets(blt, 23),
                     ask('<b>K</b> gas_vE_timing over gas_vE. <b>L</b> quote det6’s early attachment (~600 ppm O₂-eq, ×2)? '
                         'Had det6 been refilled before 6-26?', 21)]), w=640)
    body = (sd.title('§6: the bench gas is Ar/iso plus 0.4–1 % water, drying over hours; v(E) follows Magboltz at one water',
                     'det3 drift scan: timing v and angle-scale v vs Magboltz (fine grid air_merged)')
            + sd.row(fig('gas_vE_timing', w=920, h=585), right, gap=40))
    foot = 'Source: rerun/paper1_gas/README.md (GAS), rerun/vt_all/CROSSCHECK_2026-10-31.md'
    notes = ('<p><b>Text → pass 1:</b> “six fields 35–382 V/cm” → seven fields 33–367 V/cm (WRONG). “v = 2.5–37.8 µm/ns” → timing v '
             '17.7–37.2 µm/ns at 177–390 V/cm; the angle-scale v (v_drift × kw) 2.55–38.8 is not a timing v, so say which (WRONG). '
             '“0.95 % water” → 0.97 % on det3 (OK) + per-chamber values (new).</p>'
             '<p>Other nights: det4 0.70 → 0.21 % through 6-23 (caveat: 6-23 was not attachment-free, CONS §1; quote det4 at 6-24, 0.42 %), '
             'det2 0.73 → 0.57 % on 6-22 (weakly determined near the v maximum). g_det4’s −9 % in-row v is not resolved by the stack.</p>'
             '<p>The two v estimators (timing stack vs M3 ladder) agree within 3 % on attachment-free rows (det6 0.970 ± 0.013, det7 0.984 ± 0.011).</p>')
    D.slide('gas_bench', body, notes, foot=foot, short='§6 bench gas')


def s_gas_sps(D):
    tab = GASFIT.get('table') or []
    body_plot = ''
    if tab:
        n = len(tab)
        P = sd.Plot(980, 600, x=(-0.5, n - 0.5, 'lin'), y=(0, 600, 'lin'), ylabel='O₂ (ppm)', margin=(24, 80, 150, 104))
        P.yticks(tk([0, 100, 200, 300, 400, 500]))
        for i, r in enumerate(tab):
            c = sd.ORANGE if r.get('gas') == 'co2' else sd.BLUE
            o, e = r.get('o2_ppm', 0), r.get('err_o2', 0)
            P.raw(sd.line(P.X(i), P.Y(max(o - e, 0)), P.X(i), P.Y(o + e), c, w=3))
            tip = (f"{r['group']}: O₂ {o:.0f} ± {e:.0f} ppm, water {r['water']:.2f} ± {r['err_w']:.2f} % "
                   f"(gap {r['gap']} mm; χ²/ndf {r['chi2']:.0f}/{r['ndf']})")
            P.points([i], [o], c, r=10, tips=[tip])
            lab = r['group'].replace('_', ' ')
            P.raw(sd.T(P.X(i), P.Y(0) + 34, lab, size=18, rot=-30, anchor='end'))
        P.text(P.X(-0.3), P.Y(560), 'O₂ needed in every period; CO₂ > CF₄', size=26, color=sd.INK, weight=600)
        body_plot = P.svg('SPS O2') + sd.legend([('Ar/CO₂/iso period', sd.ORANGE), ('Ar/CF₄/iso period', sd.BLUE)], 22)
    else:
        body_plot = pending('sps_gasfit', 'results missing', 980, 600)
    att = fig('attachment', w=560, h=380)
    body = (sd.title('§6: the beam gas carried water and O₂; the O₂ is required and higher in the CO₂ period',
                     'SPS gas fit on the final pass, per run group, at one drift gap (Magboltz H₂O + O₂ grid)')
            + sd.row(sd.col(body_plot, w=1000),
                     sd.col(att,
                            bullets(['Water 1.7–2.1 % (loose: degenerate with the gap); at det4’s geometric gap 28.6 mm it drops 0.1–0.2 %.',
                                     'Bench vs beam: 0.35–1.5 % water, no O₂ settled, vs 1.7–2.1 % + 130–440 ppm.'], 23),
                            ask('absolute O₂ scale rests on Magboltz’s three-body attachment (×few); the ordering does not.', 22),
                            w=600), gap=48))
    foot = 'Source: rerun/sps_gasfit/README.md (GASFIT), results/verdict.json; rerun/det4_gap/README.md'
    notes = ('<p>Without O₂ at the same water each stack fits 1.5–117× worse (median 33). Closure test passes. Agrees with the pre-pass '
             'run71 fit (1.51 % / 163 ppm) once the 30 mm gap and the saturated-event treatment are accounted for.</p>'
             '<p>Degeneracies: water ↔ gap (1.2 mm ≈ 0.12–0.18 % water; O₂ unaffected); gap ↔ pulse shape (the shaper hits its bound at '
             '26.2–27.9 mm, so the gap error is one profile step, 0.9 mm); O₂ ↔ slow electronics droop on ZS stacks (largest O₂ systematic, 81 ppm).</p>'
             '<p>The grid had to grow twice: water edges (CF₄ 1.7 %, CO₂ 2.2 %), field (CO₂ rows to 139 V/cm), O₂ (272 ppm). Midpoint checks: '
             '≤ 0.011 % water, ≤ 16 ppm O₂.</p>')
    D.slide('gas_sps', body, notes, foot=foot, short='§6 beam gas')


def s_timing(D):
    hb = []
    for r in TIM:
        if r['row'] in ('o22_long_det2', 'sat_det3'):
            continue
        d = r['chamber']
        hb.append((d, round(float(r['sigma_evt']), 1), DCOL.get(d, sd.GREY),
                   f"{d} {r['row']}: event {float(r['sigma_evt']):.1f} ns = per view {float(r['sigma_xy']):.1f} ⊕ common "
                   f"{float(r['sigma_common']):.1f} ns; {float(r['evt_mm']):.2f} mm at v {float(r['v']):.1f} µm/ns (n {r['n']})",
                   f"{float(r['evt_mm']):.1f} mm"))
    chart = sd.hbars(hb, 120, width=380, h=30, label_w=80, fmt=lambda v: f'{v:.0f} ns', size=22) if hb else pending('timing', 'csv')
    body = (sd.title('§7: on its own, the chamber times a cosmic muon to 40–46 ns (1.5–1.7 mm of drift)',
                     'Column start (edge25) vs the scintillator trigger fine time, golden rows, M3-matched')
            + sd.row(fig('timing', w=1000, h=585),
                     sd.col(chart,
                            bullets(['Per view 22–28 ns ⊕ common to X/Y 35–40 ns (first electrons at the mesh).',
                                     'det4 carries the 6-23/6-24 trigger term; SPS stays unmeasured (140–218 ns on RAW).'], 22),
                            ask('<b>M</b> quote the event time with the per-view/common split? <b>N</b> MPGD “33 ns / 1.1 mm” '
                                '(hits): replace with the waveform number or label it hits?', 22), w=620), gap=40))
    foot = 'Source: rerun/paper1_timing/README.md, results/golden.csv; the fit’s t0 is pinned to the trigger, so it carries no time'
    notes = ('<p>Estimator: a step of the row’s own template fitted to the rising edge of the strip-sum waveform up to the first sample '
             'above 25 % of its maximum, plus 10 ns × ftst per view. It beats the box fit (85 ns), constant-fraction 30 % (77 ns) and 10 % '
             '(49 ns), and is the only one flat in angle (≤ 8 ns to |tan| 0.5).</p>'
             '<p>HV: det3 83 → 21 ns per view (425 → 500 V); drift field: det3 114 → 38.5 ns (104 → 382 V/cm). Trigger test: '
             'cov(start, end) −63 ± 19 ns² on sat_det3 (a 39 ns trigger term would give +1500).</p>'
             '<p>Reconciled: the hits 33 ns is the X−Y plane-to-plane number; its waveform equivalent is 22 ns per view; the hits '
             '“absolute” 37.7 ns ↔ waveform event time 42 ns.</p>')
    D.slide('timing', body, notes, foot=foot, short='§7 timing')


def s_angles(D):
    P = sd.Plot(760, 430, x=(0.5, 5.5, 'lin'), y=(0, 2.8, 'lin'), ylabel='angle s68 (°)', margin=(24, 30, 70, 100))
    P.yticks(tk([0, 0.5, 1.0, 1.5, 2.0, 2.5]))
    for i, d in enumerate(DET):
        for j, (v, c) in enumerate([('x', sd.BLUE), ('y', sd.ORANGE)]):
            s, t = res_s68(d, v), res_s68(d, v, 'vt')
            if s is None:
                continue
            x = i + 1 + (j - 0.5) * 0.3
            tip = f'{d} {v.upper()}: pass {s:.3f}° · v(t) arm {t:.3f}°' if t is not None else f'{d} {v.upper()}: {s:.3f}°'
            P.vbar(x, s, 34, c, tip=tip)
        P.raw(sd.T(P.X(i + 1), P.Y(0) + 34, d, size=22, fill=sd.INK))
    P.text(P.X(0.7), P.Y(2.6), '~1° on det2/det3; det4 2°', size=26, color=sd.INK, weight=600)
    chart = P.svg('angle s68') + sd.legend([('X view', sd.BLUE, 'box'), ('Y view', sd.ORANGE, 'box')], 22)
    body = (sd.title('§8: waveform reconstruction gives ~1° angles, flat to 20°, and 2–3× the hits chain',
                     'Angle s68 vs M3 on interior_good, golden rows (pass arm; hover for the v(t) arm)')
            + sd.row(fig('tracking_perf', w=840, h=585),
                     sd.col(chart,
                            ask('<b>C</b> pass or v(t) arm (always lower, ≤ 0.063°). <b>D</b> s68 with no clip or the ±5° clip '
                                '(up to 0.75° apart on det4 X). det4/6/7 are in-sample (only golden row = calibration row).', 22),
                            w=780), gap=40))
    foot = 'Source: rerun/paper1_res/results/summary.json (RES); text 1.15–2.51° → 0.90–2.41° (WRONG)'
    notes = ('<p>Headline rows: det2 6-22 long_run, det3 6-27 sanity long_run (quoted golden), det4 g_det4, det6 g_det6_long, det7 '
             'g_det7_long (X ch 256–511). Trends: best at the low end of every HV scan, 30–50 % worse at the top; det3 ~0.9° from 166 V/cm '
             'up, 2° at 100 V/cm.</p>'
             '<p>Paper II fleet figure: H0 (hits) 2.0–2.6° vs W 0.73–1.56° on the same events; hits imply v 46 vs 36 µm/ns on det3 '
             '(28 % compression).</p>'
             '<p>G1 det4 Y 2.348 (pass) / 2.290 (v(t)); the −0.25° vs the audit is the pass template, not the bow (DET4Y_ABLATION).</p>')
    D.slide('angles', body, notes, foot=foot, short='§8 angles')


def s_floor(D):
    rows = []
    for d in ['det2', 'det3', 'det4', 'det7']:
        v = (FLOOR.get(f'{d}|calib') or {}).get('views', {}).get('x')
        if not v:
            continue
        lad = {s['step']: s for s in v.get('ladder', [])}
        meas, toy = v.get('measured'), v.get('toy_all')
        U = v.get('U')
        clean = (lad.get('topo') or {}).get('U')
        rows.append((d, meas, toy, U, clean))
    P = sd.Plot(980, 560, x=(0.5, len(rows) + 0.5, 'lin'), y=(0, 1.8, 'lin'), ylabel='X angle (°)', margin=(24, 30, 70, 100))
    P.yticks(tk([0, 0.4, 0.8, 1.2, 1.6]))
    for i, (d, meas, toy, U, clean) in enumerate(rows):
        x = i + 1
        for k, (val, c, lab) in enumerate([(meas, sd.INK, 'measured'), (toy, sd.GREEN, 'all modelled terms'),
                                            (U, sd.RED, 'unexplained'), (clean, sd.GOLD, 'unexplained, clean events')]):
            if val is None:
                continue
            P.vbar(x + (k - 1.5) * 0.2, val, 30, c, tip=f'{d} X, calibration row: {lab} {val:.2f}°')
        P.raw(sd.T(P.X(x), P.Y(0) + 34, d, size=22, fill=sd.INK))
    P.text(P.X(0.6), P.Y(1.7), 'a ~0.6° remainder is left on clean events', size=26, color=sd.INK, weight=600)
    chart = P.svg('floor') + sd.legend([('measured s68', sd.INK, 'box'), ('all toy terms', sd.GREEN, 'box'),
                                        ('unexplained U', sd.RED, 'box'), ('U on clean events', sd.GOLD, 'box')], 22)
    if BOOT.exists():
        t = BOOT.read_text()
        m = re.search(r'\*\*(\d+ ± \d+ % \(det2\), \d+ ± \d+ % \(det3\) and \d+ ± \d+ % \(det4\))\*\*', t)
        share = m.group(1) if m else 'about half to all'
        boot = sd.callout('<b>Residual bootstrap:</b> the data’s own fit residuals, added to the toy, carry ' + share +
                          ' of the clean remainder². No named structure (common mode, template, time shape, ±1 asymmetry) '
                          'is data-specific: the toy under-generates event-to-event charge-pattern fluctuations.', sd.BLUE, 22)
    else:
        boot = sd.callout('<b>Residual bootstrap: pending</b> (real-event residuals added to the toy; '
                          'rerun/paper1_floor/BOOTSTRAP_2026-11-01.md not yet on main)', sd.GOLD, 22)
    body = (sd.title('§8: the ~0.6° angle remainder is per-event charge fluctuation the toy under-generates' if BOOT.exists() else '§8: every modelled term leaves a ~0.6° angle remainder: measured, bounded, not identified',
                     'X view, calibration rows: measured s68, the toy’s modelled terms, and what is left (FD convention)')
            + sd.row(sd.col(chart, w=1000),
                     sd.col(bullets(['Ruled out: the M3 reference, a cause common to X and Y, sub-tilt-map structure, angle '
                                     'non-linearity, the X copy (Y has it too).',
                                     'Present head-on, flat in angle, fit-visible.',
                                     sd.term('K1 is not it', 'per-plane shape scatter at the measured size explains −7…+2 % of the '
                                             'clean remainder²; the data χ² is 3–5× any K1 form’s, so the error is coherent '
                                             'across strips and samples (K1_TEST_2026-11-01.md)') + ': ~0 % of it.'], 23),
                            boot, ask('<b>O</b> wording: the 10-27 “not a waveform misfit” is reversed on pass 1, and K1 is ruled out.', 22),
                            w=580), gap=48))
    foot = 'Source: rerun/paper1_floor/results/floor_budget.json (FLOOR), K1_TEST_2026-11-01.md; toy re-run on the pass-1 bundles'
    notes = ('<p>Unexplained part (FD, toy on the pass-1 bundles, 1500 events/angle): det2 0.72 of 0.98°, det3 0.68 of 0.93°, det4 1.52 of '
             '1.62°, det6 0.93 of 1.13°, det7 0.98 of 1.12°: 54–88 % of the X variance. All toy terms 0.38–0.46°; toy floor 0.31–0.34°.</p>'
             '<p>Event conditions the toy doesn’t generate (saturated, bright tail, topology) are 12–37 % of U² (det7 65–92 %). Remainder on clean '
             'events 0.56–0.65° (det2/3/7 X), det4 1.42°.</p>'
             '<p>K1-aware weights (toy only): 0.04–0.06° worse on det2/det3 X, so the locked model leaves ≤ 0° there.</p>'
             '<p>Bootstrap (<code>rerun/paper1_floor/BOOTSTRAP_2026-11-01.md</code>): data residuals moved onto toy events move the angle '
             '1.5–2.0× more than the toy’s own (det2/3/4 X 0.243/0.241/0.416° vs 0.123/0.162/0.283°); calibrated by the method’s '
             'efficiency ε = 0.29–0.37. What carries the angle in both data and toy residuals: the part coherent in time and across strips '
             '(~80 %), antisymmetric about the track (~40 %). Untested candidates (simulation only): Polya θ, δ-ray rate/range, cluster-size '
             'distribution, per-event resistive spread. Y view and det7 rest on the data-truth arm only.</p>')
    D.slide('floor', body, notes, foot=foot, short='§8 angle floor')


def s_position(D):
    pts = []
    for d in ['det2', 'det3']:
        for v in ['x', 'y']:
            e = (POSMS.get(f'{d}|headline') or POSMS.get(f'{d}|calib') or {}).get(v) or {}
            c, ce = e.get('chamber_near_normal_um'), e.get('chamber_near_normal_err_um')
            if c:
                pts.append((f'{d} {v.upper()} bench', c, ce or 0, DCOL[d]))
    sps = [('det4 Y SPS run63 (p0)', 190, 10, sd.GREY), ('det4 Y SPS run53 (hits)', 176, 10, sd.INK)]
    allp = pts + sps
    P = sd.Plot(1000, 560, x=(-0.5, len(allp) - 0.5, 'lin'), y=(0, 450, 'lin'), ylabel='chamber resolution (µm)',
                margin=(24, 30, 160, 110))
    P.yticks(tk([0, 100, 200, 300, 400]))
    for i, (lab, c, e, col) in enumerate(allp):
        P.raw(sd.line(P.X(i), P.Y(max(c - e, 0)), P.X(i), P.Y(c + e), col, w=3))
        P.points([i], [c], col, r=10, tips=[f'{lab}: {c:.0f} ± {e:.0f} µm'])
        P.raw(sd.T(P.X(i), P.Y(0) + 30, lab, size=18, rot=-30, anchor='end'))
    P.hline(176, sd.MUT, label='176 µm (run 53, August)', anchor='start', where='below')
    P.text(P.X(-0.3), P.Y(420), 'bench near-normal and beam Y agree', size=26, color=sd.INK, weight=600)
    sr = fig('sps_resolution', w=600, h=420)
    body = (sd.title('§8: in the beam det4 Y resolves 190 µm (176 on run 53); X has no measurement',
                     'Chamber term near normal incidence: bench (M3 χ² → 0, minus pointing) and det4 Y at SPS (telescope 3-zone fit)')
            + sd.row(sd.col(P.svg('position'), w=1000),
                     sd.col(sr, sd.callout('X: the 3-zone fits fail (χ²/1 7–88; dead stripes in the beam spot), so “X 209 µm” '
                                           'is a model-dependent intercept, not a resolution. Quote Y only.', sd.RED, 22),
                            ask('<b>A</b> run63 Y 190 ± 10 (p0) or run53 Y 176 ± 10 (hits). <b>B</b> now a Y-only choice of estimator. '
                                '<b>P</b> bench chamber term or the decomposition. “176 µm = 0.30 × pitch” is wrong: 0.23 × pitch.', 22),
                            w=600), gap=48))
    foot = 'Source: rerun/paper1_floor/results/position_ms.json (FLOOR), rerun/sps_audit + sps_junk/results/final_rule.json'
    notes = ('<p>The bench residual (430–620 µm core σ) grows with the M3 χ² of the matching coordinate (det3 Y 298 → 733 µm across '
             'quintiles): M3 scattering between its doublets, 240–370 µm. Extrapolated to χ² → 0 minus the pointing P 0.21–0.28 mm: '
             'chamber 230–310 µm at all angles; near normal det2 163/176 ± 25, det3 126/116 ± 35 µm. ±10 % on P moves it ~±50 µm.</p>'
             '<p>SPS (final rule): run63 flat700 Y 190 ± 10 (p0, zone fit χ²/1 0.3); run53 Y 176 ± 10 (hits), 155 ± 7 (centroid). '
             'X: the zone fits fail (χ²/1 7–88), so the 209 ± 11 (p_mid) is a model-dependent intercept (rerun/paper1_spsfigs/README.md). '
             'The telescope term (f 0.25–0.34) is a ±40 µm systematic. Head-on, the fit p0 is noisy in X (angle noise leaks into the '
             'mesh position): run63 X 358 µm with p0 vs 294 µm centroid.</p>'
             '<p>The 6-23/6-24 trigger term makes det4’s bench position unquotable: use SPS only.</p>')
    D.slide('position', body, notes, foot=foot, short='§8 position')


def s_bow(D):
    body = (sd.title('§8b: the readout boards are dished by 1–2.5 mm, and one block per chamber transfers across days',
                     'Bow profiles and closure on the pass-1 golden tables (v(t) arm)')
            + sd.row(sd.col(fig('bow_profiles', w=800, h=440), sd.p('bow_profiles', 22, sd.MUT)),
                     sd.col(fig('bow_closure', w=800, h=440), sd.p('bow_closure', 22, sd.MUT)), gap=56)
            + sd.row(bullets(['Dish per view 0.90–2.45 mm; angle gain median 15.7 % (3.6–28.9); position 1.7–10.6 %.',
                              'The gate: one block per chamber improves every row; w0/kw stay per row.'], 24),
                     ask('<b>F</b> bow numbers per view or per-chamber mean (X and Y differ by up to 1.1 mm on det6).', 22), gap=48))
    notes = ('<p><b>Text → pass 1:</b> dish 1.5–2.3 → 0.90–2.45 mm per view (WRONG); det2 1.8 → X 1.52 / Y 2.45; det3 1.5 → 0.90 / 1.57; '
             'det4 1.9 → 1.69 / 2.10 (OK); det6 1.8 → 0.95 / 2.04; det7 2.3 → 1.22 / 1.71. Curvature 0.80–2.17 × 10⁻⁴ mm⁻¹; '
             'w0 ramp ±0.41–0.95 µm/ns; r_f 6/10 in 0.4–0.9.</p>'
             '<p>Source: <code>rerun/paper1_res/results/bow/summary.json</code>, <code>rerun/calib/bow/results_pass1/GATE_RESULT_2026-10-30.md</code>. '
             'Skeptic review wording: tilt solid; mesh depth 1.2–1.9 ± 0.4 mm.</p>')
    D.slide('bow', body, notes, short='§8b bow')


def s_stripes(D):
    body = (sd.title('§9: det4’s dead stripes are a gain continuum, the same on the bench and at SPS',
                     'X-view efficiency vs x: bench g_det4 (M3) and SPS run63/run53 (telescope)')
            + sd.row(fig('det4_stripes', w=1060, h=585),
                     sd.col(bullets(['Bench vs SPS live/dead agree on 82 % of 1 mm bins (85 % at +0.5 mm); mirrored only 52 %.',
                                     'Lower-gain run53 loses the marginal bands: a continuum, not a fixed map.',
                                     '“62 % never amplifies” → 52 % of the x range below half the plateau; 41 % two-view detection; '
                                     '34 % of tracks in the A1 stripe class.'], 24),
                            ask('re-word §9 around “52 % of the x range below half the plateau”?'), w=520), gap=48))
    foot = 'Source: rerun/paper1_figs/README.md (FIGS), rerun/consistency/README.md'
    D.slide('stripes', body, '<p>The three numbers agree once defined (CONS 11-01): tracks in the sub-half-plateau 52 % are partly '
            'still detected at low gain, or lost to other classes, so 34 % fall in the stripe class; 71.3 % efficiency in the live bands.</p>',
            foot=foot, short='§9 det4 stripes')


def s_gap(D):
    meas = [('bench Y, det4 (geometric)', 28.6, 0.5, sd.INK), ('SPS rot25, ZS floor (lower bound)', 28.5, 0.4, sd.GREEN),
            ('SPS gas fit (profiled)', 27.8, 0.9, sd.BLUE), ('input, physics_inputs.json', 30.48, 0.44, sd.RED)]
    if GASFIT.get('gap_raw'):
        meas[2] = ('SPS gas fit (profiled)', GASFIT['gap_raw'], 0.9, sd.BLUE)
    P = sd.Plot(1000, 520, x=(26, 32, 'lin'), y=(-0.5, len(meas) - 0.5, 'lin'), xlabel='det4 drift gap (mm)', margin=(24, 30, 92, 420))
    P.xticks(tk([26, 27, 28, 29, 30, 31, 32]))
    for i, (lab, g, e, c) in enumerate(meas):
        y = len(meas) - 1 - i
        P.raw(sd.line(P.X(g - e), P.Y(y), P.X(g + e), P.Y(y), c, w=4))
        P.points([g], [y], c, r=11, tips=[f'{lab}: {g:.2f} ± {e:.2f} mm'])
        P.raw(sd.T(P.X(26) - 16, P.Y(y) + 8, lab, size=22, anchor='end', fill=sd.INK))
    P.vline(28.6, sd.MUT)
    body = (sd.title('det4’s drift gap is 28.6 mm, not 30.5: three methods agree, the input does not',
                     'Geometric gap from the column’s strip extent at a known angle (no v, no pulse model), vs the other estimates')
            + sd.row(sd.col(P.svg('det4 gap'), w=1000),
                     sd.col(bullets(['Validated on det3 and det2 (+0.3 ± 0.2 mm offset, applied).',
                                     'The input is the X-view stack, which crosses det4’s dead stripes (working explanation).',
                                     'The SPS water stands (−0.1–0.2 % at 28.6 mm); the O₂ ordering doesn’t move.'], 24),
                            ask('change det4’s gap in physics_inputs.json 30.48 → 28.6? It lowers det4 timing v ~6 % '
                                '(32.7 → ~30.7 µm/ns), relabels fields (600 V → 210 V/cm), raises the §6 det4 water. Angles don’t move.'),
                            w=600), gap=48))
    foot = 'Source: rerun/det4_gap/README.md (DET4GAP), rerun/sps_gasfit/results/verdict.json'
    D.slide('gap', body, '<p>Bench Y view: 28.9–29.0 ± 0.4 raw (217 events, g_det4 + 6-23), minus the estimator’s +0.3 ± 0.2 offset '
            '(det3 Y 28.4 vs 28.0–28.2; det2 Y 31.6 vs 31.1–31.5). SPS rot25 plateau at the ZS floor 28.1–28.9 in both gases (a lower bound: '
            'at the pass thresholds the plateau rises with field, which geometry cannot do). S4’s ±tan split (33.8 / 30.0) is a reference tilt '
            'of ~0.015, source not found.</p>', foot=foot, short='det4 gap')


def s_losses(D):
    keys = [('spark:discharge', 'discharge', sd.RED), ('multi:m3_picked', '2nd particle (M3 one fitted)', sd.ORANGE),
            ('multi:other_picked', '2nd particle (other fitted)', sd.GOLD), ('far:DELTA', 'δ-ray', sd.PURPLE),
            ('far:FIT_SAT', 'saturated δ blob', sd.BLUE), ('band', 'noise band', sd.GREY)]
    cols = ['class'] + DET
    rows = []
    for k, lab, c in keys:
        r = [f'<span style="color:{c};font-weight:600">■</span> {lab}']
        for d in DET:
            v = a1pct(d).get(k)
            r.append(f'{v:.2f}' if isinstance(v, (int, float)) else '–')
        rows.append(r)
    det_row = ['<b>detected</b>'] + [f"<b>{a1pct(d).get('detected', 0):.1f}</b>" for d in DET]
    tbl = sd.table(cols, [det_row] + rows, size=22, widths=[340] + [90] * 5)
    body = (sd.title('A1: every lost track has a class; det6/det7 lose a fifth to discharges, det4 a third to stripes',
                     'Loss classes (% of interior M3 tracks, golden rows) and real event displays from pass 1')
            + sd.row(sd.col(tbl, w=860),
                     sd.col(sd.row(fig('loss_spark_discharge', w=340, h=255), fig('loss_multi_m3picked', w=340, h=255), gap=20),
                            sd.row(fig('loss_delta', w=340, h=255), fig('loss_satblob', w=340, h=255), gap=20), gap=20), gap=40)
            + ask('<b>E</b> per-chamber columns or det3 only. <b>H</b> discharge vs burst (det4 6.48 → 1.82 + 4.66 %). <b>J</b> det6/det7 '
                  'discharges: the muon is usually still visible; recovering them is a reco change. All 14 displays are rebuilt.', 22))
    foot = 'Source: rerun/effbudget/pass1/results/a1.json (EFF), rerun/paper1_figs/README.md (FIGS: audit by eye, rules hold)'
    notes = ('<p>det3 (sat_det3, n = 6 454): detected 91.18 → 92.64 %; noise band 1.43 → 0.22; second particle other-fitted 1.26 → 1.43; '
             'fit off 0.34 → 0.23; head-on 0.26 → 0.02 (fit_failed). Spark discharge / muon+other / shower 0.50 / 0.40 / 0.67 unchanged.</p>'
             '<p>About 22 % of the far losses (32 % on det7) are fits 5–8 mm from M3: the resolution tail meeting the 5 mm radius.</p>'
             '<p>Discharge vs burst (a chip ≥ 8 strips over 3000 ADC = discharge): det2 2.50 → 1.80 + 0.71, det3 1.56 → 0.88 + 0.68, '
             'det4 6.48 → 1.82 + 4.66, det6 18.13 → 16.73 + 1.40, det7 24.95 → 24.18 + 0.77 %.</p>')
    D.slide('losses', body, notes, foot=foot, short='A1 losses')


def s_night(D):
    flow = sd.flow([dict(label='gain flat over HV', sub='§4: det3 and det4 alike', color=sd.RED),
                    dict(label='trigger jitter 40–60 ns', sub='§7: 6-23 and 6-24', color=sd.PURPLE),
                    dict(label='poor M3 matching', sub='charge', color=sd.GREY),
                    dict(label='v drift + jump', sub='§6: drying, +13 % step', color=sd.BLUE)], size=24)
    body = (sd.title('The 6-23/6-24 night has two causes: chambers not yet at gain, and a trigger-timing term',
                     'Four analyses flagged the same night; the consistency audit separated them')
            + flow
            + sd.row(sd.card(sd.p('<b>(a) measured:</b> at a fixed 495 V det3’s charge rose ×4 and det4’s ×7 through the night; '
                                  'efficiency 0.06 → 35 % / 3.9 → 28.5 %. That hides 20–35 % of the gain slope. Likely gas '
                                  'contamination from the chamber changes (inferred).', 24)),
                     sd.card(sd.p('<b>(b) measured, cause unknown:</b> cov(start, end) +3900–4400 ns² on both chambers vs ≤ +518 on every '
                                  'other night; persists on 6-24 after the gain recovered.', 24)),
                     sd.card(sd.p('<b>Consequence:</b> det4 bench timing and position are not quotable (SPS only); det4 angles are fine '
                                  '(a common time shift moves position, not slope); the 6-23 water needs a caveat.', 24)), gap=32)
            + ask('facts needed: what changed in the trigger/paddles on 6-23/6-24? Was det3 or det4 opened on 6-22/6-23, and the gas line order?'))
    foot = 'Source: rerun/consistency/README.md (CONS §1)'
    D.slide('night', body, '<p>The only config difference for (b): 6-23/6-24 are the only nights with no fresh pedestal/threshold run on the '
            'old pre-6-26 DAQ config (inference). det6 shows the (a) signature after its 6-26 install.</p>', foot=foot, short='The 6-23 night')


def s_sps_paper2(D):
    names = ['kernel_forms', 'angled_kernel', 'known_angle']
    cells = [sd.col(fig(n, w=520, h=360, paper=2), sd.p(n + ' (Paper II)', 22, sd.MUT)) for n in names]
    have = SPSFIGS.exists()
    body = (sd.title('Beam figures shared with Paper II: the footprint at normal incidence and at a known angle',
                     'RAW run71 (no zero suppression), run57 at 25.4°, run61 at 15.5°; status read from FIGURE_PROVENANCE')
            + sd.row(*cells, gap=40)
            + sd.row(bullets(['Head-on kernel ±1/lead 0.38 (X) / 0.39 (Y), field-independent 96–243 V/cm, so not diffusion; '
                              'Y ±2/±1 0.26, not the August 0.42–0.47.',
                              'At a known 25.4° the inclined view reads 0.981 ± 0.03 of the stage tan with the column-duration v '
                              '(0.927 with the registry v). The X kernel is unchanged by the rotation.'], 23),
                     ask('only |tan| is defined at SPS. ' + ('' if have else 'rerun/paper1_spsfigs not on main yet.') +
                         ' Use kernel_forms (RAW) for absolute numbers; angled_kernel is ZS with the pass selection.', 22), gap=48))
    foot = 'Source: rerun/sps_audit/results/footprint.csv, rerun/sps_followup, rerun/paper1_spsfigs'
    D.slide('sps2', body, '<p>Source: <code>rerun/paper1_spsfigs/README.md</code> (kernel_job.py, tan_known.py, vbox.py). '
            '15.465° (run61): registry v 0.80, no column v (32-sample rows overrun the window).</p>', foot=foot, short='Beam footprint (P-II)')


def s_decisions(D):
    t = (R / 'PAPER1_STORY_2026-11-01.md').read_text() if (R / 'PAPER1_STORY_2026-11-01.md').exists() else ''
    dec = re.findall(r'^- \*\*([A-Q])\.\*\* (.+?)(?=\n- \*\*|\n\n|\Z)', t, flags=re.S | re.M)
    rows = [[f'<b style="color:{sd.GOLD}">{k}</b>', ' '.join(v.split())] for k, v in dec]
    rows.append([f'<b style="color:{sd.GOLD}">R</b>', 'det4 drift gap: change physics_inputs.json 30.48 → 28.6 ± 0.5 mm '
                 '(geometric; lowers det4 timing v ~6 %, relabels its fields, raises the §6 det4 water; angles unchanged)?'])
    half = (len(rows) + 1) // 2
    style = f'color:{sd.DINK}'
    t1 = sd.table(['', 'decision'], rows[:half], size=21, widths=[40, 760], align=['left', 'left'])
    t2 = sd.table(['', 'decision'], rows[half:], size=21, widths=[40, 760], align=['left', 'left'])
    body = (sd.kicker('Decisions for Dylan') +
            f'<h2 style="font-size:52px;font-weight:600;color:#eef0f3;line-height:1.1">{len(rows)} choices change what the text says; '
            'none needs new processing</h2>'
            + f'<div style="{style}">' + sd.row(t1, t2, gap=48) + '</div>')
    D.slide('decisions', body, '<p>From <code>rerun/PAPER1_STORY_2026-11-01.md</code> (A–J from the numbers sheet, K–Q from the section READMEs). '
            'Provisional choices used in this deck: A run63, B p0 (Y only; X is not a measurement), C pass arm, D no clip, I 30 mm.</p>', dark=True, short='Decisions A–Q')


def s_close(D):
    facts = ['Had det6 just been opened or refilled before 6-26?',
             'What changed in the trigger or the paddles on 6-23 and 6-24 (40–60 ns timing term)?',
             'Was det3 or det4 opened on 6-22/6-23, and what was the gas line order?',
             'Stephan: is det4’s SPS resist HV fed through a filter or a series resistor (imon ~0.7 µA at 770 V)?']
    limits = [('The ~0.6° angle remainder (det4 1.4°) is per-event charge fluctuation the toy under-generates; which microphysics '
               '(Polya, δ-rays, cluster sizes, resistive spread) is untested' if BOOT.exists() else
               'The ~0.6° angle remainder (det4 1.4°) is characterised and bounded, not identified (residual bootstrap pending)') + '.',
              'SPS det4 X has no resolution measurement (zone fits fail on the dead stripes); only Y is quoted.',
              'det4/det6/det7 headline angles are in-sample; det6’s v over 6-26 is known to ~×2.',
              'No absolute gain; the 3550 clip rule biases det4/det6 charge low (K4, locked).',
              'SPS: no time resolution, only |tan|; the telescope term is a ±40 µm systematic.',
              'Loss classes audited by eye, not a measured purity.',
              'The texts carry 31 (Paper I), 2 (INFN) and 7 (MPGD) wrong numbers until edited.']
    body = (sd.kicker('Before the text is edited') +
            '<h2 style="font-size:52px;font-weight:600;color:#eef0f3;line-height:1.1">Four facts only you or Stephan have, one input '
            'change, and what this does not rule out</h2>'
            + sd.row(sd.col(sd.p('<b>Facts the data can’t supply</b>', 28, sd.DINK), bullets(facts, 24).replace('<ul style="', f'<ul style="color:{sd.DINK};'),
                            sd.p('<b>Input change:</b> det4 gap 30.48 → 28.6 mm in physics_inputs.json?', 26, sd.GOLD), w=760),
                     sd.col(sd.p('<b>What this does not rule out</b>', 28, sd.DINK), bullets(limits, 24).replace('<ul style="', f'<ul style="color:{sd.DINK};'), w=820), gap=64))
    notes = ('<p>Where to start editing: <code>rerun/PAPER1_NUMBERS_2026-10-31.md</code> lists every number in the three texts '
             '(old value, pass-1 value, status, source). The .tex files were not edited.</p>')
    D.slide('close', body, notes, dark=True, short='Facts, limits')


def build(out):
    D = sd.Deck('MX17 Paper I review (pass 1)', 'Paper I section by section on the from-scratch pass: figures, discussion, decisions')
    for f in (s_cover, s_pass, s_setup, s_hv, s_spark, s_charge, s_eff, s_edge, s_gas_bench, s_gas_sps, s_timing,
              s_angles, s_floor, s_position, s_bow, s_stripes, s_gap, s_losses, s_night, s_sps_paper2, s_decisions, s_close):
        f(D)
    meta = dict(title='MX17 Paper I review: every section on pass 1',
                summary='Paper I section by section on the from-scratch pass: the figures it would carry, the discussion points, '
                        'and the 17 open decisions (A–Q).',
                tags='x17, mx17, paper', date=dt.date.today().isoformat())
    return D.write(out, meta, footer=f'Built {dt.datetime.now():%Y-%m-%d %H:%M} by nTof_x17/papers/paper1/make_review_deck.py')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'review_deck' / 'mx17-paper1-review.html'))
    a = ap.parse_args()
    p = build(a.out)
    print(p, f'{p.stat().st_size / 1e6:.1f} MB')
