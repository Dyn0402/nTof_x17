#!/usr/bin/env python3
"""make_figures_deck.py [--out PATH] -- Paper I figure review (slide note), 2026-11-02.

Every figure the paper includes, in the order of main.tex: the figure as built (figures/<id>.pdf), its state in
figures.toml, its caption, its producer, and a review (verdict + what to fix). Figure order, captions, states,
sources and the list of manifest entries the text no longer uses are read from the paper sources, so a rebuild
after figures change shows the current set; the REVIEW table below is the only hand-written part (one look at each
figure on 2026-11-02).

Supersedes the section-by-section review deck (make_review_deck.py, notes/mx17-paper1-review) for figure work.
Publish: python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <out> --slug mx17-paper1-figures --force --deploy
"""
import argparse
import datetime as dt
import io
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
CACHE = Path(os.environ.get('CLAUDE_JOB_DIR', '/tmp')) / 'tmp' / 'paper1_figdeck_png'

VCOL = {'keep': sd.GREEN, 'polish': sd.GOLD, 'revise': sd.ORANGE, 'redo': sd.RED, 'make': sd.PURPLE,
        'placeholder': sd.GREY, 'move': sd.BLUE}
VTXT = {'keep': 'keep', 'polish': 'polish', 'revise': 'revise', 'redo': 'redo', 'make': 'to make',
        'placeholder': 'placeholder', 'move': 'move / demote'}

# --------------------------------------------------------------------------- #
# the review (2026-11-02): verdict, claim title, issues (on the slide), fix (Details)
# --------------------------------------------------------------------------- #
REVIEW = {
    'microtpc_principle': ('polish', 'The principle figure now shows what the fit uses: a drift sketch and a real waveform with its model', ['Redone 11-02; check the sketch labels at print size', 'Time runs up in (b) (the only display that does)'], 'papers/paper1/figs/make_principle.py.'),
    'chamber_stack': ('placeholder', 'Cross-section of the module: still a placeholder, waiting on Stephan',
        ['Needs Stephan’s layer stack-up (spec page: 9 unknown, 31 assumed numbers)',
         'Draw from docs/detector_spec/spec.py so paper, spec page and Geant4 agree'],
        'Ask Stephan for the stack-up drawing or the answers on the spec page; then generate it from spec.py.'),
    'readout_layout': ('placeholder', 'Readout board drawing: placeholder; should also show the passivation',
        ['Pads → X/Y strips, the 18 mm passivated y ends, the unpassivated x sides',
         'Ties the design notes (§2.1) to the edge section'],
        'A photo or Gerber render from Stephan plus an overlay of the measured efficient area 399.4 × 359.9 mm².'),
    'bench_setup': ('polish', 'The bench schematic works; add the numbers the text uses',
        ['Add z of the M3 stations and the two slots (24/144/1185/1302; 232/702 mm)',
         'Mark which slot det3 used on the golden rows'],
        'Small edit of mx_june_cosmic_qa/engineer_package/make_bench_diagram.py.'),
    'sps_setup': ('placeholder', 'SPS setup: placeholder; keep it small now that SPS is a cross-check',
        ['One schematic: beam, uRWELL telescope, det4, rotation stage',
         'Could go to the SPS appendix if the section is demoted'],
        'From the SPS logbook photos / the sps_beam_test_26 geometry.'),
    'currents_vs_hv': ('polish', 'The dark-current figure is new and readable; tidy the legend',
        ['Two-column legend of night labels is dense: label curves directly or by board only',
         'Use the paper’s chamber colours (see the colour slide)',
         'Add the gain curve (×2 / 16.7 V) as a reference slope'],
        'rerun/paper1_ops/currents/hv_currents.py; boards from 6-17 only (Dylan).'),
    'hv_scan': ('revise', 'hv_scan already is the “efficiency vs HV with the discharge loss” figure: promote it',
        ['Top two panels = the plot asked for in §5 (eff_vs_hv_discharge placeholder): use them there',
         'det4’s golden star (41 %) has no curve and no legend entry',
         'Middle-panel annotation is tiny; colours differ from every other figure'],
        'Split: (efficiency, lost-to-discharge) vs HV for det3 / det6 / det7 into §5; the discharges-per-trigger '
        'panel stays in §4. Drop the placeholder eff_vs_hv_discharge.'),
    'spark_deadtime': ('revise', 'No dead time after a discharge: the claim is right, det4 swamps the picture',
        ['det4 (4 303 discharges) runs off the axis at 3 ms and dominates the eye',
         'Gain panel has no error bars',
         'Lead with det3 (45 500 discharges); others lighter'],
        'Clip/omit det4 or give it its own small panel; add bootstrap errors to the gain ratio.'),
    'charge_balance': ('polish', 'X/Y charge balance: clear; match the text’s quantity and the colours',
        ['Text quotes f = qX/(qX+qY) = 0.485 / 0.513; the figure plots ln(qX/qY)',
         '“Q_x” renders with a literal underscore',
         'Legend sits on the det7 peak'],
        'Either quote ln ratio medians in the text (−0.060 det3) or plot f; move legend; mathtext labels.'),
    'eff_vs_radius': ('keep', 'Efficiency vs matching radius: ready',
        ['Built this week; knee at R = 5 mm marked, accidentals on the right axis'],
        'rerun/effbudget/single/make_figures.py.'),
    'residuals': ('keep', 'Full residual distributions behind the radius choice: ready',
        ['Per view (log) and radial with event-mixed accidentals'],
        'rerun/effbudget/single/make_figures.py.'),
    'efficiency_map': ('revise', 'The five maps show the production defects; make det3 the big one',
        ['Panel titles pool many rows (“30 rows, 70 470 rays”): the text quotes golden rows',
         'det3 deserves a large panel; the four others small (det4 stripes, det6/det7 discharges, det7 X fault)',
         'det7’s “X ch 0–255” label unreadable; det4 drawn outside its active area'],
        'rerun/paper1_figs/make_effmap_figure.py: det3 large + 2×2 small; golden-row pools; mask outside-area pixels.'),
    'eff_vs_hv_discharge': ('polish', 'det3 reaches its plateau before discharges cost anything; det6/det7 do not', ['New 11-02 (split from hv_scan)'], 'rerun/paper1_ops/make_report.py.'),
    'xy_pairing': ('polish', 'X/Y pairing by charge: ready; add the production result after the fix',
        ['Annotation should add the production pairing (96.7 % after the fix)',
         'Could show the similar-charge subset separately'],
        'rerun/effbudget/single/make_figures.py.'),
    'gas_vE': ('polish', 'Drift scan: both velocities on one Magboltz curve, one water value', ['Redone 11-02 with timing v + angle-scale v and a ratio panel'], 'rerun/paper1_gas/pg_figs.py.'),
    'attachment': ('polish', 'SPS gas composition: legend and readable labels', ['Restyled 11-02; bench shown as grey bands; appendix candidate'], 'rerun/paper1_spsfigs.'),
    'tracking_perf': ('revise', 'Angular resolution vs angle: det3 is the headline, det4 sets the scale',
        ['Right panel repeats the efficiency (old interior values) and belongs in §5: drop it',
         'det4 (2–2.9°) stretches the axis; det3 (0.9–1.0°) is a thin line at the bottom',
         'Label “σθ (s68)”: define once, in the caption'],
        'det3 X/Y large with its error band; others light grey; det4 in an inset or the table.'),
    'sps_resolution': ('polish', 'SPS cross-check on the working region R: 138 / 120 µm', ['Redone 11-02; appendix candidate'], 'rerun/paper1_spsfigs.'),
    'bow_profiles': ('polish', 'Mesh bow per chamber: clear; colours and det3 emphasis',
        ['Five chambers, five colours unlike the other figures'],
        'Restyle with the paper palette.'),
    'bow_closure': ('polish', 'Bow closure: fine for an appendix; axis labels are code',
        ['“dw0/dp”, “v h″” need words in the caption or axis'],
        'Move to the bow appendix if the section is shortened.'),
    'edge_turnon': ('polish', 'Edge zone, three signals: good; the x axis needs a plain scale',
        ['Symlog axis with a negative region is hard to read',
         'Legend only in (a); colour = side, consistent across panels (good)'],
        'Linear axis 0–60 mm (log not needed), shade the 30 mm fiducial.'),
    'timing': ('revise', 'Timing: three dense panels; the paper needs one number per chamber',
        ['Panel (b) mixes SPS det4 with bench chambers',
         'Order det2, det3, det7, det6, det4 is arbitrary; colours another palette',
         'Lead with det3 (40 ns event time); others lighter'],
        'Keep (a) and (c) for det3 + fleet in the paper; (b) to an appendix.'),
    'det4_stripes': ('keep', 'det4 stripes bench vs SPS: clear and convincing',
        ['Legend line colours too thin to match the curves'],
        'Thicker legend handles.'),
    'loss_budget': ('polish', 'Where det3 single-track inefficiency goes: δ-rays first, then the edge-zone veto', ['New 11-02'], 'rerun/effbudget/single/make_figures.py.'),
    'two_particle': ('polish', 'With a second particle the muon is found unless the two merge or trip the veto', ['New 11-02; rebuild after the two-track and veto fixes'], 'rerun/effbudget/single/make_figures.py.'),
    'two_velocities': ('polish', 'Two independent drift velocities; their ratio measures the gap: 28.2 ± 0.5 mm', ['New 11-02'], 'rerun/paper1_gas/pg_twov.py.'),
    'gas_drying': ('polish', 'The bench gas dried over hours with one time shape', ['New 11-02'], 'rerun/paper1_gas.'),
    'bench_position_res': ('polish', 'det3 intrinsic resolution ~120 µm once scattering and the M3 pointing are removed', ['New 11-02; the P subtraction dominates the error'], 'rerun/paper1_floor/position_ms_fig.py.'),
    'display_two_separate': ('polish', 'Two separate particles: both fitted', ['New 11-02'], 'papers/paper1/figs/make_two_particle_displays.py.'),
    'display_merged_projection': ('polish', 'Two particles merged in one projection: the muon is pulled', ['New 11-02'], 'papers/paper1/figs/make_two_particle_displays.py.'),
    'display_muon_deposit': ('polish', 'A muon and a separate deposit trip the discharge veto', ['New 11-02'], 'papers/paper1/figs/make_two_particle_displays.py.'),
}
A1_REVIEW = ('revise', 'Loss-class displays: real events, readable; titles use code names and one is mislabelled',
             ['Titles show internal codes (“far:DELTA”, “multi:m3_picked”): use the paper’s class names',
              'loss_headon shows a fit_failed event (head-on class ~0 on pass 1): caption no longer matches',
              'Fractions are interior: the headline is now the full area',
              'Add the δ size continuum (12 / 19 / 34 / 67 strips) as one summary panel'],
             'rerun/satstudy/appendix_figs.py; captions in A1_loss_categories.tex.')

MISSING = []   # all six made on 2026-11-02 (bench resolution, two velocities, drying, loss budget, two-particle, displays)

PALETTES = {   # colours the figures use per chamber (read off the figures, 2026-11-02)
    'hv_scan':        {'det2': '#2c7bd6', 'det3': '#e8562a', 'det6': '#f0a30a', 'det7': '#e57ab4'},
    'charge_balance': {'det2': '#2c7bd6', 'det3': '#e8562a', 'det4': '#1aa36b', 'det6': '#f0a30a', 'det7': '#e57ab4'},
    'tracking_perf':  {'det2': '#1f77b4', 'det3': '#d62728', 'det4': '#2ca02c', 'det6': '#9467bd', 'det7': '#ff7f0e'},
    'bow_profiles':   {'det2': '#2459a8', 'det3': '#2f8a5b', 'det4': '#c63d4f', 'det6': '#7a52b8', 'det7': '#c99318'},
    'timing':         {'det2': '#1f77b4', 'det3': '#d62728', 'det4': '#7f7f7f', 'det6': '#2ca02c', 'det7': '#9467bd'},
}
PROPOSED = {'det2': '#56B4E9', 'det3': '#0b3d91', 'det4': '#999999', 'det6': '#E69F00', 'det7': '#CC79A7'}


# --------------------------------------------------------------------------- #
# paper sources
# --------------------------------------------------------------------------- #
def braced(s, i):
    """Text inside the brace group starting at s[i] == '{'."""
    d = 0
    for j in range(i, len(s)):
        d += {'{': 1, '}': -1}.get(s[j], 0)
        if d == 0:
            return s[i + 1:j]
    return s[i + 1:]


def detex(s):
    s = re.sub(r'\\(tbw|requote|emph|textbf|texttt|mathrm)\{', '{', s)
    s = re.sub(r'\\(cref|ref|label)\{[^}]*\}', '§', s)
    rep = {r'\,': ' ', '~': ' ', r'\%': '%', r'\textmu': 'µ', r'\textdegree': '°', r'\ldots': '…', '$': '',
           r'\times': '×', r'\le': '≤', r'\ge': '≥', r'\sigma': 'σ', r'\delta': 'δ', r'\chi': 'χ', r'\pm': '±',
           r'\theta': 'θ', r'\varepsilon': 'ε', '--': '–', '``': '“', "''": '”', r'\_': '_'}
    for a, b in rep.items():
        s = s.replace(a, b)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    return re.sub(r'\s+', ' ', s.replace('{', '').replace('}', '')).strip()


def paper_figures():
    order = re.findall(r'\\input\{sections/([^}]+)\}', (HERE / 'main.tex').read_text())
    out = []
    for sec in order:
        t = (HERE / 'sections' / f'{sec}.tex').read_text()
        stitle = re.search(r'\\(?:sub)?section\{([^}]*)\}', t)
        stitle = detex(stitle.group(1)) if stitle else sec
        for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}#]+)\}', t):
            name = m.group(1)
            c = t.find(r'\caption{', m.end())
            cap = detex(braced(t, c + len(r'\caption'))) if c > 0 and c - m.end() < 400 else ''
            out.append(dict(sec=sec, stitle=stitle, name=name, caption=cap))
        for m in re.finditer(r'\\lossfig\{([^}]+)\}\{', t):
            out.append(dict(sec=sec, stitle=stitle, name=m.group(1), caption=detex(braced(t, m.end() - 1))))
    return out


def pdf_png(name, dpi=150):
    CACHE.mkdir(parents=True, exist_ok=True)
    src = next((p for p in (FIG / f'{name}.pdf', FIG / f'{name}.png') if p.exists()), None)
    if src is None:
        return None
    if src.suffix == '.png':
        return src
    out = CACHE / f'{name}.png'
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(['pdftoppm', '-png', '-r', str(dpi), '-singlefile', str(src), str(out.with_suffix(''))], check=True)
    return out


def fig(name, w, h, tip=None):
    p = pdf_png(name)
    if p is None:
        return sd.card(sd.p(f'{name}: no file', 26, sd.MUT), w=w)
    im = Image.open(p).convert('RGB')
    tw = int(min(im.width, min(w, h * im.width / im.height) * 1.5))
    if tw < im.width:
        im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(b, 'PNG', optimize=True)
    return sd.img(b.getvalue(), w=w, h=h, tip=tip or f'figures/{name}')


SCOL = {'not ready': sd.RED, 'in work': sd.GOLD, 'for review': sd.BLUE, 'accepted': sd.GREEN}


def load_log():
    p = HERE / 'figure_log.toml'
    return tomllib.loads(p.read_text()).get('entry', []) if p.exists() else []


def status_of(name, log):
    st = None
    for e in log:
        if e.get('status') and e['fig'] in ('*', name):
            st = e['status']
    return st or 'not ready'


def schip(st, size=22):
    return (f'<span style="display:inline-block;border:2px solid {SCOL[st]};color:{SCOL[st]};font-size:{size}px;'
            f'font-weight:600;padding:2px 12px;border-radius:999px">{st}</span>')


def log_html(name, log):
    rows = [e for e in log if e['fig'] in ('*', name)]
    if not rows:
        return '<p><b>Log:</b> —</p>'
    tr = ''.join(f'<tr><td style="white-space:nowrap;padding-right:16px">{e["date"]}</td>'
                 f'<td style="padding-right:16px">{e.get("status", "")}</td><td>{sd.esc(e["change"])}</td></tr>'
                 for e in reversed(rows))
    return f'<p><b>Change log</b> (newest first; <code>papers/paper1/figure_log.toml</code>):</p><table>{tr}</table>'


def chip(verdict, size=24):
    c = VCOL[verdict]
    return (f'<span style="display:inline-block;background:{c};color:#fff;font-size:{size}px;font-weight:600;'
            f'padding:4px 14px;border-radius:999px">{VTXT[verdict]}</span>')


def verdict_of(name, state):
    if name in REVIEW:
        return REVIEW[name][0]
    if name.startswith('loss_'):
        return A1_REVIEW[0]
    return 'placeholder' if state == 'placeholder' else 'keep'


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D, F, man, log=()):
    body_figs = [f for f in F if not f['name'].startswith('loss_')]
    st = [status_of(f['name'], log) for f in body_figs]
    n_rev = sum(x == 'for review' for x in st)
    n_acc = sum(x == 'accepted' for x in st)
    n_ph = sum(man.get(f['name'], {}).get('state') == 'placeholder' for f in body_figs)
    body = (sd.kicker('MX17 Paper I · figures · ' + dt.date.today().isoformat())
            + sd.title(f'{n_rev} of {len(body_figs)} body figures were changed and wait for your review; '
                       f'{n_acc} accepted so far')
            + sd.row(sd.bignum(str(len(body_figs)), 'body figures', sd.DBLUE, f'plus {len(F) - len(body_figs)} loss-class displays (A1)'),
                     sd.bignum(str(n_rev), 'for review', sd.DBLUE, 'changed since your last look (see the log)'),
                     sd.bignum(str(n_acc), 'accepted', sd.DGREEN, 'by Dylan'),
                     sd.bignum(str(n_ph), 'placeholders', sd.DGREY, 'stack, readout board, SPS setup (Stephan / photos)'), gap=48))
    D.slide('cover', body, '<p>Every figure included by main.tex, in paper order. Status from the change log '
            '<code>papers/paper1/figure_log.toml</code> (not ready → in work → for review → accepted); the 11-02 review '
            'verdicts stay on each slide as the second chip. Rebuild: <code>papers/paper1/make_figures_deck.py</code>.</p>',
            dark=True, short='Cover')


def s_board(D, F, man, log=()):
    rows, cur = [], None
    for f in F:
        if f['name'].startswith('loss_'):
            continue
        v = verdict_of(f['name'], man.get(f['name'], {}).get('state'))
        sec = f['stitle'] if f['stitle'] != cur else ''
        cur = f['stitle']
        rows.append([sec, f'<code>{f["name"]}</code>', schip(status_of(f['name'], log), 18), chip(v, 18),
                     REVIEW.get(f['name'], ('', ''))[1][:80]])
    leg = ' '.join(chip(k, 20) for k in ('keep', 'polish', 'revise', 'redo', 'move', 'placeholder'))
    per = 13
    parts = [rows[i:i + per] for i in range(0, len(rows), per)]
    for k, part in enumerate(parts):
        t = sd.table(['section', 'figure', 'status', 'review', 'what it needed (11-02 review)'], part, size=19,
                     widths=[230, 250, 160, 150, 870], align=['left'] * 5)
        body = (sd.title(f'The board ({k + 1}/{len(parts)}): every body figure, in paper order',
                         'One line per figure; each has its own slide below') + t + (sd.p(leg, 20) if k == 0 else ''))
        D.slide(f'board{k + 1}', body, '<p>A1’s 14 loss displays are one slide at the end. The section column follows '
                'main.tex (timing now after tracking, edge as its own subsection).</p>', short=f'Board {k + 1}/{len(parts)}')


def s_colours(D):
    dets = ['det2', 'det3', 'det4', 'det6', 'det7']
    rows = []
    for figname, pal in list(PALETTES.items()) + [('proposed', PROPOSED)]:
        cells = []
        for d in dets:
            c = pal.get(d)
            cells.append(f'<span style="display:inline-block;width:64px;height:30px;border-radius:6px;background:{c}"></span>'
                         if c else '<span style="color:#aaa">—</span>')
        rows.append([('<b>proposed</b>' if figname == 'proposed' else f'<code>{figname}</code>')] + cells)
    t = sd.table(['figure'] + dets, rows, size=22, widths=[300] + [150] * 5, align=['left'] + ['center'] * 5)
    side = sd.col(sd.callout('det3 is red in two figures, orange-red in two, green in one: a reader cannot follow it '
                             'from section to section.', sd.RED, 24),
                  sd.callout('One palette for the whole paper, det3 the strongest and darkest (it is the paper’s chamber), '
                             'the others colour-blind-safe and lighter.', sd.GREEN, 24),
                  sd.callout('One module (figstyle-like) imported by every producer, so a rebuild fixes them all.', sd.BLUE, 24),
                  w=560)
    body = sd.title('Adopted 11-02: one colour per chamber (before: five figures, five palettes)',
                    'Colours read off each figure before the restyle; bottom row the palette now in papers/paper_style.py') + sd.row(t, side, gap=60)
    D.slide('colours', body, '<p>Proposal: det3 #0b3d91 (dark blue), det2 #56B4E9 (sky), det6 #E69F00 (orange), det7 #CC79A7 '
            '(pink), det4 #999999 (grey: the defective chamber, background in most figures). Put it in a shared '
            '<code>papers/paper_style.py</code> (rcParams + DET_COLOR + label helpers: mathtext, no raw underscores) and '
            'import it from every producer listed in figures.toml.</p>', short='One palette')


def s_missing(D):
    cards = [sd.card(sd.p(f'<b>{t}</b>', 26) + sd.p(d, 22, sd.MUT) + sd.p(s, 20, sd.BLUE), w=500) for t, d, s in MISSING]
    body = (sd.title('Six figures the text needs do not exist yet', 'The story moved to det3, cosmics-first and two-particle events this week')
            + sd.col(sd.row(*cards[:3], gap=32), sd.row(*cards[3:], gap=32), gap=32))
    D.slide('missing', body, '<p>The bench intrinsic resolution became the headline position result when SPS was demoted '
            'to a cross-check (Dylan 11-02); the two-velocity / gap figure and the drift scan with both estimators are '
            'from Dylan’s outline (laptop session 11-01); the loss budget and the two-particle classes from this week’s '
            'efficiency work (<code>rerun/effbudget/single/README.md</code>).</p>', short='Missing')


def s_figure(D, f, man, log):
    name = f['name']
    e = man.get(name, {})
    v, claim, issues, fix = REVIEW.get(name, (verdict_of(name, e.get('state')), name, [], ''))
    st = status_of(name, log)
    last = next((x for x in reversed(log) if x['fig'] == name), None)
    if last:
        issues = [f'{last["date"]}: ' + last['change'][:150] + ('…' if len(last['change']) > 150 else '')] + \
                 [i for i in issues][:2]
    figbox = fig(name, 1040, 640, tip=f'figures/{name} · state {e.get("state", "?")}')
    side = sd.col(sd.p(schip(st) + '&nbsp;' + chip(v, 20) + f'&nbsp;&nbsp;<span style="color:{sd.MUT};font-size:20px">{f["stitle"]}</span>', 22),
                  *[sd.callout(i, VCOL[v], 23) for i in issues], w=560, gap=20)
    body = sd.title(claim, f'{name} · figures.toml: {e.get("state", "not in manifest")}') + sd.row(figbox, side, gap=48)
    notes = (f'<p><b>Caption (draft):</b> {sd.esc(f["caption"]) or "—"}</p>'
             f'<p><b>Fix:</b> {sd.esc(fix)}</p>'
             f'<p><b>Source:</b> <code>{sd.esc(e.get("source", "—"))}</code><br><b>Producer:</b> '
             f'<code>{sd.esc(e.get("producer", "—"))}</code></p>'
             f'<p><b>Manifest note:</b> {sd.esc(e.get("note", "—"))}</p>' + log_html(name, log))
    D.slide(f'f_{name}', body, notes, short=name)


def s_a1(D, F, man):
    names = [f['name'] for f in F if f['name'].startswith('loss_')]
    ims = [Image.open(pdf_png(n)).convert('RGB') for n in names if pdf_png(n)]
    w, h = ims[0].size
    cols = 4
    rows_ = (len(ims) + cols - 1) // cols
    S = Image.new('RGB', (w * cols, h * rows_), 'white')
    for i, im in enumerate(ims):
        S.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    S.thumbnail((2400, 2400))
    b = io.BytesIO()
    S.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(b, 'PNG', optimize=True)
    v, claim, issues, fix = A1_REVIEW
    side = sd.col(sd.p(chip(v) + f'&nbsp;&nbsp;<span style="color:{sd.MUT};font-size:22px">Appendix A1</span>', 22),
                  *[sd.callout(i, VCOL[v], 22) for i in issues], w=540, gap=18)
    body = sd.title(claim, f'{len(names)} displays, one real det3 event per class') + sd.row(sd.img(b.getvalue(), w=1080, h=650), side, gap=44)
    D.slide('a1', body, f'<p><b>Fix:</b> {sd.esc(fix)}</p><p>Displays: ' + ', '.join(f'<code>{n}</code>' for n in names) + '</p>',
            short='A1 displays')


EVD_GALLERY = Path(os.environ.get('EVD_GALLERY', os.environ.get('WORK_DATA', '/media/ucla/work') + '/claude-jobs/1ccd2342/evd/gallery'))
#: colour map -> (claim, what to look at); each is shown in both normalisations (EVD_NORMS)
EVD_MAPS = [
    ('reds', 'White to red (the paper default)', 'Strongest contrast for the black reference and the blue fit; prints well; faint charge reads as pink.'),
    ('greens', 'White to green', 'Fits drawn magenta; the green maximum is a little lighter than the red, so the core stands out less.'),
    ('fire', 'White to yellow to red', 'Faint charge in yellow: brighter on screen, but pale yellow washes out on white paper.'),
    ('jet', 'Jet, zero white', 'Every sample above threshold coloured (blue), so noise and diffusion tails look as loud as the track; fits magenta.'),
]
EVD_NORMS = [('full', 'normalised to the ADC full scale (4096)', 'One scale for every event in the paper: the colour bar means the same in each figure.'),
             ('event', 'normalised to the largest amplitude in the event', 'X and Y together; each event uses the whole colour range. Here it looks almost the same as full scale, because most events reach the 3550 ADC clip.')]


def s_evd_styles(D):
    """Appendix: event-display colour maps x amplitude normalisations (Dylan 11-03, second pass), same events in each."""
    pages = [(m, mc, mw, n, nc, nw) for m, mc, mw in EVD_MAPS for n, nc, nw in EVD_NORMS
             if (EVD_GALLERY / f'style_{m}_{n}.png').exists()]
    if not pages:
        return
    rows = [[f'<b>{mc}</b>', sd.esc(mw)] for m, mc, mw in EVD_MAPS]
    body = (sd.title('Appendix: event-display colour maps and amplitude scales',
                     'Four chamber A events (Run 1), one display per row, in 4 colour maps × 2 normalisations '
                     '(papers/paper1/figs/make_displays.py gallery)')
            + sd.table(['colour map', 'what to look at'], rows, size=21, widths=[420, 1240], align=['left', 'left'])
            + sd.p('Normalisation: ' + '; '.join(f'<b>{n}</b> = {nc}' for n, nc, _ in EVD_NORMS) + '.', 21, sd.MUT))
    D.slide('evd_index', body, '<p>Fixed in every option (11-03 second pass): mesh and cathode lines kept; the reference '
            'track a solid black line running past them; fits thin and dashed, in a colour distinct from the data and '
            'the reference; zero amplitude white (samples below 60 ADC are not drawn); a linear amplitude scale; X and Y '
            'views of one event side by side on one row; broken axes only where a figure asks for them (none here). '
            'The choice is one line in papers/paper1/figs/evd.py (DEFAULT, OPT["norm"]).</p>', short='Display styles')
    for m, mc, mw, n, nc, nw in pages:
        im = Image.open(EVD_GALLERY / f'style_{m}_{n}.png').convert('RGB')
        im.thumbnail((1500, 1500))
        b = io.BytesIO(); im.save(b, 'PNG', optimize=True)
        h = 680
        body = sd.row(sd.img(b.getvalue(), w=int(h * im.size[0] / im.size[1]), h=h),
                      sd.col(sd.title(f'{mc}, {nc}', ''), sd.p(sd.esc(mw), 22), sd.p(sd.esc(nw), 22, sd.MUT), w=900, gap=14),
                      gap=40)
        D.slide(f'evd_{m}_{n}', body, f'<p><code>{EVD_GALLERY}/style_{m}_{n}.png</code></p>', short=f'{m}, {n}')


def s_unused(D, F, man):
    used = {f['name'] for f in F}
    rows = [[f'<code>{k}</code>', e.get('state', ''), sd.esc(e.get('title', ''))[:110]] for k, e in man.items() if k not in used]
    t = sd.table(['manifest entry', 'state', 'title'], rows, size=19, widths=[330, 150, 1180], align=['left'] * 3)
    head = (f'{len(rows)} manifest figures are built but not used by the text' if len(rows) != 1 else
            'One manifest figure is built but not used by the text: gas_vE_timing, the drift-scan figure the story wants' if 'gas_vE_timing' in rows[0][0] else 'One manifest figure is built but not used by the text')
    body = sd.title(head, 'Candidates to bring back, move to Paper II, or delete') + t
    D.slide('unused', body, '<p>gas_vE_timing is the one to bring back (the drift-scan figure the story wants).</p>', short='Unused')


CHECKLIST = [
    ('Size', 'drawn at the width it is printed (full width 6.1 in, half 3.0 in); no rescaling in LaTeX; type 7–9 pt in print'),
    ('No titles inside', 'the caption carries the title; panels labelled (a), (b), … in a corner'),
    ('Axes', 'every axis labelled with quantity and unit in mathtext; sensible ticks; ranges that show the data, shared where compared'),
    ('Chambers', 'paper palette only (det3 dark blue on top); X solid, Y dashed open; det4 grey'),
    ('Data vs model', 'data as markers with error bars; models/fits as lines; the claim visible without reading the caption'),
    ('Legend', 'only what the reader needs; off the data; one per figure if panels share it'),
    ('Ink', 'no gridline clutter, no boxes, no superfluous annotations; text in the plot only for the one point it makes'),
    ('Numbers', 'every plotted number traceable: the figure ships its CSV; caption and text quote the same values'),
    ('Vector', 'PDF, embedded fonts; rasters only for maps/displays, at ≥ 300 dpi'),
]


def s_checklist(D):
    t = sd.table(['', 'standard'], [[f'<b>{a}</b>', b] for a, b in CHECKLIST], size=24, widths=[260, 1400], align=['left', 'left'])
    body = sd.title('What “publication-ready” means here: the checklist every figure has to pass',
                    'JINST, full text width ≈ 15.5 cm; applied one figure at a time with Dylan') + t
    D.slide('checklist', body, '<p>Proposed 2026-11-02 after Dylan judged the figures not publication-ready. Edit the list '
            'in make_figures_deck.py (CHECKLIST) as standards are agreed.</p>', short='Checklist')


def s_log(D, log):
    rows = [[e['date'], f'<code>{e["fig"]}</code>', e.get('status', ''), sd.esc(e['change'])[:150]] for e in reversed(log)]
    t = sd.table(['date', 'figure', 'status', 'change'], rows[:12], size=18, widths=[150, 260, 150, 1100], align=['left'] * 4)
    body = sd.title(f'Change log: {len(log)} entries', 'Newest first (latest 12 shown; all in Details and in figure_log.toml)') + t
    full = ''.join(f'<tr><td>{e["date"]}</td><td><code>{e["fig"]}</code></td><td>{e.get("status", "")}</td><td>{sd.esc(e["change"])}</td></tr>'
                   for e in reversed(log))
    D.slide('log', body, f'<table>{full}</table>', short='Change log')


def s_close(D):
    order = [('1', 'One palette + style module', 'every figure inherits it; do first, it touches all producers'),
             ('2', 'Redo the principle figure', 'it shows the hit-time method the paper rejects'),
             ('3', 'Make the bench position-resolution figure', 'the position headline has no figure'),
             ('4', 'Drift scan with both velocities (gas_vE_timing + angle v)', 'and the gap from their ratio'),
             ('5', 'Split hv_scan into §4 / §5', 'drops the eff_vs_hv placeholder'),
             ('6', 'det3-first versions', 'efficiency_map, tracking_perf, spark_deadtime, timing'),
             ('7', 'Loss budget + two-particle figures', 'after the three reconstruction fixes land'),
             ('8', 'SPS figures to the SPS section/appendix', 'region R, not the full-beam zone fits')]
    t = sd.table(['', 'do', 'why'], [[a, f'<b>{b}</b>', c] for a, b, c in order], size=24, widths=[50, 760, 850],
                 align=['left'] * 3)
    body = (sd.kicker('Order of work') + sd.title('Palette first, then the two figures that contradict the text') + t
            + sd.p('Waiting on others: chamber_stack, readout_layout (Stephan); two-particle numbers (the merged-projection and '
                   'discharge-veto sessions).', 22, sd.DMUT))
    D.slide('close', body, '<p>Decisions for Dylan: (a) the palette; (b) whether bow_closure and timing (b) go to '
            'appendices; (c) whether the SPS figures stay in the body.</p>', dark=True, short='Order of work')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(os.environ.get('CLAUDE_JOB_DIR', '/tmp')) / 'tmp' / 'mx17-paper1-figures.html'))
    a = ap.parse_args()
    man = tomllib.loads((HERE / 'figures.toml').read_text())['fig']
    F = paper_figures()
    log = load_log()
    D = sd.Deck('MX17 Paper I: figure review', 'Every Paper I figure in paper order, with its status and what it needs')
    s_cover(D, F, man, log); s_checklist(D); s_board(D, F, man, log); s_log(D, log); s_colours(D)
    if MISSING:
        s_missing(D)
    for f in F:
        if not f['name'].startswith('loss_'):
            s_figure(D, f, man, log)
    s_a1(D, F, man); s_unused(D, F, man); s_close(D); s_evd_styles(D)
    D.write(a.out, note_meta=dict(title='MX17 Paper I: figure review',
                                  summary='Every Paper I figure in paper order: status, verdict and what to fix; one palette, '
                                          'six missing figures, the order of work.',
                                  tags=['x17', 'paper', 'figures'], date=dt.date.today().isoformat()))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
