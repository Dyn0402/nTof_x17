#!/usr/bin/env python3
"""make_collection_deck.py [--out PATH] -- the Paper II figure collection and outline as a slide note.

Reads papers/paper2/collection/index.json (built by collect_figures.py) and embeds the figures themselves, section
by section, then the proposed outline, the work plan, the gaps and the open decisions. A first dump to look at
what we have, not a polished deck.

Publish: python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <out> --slug mx17-paper2-collection --force --deploy
"""
import argparse
import datetime as dt
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

HERE = Path(__file__).resolve().parent
COL = HERE / 'collection'
CACHE = Path(os.environ.get('CLAUDE_JOB_DIR', '/tmp')) / 'tmp' / 'paper2_deck_png'
CAT = json.loads((COL / 'index.json').read_text())
ITEMS = {i['name']: i for i in CAT['items']}
RC = {'A': sd.GREEN, 'A-': sd.GREEN, 'B': sd.GOLD, 'C': sd.GREY}


def rating(r):
    return (r or '').split(' ')[0] or '?'


def png_of(path, dpi=130):
    if path.suffix.lower() != '.pdf':
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f'{path.parent.name}_{path.stem}_{dpi}'
    png = Path(str(out) + '.png')
    if not png.exists() or png.stat().st_mtime < path.stat().st_mtime:
        subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-singlefile', str(path), str(out)], check=True)
    return png


def fig(name, w=None, h=None, k=0, label=True):
    """One collection figure as an embedded PNG with a one-line label underneath and the provenance on hover."""
    it = ITEMS[name]
    path = COL / it['section'] / it['files'][k]
    im = Image.open(png_of(path)).convert('RGB')
    box_w = w or (h * im.width / im.height)
    if h and w:
        box_w = min(w, h * im.width / im.height)
    tw = int(min(im.width, box_w * 1.5))
    if tw < im.width:
        im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(b, 'PNG', optimize=True)
    tip = (f"{it['caption']}\nbasis: {it['basis']}\nrating {it['rating']}\nsource: {it['src']}\nproducer: {it['producer']}")
    r = rating(it['rating'])
    cap = (f'<p style="font-size:21px;color:{sd.MUT};line-height:1.25;max-width:{int(box_w)}px">'
           f'<span style="color:{RC.get(r, sd.GREY)};font-weight:600">{r}</span> · {name} · {it["basis"]}</p>') if label else ''
    return sd.col(sd.img(b.getvalue(), w=w, h=h, tip=tip), cap, gap=8)


def bullets(items, size=26):
    li = ''.join(f'<li style="margin-bottom:10px">{x}</li>' for x in items)
    return f'<ul style="font-size:{size}px;line-height:1.35;padding-left:30px;margin:0">{li}</ul>'


def notes_table(secs):
    rows = []
    for it in CAT['items']:
        if it['section'] in secs:
            rows.append([it['name'], rating(it['rating']), it['basis'] or '—', it['caption'], f"<code>{it['src']}</code>"])
    return sd.table(['item', 'rating', 'basis', 'what it shows', 'source'], rows, size=19,
                    widths=[200, 60, 190, 560, 460], align=['left'] * 5)


def counts():
    real = [i for i in CAT['items'] if i['files']]
    gaps = [i for i in CAT['items'] if i['section'] == '9_gaps']
    a = [i for i in real if rating(i['rating']) in ('A', 'A-')]
    bc = [i for i in real if rating(i['rating']) in ('B',)]
    return len(real), len(a), len(bc), len(gaps)


# --------------------------------------------------------------------------- #
def s_cover(D):
    nreal, na, nb, ng = counts()
    body = (sd.kicker('MX17 · Paper II (reconstruction) · figure collection and outline · 5 Nov 2026') +
            '<h1 style="font-size:68px;font-weight:600;line-height:1.08;letter-spacing:-1.5px;color:#eef0f3">'
            'Paper II has its kernel figures and a pass-1 hits-vs-waveforms headline. The two figures it is built around, '
            'the ladder compression and the tier benchmark, do not exist on pass 1</h1>' +
            sd.p('Every figure and metric found in the repo that could serve the reconstruction paper, sorted into paper sections, '
                 'with what is paper-ready, what needs a remake, and what has never been made.', size=30, color=sd.DMUT) +
            sd.row(sd.bignum(str(nreal), 'catalogued figures', sd.DBLUE, 'copied into papers/paper2/collection/, one folder per section'),
                   sd.bignum(str(na), 'rated A (paper-ready or near)', sd.DGREEN, 'mostly pass-1 SPS and fleet figures, plus the July flagship layouts'),
                   sd.bignum(str(nb), 'rated B (need a remake)', sd.GOLD, 'old model (July R&D), hits basis, or wrong framing'),
                   sd.bignum(str(ng), 'gaps', sd.DRED, 'figures the paper needs that were never made', ), gap=48))
    D.slide('cover', body, '<p>Built by <code>papers/paper2/make_collection_deck.py</code> from <code>papers/paper2/collection/index.json</code>. '
            'Outline and plan: <code>papers/paper2/OUTLINE_PLAN_2026-11-05.md</code>. Three catalogue sweeps (waveform-first and June QA; '
            'wft, SPS and kernel; rerun and published notes) fed the list; ratings are mine, from reading the producer and, for most, '
            'looking at the image.</p>', dark=True, short='Cover')


def s_story(D):
    steps = [dict(label='Kernel', sub='how charge reaches the neighbours', color=sd.BLUE,
                  tip='±1/lead 0.38; ±2/±1 0.19 (X), 0.26 (Y); field-independent'),
             dict(label='Hits fail', sub='ladder compressed 20–30 %', color=sd.RED,
                  tip='estimator-independent: leading edge, CFD, matched filter'),
             dict(label='Forward model', sub='template × kernel × drift', color=sd.GREEN),
             dict(label='Floor', sub='0.30–0.35° diffusion', color=sd.PURPLE,
                  tip='measured is 3–6× the floor; 0.56–0.65° remainder unexplained'),
             dict(label='Hits vs waveforms', sub='same events, tiers H0…W', color=sd.ORANGE,
                  tip='only H0 vs W exists on pass 1'),
             dict(label='Readout', sub='threshold, neighbours', color=sd.GREY, tip='no data yet: software re-extraction of hits'),
             dict(label='Known angle', sub='det4 at 25.4°', color=sd.GOLD)]
    body = (sd.title('The paper in one line: hit times cannot be fixed by a better estimator, a forward model can fix them',
                     'Section order of the proposed paper; hover the boxes for the numbers; figures are in the slides that follow') +
            sd.flow(steps, size=26) +
            sd.row(sd.card(sd.p('<b>The claim</b>', 26) + sd.p('A per-strip hit time mixes the strip’s own charge with delayed copies of its '
                                                               'neighbours’. That compresses the drift ladder whatever the estimator.', 24, sd.MUT), w=520),
                   sd.card(sd.p('<b>The result</b>', 26) + sd.p('Same events, five chambers: angle s68 2.0–2.6° (hits) against '
                                                                 '0.73–1.56° (waveforms).', 24, sd.MUT), w=520),
                   sd.card(sd.p('<b>The open question</b>', 26) + sd.p('How much of that gap a sharing-aware fit on hit lists recovers (H2c), '
                                                                        'and what a hit front-end must record.', 24, sd.MUT), w=520), gap=40))
    D.slide('story', body, '<p>Pass-1 numbers: <code>rerun/paper1_floor/figures/fleet.pdf</code>, README of <code>rerun/paper1_floor</code>. '
            'Hover the dotted boxes. Figure thumbnails carry their provenance on hover too.</p>', short='The story')


def s_inventory(D):
    secs = [(k, v) for k, v in sorted(CAT['sections'].items()) if k != '9_gaps']
    rows = []
    mx = 1
    for k, v in secs:
        its = [i for i in CAT['items'] if i['section'] == k and i['files']]
        a = sum(rating(i['rating']) in ('A', 'A-') for i in its)
        b = sum(rating(i['rating']) == 'B' for i in its)
        c = sum(rating(i['rating']) == 'C' for i in its)
        rows.append((k, v, a, b, c))
        mx = max(mx, a + b + c)
    ng = sum(1 for i in CAT['items'] if i['section'] == '9_gaps')
    W, rh = 1000, 54
    inner = ''
    for n, (k, v, a, b, c) in enumerate(rows):
        y = 20 + n * rh
        inner += sd.T(0, y + 32, f"{v}", 24, sd.INK, 'start')
        x = 640
        for val, col, nm in ((a, sd.GREEN, 'A'), (b, sd.GOLD, 'B'), (c, sd.GREY, 'C')):
            w = val * 70
            if val:
                inner += (f'<rect x="{x}" y="{y}" width="{w}" height="38" fill="{col}" rx="4"{sd.tipattr(f"{v}: {val} rated {nm}")}/>'
                          + sd.T(x + w / 2, y + 28, str(val), 22, '#fff', 'middle', 600))
            x += w
    H = 40 + len(rows) * rh
    chart = sd.svg(1560, H, inner, 'Figures per paper section by rating')
    body = (sd.title('The kernel and fleet ends are well covered; the floor, hits-vs-waveforms and readout sections are thin',
                     'Catalogued figures per proposed paper section, by rating') + chart +
            sd.legend([('A  paper-ready or near', sd.GREEN, 'box'), ('B  needs a remake', sd.GOLD, 'box'), ('C  context only', sd.GREY, 'box')]) +
            sd.p('“Readout” has no figure at all.', 24, sd.MUT) +
            sd.p(f'Plus {ng} gaps: figures the paper needs that nobody has made (last slides).', 24, sd.MUT))
    D.slide('inventory', body, notes_table(set(CAT['sections'])), short='Inventory')


def two_by(D, sid, ttl, sub, names, notes, w=800, h=420, secs=None, foot=None, short=None, cols=2):
    cells = [fig(n, w=w, h=h) for n in names]
    rows = [sd.row(*cells[i:i + cols], gap=44) for i in range(0, len(cells), cols)]
    D.slide(sid, sd.title(ttl, sub) + ''.join(rows), notes + (notes_table(secs) if secs else ''), foot=foot, short=short or sid)


def s_kernel(D):
    two_by(D, 'kernel', 'The kernel is measured head-on in the beam and does not change with field or rotation',
           'SPS det4, run71 RAW (no zero suppression) and run63 at 25.4°, pass cache, 1 Nov',
           ['kernel_forms', 'angled_kernel'],
           '<p>±1/lead 0.377 X, 0.390 Y; ±2/±1 0.19 X, 0.26 Y at 243 V/cm, flat from 96 to 243 V/cm. The old Y c2/c1 0.42–0.47 is superseded. '
           'Also in the collection: model-free ±2/±1 per plane on all five chambers, the angled-kernel stacks and width tests, the old hits-based '
           'kernel as the “before”.</p>',
           secs={'1_kernel'}, w=800, h=640, foot='Source: rerun/paper1_spsfigs (kernel_job.py on condor)', short='Kernel')


def s_kernel2(D):
    two_by(D, 'kernel2', 'Supporting kernel evidence: all five chambers, the rotation test, and the schematic that must show c2 < c1',
           'Model-free plane ratio (cosmics), angled-kernel stacks (ZS, August), the existing sharing schematic (July)',
           ['modelfree_ratio', 'angled_stacks', 'sharing_schematic', 'angled_width'],
           '<p>The schematic’s cartoon uses c1≈0.5 and c2 0.05–0.15, a hits-era picture; redraw with the measured values. '
           'angled_stacks and angled_width are zero-suppressed August products and predate the pass-1 rebuild.</p>',
           w=800, h=290, short='Kernel: more')


def s_fail(D):
    two_by(D, 'fail', 'Every hit-time estimator reads late at the mesh and early at the cathode: the ladder is compressed',
           'det3, matched-filter t50 and leading edge 20 %, residual to the reference drift time (July R&D model: remake on pass 1, all chambers)',
           ['estimator_residual_curves', 'unsharing_depth_bias'],
           '<p>+40 ns late at the mesh, −100…−130 ns early at the cathode (20–30 % compression; ~4° too steep). The second figure is the '
           'hits-era evidence that the cluster fans away from the track with depth (+19 µm/mm production, +2 µm/mm unshared). '
           'Both are det3 only. A pass-1 fleet version is the first work item.</p>',
           secs={'2_hits_fail'}, w=800, h=560, foot='Source: waveform_first_threading/05 (report PDF), engineer_package 09. Hits-based by design: they show the failure.',
           short='Hits fail')


def s_fail2(D):
    two_by(D, 'fail2', 'Raw waveforms already show the mechanism: the stripe is later when deeper, and neighbours carry copies',
           'Strip × time matrices of single det3 cosmics (opener candidates for “Why hit times fail”)',
           ['event_waveforms', 'event_raw_x_y', 'angle_bias_schematic', 'event_3d_hits_vs_m3'],
           '<p>event_waveforms is already in the paper skeleton. The 3-D display and the angle-bias schematic are hits-era (July 12–18) '
           'and carry hit positions.</p>', w=800, h=290, short='Hits fail: displays')


def s_walk(D):
    cells = [fig('forward_fit_walkthrough', w=312, h=210, k=k, label=False) for k in range(10)]
    rows = [sd.row(*cells[i:i + 5], gap=20) for i in range(0, 10, 5)]
    body = (sd.title('One muon through the fit, step by step: ten panels from the published forward-fit note',
                     'forward-fit-det3 note, September, R&D model; a candidate storyboard for the forward-model diagram') + ''.join(rows) +
            sd.p('Hover for provenance. Pick the panels worth redrawing on the physical RC model.', 22, sd.MUT))
    D.slide('walk', body, '<p>Images extracted from <code>dylan-cern-site/notes/forward-fit-det3.html</code> into '
            '<code>collection/2_hits_fail/forward_fit_walkthrough_NN.png</code>. Slide titles of the note: What comes in; The three numbers; '
            'Geometry decides where each slice lands; The resistive kernel copies charge sideways, late; One slice → five waveforms; '
            'The charges are solved, not searched; What each strip’s prediction is made of; The whole window at once; What is actually searched; '
            'What it cost; Does it work.</p>', short='Fit walk-through')


def s_model(D):
    two_by(D, 'model', 'The forward model reproduces a strip’s waveform from its own charge plus its neighbours’ delayed copies',
           'det3, July R&D model and an mpgd26 render (old bundle); the diagram on the production model does not exist yet',
           ['event_fit', 'share_decompose', 'templates_perplane', 'vtension_resolution'],
           '<p>The calibration (template per plane, kernel, v, t0) is per detector and per run condition. The two August–September pictures '
           '(share_decompose, event_fit) are the best “what the model does” figures; remake on pass 1. '
           'Pass-1 calibration closure (G1) passes on all five chambers: <code>rerun/calib/G1_RESULT_2026-10-30.md</code>. '
           'Bow maps and closure are in the collection too.</p>',
           secs={'3_model'}, w=800, h=290, short='Model')


def s_floor(D):
    two_by(D, 'floor', 'The floor is 0.30–0.35° and set by diffusion; the measured ~1° sits 3–6× above it',
           'Toy, det3 (bench), perfect model at 60 and 20 ns depth bins; correction of the July "floor is physics" claim',
           ['floor_vs_angle_det3', 'floor_breakdown_det3', 'floor_bias_grid_det3', 'floor_fleet'],
           '<p>Closed form σθ ≈ D_T√(6(1+f)/(N·G)) gives 0.29° on det3. 60 ns depth bins bias steep tracks shallow (−0.08/−0.23/−0.51° at 25/30/40°); '
           '20 ns bins are flat to 45°. Every measured imperfection together reaches 31–46 % of the measured resolution. '
           'Pass-1 remainder: 0.56–0.65° on det2/3/7 X. The July claim of 0.02° electronics plus ~1° centroid jitter was withdrawn on 2026-10-27.</p>',
           secs={'4_floor'}, w=800, h=290, short='Floor')


def s_tiers(D):
    two_by(D, 'tiers', 'Same events, five chambers: waveforms cut the hits chain’s angle error on every chamber, 2.0–2.6° to 0.73–1.56°',
           'Pass 1 (left) and the July det3 three-way prototype (right, R&D model): the tier figure would extend the right panel to H0, H1, H2a, H2b, H2c, W',
           ['fleet_H0_vs_W', 'three_way_comparison'],
           '<p>Pass 1: angle s68 H0 2.0–2.6° vs W 0.73–1.56°; position core σ 550–805 vs 430–620 µm; efficiency within 5 mm, 1–5 points. '
           'July det3 (2000 events, |tan| 0.10–0.40): σθ X/Y production 4.97/5.74°, hybrid 1.58/1.53°, slope remap 1.51/1.86°, forward model 1.06/1.10° '
           '(do not quote: R&D model). H1, H2a, H2b, H2c have no pass-1 product.</p>',
           secs={'5_hits_vs_waveforms'}, w=800, h=560, short='Hits vs waveforms')


def s_known(D):
    two_by(D, 'known', 'At a known 25.4° mount the inclined view reads 0.981 ± 0.03 of the stage angle with the column-duration v',
           'det4 at SPS, pass-1 final tables; 0.927 with the registry v; the 15.5° rows have no column v',
           ['known_angle', 'sps_resolution'],
           '<p>Only |tan| is defined (COORDINATES.md 4b). The geometry figure in the paper skeleton is not a geometry schematic '
           '(it is a Y-view bar chart): a real schematic is a gap. Mount angle appears as 25.4, 25.6 and 25.64° in different files.</p>',
           secs={'7_known_angle'}, w=800, h=560, short='Known angle')


def s_fleet(D):
    two_by(D, 'fleet', 'On pass 1 the waveform chain gives 0.9–1.5° on four chambers, flat to 20°; det4 is the exception',
           'σθ against reference angle per chamber (left), bench position resolution against M3 (right); Paper I figures reused',
           ['tracking_perf_all', 'bench_position_res'],
           '<p>Headline s68 X/Y (°): det2 1.045/1.125, det3 0.991/0.901, det4 2.412/2.076, det6 1.256/1.506, det7 1.232/1.461. '
           'Position core σ vs M3 (µm, X/Y, upper bounds since multiple scattering is not subtracted): det2 456/444, det3 447/443, det4 663/526, det6 348/382, det7 434/488. '
           'Cost table (R10) is not made.</p>',
           secs={'8_fleet'}, w=800, h=560, short='Fleet')


def s_outline(D):
    rows = [['1', 'Introduction', 'claim schematic', 'write'],
            ['2', 'Sharing kernel', 'kernel_forms, angled_kernel', '<b style="color:#2f8a5b">ready</b>'],
            ['3', 'Why hit times fail', 'ladder compression, five chambers', '<b style="color:#c63d4f">remake</b>'],
            ['4', 'Forward model, calibration', 'model diagram, templates, fit example', '<b style="color:#c99318">partial</b>'],
            ['5', 'How far from the floor', 'floor_vs_angle, breakdown', '<b style="color:#c99318">partial</b>'],
            ['6', 'Hits vs waveforms', 'tier benchmark, fleet', '<b style="color:#c63d4f">open (core)</b>'],
            ['7', 'What a hit readout must record', 'threshold × neighbours', '<b style="color:#c63d4f">open</b>'],
            ['8', 'Known angle in the beam', 'known_angle + schematic', '<b style="color:#c99318">partial</b>'],
            ['9', 'Fleet, cost, outlook', 'tracking_perf_all, cost table', '<b style="color:#c99318">partial</b>'],
            ['10', 'Conclusions', '', '']]
    body = (sd.title('Proposed outline: ten sections, about ten figures, for NIM A',
                     'Merge candidates for length: 7 into 6, and 8 into 9') +
            sd.table(['§', 'section', 'main figures', 'state'], rows, size=27, widths=[70, 560, 760, 240], align=['left'] * 4))
    D.slide('outline', body, open(HERE / 'OUTLINE_PLAN_2026-11-05.md').read().replace('\n', '<br>') if False else
            '<p>Full text, with what changed since the <code>.tex</code> skeleton: <code>papers/paper2/OUTLINE_PLAN_2026-11-05.md</code>. '
            'Differences from the skeleton: the floor section is reframed (the July claim was withdrawn), the kernel and fleet numbers are the pass-1 ones, '
            'and readout requirements is a software study on stored waveforms. No new data is possible.</p>', short='Outline')


def s_plan(D):
    steps = [dict(label='R1', sub='hit tiers H0/H1 on pass 1: ladder figure', color=sd.RED, tip='M. hits_arm.py already builds H0 on the same events'),
             dict(label='R2', sub='H2a/H2b refit on pass 1', color=sd.ORANGE, tip='M. hybrid constants were fitted on hits: refit, do not reuse'),
             dict(label='R3', sub='H2c: forward fit on hit lists', color=sd.PURPLE, tip='L, new work. The key new result'),
             dict(label='R5', sub='tier benchmark figure', color=sd.GREEN),
             dict(label='R4', sub='threshold × neighbours', color=sd.BLUE, tip='re-extract hits from stored waveforms; no new data')]
    body = (sd.title('The plan: the ladder figure first, then the tiers, then the readout scan; H2c decides how strong the message is',
                     'Work order by what unblocks most (ids from mx_june_cosmic_qa/PAPER_PLAN.md)') +
            sd.flow(steps, size=28) +
            sd.row(bullets(['<b>Diagrams (S):</b> forward-model diagram, known-angle geometry, a sharing schematic with c2 < c1.',
                            '<b>Floor (S):</b> paper-style figure from ws_floor and the pass-1 budget; write the withdrawn claim into §5.',
                            '<b>Write-up (S):</b> calibration protocol and traps (R9), cost table (R10), X asymmetry as the stated mismatch (X1).'], 26),
                   bullets(['<b>Limitations to state, not chase:</b> τ and c2 are bounds; X-width excess and X asymmetry unexplained; '
                            '0.56–0.65° remainder; |tan| only at one mount angle; the VMM comparison is an emulation.'], 26), gap=64))
    D.slide('plan', body, '<p>Sizes and dependencies: work list in <code>mx_june_cosmic_qa/PAPER_PLAN.md</code>. '
            'The dataset is final, so every item is computation on existing waveforms or simulation.</p>', short='Plan')


def s_gaps(D):
    rows = [[i['name'], i['caption'], f"<code>{i['producer']}</code>"] for i in CAT['items'] if i['section'] == '9_gaps']
    body = (sd.title('Nine figures the paper needs do not exist: five are placeholders, four were never planned',
                     'Gaps found while cataloguing') +
            sd.table(['figure', 'what is missing', 'start from'], rows, size=20, widths=[310, 900, 450], align=['left'] * 3))
    D.slide('gaps', body, '<p>Placeholders in <code>papers/paper2/figures.toml</code>: ladder_compression, forward_model, toy_floor, tiers_benchmark, '
            'threshold_neighbours. Two entries there are also wrong: known_angle_geometry is not a geometry schematic, and the toy_floor note carries the withdrawn claim.</p>',
            short='Gaps')


def s_close(D):
    body = (sd.kicker('What this does not settle · decisions') +
            '<h1 style="font-size:60px;font-weight:600;line-height:1.1;color:#eef0f3">Six decisions, and what the collection cannot tell us</h1>' +
            sd.row(bullets(['<b>H2c in scope?</b> The one new-result item and the largest piece of work.',
                            '<b>Merge 7→6 and 8→9</b> for NIM A length?',
                            '<b>Headline chambers:</b> all five, or the calibration-row subset of fleet.pdf?',
                            '<b>Floor:</b> report the 0.30–0.35° diffusion floor and the 3–6× gap in Paper II, or keep it in Paper I?',
                            '<b>60 ns depth-bin bias</b> (−0.5° at 40°): a paragraph, given n_TOF steep tracks?',
                            '<b>Mount angle:</b> 25.4, 25.6 or 25.64° in the text?'], 27),
                   bullets(['Whether the ladder compression holds on the other four chambers: only det3 has it, from July.',
                            'Whether a sharing-aware fit on hits (H2c) closes the gap. Not run.',
                            'Anything about a VMM front-end beyond emulation.',
                            'The ratings are mine: I opened about a third of the images; the rest I judged from the producer and its notes.'], 27), gap=64) )
    D.slide('close', body, '<p>Decisions numbered as in <code>papers/paper2/OUTLINE_PLAN_2026-11-05.md</code>.</p>', dark=True, short='Decisions')


def build(out):
    D = sd.Deck('MX17 Paper II: figure collection and outline',
                'Every figure and metric in the repo that could serve the reconstruction paper, an outline, a plan, the gaps')
    for f in (s_cover, s_story, s_inventory, s_kernel, s_kernel2, s_fail, s_fail2, s_walk, s_model, s_floor, s_tiers,
              s_known, s_fleet, s_outline, s_plan, s_gaps, s_close):
        f(D)
    meta = dict(title='MX17 Paper II: figure collection and outline',
                summary='Figures, metrics and diagrams found for the reconstruction paper, sorted by section, with an outline, work plan and gaps.',
                tags='x17, mx17, paper', date=dt.date.today().isoformat())
    return D.write(out, meta, footer=f'Built {dt.datetime.now():%Y-%m-%d %H:%M} by nTof_x17/papers/paper2/make_collection_deck.py')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'collection_deck' / 'mx17-paper2-collection.html'))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    p = build(a.out)
    print(p, f'{p.stat().st_size / 1e6:.1f} MB')
