#!/usr/bin/env python3
"""make_xasym_plan_deck.py -- short slide note: the contingency plan if the X neighbour asymmetry stays unexplained
(XASYM_CONTINGENCY.md). How big it is, why we do not absorb it into the calibration, and what each workstream does.

Maps, K and the leave-one-out validation come from wft/xasym_maps.json; the re-alignment resolutions are the
MX17_Geant asym_deck FINDINGS eighth round §6 table (realign_xasym.sh), copied below with that provenance.

    make_xasym_plan_deck.py [--out ~/x17/cosmic_bench/cloud_basics/xasym_plan/mx17-xasym-plan.html]
    then: python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <out> --slug mx17-xasym-plan --force --deploy
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.expanduser(os.environ.get('SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd                                          # noqa: E402
from slidedoc import BLUE, ORANGE, RED, GREY, GREEN, INK, MUT, DBLUE, DRED, DGREEN, DGREY  # noqa: E402

CB = os.path.expanduser('~/x17/cosmic_bench/cloud_basics')
M = json.load(open(os.path.join(REPO, 'wft', 'xasym_maps.json')))
K, KERR, CELL, NC = M['K'], M['K_err'], M['cell_mm'], M['ncell']
#: X resolution (mm) before / after the correction and a full re-alignment; MX17_Geant
#: scratchpad/2026-10-19/asym_deck/FINDINGS.md, eighth round §6 (production 01_alignment.py into scratch)
REALIGN = [('det3 6-22 (scale 0.96)', 0.457, 0.453), ('det3 6-27 (0.44)', 0.553, 0.554),
           ('det2 6-22 long', 0.563, 0.563), ('det2 6-22 longer', 0.546, 0.546)]
DECISION_DAY = '2026-10-23'
BOARDS = [('det3 June-15', 'chamber A', BLUE), ('det2', 'chamber B', GREEN)]


def carried(build):
    """Production X position bias, µm: K · dx, board mean removed (the alignment absorbs it)."""
    G = np.array([[np.nan if v is None else v for v in r] for r in M['builds'][build]['dx_um']], float) * K
    return G - np.nanmean(G)


def cover(D):
    nb = len(M['builds'])
    body = (sd.kicker('MX17 · X neighbour asymmetry · contingency') +
            sd.title(f'If it is not explained by {DECISION_DAY}: freeze the symmetric model and report the asymmetry '
                     'as a measured detector property') +
            sd.row(sd.bignum('5 / 5', 'June boards carry it', DBLUE, 'X view only; Y flat everywhere'),
                   sd.bignum('40', 'routes tried', DGREY, 'ledger in MX17_Geant asym_deck/FINDINGS.md'),
                   sd.bignum(f'{nb}', 'boards with a validated correction', DGREEN, 'det2 and det3 June-15 (chambers B, A)'),
                   gap=64))
    D.slide('cover', body, f'''<p>One more day on the mechanism ({DECISION_DAY}). If it is still open at the end of that
day, this plan replaces it. Nothing in the plan needs new data.</p>
<p>Written up in <code>sps_beam_test_26/analysis/cloud_basics/XASYM_CONTINGENCY.md</code>.</p>''',
            dark=True, short='plan')


def size(D):
    vmax = 110
    panels = []
    for b, ch, _c in BOARDS:
        G = carried(b)
        P = sd.Plot(400, 470, x=(0, NC * CELL), y=(0, NC * CELL), xlabel='x (mm)', ylabel='y (mm)',
                    title=f'{b} = {ch}', margin=(24, 20, 80, 96))
        P.xticks([(v, f'{v:.0f}') for v in (0, 100, 200, 300, 400)])
        P.yticks([(v, f'{v:.0f}') for v in (0, 100, 200, 300, 400)])
        cells = []
        for i in range(NC):
            for j in range(NC):
                v = G[i, j]
                tp = (f'{b}, cell x {i * CELL:.0f}–{(i + 1) * CELL:.0f}, y {j * CELL:.0f}–{(j + 1) * CELL:.0f} mm\n'
                      + ('empty cell' if np.isnan(v) else f'carried bias {v:+.0f} µm'))
                cells.append((i * CELL, (i + 1) * CELL, j * CELL, (j + 1) * CELL, None if np.isnan(v) else float(v), tp))
        P.cells(cells, -vmax, vmax, sd.DIVERGE, missing='#e4e0d8', gap=1)
        panels.append(P.svg(f'{b} carried X bias map'))
    cb = sd.colorbar(130, 470, -vmax, vmax, [(v, f'{v:+d}') for v in (-100, -50, 0, 50, 100)],
                     'bias (µm)', sd.DIVERGE)
    a3 = float(np.nanmax(np.abs(carried(BOARDS[0][0]))))
    a2 = float(np.nanmax(np.abs(carried(BOARDS[1][0]))))
    dres = max(abs(b - a) for _n, b, a in REALIGN) * 1e3
    bars = sd.hbars([
        ('det2 bias (max cell)', a2, GREEN),
        ('det3 bias (max cell)', a3, BLUE),
        ('resolution change', max(dres, 1.0), RED),
        ('SPS intrinsic (det4)', 176, GREY),
        ('bench X resolution', REALIGN[0][1] * 1e3, GREY),
    ], 600, width=200, log=True, vmin=1, label_w=250, fmt=lambda v: f'{v:.0f} µm', size=22, h=30)
    body = (sd.title('The asymmetry moves X positions by up to ~0.1 mm and does not change the resolution',
                     f'Bias the production X position carries (K = {K:.2f} ± {KERR:.2f} × raw centroid map), '
                     '50 mm cells, bench cosmics') +
            sd.row(panels[0], panels[1], cb, sd.col(sd.p('µm, log scale', 22, MUT), bars,
                                                        sd.p('resolution change: after correction + re-alignment, 4 bench runs', 20, MUT), gap=14), gap=28))
    rows = ''.join(f'<tr><td>{n}</td><td>{b:.3f}</td><td>{a:.3f}</td></tr>' for n, b, a in REALIGN)
    val = ''.join(f'<tr><td>{n}</td><td>{v["chi2_before"]:.0f} → {v["chi2_after"]:.0f}</td>'
                  f'<td>{v["rms_before"]:.0f} → {v["rms_after"]:.0f}</td><td>{v["ncell"]}</td></tr>'
                  for b in M['builds'] for n, v in M['builds'][b]['validation'].items())
    D.slide('size', body, f'''<p>Maps: <code>wft/xasym_maps.json</code> (raw-ADC 3-strip centroid bias per cell), times
K, the measured transfer to the forward-model position against M3. Board mean removed: the alignment absorbs it
(and a rotation). Grey cells are empty.</p>
<p><b>Resolution after correction and full re-alignment</b> (X, mm):</p>
<table><tr><th>run</th><th>original</th><th>corrected</th></tr>{rows}</table>
<p><b>Leave-one-out validation</b> of the M3 x-residual cell map (offset + rotation removed):</p>
<table><tr><th>run</th><th>cell χ²</th><th>cell rms (µm)</th><th>cells</th></tr>{val}</table>
<p>A 30–100 µm cell bias adds in quadrature to 450–560 µm, so it shows only as a position-dependent offset. That offset
matters for anything comparing positions across the board or between chambers (vertexing, inter-chamber alignment),
not for resolution. Angle: the quoted 1.15–2.51° already includes the asymmetry, since it is in the data. The angle
scale slopes do not follow it either: det3, the strongest board, has the slope closest to 1 (0.963; §38).</p>
<p>det4/det6/det7 also carry a field but their maps are too sparse to correct (det4 got worse, 162 → 174). Chamber C
(det6) therefore has no correction.</p>''', foot='Source: wft/xasym_maps.json; MX17_Geant asym_deck FINDINGS routes 22–24, 40; cloud_basics FINDINGS §38, §42')


def options(D):
    st = [
        dict(label='✗ Model it in the kernel', sub='a one-sided ±1 term per board<br>no mechanism: best sim ×1.8, data need ×2.5–5', color=RED,
             tip='No mechanism reproduces the size: best Y-compatible sim ×1.8 vs ×2.5–5 needed.\n'
                 'Without a mechanism it is a free fit parameter per board and per run condition.'),
        dict(label='✗ Empirical correction as nominal', sub='wft.xasym on every chamber<br>maps for det2, det3 only; chamber C has none', color=ORANGE,
             tip='Validated on det2 and det3 only; det4/6/7 maps too sparse (chamber C uncorrectable).\n'
                 'Amplitude scales with cloud width, so each run condition needs its own scale.'),
        dict(label='✓ Symmetric model + systematic', sub='freeze r06; xasym as a ± variation; write it up<br>resolution unchanged; positions ±0.1 mm', color=GREEN,
             tip='Resolution unchanged within 5 µm; positions shift ≤ ~0.1 mm.\n'
                 'Quote the difference where positions are compared across a board or between chambers.'),
    ]
    body = (sd.title('Do not absorb it: freeze the symmetric model and carry the correction as a systematic',
                     'Three ways to handle an effect we cannot explain') +
            sd.flow(st, size=30, arrow='·') +
            sd.callout('The cost is fit quality (X χ², the σ0_x tension) and understanding, not the numbers we quote.',
                       GREEN, 30))
    D.slide('options', body, '''<p><b>Why not a kernel term.</b> The best simulated mechanism consistent with the Y view
(700 µm strips + funnelling + across-strip anisotropy at 5 MΩ/sq) gives ×1.8 of the sim baseline; the data need
×2.5–5. ρ_s below 5 MΩ/sq breaks the Y view. A fitted term would be calibrated per board and per run condition with
no physics behind it.</p>
<p><b>Why not the correction as nominal.</b> It exists for det2 and det3 June-15 only, needs the run's amplitude scale
(det3 sessions 0.44–1.57) and a re-alignment, and buys no resolution.</p>
<p><b>Where it shows.</b> The symmetric footprint cannot follow the ±1/±2 residual asymmetry (§38): the X χ² structure,
and the per-event σ0_x 0.20–0.25 vs stacked 0.35–0.43.</p>''', short='decision')


def plan(D):
    def card(head, color, items, tip=None):
        li = ''.join(sd.p('• ' + t, 25) for t in items)
        return sd.card(sd.p(head, 30, color, 600) + li, tip=tip)
    body = (sd.title('What each workstream does instead',
                     'No new data; one simulation check bounds the angle cost') +
            sd.row(
                card('Detector characterization', BLUE, [
                    'Lift the kernel-freeze hold: freeze r06 (paper 1 C2)',
                    'σ0_x 0.25 per event (§38)',
                    'wft.xasym = systematic variation, not nominal',
                    'det3 D_T excess + asymmetry → one det3 limitation; gas transport validated on det2/det4']),
                card('n_TOF', ORANGE, [
                    'Nominal reconstruction unchanged',
                    'Chamber A: session scale from n_TOF raw ADC; xasym as ± in capsule pointing / alignment / vertexing',
                    'Chamber C: flat ±0.1 mm X systematic',
                    'Same-chamber pairs: common bias cancels (mm apart in 50 mm cells)']),
                card('Papers', GREEN, [
                    'Paper 1: a subsection on the X-only, pitch-locked, board-fixed field; exclusions; leading hypothesis; '
                    'position effect; mask never obtained',
                    'Paper 2: model is symmetric in X; this is the main known mismatch; quote its angle cost']),
                gap=36) +
            sd.card(sd.p('<b>The one check (simulation, ~1 day):</b> inject the measured one-sided ±1 term '
                         '(aQ ≈ 0.165, ~110 ns late) into the paper-2 physics-floor toy, reconstruct with the symmetric '
                         'model → the angle bias and resolution it costs. Expected: a few hundredths of a degree '
                         '(§38 head-on +0.04°), unmeasured until it runs.', 26), bg='#eef4ec'))
    D.slide('plan', body, '''<p><b>Stop:</b> the mechanism simulations (strip width, funnelling, ρ_s, anisotropy) and the
strip-mask question. No result above needs them.</p>
<p>Chamber map: A = mx17_3 (det3 June-15), B = mx17_2 (det2), C = mx17_6 (det6), D = mx17_7 (det7);
<code>sept26_prelim_analysis/capsule_y.py</code>.</p>
<p>Correction API: <code>wft.xasym.correct_x(x_p0, y_p0, build, scale)</code>; re-align on the corrected positions,
since the map has a board-mean part.</p>''', short='plan')


def closing(D):
    body = (sd.kicker('What this does not rule out') +
            sd.col(*[sd.p('• ' + t, 30, sd.DINK) for t in [
                'An angle cost larger than §38 suggests, until the toy injection runs.',
                'A position bias on chamber C (det6) that we cannot map: carried as a flat ±0.1 mm.',
                'That det3’s D_T excess is the same effect — a plausible link, not tested.',
                'A mechanism that would also change the Y view or the gap; nothing seen there so far.',
            ]], gap=18) +
            sd.kicker('Decision') +
            sd.p(f'{DECISION_DAY}: one more day on the mechanism. Unsolved at the end of it → this plan.', 32, DGREEN, 600))
    D.slide('closing', body, dark=True, short='limits')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(CB, 'xasym_plan', 'mx17-xasym-plan.html'))
    a = ap.parse_args()
    D = sd.Deck('X asymmetry contingency', 'MX17: what to do if the X neighbour asymmetry stays unexplained')
    for f in (cover, size, options, plan, closing):
        f(D)
    D.write(a.out, note_meta=dict(
        title='MX17 X asymmetry: the contingency plan',
        summary='If the X neighbour asymmetry is not explained by 2026-10-23: freeze the symmetric r06 kernel, carry the '
                'det2/det3 position correction as a systematic, and write the asymmetry up as a measured detector '
                'property. It shifts X positions by up to ~0.1 mm and leaves the resolution unchanged.',
        tags='X17, cosmic bench, micromegas, detector model', date='2026-10-22'),
        footer='Built by sps_beam_test_26/analysis/cloud_basics/make_xasym_plan_deck.py from wft/xasym_maps.json.')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
