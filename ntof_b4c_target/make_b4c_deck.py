#!/usr/bin/env python3
"""make_b4c_deck.py -- slide note: the July B4C night, and why it cannot show pairs.

    python -m ntof_b4c_target.expect        # expect.json      (sim rates)
    python -m ntof_b4c_target.deck_data     # deck_data.json   (per-line, ³He, data census)
    python -m ntof_b4c_target.make_b4c_deck # OUT/b4c-target-pairs.html

Every number on a slide is read from expect.json / deck_data.json or the
Geant4 capture-model check (data/hp3k.json, data/pe3k.json).
"""
import collections
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

from .expect import OUT  # noqa: E402

HERE = Path(__file__).resolve().parent
E = json.loads((OUT / 'expect.json').read_text())
D = json.loads((OUT / 'deck_data.json').read_text())
HP = json.loads((HERE / 'data' / 'hp3k.json').read_text())
PE = json.loads((HERE / 'data' / 'pe3k.json').read_text())

C = E['constants']
NPULSE_EXP = C['EXPOSURE_PULSES']          # nominal-pulse equivalents of run_39 + run_40
PP = D['per_pulse']
B4C, HE3, AL, X17C, E0C, DATA = sd.ORANGE, sd.BLUE, sd.GREY, sd.GREEN, sd.PURPLE, sd.INK
LINES = D['lines']


def s(v, d=1):
    return sd.sci(v, d)


# --------------------------------------------------------------------------- #
# derived numbers
# --------------------------------------------------------------------------- #
def line_rate(e):
    """IPC pairs per nominal pulse made by line e (str key)."""
    L = LINES[e]
    return C['CAPT_PER_PULSE'] * L['intensity'] / 100 * L['alpha']


b4c_stage = {k: sum(line_rate(e) * LINES[e][k] for e in LINES) for k in ('p_split', 'p_leg2', 'p_pair')}
b4c_made = sum(line_rate(e) for e in LINES)
he = D['he3']['ipcM1']
he3_stage = {k: PP['he3_ipc'] * he[k] for k in ('p_split', 'p_leg2', 'p_pair')}
eff_b4c = b4c_stage['p_pair'] / b4c_made
eff_he3 = he['p_pair']
n_pair_exp = b4c_stage['p_pair'] * NPULSE_EXP
census = D['census']
n_trig = sum(r['ev'] for r in census if r['run'] in ('run_39', 'run_40'))   # events with a hit
n_trig_all = sum(D['timeline'][r].get('scint', 0) for r in ('run_39', 'run_40'))   # all recorded triggers
n_multi = sum(r['ge2'] for r in census if r['run'] in ('run_39', 'run_40'))
hours = sum(r['hours'] for r in census if r['run'] in ('run_39', 'run_40'))

deck = sd.Deck('B4C target: why no pairs',
               'The July 2026 B4C night at n_TOF EAR2: what data we have, what Geant4 says it should '
               'contain, and the physics reason its pairs are invisible to the trigger.')

# --------------------------------------------------------------------------- #
# 1  cover
# --------------------------------------------------------------------------- #
body = (sd.kicker('n_TOF EAR2 · X17 · run_39 + run_40 · 15 July 2026')
        + sd.title(f'The B4C disc made {b4c_made:.1f} internal pairs per pulse.<br>'
                   f'Our trigger keeps 1 in {1 / eff_b4c:,.0f} of them.')
        + sd.row(
            sd.bignum(f'{n_trig_all / 1e3:.0f} k', 'scintillator triggers on B4C', sd.DGREY,
                      f'{hours:.1f} h, runs 39 + 40, 2635 beam pulses', tip='DREAM events in the B4C runs with the scintillator trigger (Singles OR Doubles); run_38 was flash-only.'),
            sd.bignum(f'{b4c_made:.1f}', 'IPC pairs made per pulse', '#e9a46b',
                      f'{PP["b4c_ng"]:.0f} ¹⁰B(n,γ) captures × 4.1×10⁻³ pairs each',
                      tip='Nominal 7×10¹² pulse; EAR2 flux 2.2 meV–2 eV, 98.4 % absorbed in the disc.'),
            sd.bignum(f'{n_pair_exp:.2f}', 'pair events expected in the data', sd.DRED,
                      'two trigger legs + both chambers, whole B4C exposure',
                      tip='Geant4, production thresholds (SiPM 0.5 MIP, plastic 0.9 MIP), 3107 nominal-pulse equivalents.'),
            gap=64)
        + sd.p('The leptons of ≤ 11.45 MeV transitions stop before the plastic scintillators. '
               'The ³He trigger was built for 20.6 MeV.', 30, sd.DINK))
deck.slide('cover', body, dark=True, short='cover', notes=(
    '<p>Question (Stephan, 2026-10-19): the early July running had other targets in the beam; '
    'the B4C night ran with the scintillator trigger, so could it give pairs?</p>'
    '<p>Answer: no. This note shows the data that exist, the Geant4 expectation, and the reason, '
    'and compares with the ³He running.</p>'
    '<p>Built by <code>nTof_x17/ntof_b4c_target/make_b4c_deck.py</code> from <code>expect.json</code> '
    'and <code>deck_data.json</code> (<code>/media/dylan/data/x17/b4c_target/</code>).</p>'))

# --------------------------------------------------------------------------- #
# 2  setup diagram
# --------------------------------------------------------------------------- #
W, H = 820, 690
cx, cy, k = 400, 345, 7.0            # px per cm
layers = [  # (name, r_in cm, r_out cm, colour, tip)
    ('Micromegas (30 mm drift)', 25.0, 28.5, '#8fb3dc', 'MX17 resistive-strip Micromegas: 30 mm drift, 40 × 36 cm. Tracks the leptons.'),
    ('SiPM wall', 31.0, 32.0, '#5f8f6b', 'SiPM-read scintillator bars, 2.5 cm pitch. Trigger leg needs one bar ≥ 0.5 MIP (0.475 MeV).'),
    ('liquid-scintillator vessel', 34.0, 37.5, '#c9b38a', '2.6 mm CFRP shell, ~2 cm LAB slab with bulged faces. Out of the trigger since 14 July: here it is only material.'),
    ('plastic bars (2 cm)', 39.0, 41.0, '#9a7fc4', 'Two 20 × 30 × 2 cm PVT bars per arm. Trigger leg needs ≥ 0.9 MIP (3.0 MeV) in one bar.'),
]
g = []
for ang, lab in ((0, 'D (+x)'), (180, 'B (−x)'), (90, 'C'), (270, 'A')):
    for name, r0, r1, col, tp in layers:
        half = 20.0 * k
        if ang in (0, 180):
            sx = 1 if ang == 0 else -1
            x0 = cx + sx * r0 * k; x1 = cx + sx * r1 * k
            g.append(f'<rect x="{min(x0, x1):.0f}" y="{cy - half:.0f}" width="{abs(x1 - x0):.0f}" height="{2 * half:.0f}" '
                     f'fill="{col}" rx="3"{sd.tipattr(name + ": " + tp)}/>')
        else:
            sy = 1 if ang == 90 else -1
            y0 = cy + sy * r0 * k; y1 = cy + sy * r1 * k
            g.append(f'<rect x="{cx - half:.0f}" y="{min(y0, y1):.0f}" width="{2 * half:.0f}" height="{abs(y1 - y0):.0f}" '
                     f'fill="{col}" rx="3"{sd.tipattr(name + ": " + tp)}/>')
# arm letters
g.append(sd.T(cx + 44 * k, cy + 8, 'D', 26, sd.INK, weight=600) + sd.T(cx - 44 * k, cy + 8, 'B', 26, sd.INK, weight=600)
         + sd.T(cx, cy + 45 * k, 'C', 26, sd.INK, weight=600) + sd.T(cx, cy - 43 * k, 'A', 26, sd.INK, weight=600))
# a B4C pair: e+ stops in the LS vessel of arm D, e- reaches but under-deposits in the plastic of arm A
g.append(sd.arrow(cx, cy, cx + 35.5 * k, cy - 6 * k, sd.RED, 3.5, 14))
g.append(sd.T(cx + 35.5 * k + 6, cy - 6 * k - 14, '✕', 30, sd.RED, 'start'))
g.append(sd.arrow(cx, cy, cx - 4 * k, cy - 36 * k, sd.RED, 3.5, 14))
g.append(sd.T(cx - 4 * k - 14, cy - 36 * k - 4, '✕', 30, sd.RED, 'end'))
g.append(sd.T(cx, cy + 125, 'e⁺e⁻ from a 7 MeV line', 22, sd.RED, 'middle',
              tip='Typical ¹¹B internal pair: each lepton carries ~3 MeV and stops in the LS vessel or in the first mm of plastic.'))
# disc and beam
g.append(f'<circle cx="{cx}" cy="{cy}" r="{2.5 * k:.0f}" fill="{B4C}"{sd.tipattr("B4C disc, 2.5 mm thick, Ø unknown (Ø100 and Ø50 mm simulated); beam footprint r ≤ 22 mm.")}/>')
g.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{sd.BG}" stroke="{sd.INK}" stroke-width="2.5"/><circle cx="{cx}" cy="{cy}" r="2.5" fill="{sd.INK}"/>')
g.append(sd.T(cx + 26, cy + 40, 'B4C disc', 21, sd.INK, 'start'))
g.append(sd.T(cx + 26, cy + 64, 'beam into page', 21, sd.INK, 'start'))
diagram = sd.svg(W, H, ''.join(g), 'top-down schematic of the four arms around the B4C disc')
leg = sd.legend([(n, c, 'box') for n, _a, _b, c, _t in layers], 24)
side = sd.col(
    sd.p('Top-down view, beam into the page. Four identical arms, '
         'each a tracking chamber in front of the trigger scintillators.', 26),
    leg,
    sd.callout(f'A {sd.term("trigger leg", "One arm with a SiPM bar ≥ 0.5 MIP AND a plastic bar ≥ 0.9 MIP within 20 ns (N1081B M3 sector AND).")} '
               f'needs the lepton to cross the chamber, the wall and the LS vessel, and still deposit 3 MeV in 2 cm of plastic.', sd.INK),
    sd.p('Hover the dotted words, the layers, bars and points for numbers and sources.', 22, sd.MUT),
    gap=28, w=720)
deck.slide('setup', sd.title('Both leptons must punch through to the plastics',
                             'The July set-up around the B4C disc (schematic, roughly to scale)')
           + sd.row(diagram, side, gap=60, align='center'),
           short='set-up', notes=(
    '<p>Geometry as in <code>MX17_Full_Geant</code> (July survey): Micromegas front face 25 cm from the beam axis, '
    'SiPM wall behind it, the liquid-scintillator vessel, then two 2 cm plastic bars. The layer radii drawn are rounded.</p>'
    '<p>The trigger in run_39/40 was N1081B mode 3: Singles (any one leg) OR Doubles (two legs), gated 30 ms after the PS '
    'pulse. A "pair event" in this note is two legs in two arms with charge in both of those arms\' chambers.</p>'
    '<p>Thresholds used in the simulation are the production ones (SiPM 0.5 MIP, plastic 0.9 MIP; MIP = 0.475 / 3.349 MeV '
    'from the 2 cm calibration run). In run_39 the wall thresholds were re-tuned through the night and the plastics sat at '
    '−15 mV until 09:04; slide 10 scans the plastic threshold.</p>'))

# --------------------------------------------------------------------------- #
# 3  timeline of targets and runs
# --------------------------------------------------------------------------- #
t0 = datetime(2026, 7, 10, 0, 0)


def hrs(ts):
    return (datetime.fromisoformat(ts) - t0).total_seconds() / 3600


P = sd.Plot(1664, 600, x=(0, 144, 'lin'), y=(10, 1e6, 'log'),
            xlabel='date (July 2026)', ylabel='recorded DREAM events per run', margin=(70, 30, 92, 120))
P.xticks([(24 * d, f'{10 + d} Jul') for d in range(0, 7)])
P.yticks(sd.log_ticks(1, 6))
bands = [(0, 17.75, 'none', '#e3e0d8'), (17.75, 111, '“Marex” target', '#d7d3ca'),
         (111, 116, '', '#f5f3ee'), (116, 132, 'B4C', '#f3d2b8'), (132, 144, '³He', '#cfdcec')]
for a, b, lab, col in bands:
    P.raw(f'<rect x="{P.X(a):.1f}" y="{P.y0}" width="{P.X(b) - P.X(a):.1f}" height="{P.ph}" fill="{col}"/>', back=True)
    P.raw(sd.T((P.X(a) + P.X(b)) / 2, P.y0 - 14, lab, 24, sd.INK, weight=600), back=False)
fam_col = {'scint': sd.INK, 'flash': '#9aa1ad', 'random': '#c3c7cf', 'other': '#c3c7cf'}
for run, d in sorted(D['timeline'].items(), key=lambda kv: kv[1]['stamp']):
    x = hrs(d['stamp'])
    if not 0 <= x <= 144:
        continue
    off = 0
    for fam in ('scint', 'flash', 'random', 'other'):
        v = d.get(fam, 0)
        if v:
            tp = f'{run} · {fam} trigger · {v:,} events · starts {d["stamp"]}'
            P.vbar(x + off, v, 9, fam_col[fam], tip=tp, base=10)
            off += 1.0
    if run in ('run_39', 'run_40', 'run_38', 'run_41', 'run_37'):
        P.text(x + 0.3, max(d.get('scint', 0), d.get('flash', 0), 20) * 1.8, run.replace('run_', ''), 20, sd.INK, 'middle', 600)
P.text(100, 6e5, f'B4C, scintillator trigger: {n_trig_all / 1e3:.0f} k events →', 26, B4C, 'end', 600)
deck.slide('timeline', sd.title('Only runs 39 and 40 put the scintillator trigger on the B4C disc',
                                'Recorded events per run, 10–15 July, by trigger; shaded by the target in the beam')
           + P.svg('runs and targets, 10-15 July')
           + sd.legend([('scintillator trigger', fam_col['scint'], 'box'), ('γ-flash trigger', fam_col['flash'], 'box'),
                        ('random / pulser trigger', fam_col['random'], 'box')], 22),
           short='the B4C runs', foot='Logbook 10–15 July; run_config.json on EOS; event census ntof_run_report/data/events_per_subrun.csv.',
           notes=(
    '<table><tr><th>period</th><th>target</th><th>DREAM runs</th></tr>'
    '<tr><td>2 – 10 Jul 17:45</td><td>none</td><td>run_1 – run_24</td></tr>'
    '<tr><td>10 Jul 17:45 – 14 Jul 15:00</td><td>“Marex”, on strings, maybe missing the beam</td><td>run_25 – run_37</td></tr>'
    '<tr><td>14 Jul 20:00 – 15 Jul ~12:00</td><td>B4C 2.5 mm</td><td>run_38 (flash only), run_39, run_40</td></tr>'
    '<tr><td>15 Jul 12:00 →</td><td>³He</td><td>run_41 →</td></tr></table>'
    '<p>run_38 (22:12–01:09) is a flash-trigger HV scan with no scintillator trigger; its run_config still says “Marex” '
    'because the target field was only corrected at 05:08. Nothing ran 01:09–05:05.</p>'
    '<p>run_39: 10 × 24 min, 05:06–09:03, all chambers 560 → 530 V, drift 800 V, 32 × 60 ns. run_40: 2 sub-runs on EOS at '
    '545 V, 09:13–09:59 (the logbook says it ran to midday; three sub-runs are missing). Beam: 2635 pulses, 2.175×10¹⁶ '
    'protons (8.25×10¹² per pulse). n_TOF DAQ runs 224451–224455 recorded the walls, plastics and liquids over the window.</p>'
    '<p>The Marex period has more scintillator-triggered data (runs 32/33/36/37), but with the mesh-injection circuits on, '
    'half the trigger on liquid scintillators, and an unknown target that may have missed the beam.</p>'))

# --------------------------------------------------------------------------- #
# 4  what the data hold
# --------------------------------------------------------------------------- #
groups = [('run_37', '“Marex”', '#8d8a84'), ('run_39', 'B4C', B4C), ('run_40', 'B4C', B4C), ('run_79', '³He', HE3)]
xi = 0
xt = []
for run, tgt, col in groups:
    rows = [r for r in census if r['run'] == run and r['ev'] >= 1000]
    xt.append((run, tgt, col, xi, xi + len(rows) - 1, rows))
    xi += len(rows) + 1
P = sd.Plot(1100, 620, x=(-0.8, xi - 1.2), y=(0, 6), xlabel='sub-runs (run 39 ordered 530 → 560 V resist)',
            ylabel='triggers with ≥ 2 chambers active [%]', margin=(40, 30, 92, 104))
P.yticks([(v, f'{v}') for v in range(0, 7)])
for run, tgt, col, a, b, rows in xt:
    xs = list(range(a, b + 1))
    ys = [r['ge2'] / r['ev'] * 100 for r in rows]
    tips = [f'{run} {r["sub"]}\n{r["ge2"]:,} of {r["ev"]:,} triggers ({y:.1f} %)\n≥ 3 chambers: {r["ge3"]}'
            for r, y in zip(rows, ys)]
    P.points(xs, ys, col, r=9, tips=tips)
    lab_y = 5.6 if run != 'run_40' else 5.15
    P.text((a + b) / 2, lab_y, run.replace('run_', 'run ') + ' · ' + tgt, 22, col, 'middle', 600)
    if b < xi - 2:
        P.raw(sd.line(P.X(b + 1), P.y0, P.X(b + 1), P.y0 + P.ph, sd.RULE, 2))
f_b4c = n_multi / n_trig * 100
r79 = [r for r in census if r['run'] == 'run_79']
f_79 = sum(r['ge2'] for r in r79) / sum(r['ev'] for r in r79) * 100
P.text(10.5, 4.85, f'Geant4, pairs from the disc: {n_pair_exp:.2f} events in the whole night', 22, sd.RED, 'middle', 600)
side = sd.col(
    sd.bignum(f'{n_multi:,}', 'B4C triggers with ≥ 2 chambers', B4C, f'{f_b4c:.1f} % of the {n_trig:,} with any hit', dark=False, size=64),
    sd.bignum(f'{f_79:.1f} %', 'the same in ³He production', HE3, 'run_79, first 4 h', dark=False, size=64),
    sd.p('The fraction follows the chamber HV, not the target: accidental activity and background, '
         'not pairs from the disc.', 24, sd.MUT),
    gap=24, w=500)
deck.slide('data', sd.title('Two-chamber events run at a few % for every target',
                            'Fraction of scintillator triggers with x+y activity in two or more chambers, per sub-run')
           + sd.row(P.svg('multi-chamber fraction per sub-run'), side, gap=56, align='center'),
           short='what the data hold', foot='Hits used for candidate counting only (no positions): ≥ 2 strips above 500 ADC in both views, 2–40 strips, in the signal samples.',
           notes=(
    '<p>Census: <code>ntof_b4c_target/lxplus/data_activity_census.py</code> on EOS <code>combined_hits_root</code>; log '
    'copied to <code>ntof_b4c_target/data/activity_census.log</code>. A chamber is “active” when both its x FEU and its y FEU '
    'have 2–40 hits with amplitude &gt; 500 ADC peaking in samples 6–28 (B4C/Marex, 32-sample window) or 0–19 (run_79, '
    '20-sample window).</p>'
    '<p>run_37 “mesh on” sub-runs show a B–C excess that looks like charge injection, not physics; run_37 sub-runs with '
    'fewer than 1000 triggers are not drawn. Chamber D was noisy in run_39 (~120 D-x hits per event against 55 in run_79), '
    'so pairs involving D are inflated.</p>'
    f'<p>Totals: B4C {n_multi:,} of {n_trig:,} triggers ({f_b4c:.1f} %); run_79 {f_79:.1f} %. About one production hour of '
    'multi-chamber candidates in total.</p>'))

# --------------------------------------------------------------------------- #
# 5  the simulation chain + what the disc makes
# --------------------------------------------------------------------------- #
fl = sd.flow([
    dict(label='EAR2 beam', sub=f'{sd.sci(C["N_PULSE"])} n/pulse<br>2.2 meV – 2 eV (1–30 ms)', color=sd.GREY,
         tip='Phase-4 evaluated flux, 2D profile at 19.5 m (beam ends at r ≈ 22 mm), per nominal 7×10¹² pulse.'),
    dict(label='absorbed in the disc', sub=f'{C["F_ABS"] * 100:.1f} %, median depth 0.12 mm', color=sd.GREY,
         tip='Geant4, 100 k neutrons; Ø100 and Ø50 mm discs absorb the same.'),
    dict(label='¹⁰B(n,α)⁷Li', sub='99.992 %<br>478 keV γ: no pairs', color=sd.GREY,
         tip='3837 b. The 478 keV ⁷Li* line (94 %) is below the pair threshold; it lights the SiPM walls and chambers.'),
    dict(label='¹⁰B(n,γ)¹¹B', sub=f'{PP["b4c_ng"]:.0f} cascades / pulse<br>up to 11.45 MeV', color=B4C,
         tip='σγ = 0.29 b (TUNL/EGAF), 7.6×10⁻⁵ of absorptions. Measured EGAF cascade generated (slide 6).'),
    dict(label='internal pairs', sub=f'{b4c_made:.2f} / pulse<br>4.1×10⁻³ per capture', color=B4C,
         tip='Born IPC (ipc_born.alpha_pair) for the seven strongest ¹¹B lines.'),
], size=24)
L_sorted = sorted(LINES, key=float)
P = sd.Plot(1664, 470, x=(0, 12.4), y=(0, 0.75), xlabel='transition energy [MeV]',
            ylabel='IPC pairs / pulse', margin=(30, 30, 92, 104))
P.xticks([(v, str(v)) for v in range(0, 13, 1)])
P.yticks([(v / 10, f'{v / 10:.1f}') for v in range(0, 8)])
for e in L_sorted:
    L = LINES[e]
    v = line_rate(e)
    tp = (f'{float(e):.3f} MeV ({L["multipole"]})\nintensity {L["intensity"]} per 100 captures\n'
          f'α(IPC) {L["alpha"]:.2e}\n{v:.3f} pairs / pulse')
    dx = {'6.740': -0.1, '7.007': 0.1, '4.444': -0.1, '4.711': 0.1}.get(e, 0)
    P.vbar(float(e) + dx, v, 26, B4C if L['multipole'] == 'E1' else '#e9a46b', tip=tp, base=0,
           label=f'{v:.2f}' if e not in ('6.740',) else None)
P.text(0.3, 0.62, f'{b4c_made:.1f} pairs per pulse in total — ~{b4c_made / PP["he3_ipc"]:,.0f}× the ³He(n,γ) yield', 26, sd.INK, 'start', 600)
deck.slide('sim', sd.title(f'The disc makes {b4c_made:.1f} pairs per pulse, all ≤ 11.45 MeV',
                           'Simulation chain per nominal 7×10¹² pulse, and the pairs made by each ¹¹B line')
           + fl + P.svg('IPC pairs per pulse per 11B line')
           + sd.legend([('E1 (primary lines)', B4C, 'box'), ('M1 / E2 (secondary lines)', '#e9a46b', 'box')], 22),
           short='the simulation', notes=(
    '<p><code>MX17_Full_Geant</code> branch merged to main (<code>--slab B4C:2.5 --slab-radius R</code>, '
    '<code>--bias-slab</code>, cascade entries in <code>--gamma-lines</code>). Three sets per disc radius (Ø100, Ø50 mm): '
    '2×10⁷ analog beam neutrons (photon evaporation on; events with a slab ¹⁰B(n,γ) removed), 10⁷ EGAF cascades from the '
    'slab capture vertices, and 10⁶ Born IPC pairs per line. Outputs: <code>/eos/experiment/ntof/data/x17/full_sim/b4c_july/</code>.</p>'
    '<table><tr><th>line [MeV]</th><th>multipole</th><th>I / 100 capt.</th><th>α(IPC)</th><th>pairs / pulse</th></tr>'
    + ''.join(f'<tr><td>{float(e):.3f}</td><td>{LINES[e]["multipole"]}</td><td>{LINES[e]["intensity"]}</td>'
              f'<td>{LINES[e]["alpha"]:.2e}</td><td>{line_rate(e):.3f}</td></tr>' for e in L_sorted)
    + '</table><p>The 6.740 MeV line is E2; the generator has no E2, so it uses the E2 coefficient with M1 kinematics. '
    'The 2.297 MeV line (4×10⁻⁵ pairs per capture) is left out. ¹¹B(n,γ)¹²B (3.37 MeV, 7 % of the (n,γ)) and the ¹²B β '
    'decay are not simulated.</p>'))

# --------------------------------------------------------------------------- #
# 6  Geant4 capture model
# --------------------------------------------------------------------------- #
sums_hp = [sum(c) for c in HP['b11_cascades']]
sums_pe = [sum(c) for c in PE['b11_cascades']]
edges = [i * 2.5 for i in range(0, 25)]


def hist(v):
    h = [0] * (len(edges) - 1)
    for x in v:
        i = min(int(x // 2.5), len(h) - 1)
        h[i] += 1
    return h


hh, hp_ = hist(sums_hp), hist(sums_pe)
n_hp = len(sums_hp)
P1 = sd.Plot(810, 560, x=(0, 60), y=(0, 1.0), xlabel='sum of γ energies per ¹⁰B(n,γ) capture [MeV]',
             ylabel='fraction of captures', margin=(30, 30, 92, 104))
P1.xticks([(v, str(v)) for v in range(0, 61, 10)]); P1.yticks([(v / 5, f'{v / 5:.1f}') for v in range(0, 6)])
for i in range(len(hh)):
    xm = (edges[i] + edges[i + 1]) / 2
    if hh[i]:
        P1.vbar(xm - 0.55, hh[i] / n_hp, 11, sd.RED, base=0, tip=f'Geant4 ParticleHP: {hh[i]} of {n_hp} captures in {edges[i]:.1f}–{edges[i + 1]:.1f} MeV')
    if hp_[i]:
        P1.vbar(xm + 0.55, hp_[i] / len(sums_pe), 11, sd.GREY, base=0, tip=f'Geant4 photon evaporation: {hp_[i]} of {len(sums_pe)} in {edges[i]:.1f}–{edges[i + 1]:.1f} MeV')
P1.vline(11.454, sd.INK, '6 5', 2.5, label='Q = 11.45 MeV', tip='Neutron separation energy of ¹¹B: every real cascade sums to this.')
over = sum(1 for x in sums_hp if x > 12) / n_hp
P1.text(24, 0.62, f'ParticleHP: {over * 100:.0f} % of cascades', 24, sd.RED, 'start', 600)
P1.text(24, 0.55, 'carry more than Q', 24, sd.RED, 'start', 600)


def route(c):
    e = sorted(c, reverse=True)
    if any(abs(x - 11.45) < 0.1 for x in c) and len(c) == 1:
        return '11.45 → g.s.'
    if any(abs(x - 7.01) < 0.03 for x in c):
        return '7.01 → 4.44'
    if any(abs(x - 4.71) < 0.03 for x in c):
        return '4.71 → 6.74'
    if any(abs(x - 2.53) < 0.03 for x in c) or any(abs(x - 8.92) < 0.03 for x in c):
        return '2.53 → 8.92'
    if any(abs(x - 9.33) < 0.05 for x in c):
        return '9.33 → 2.12'
    return 'other'


routes = ['7.01 → 4.44', '4.71 → 6.74', '2.53 → 8.92', '11.45 → g.s.', '9.33 → 2.12', 'other']
egaf = {'7.01 → 4.44': 54.2, '4.71 → 6.74': 25.7, '2.53 → 8.92': 13.8, '11.45 → g.s.': 4.64, '9.33 → 2.12': 0, 'other': 1.7}
pe_c = collections.Counter(route(c) for c in PE['b11_cascades'])
P2 = sd.Plot(810, 560, x=(-0.6, 5.6), y=(0, 60), xlabel='primary transition from the capture state [MeV]',
             ylabel='per 100 captures', margin=(30, 30, 92, 104))
P2.xticks([(i, r.split(' ')[0]) for i, r in enumerate(routes)]); P2.yticks([(v, str(v)) for v in range(0, 61, 10)])
for i, r in enumerate(routes):
    v_e = egaf[r]; v_p = pe_c[r] / len(PE['b11_cascades']) * 100
    P2.vbar(i - 0.17, v_e, 34, B4C, base=0, tip=f'{r}: measured (EGAF) {v_e} per 100 captures')
    P2.vbar(i + 0.17, v_p, 34, sd.GREY, base=0, tip=f'{r}: Geant4 photon evaporation {v_p:.1f} per 100 ({pe_c[r]} of {len(PE["b11_cascades"])})')
P2.text(2.6, 48, 'no 2.12 MeV level in the data', 24, sd.INK, 'middle', 600)
deck.slide('g4model', sd.title('Geant4’s ¹⁰B(n,γ) is wrong, so the measured cascade is generated',
                               f'{n_hp} biased captures in the B4C disc, Geant4 11.2.1, two capture models vs the EGAF thermal-capture data')
           + sd.row(P1.svg('cascade energy sums'), P2.svg('primary branching'), gap=44)
           + sd.legend([('Geant4 ParticleHP (default)', sd.RED, 'box'), ('Geant4 photon evaporation', sd.GREY, 'box'),
                        ('measured, EGAF (what the sim emits)', B4C, 'box')], 22),
           short='Geant4 capture model', notes=(
    '<p>Default ParticleHP samples the ¹⁰B capture γ lines independently from the evaluated data, so a capture can emit '
    'two 11.45 MeV γs and three 7 MeV γs at once. <code>/process/had/particle_hp/use_photo_evaporation true</code> '
    'conserves energy but chooses primaries from a statistical model: a third of its cascades go through a 2.12 MeV level '
    'that the thermal-capture data do not show.</p>'
    '<p>The simulation therefore emits the EGAF scheme (TUNL compilation; per 100 captures): 7.007+4.444 (54.2), '
    '4.711+6.740 (17.4), 4.711+2.297+4.444 (8.3), 2.533+8.917 (13.0), 2.533+4.475+4.444 (0.8), 11.448 (4.64), each γ '
    'isotropic from a vertex drawn from the slab captures of the neutron run. The 4.444 line then comes out at 63 % '
    '(EGAF 64.6 %).</p>'
    f'<p>Also from this check: Geant4 emits the 478 keV ⁷Li* line in {HP["n_alpha_478"] / HP["n_alpha"] * 100:.0f} % of (n,α) '
    f'({HP["n_alpha_478"]} of {HP["n_alpha"]}) instead of 94 %. It cannot make a pair, but it sets the wall and chamber singles rates.</p>'
    '<p>Source: 3000 beam neutrons with <code>--bias-slab 1000</code> and <code>/tracking/verbose 1</code>, parsed by '
    '<code>parse_trk2.py</code>; <code>ntof_b4c_target/data/{hp3k,pe3k}.json</code>.</p>'))

# --------------------------------------------------------------------------- #
# 7  the funnel, B4C vs 3He
# --------------------------------------------------------------------------- #
stages = [('made', 'pairs made', None), ('p_split', 'e⁺ and e⁻ in two\ndifferent chambers', None),
          ('p_leg2', 'two trigger legs', None), ('p_pair', 'two legs +\nboth chambers lit', None)]
P = sd.Plot(1664, 640, x=(-0.6, 3.6), y=(1e-9, 10, 'log'), ylabel='per nominal pulse', margin=(40, 30, 110, 120))
P.yticks(sd.log_ticks(-9, 1))
P.xticks([])
for i, (key, lab, _) in enumerate(stages):
    vb = b4c_made if key == 'made' else b4c_stage[key]
    vh = PP['he3_ipc'] if key == 'made' else he3_stage[key]
    for j, l in enumerate(lab.split('\n')):
        P.raw(sd.T(P.X(i), P.y0 + P.ph + 34 + 26 * j, l, 22, sd.INK))
    tb = f'B4C ¹¹B lines, {lab.replace(chr(10), " ")}: {vb:.2e} / pulse\n= {vb * NPULSE_EXP:.3g} in the B4C exposure'
    th = f'³He(n,γ) 20.6 MeV M1, {lab.replace(chr(10), " ")}: {vh:.2e} / pulse'
    P.vbar(i - 0.17, vb, 120, B4C, tip=tb, base=1e-9, label=s(vb))
    P.vbar(i + 0.17, vh, 120, HE3, tip=th, base=1e-9, label=s(vh))
P.hline(1 / NPULSE_EXP, sd.RED, '8 6', 2.5, label='1 event in the whole B4C night', tip=f'1 / {NPULSE_EXP:.0f} nominal pulses')
P.text(1.6, 1.5, f'B4C keeps 1 in {1 / eff_b4c:,.0f} pairs, ³He 1 in {1 / eff_he3:,.0f}', 26, sd.INK, 'middle', 600)
deck.slide('funnel', sd.title(f'B4C makes {b4c_made / PP["he3_ipc"]:,.0f}× more pairs than ³He(n,γ), and loses almost all of them',
                              'Internal pairs per nominal pulse at each step of the selection, Geant4 (Ø100 mm disc; ³He capsule, same geometry)')
           + P.svg('selection funnel')
           + sd.legend([('B4C disc: ¹⁰B(n,γ) internal pairs', B4C, 'box'), ('³He capsule: ³He(n,γ) 20.6 MeV M1 internal pairs', HE3, 'box')], 22),
           short='the funnel', notes=(
    f'<p>B4C: {b4c_made:.2f} pairs per pulse made; {b4c_stage["p_split"]:.3f} with the e⁺ and e⁻ in two different chambers; '
    f'{b4c_stage["p_leg2"]:.2e} with two trigger legs; {b4c_stage["p_pair"]:.2e} with two legs and charge in both leg chambers. '
    f'Over the exposure ({NPULSE_EXP:.0f} nominal pulses) that is {n_pair_exp:.3f} events. Ø50 mm gives the same within MC error.</p>'
    f'<p>³He: 44.8 % of beam neutrons are absorbed in the gas ((n,p), current EAR2 profile run '
    f'<code>neutrons_thermal_ear2_L19p5</code>), ³He(n,γ)/(n,p) = 55 µb / 5333 b, α(M1, 20.58 MeV) = 3.66×10⁻³: '
    f'{PP["he3_ipc"]:.1e} pairs per pulse. Per pair: two legs + both chambers {eff_he3:.2e} (M1), '
    f'{D["he3"]["ipcE0"]["p_pair"]:.2e} (E0), {D["he3"]["x17"]["p_pair"]:.2e} (X17). The ³He pair runs: 2×10⁵ each, '
    'vertices from the capsule gas library, same build and thresholds.</p>'
    '<p>Even with more pairs surviving per pulse than ³He(n,γ), B4C stays far below one event: the thermal gate is not '
    'where either target’s pair physics is visible.</p>'))

# --------------------------------------------------------------------------- #
# 8  the physics reason: efficiency vs transition energy
# --------------------------------------------------------------------------- #
P1 = sd.Plot(810, 600, x=(0, 22), y=(0, 25), xlabel='transition energy [MeV]',
             ylabel='lepton reaches plastic with ≥ 0.9 MIP [%]', margin=(30, 30, 92, 104))
P1.xticks([(v, str(v)) for v in range(0, 23, 4)]); P1.yticks([(v, str(v)) for v in range(0, 26, 5)])
xs = [float(e) for e in L_sorted]
ys = [LINES[e]['reach_plastic_thr'] * 100 for e in L_sorted]
tps = [f'{float(e):.3f} MeV {LINES[e]["multipole"]}: {LINES[e]["reach_plastic_thr"] * 100:.2f} % of {LINES[e]["n_lepton_arms"]:,} lepton-arms\n'
       f'reach the wall ≥ 0.5 MIP: {LINES[e]["reach_wall"] * 100:.0f} %; any plastic deposit: {LINES[e]["reach_plastic_any"] * 100:.0f} %'
       for e in L_sorted]
P1.line(xs, ys, B4C, tips=tps)
for tag, col, lab in (('ipcM1', HE3, 'M1'), ('ipcE0', E0C, 'E0'), ('x17', X17C, 'X17')):
    h = D['he3'][tag]
    P1.points([20.58], [h['reach_plastic_thr'] * 100], col, r=11,
              tips=[f'³He 20.58 MeV {lab}: {h["reach_plastic_thr"] * 100:.1f} % of {h["n_lepton_arms"]:,} lepton-arms'])
P1.text(1, 21, 'one lepton', 26, sd.INK, 'start', 600)
P2 = sd.Plot(810, 600, x=(0, 22), y=(1e-7, 1e-1, 'log'), xlabel='transition energy [MeV]',
             ylabel='pair passes: two legs + both chambers', margin=(30, 30, 92, 104))
P2.xticks([(v, str(v)) for v in range(0, 23, 4)]); P2.yticks(sd.log_ticks(-7, -1))
floor = 1.5e-7
ys2 = [max(LINES[e]['p_pair'], floor) for e in L_sorted]
tps2 = [f'{float(e):.3f} MeV: {LINES[e]["n_pair"]} of {LINES[e]["ngen"]:,} pairs pass' + (' (0: drawn at the floor)' if LINES[e]['p_pair'] == 0 else '')
        for e in L_sorted]
P2.line(xs, ys2, B4C, tips=tps2, marker='circle')
for tag, col, lab in (('ipcM1', HE3, '³He IPC M1'), ('ipcE0', E0C, '³He IPC E0'), ('x17', X17C, 'X17 → e⁺e⁻')):
    h = D['he3'][tag]
    P2.points([20.58], [h['p_pair']], col, r=11, tips=[f'{lab}, 20.58 MeV: {h["n_pair"]} of {h["ngen"]:,} pass ({h["p_pair"]:.2e})'])
P2.text(1, 3e-2, 'the pair', 26, sd.INK, 'start', 600)
P2.text(0.6, 3e-7, 'zero MC events', 20, sd.MUT, 'start')
P2.raw(sd.line(P2.x0, P2.Y(floor), P2.X(10), P2.Y(floor), sd.RULE, 2, '4 4'))
deck.slide('physics', sd.title('The trigger is an energy threshold: below ~10 MeV the leptons never reach the plastics',
                               'Geant4, per lepton (left) and per pair (right), against the transition energy; ¹¹B lines from the B4C disc, ³He at 20.6 MeV')
           + sd.row(P1.svg('lepton reaches plastic vs energy'), P2.svg('pair efficiency vs energy'), gap=44)
           + sd.legend([('¹⁰B(n,γ) ¹¹B lines (B4C)', B4C), ('³He(n,γ) IPC, M1', HE3, 'dot'),
                        ('³He IPC, E0', E0C, 'dot'), ('X17 (16.8 MeV) → e⁺e⁻', X17C, 'dot')], 22),
           short='the physics reason', notes=(
    '<p>A lepton must cross the Micromegas, the SiPM wall and the liquid-scintillator vessel (2.6 mm CFRP plus ~2 cm of '
    'LAB) and still leave 3.0 MeV (0.9 MIP) in a 2 cm plastic bar. That takes roughly ≥ 8–10 MeV of kinetic energy at '
    'normal incidence. A pair from a W MeV transition shares W − 1.02 MeV between its two leptons, so below ~10 MeV at '
    'most one of them can make it.</p>'
    '<table><tr><th>line</th><th>lepton → wall ≥ 0.5 MIP</th><th>→ any plastic</th><th>→ plastic ≥ 0.9 MIP</th><th>pair passes</th></tr>'
    + ''.join(f'<tr><td>{float(e):.3f}</td><td>{LINES[e]["reach_wall"] * 100:.0f} %</td><td>{LINES[e]["reach_plastic_any"] * 100:.1f} %</td>'
              f'<td>{LINES[e]["reach_plastic_thr"] * 100:.2f} %</td><td>{LINES[e]["p_pair"]:.1e}</td></tr>' for e in L_sorted)
    + ''.join(f'<tr><td>³He {lab}</td><td>{D["he3"][t]["reach_wall"] * 100:.0f} %</td><td>{D["he3"][t]["reach_plastic_any"] * 100:.1f} %</td>'
              f'<td>{D["he3"][t]["reach_plastic_thr"] * 100:.2f} %</td><td>{D["he3"][t]["p_pair"]:.1e}</td></tr>'
              for t, lab in (('ipcM1', 'M1'), ('ipcE0', 'E0'), ('x17', 'X17')))
    + '</table><p>Only the 11.448 MeV ground-state line (4.6 % of captures) ever gives a two-leg pair. ³He E0 and X17 '
    'pairs are more open than M1, so more of them split over two arms.</p>'))

# --------------------------------------------------------------------------- #
# 9  comparison with the 3He running
# --------------------------------------------------------------------------- #
rows3 = [('B4C disc: ¹⁰B(n,γ)', b4c_made, b4c_stage['p_pair'], B4C, '≤ 11.45 MeV'),
         ('³He capsule: ³He(n,γ)', PP['he3_ipc'], he3_stage['p_pair'], HE3, '20.6 MeV, M1'),
         ('³He capsule: its Al walls', PP['he3_al_ipc'], None, AL, '≤ 7.7 MeV ²⁸Al lines')]
P = sd.Plot(1080, 620, x=(-0.6, 2.6), y=(1e-8, 1e3, 'log'), ylabel='internal pairs per nominal pulse',
            margin=(40, 30, 110, 120))
P.yticks(sd.log_ticks(-8, 3)); P.xticks([])
for i, (lab, made, det, col, sub) in enumerate(rows3):
    P.raw(sd.T(P.X(i), P.y0 + P.ph + 34, lab, 22, sd.INK, weight=600) + sd.T(P.X(i), P.y0 + P.ph + 62, sub, 20, sd.MUT))
    P.vbar(i - 0.17, made, 110, col, base=1e-8, label=s(made), tip=f'{lab}: {made:.3g} pairs made per pulse')
    if det is not None:
        P.vbar(i + 0.17, det, 110, col, base=1e-8, label=s(det), opacity=0.45,
               tip=f'{lab}: {det:.2e} pass two legs + both chambers per pulse')
    else:
        P.text(i + 0.17, 3e-7, '≈ 0', 26, col, 'middle', 600,
               tip='Not simulated pair by pair: the ²⁸Al lines are ≤ 7.7 MeV, where the ¹¹B lines give zero passing pairs (slide 8).')
P.hline(1 / NPULSE_EXP, sd.RED, '8 6', 2.5, label='1 event per B4C night', tip=f'1 / {NPULSE_EXP:.0f} nominal pulses',
        anchor='start', where='below')
side = sd.col(
    sd.legend([('pairs made', sd.INK, 'box'), ('pairs that pass (lighter)', '#9aa1ad', 'box')], 22),
    sd.callout(f'Per pulse, the ³He capsule’s own aluminium makes {PP["he3_al_ipc"] / b4c_made:.0f}× more few-MeV pairs '
               'than the B4C disc did, through the whole production run.', AL),
    sd.callout('So the B4C night adds nothing the ³He data lack: both see these low-energy pairs equally poorly.', B4C),
    sd.callout('The ³He physics is the 20.6 MeV line, which the trigger was built for — and only in the MeV window '
               'is it frequent enough to show.', HE3),
    gap=26, w=540)
deck.slide('he3', sd.title('The ³He capsule’s aluminium already makes more of these pairs than the B4C did',
                           'Internal pairs made and passing per nominal pulse: the B4C night against the ³He running in the 1–30 ms gate')
           + sd.row(P.svg('B4C vs 3He comparison'), side, gap=44, align='center'),
           short='B4C vs ³He', notes=(
    f'<p>³He(n,γ): {PP["he3_ng"]:.3f} captures per pulse in the gas, {PP["he3_ipc"]:.1e} M1 internal pairs. The E0 channel '
    'is not bounded by the (n,γ) cross section (CLAUDE.md conventions, ipc_born); it is left out of the bar.</p>'
    f'<p>Capsule Al: {PP["he3_al_ng"]:,.0f} captures per pulse (He3Cap_Al, 0.285 % of beam neutrons, analog run '
    f'<code>neutrons_thermal_ear2_L19p5</code>), × Σ I·α = 2.56×10⁻³ over the ²⁸Al lines above 2 MeV, all taken as E1: '
    f'{PP["he3_al_ipc"]:.0f} pairs per pulse. Approximate (secondary lines are not all E1), but the order of magnitude '
    'is the point.</p>'
    '<p>External conversion is not in these bars. It is larger than IPC for both targets (in the capsule, 84.5 % of the '
    'capsule pairs reaching a drift gap are Al external conversions — CAMPAIGN_STATUS 2026-09-15), and it has the same '
    'energy problem.</p>'))

# --------------------------------------------------------------------------- #
# 10  no threshold rescues it
# --------------------------------------------------------------------------- #
sc = E['scan_r50']; sc25 = E['scan_r25']
P = sd.Plot(1664, 600, x=(0.1, 1.5), y=(1e-4, 100, 'log'), xlabel='plastic threshold [MIP]  (SiPM wall at 0.5 MIP)',
            ylabel='Doubles in the whole B4C night', margin=(30, 30, 92, 120))
P.xticks([(v / 10, f'{v / 10:.1f}') for v in range(1, 16, 2)]); P.yticks(sd.log_ticks(-4, 2))
for src, col, lab in (('casc', B4C, 'cascade γs'), ('ipc', sd.RED, 'internal pairs')):
    ys = [v * NPULSE_EXP if v > 0 else None for v in sc[src]]
    tps = [f'{lab}, plastic ≥ {b} MIP: {v * NPULSE_EXP:.2f} Doubles (Ø100 mm); Ø50 mm {w * NPULSE_EXP:.2f}' if v > 0 else None
           for b, v, w in zip(sc['b'], sc[src], sc25[src])]
    P.line(sc['b'], ys, col, tips=tps)
P.vline(0.9, sd.INK, '6 5', 2, label='production 0.9 MIP')
P.hline(1, sd.MUT, '4 4', 1.5, label='1 event', anchor='end')
i01 = sc['b'].index(0.1)
P.text(0.13, 3e-4, f'At 0.1 MIP: {sc["ipc"][i01] * NPULSE_EXP:.1f} pair Doubles in the whole night', 26, sd.INK, 'start', 600)
deck.slide('threshold', sd.title('No plastic threshold rescues it',
                                 'Two-arm (Doubles) events from the disc over the whole exposure, against the plastic threshold')
           + P.svg('Doubles vs plastic threshold')
           + sd.legend([('¹⁰B(n,γ) cascade γs (two arms, Compton)', B4C), ('¹⁰B(n,γ) internal pairs', sd.RED)], 22),
           short='thresholds', notes=(
    '<p>The run_39 thresholds moved during the night: the wall (M1) thresholds were tuned 05:06–09:01 (A 30 → 20 mV, '
    'B 45 → 55 mV, then up to 133–160 mV), and the plastics (M2) stayed at −15 mV until 09:04, when they went to '
    '−65/−80 mV. The pre-FIFO −15 mV is roughly the production 0.9 MIP (the 17 July fan-out doubled the amplitude and the '
    'threshold went to −30 mV). This scan covers that whole range.</p>'
    '<p>The beam-neutron source gives no Doubles at any threshold (no MC events in 2×10⁷ neutrons). Doubles from the '
    'cascade γs are two Compton electrons in two arms, not pairs.</p>'))

# --------------------------------------------------------------------------- #
# 11  closing
# --------------------------------------------------------------------------- #
body = (sd.title('What this does not rule out')
        + sd.row(
            sd.col(
                sd.p('<b>The chamber occupancy does not match.</b> Geant4 lights each chamber ~1900 times per pulse '
                     '(Compton electrons from the 478 keV ⁷Li line), which would put two chambers in almost every readout. '
                     'The data show a few %, as in production. The data criterion is much harder than a 3 keV deposit, so '
                     'this is open; it could also mean the disc was small or off axis (the May test found it misaligned).',
                     26, sd.DINK),
                sd.p('<b>The target itself is undocumented:</b> diameter, density (2.52 g/cm³ assumed), position along the '
                     'beam, and “2.5 mm, I think?”. None of these changes the verdict: the pairs are lost at the plastics, '
                     'not at the target.', 26, sd.DINK), gap=28),
            sd.col(
                sd.p('<b>Not simulated:</b> room-return and epithermal neutrons, the flash and 0–1 ms, accidental '
                     'coincidences between captures, ¹¹B(n,γ) and the ¹²B β decay, E2 kinematics for the 6.74 MeV line.',
                     26, sd.DINK),
                sd.p('<b>What it does say:</b> any low-energy pair source in the thermal gate — B4C, or the capsule’s own '
                     'aluminium — is invisible to this trigger. The early-July targets offer no pair sample beyond what the '
                     '³He production data already contain.', 26, sd.DINK), gap=28),
            gap=72))
deck.slide('close', body, dark=True, short='not ruled out', notes=(
    '<p>Code: <code>nTof_x17/ntof_b4c_target/</code> (README, <code>expect.py</code>, <code>deck_data.py</code>, '
    '<code>make_report.py</code>, <code>make_b4c_deck.py</code>, <code>lxplus/</code> job and reduction scripts). '
    'Geant4: <code>MX17_Full_Geant</code> main (merged from <code>b4c-slab</code>).</p>'
    '<p>The dataset is final; nothing here asks for a new measurement.</p>'))

out = OUT / 'b4c-target-pairs.html'
deck.write(out, note_meta=dict(
    title='B4C target: why the July B4C night shows no pairs',
    summary=(f'The B4C disc made {b4c_made:.1f} internal pairs per pulse, but at ≤ 11.45 MeV their leptons stop before the '
             f'plastics: Geant4 expects {n_pair_exp:.2f} pair events in the whole night. Data inventory, the simulation, '
             'and the comparison with ³He.'),
    tags='X17, n_TOF, Geant4, B4C, pairs', date='2026-10-20'))
print(out)
