#!/usr/bin/env python3
"""make_transfer_deck.py -- what the SPS det4 measurements give the cosmic-bench model, given that
the beam gas attached electrons (FINDINGS §27).  A figure-first slide note, companion to
make_attachment_deck.py (notes/mx17-attachment-slides).

Inputs (all in results/):
  attach_transfer.json   injection toy: the SPS measurement code on events with known constants,
                         attachment on/off with common random numbers (attach_transfer.py)
  rc_drift.json          the RC fit with drift diffusion written in, beam run_71 and five bench
                         chambers, D_T arms (rc_drift.py)
  rc_diffusion.json      the original RC fit (no drift diffusion), same stacks
  ../sharing_kernel/fit_kernel.json, decompose_tmax1300.json   the beam kernel numbers

    ../../../../.venv/bin/python make_transfer_deck.py [--out PATH]
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug mx17-sps-to-bench --force --deploy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd                                                   # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN,      # noqa: E402,F401
                      INK, MUT, RULE, DINK, DMUT)

RES = os.path.join(HERE, 'results')
OUT = os.path.expanduser('~/x17/cosmic_bench/cloud_basics/transfer')
BENCH = ('det2', 'det3', 'det4', 'det6', 'det7')
BEAM = (('sps700', 243), ('sps450', 150), ('sps275', 92))
FC = {243: BLUE, 150: ORANGE, 92: PURPLE}
ATT_C, ATT2_C, GAS_C = RED, GOLD, GREEN


def J(name, base=RES):
    return json.load(open(os.path.join(base, name)))


# --------------------------------------------------------------------------- #
# the injection toy: paired shifts
# --------------------------------------------------------------------------- #
# (key in measure(), view, label, what it is, how the bench model uses it)
PARAMS = [
    ('lead_sig_eq', 'x', 'X prompt width, leading edge', 'σ-equivalent of ±1/centre at the first charge'),
    ('rc_sig0', 'x', 'X prompt width, RC fit', 'σ0 of the head-on (offset, time) fit, X'),
    ('lead_sig_eq', 'y', 'Y prompt width, leading edge', 'σ-equivalent of ±1/centre at the first charge'),
    ('rc_sig0', 'y', 'Y prompt width, RC fit', 'σ0 of the head-on (offset, time) fit, Y'),
    ('rc_D', 'y', 'Y resistive spread D_rc', 'RC diffusion constant along the resistive strip'),
    ('kern_delay_c2_over_c1', 'y', 'Y c2/c1 (delay form)', 'the ratio production pins (r06: 0.6)'),
    ('kern_cascade_c', 'y', 'Y cascade c (lp form)', 'one-pole ladder amplitude'),
    ('dec_b1', 'y', 'Y delayed fraction b1', 'decompose.py: slow neighbour charge'),
    ('dec_tau', 'y', 'Y delay τ', 'decompose.py: low-pass time constant'),
]


class Toy:
    def __init__(self):
        self.d = J('attach_transfer.json')
        self.C = self.d['cases']
        self.truth = self.d['truth']

    def vals(self, case, key, view):
        return np.array([r[view][key] for r in self.C[case]['runs']], float)

    def shift(self, base, alt, key, view):
        """paired relative shift alt/base - 1 per seed: mean, error of the mean"""
        a, b = self.vals(base, key, view), self.vals(alt, key, view)
        s = b / a - 1.0
        return float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s))), float(a.mean()), float(b.mean())


# --------------------------------------------------------------------------- #
G = dict(
    att=('Attachment here = the run_71 beam gas: 1.51 % water + 0.078 % air (163 ppm O₂), η·v = 2.2×10⁻⁴/ns '
         'at 243 V/cm from Magboltz. About 25 % of the charge from the cathode end is lost (FINDINGS §16, §25).'),
    toy=('Injection toy (attach_transfer.py): head-on tracks, 3 primary clusters/mm with 1/n² sizes, per-electron '
         'survival exp(−η·v·t), Magboltz drift diffusion, prompt footprint σ0, Y resistive spread D_rc, the det3 X '
         'template, 60 ns sampling, 6.5 ADC noise, 9 strips. 4 × 3000 events per case. On and off share every random '
         'number except which electrons survive.'),
    paired=('Paired: the same simulated tracks with and without the loss, so the difference is the attachment '
            'effect with the statistical noise largely cancelled. Error = spread over 4 seeds / √4.'),
    lead=('Leading edge: the ±1/centre ratio at the first 30 % of the centre pulse, turned into the equivalent '
          'Gaussian width. It sees the first-arriving (shallow) charge, before any loss has acted.'),
    rcfit=('RC fit (rc_diffusion.py): W_o = S ⊗ dF_o/ds, with S the all-strip sum taken from the data. The arrival '
           'current, attachment included, is in S and cancels; only the lateral spread is fitted.'),
    cross=('Cross-relation (sharing_kernel/): n_0 ⊗ W_d = n_d ⊗ W_0. The common drive C(t) — arrival current × '
           'electronics, attachment included — cancels identically for head-on tracks.'),
    Drc=('D_rc: the resistive spread along the resistive strips (Y view). Width² grows as σ0² + 2·D_rc·s with s '
         'the time since the charge landed. X, across the strips, has none.'),
    DT=('Transverse drift diffusion: width² grows by D_T²·z with the drift depth z. Beam gas (Ar/CF₄/iso + 1.5 % '
        'water) D_T 160–240 µm/√cm; bench gas (Ar/iso 95/5) 270 (≈1 % water) to 480 (dry) µm/√cm.'),
)


# --------------------------------------------------------------------------- #
def s_cover(D, T, R):
    sD, eD, *_ = T.shift('beam243_noatt', 'beam243_att', 'rc_D', 'y')
    sX, eX, *_ = T.shift('beam243_noatt', 'beam243_att', 'lead_sig_eq', 'x')
    sR, eR, *_ = T.shift('beam243_noatt', 'beam243_att', 'kern_delay_c2_over_c1', 'y')
    sB, eB, *_ = T.shift('beam243_noatt', 'beam243_att', 'dec_b1', 'y')
    beamD = [R['beam'][l]['y']['magboltz']['D'] for l, _ in BEAM]
    body = (sd.kicker('det4 at SPS H4 → the June cosmic bench')
            + sd.title('The beam O₂ barely touches the readout constants; the gas does',
                       'What survives the transfer from the SPS beam (wet, O₂-attaching CF₄ gas) to the bench '
                       '(moist Ar/iso, no O₂)')
            + sd.row(
                sd.bignum(f'{sX:+.0%}', 'prompt footprint', GREEN,
                          'leading-edge width, X, at the run_71 O₂ level (toy, paired)', tip=G['lead']),
                sd.bignum(f'{sD:+.0%}', 'Y resistive spread D_rc', GOLD,
                          f'beam det4 {min(beamD) * 1e4:.2f}–{max(beamD) * 1e4:.2f}×10⁻⁴ mm²/ns over 3 fields', tip=G['Drc']),
                sd.bignum(f'{sR:+.0%}', 'c2/c1 (production kernel)', GREEN,
                          'the cross-relation cancels the arrival current', tip=G['cross']),
                sd.bignum(f'{sB:+.0%}', 'delayed fraction b1', RED,
                          'the one SPS number attachment does move; the physical kernel does not use it'),
                gap=40))
    D.slide('cover', body, f'''
<p>Question (Dylan, 2026-10-11): the SPS beam gas attached electrons (O₂, ~150–230 ppm; the bench gas is only moist).
Does that contaminate the model parameters measured at SPS so that they no longer apply where there is no attachment?</p>
<p>Answer: almost entirely no. Every SPS constant the bench model takes is measured either at the leading edge (before any
loss acts) or through a relation in which the arrival current, attachment included, cancels. An injection test that runs
the SPS measurement code itself on events with known constants, with the run_71 loss on and off, moves them by ≤ {max(abs(sX), abs(sD), abs(sR)):.0%}.
The exception is the prompt/delayed split of the old decomposition (b1 {sB:+.0%}, τ), which the physical model no longer uses.</p>
<p>What does not transfer is what never would have: the gas (v, drift diffusion, the loss itself), and setup-specific
calibrations. The bigger transfer issue turns out to be bench drift diffusion, which is degenerate with D_rc and σ0 on
the bench (slide §§bench§§).</p>''', dark=True, short='Answer')


def s_inventory(D, T, R):
    """what SPS measured, sorted by how the measurement sees the arrival current"""
    sB, *_ = T.shift('beam243_noatt', 'beam243_att', 'dec_b1', 'y')
    steps = [
        dict(label='Leading edge', sub='σ0 X / Y, prompt ±1 ratio', color=GREEN,
             tip='Measured on the first charge to arrive, from the shallowest depths. Attachment has had no time to act.'),
        dict(label='Arrival current cancels', sub='D_rc, X has no RC, c2/c1, cascade c, gain invariance', color=GREEN,
             tip='RC fit with S from the data; cross-relation n_0⊗W_d = n_d⊗W_0; ratios at fixed time. Attachment lives in '
                 'the common drive and drops out — up to the depth dependence of the footprint.'),
        dict(label='Time split of the neighbour', sub='decompose a1 / b1 / τ, tail τ', color=GOLD,
             tip='Prompt vs delayed shares of the neighbour signal. The delayed share is weighted by late arrivals, '
                 'which attachment removes.'),
        dict(label='Assumes a charge profile', sub='beam forward fit (bench template), v from fits w/o loss', color=RED,
             tip='Any fit that assumes uniform ionisation along the drift. beam_fit.py did not close and is not used; '
                 'wft solves the per-depth charge by NNLS, so it absorbs the loss.'),
        dict(label='Gas properties', sub='v(E), D_T, D_L, η — and the 176 µm resolution', color=GREY,
             tip='Properties of the H4 gas, not of the chamber. They never transfer to the bench, attachment or not.'),
    ]
    body = sd.title('Five kinds of SPS measurement, sorted by how they see the arrival current',
                    'Green: attachment cannot reach them in principle. Gold: it can. Red / grey: they do not transfer anyway.')
    body += sd.flow(steps, size=24, arrow='')
    rows = [
        ['σ0 leading edge (X, Y)', 'det4 X 0.42–0.51, Y 0.36–0.38 mm', 'readout', 'leading edge', '<b style="color:#2a8a4a">transfers</b>'],
        ['X has no resistive spread', 'D_rc,x ≈ 0 (1–2×10⁻⁵)', 'board geometry', 'linear, S from data', '<b style="color:#2a8a4a">transfers</b>'],
        ['Y resistive spread D_rc', '1.9–2.3×10⁻⁴ mm²/ns', 'chamber (R_s·C)', 'linear, S from data', '<b style="color:#b8860b">det4 only, ±10 %</b>'],
        ['c2/c1, cascade c (old kernel)', '0.42–0.47; 0.63–0.66', 'readout', 'cross-relation', '<b style="color:#2a8a4a">transfers</b> (superseded)'],
        ['prompt/delayed split b1, τ', 'b1 0.21–0.25, τ 530–590 ns', 'mixed', 'arrival-weighted', f'<b style="color:#c0392b">biased {sB:+.0%}</b>'],
        ['X / Y footprint ratio', 'X wider (one board layer deeper)', 'board geometry', 'leading edge', '<b style="color:#2a8a4a">transfers</b>'],
        ['v(E), D_T, η; 176 µm', 'H4 gas', 'gas / setup', '—', '<b style="color:#777">never transfers</b>'],
    ]
    body += sd.table(['quantity', 'SPS value (det4, run_71)', 'property of', 'how measured', 'to the bench'], rows, size=22,
                     widths=[420, 420, 240, 270, 300], align=['left', 'left', 'left', 'left', 'left'])
    D.slide('inventory', body, '''
<p>Hover the dotted terms and the points on every chart for definitions, counts and sources.</p>
<p>Sources: leading edge and RC fits <code>cloud_basics/rc_diffusion.py</code>, <code>neighbour_vs_time.py</code> (FINDINGS §1, §3, §10);
cross-relation kernel <code>sharing_kernel/fit_kernel.py</code>; decomposition <code>decompose.py</code> (§2); intrinsic resolution
<code>spatial_resolution/</code>. All on det4 run_71 RAW at 243 / 150 / 92 V/cm (31 Jul – 3 Aug 2026), head-on (|tan| &lt; 0.03).</p>
<p>The wft reconstruction itself solves the charge per 60 ns depth slice by NNLS, so it assumes no charge profile and needs
no attachment term to reconstruct beam tracks; only a fit that imposes uniform ionisation would be biased.</p>''',
            short='Inventory')


def s_cancel(D, T, R):
    """schematic: the drive C(t) carries the loss and is common to every strip"""
    W, H = 1660, 560
    inner = []
    # arrival current with loss
    x0, y0, w, h = 60, 90, 420, 300
    inner.append(sd.line(x0, y0 + h, x0 + w, y0 + h, INK, 2))
    inner.append(sd.line(x0, y0, x0, y0 + h, INK, 2))
    ts = np.linspace(0, 1, 50)
    flat = [y0 + h - 0.8 * h * (1 if t < 0.85 else 0) for t in ts]
    loss = [y0 + h - 0.8 * h * (np.exp(-0.45 * t / 0.85) if t < 0.85 else 0) for t in ts]
    inner.append(sd.poly(list(x0 + w * ts), flat, GREY, 3, dash='10 7', tip='no loss: uniform ionisation arrives as a box'))
    inner.append(sd.poly(list(x0 + w * ts), loss, RED, 4, tip='run_71: the late (deep) charge is reduced by O₂ attachment'))
    inner.append(sd.T(x0 + w / 2, y0 - 30, 'arrival current C(t)', 26, INK, weight=600))
    inner.append(sd.T(x0 + w / 2, y0 + h + 40, 'drift time →', 22))
    inner.append(sd.arrow(x0 + w + 30, y0 + h / 2, x0 + w + 150, y0 + h / 2, INK, 3))
    # lateral transfer boxes
    bx = x0 + w + 170
    for i, (lab, sub) in enumerate([('n₀', 'centre'), ('n₁', '±1'), ('n₂', '±2')]):
        yy = y0 + i * 110
        inner.append(f'<rect x="{bx}" y="{yy}" width="190" height="80" rx="12" fill="#f4f6fa" stroke="{BLUE}" stroke-width="2"/>')
        inner.append(sd.T(bx + 95, yy + 38, lab, 30, BLUE, weight=600))
        inner.append(sd.T(bx + 95, yy + 66, sub, 20))
        inner.append(sd.arrow(bx + 200, yy + 40, bx + 330, yy + 40, INK, 2.5))
        inner.append(sd.T(bx + 430, yy + 48, f'W{["₀", "₁", "₂"][i]} = n{["₀", "₁", "₂"][i]} ⊗ C', 26, INK))
    inner.append(sd.T(bx + 95, y0 + 345, 'lateral transfer', 22, BLUE))
    inner.append(sd.T(bx + 95, y0 + 372, '(footprint + RC)', 20))
    # identity
    ix = bx + 560
    inner.append(sd.T(ix + 290, y0 + 60, 'n₀ ⊗ W_d = n_d ⊗ W₀', 34, GREEN, weight=600,
                      tip=G['cross']))
    inner.append(sd.T(ix + 290, y0 + 110, 'C cancels — loss and all', 26, GREEN))
    inner.append(sd.T(ix + 290, y0 + 190, 'W_o = S ⊗ dF_o/ds,  S = Σ W', 30, GREEN, weight=600, tip=G['rcfit']))
    inner.append(sd.T(ix + 290, y0 + 236, 'S is measured: the loss is in it', 26, GREEN))
    inner.append(sd.T(ix + 290, y0 + 330, '…unless n depends on depth', 26, RED, weight=600,
                      tip='Drift diffusion widens deep charge (D_T²·z). Then each depth has its own n, and the weights of '
                          'the depths — which attachment changes — enter. That is the residual the toy measures.'))
    inner.append(sd.T(ix + 290, y0 + 370, 'drift diffusion: width² + D_T² z', 22, RED))
    body = sd.title('Why attachment mostly cannot reach them: it lives in the drive every strip shares',
                    'Head-on, every strip sees the same arrival current through its own lateral filter')
    body += sd.svg(W, H, ''.join(inner), 'cancel')
    D.slide('cancel', body, '''
<p>For a track normal to the view, every depth lands at the same lateral position. Each strip's signal is then the same
drive C(t) (arrival current × electronics) passed through that strip's lateral transfer n_d. Attachment changes C(t) and nothing
else, so any relation built from the strips' own signals — the cross-relation, or the RC fit with the measured strip sum as the
source — is blind to it.</p>
<p>The cancellation is exact only if the lateral transfer is the same for every depth. Drift diffusion breaks that: charge from
deeper lands wider. With a depth-dependent footprint the fit sees a mixture whose depth weights attachment changes. The size of
that leak is what the injection test measures, and it is small because the beam gas diffuses little (D_T 160–240 µm/√cm,
σ ≤ 0.4 mm at the cathode).</p>''', short='Why it cancels')


def s_toy(D, T, R):
    rows = []
    for key, view, lab, what in PARAMS:
        s1, e1, a, b = T.shift('beam243_noatt', 'beam243_att', key, view)
        s2, e2, _, c = T.shift('beam243_noatt', 'beam243_att2x', key, view)
        sg, eg, _, g = T.shift('beam243_noatt', 'bench267_noatt', key, view)
        rows.append((lab, what, s1, e1, s2, e2, sg, a, b, c, g))
    n = len(rows)
    P = sd.Plot(1100, 640, x=(-0.45, 0.45), y=(-0.5, n - 0.5), xlabel='relative shift of the measured constant',
                margin=(10, 30, 86, 420))
    P.xticks([(v, f'{v:+.0%}' if v else '0') for v in (-0.4, -0.2, 0, 0.2, 0.4)])
    P.yticks([])
    P.raw(f'<rect x="{P.X(-0.05):.0f}" y="{P.Y(n - 0.5):.0f}" width="{P.X(0.05) - P.X(-0.05):.0f}" '
          f'height="{P.Y(-0.5) - P.Y(n - 0.5):.0f}" fill="#eaf5ec" />', back=True)
    P.vline(0, INK, None, 1.5)
    for i, (lab, what, s1, e1, s2, e2, sg, a, b, c, g) in enumerate(rows):
        y = n - 1 - i
        P.text(-0.47, y - 0.05, lab, 22, INK, anchor='end')
        for dy, s, e, col, nm, v in ((0.22, s1, e1, ATT_C, 'run_71 O₂ level', b), (-0.0, s2, e2, ATT2_C, '2× (CO₂-period level)', c),
                                    (-0.22, sg, 0, GAS_C, 'bench gas instead (no loss)', g)):
            sc = max(min(s, 0.44), -0.44)
            P.rect(min(0, sc), max(0, sc), y + dy - 0.09, y + dy + 0.09, col, 1, fill=col,
                   tip=f'{lab}: {nm}: {s:+.1%}' + (f' ± {e:.1%}' if e else '') + f'\n{a:.4g} → {v:.4g} (toy truth: '
                       f'σ0 X {T.truth["sig0_x"]}, Y {T.truth["sig0_y"]} mm, D_rc {T.truth["drc_y"]:.1e})'
                       + ('\nclipped at ±44 %' if sc != s else ''))
    side = sd.col(
        sd.legend([('run_71 O₂ (η·v 2.2×10⁻⁴/ns)', ATT_C, 'box'), ('2× that (CO₂ period)', ATT2_C, 'box'),
                   ('bench gas, no O₂', GAS_C, 'box')], size=22),
        sd.p('Green band: ±5 %, about the field-to-field scatter of the beam measurements themselves.', 22, MUT),
        sd.callout('Footprint, D_rc and c2/c1 move ≤ 7 %. Only the prompt/delayed split (b1, τ) moves a lot.', GREEN, 24),
        sd.callout('Swapping the gas (green bars) moves the widths more than the oxygen does.', GOLD, 24),
        gap=24, w=520)
    body = sd.title('Injection test: the SPS code on known events, with and without the run_71 loss',
                    '243 V/cm, beam gas, 4 × 3000 head-on events per case; each bar is a paired shift against no loss')
    body += sd.row(P.svg('toy'), side, gap=36)
    tr = ''.join(f'<tr><td>{lab}</td><td>{a:.4g}</td><td>{s1:+.1%} ± {e1:.1%}</td><td>{s2:+.1%} ± {e2:.1%}</td><td>{sg:+.1%}</td></tr>'
                 for lab, what, s1, e1, s2, e2, sg, a, b, c, g in rows)
    D.slide('toy', body, f'''
<p>{G['toy']}</p><p>{G['paired']}</p>
<table><tr><th>constant</th><th>no loss</th><th>run_71 O₂</th><th>2×</th><th>bench gas</th></tr>{tr}</table>
<p>Realism check at 243 V/cm, toy against run_71: X σ0 (RC fit) 0.52 vs 0.54 mm, Y D_rc 2.4 vs 2.1×10⁻⁴ mm²/ns, delay c2/c1 0.43 vs
0.47, cascade c 0.62 vs 0.66, b1 0.27 vs 0.22. The toy is NOT realistic at 150 and 92 V/cm: there the data's widths shrink with
falling field and the toy's grow (results kept in the JSON, not shown). The 243 V/cm shifts are quoted for all fields; the loss
per unit time is the same at all three, so the arrival-current argument is too.</p>
<p>Halving the toy's drift diffusion still leaves a D_rc shift of −4 %, so drift diffusion is not the only path: part of the shift is
the ±4-strip window (Y's RC spread carries late charge outside the strip sum that serves as the source).</p>
<p><code>attach_transfer.py --nev 3000 --seeds 4</code> → <code>results/attach_transfer.json</code>.</p>''', short='Injection test')


def s_beam_dt(D, T, R):
    P = sd.Plot(900, 520, x=(70, 265), y=(1.5e-4, 2.6e-4), xlabel='drift field [V/cm]', ylabel='Y D_rc [10⁻⁴ mm²/ns]',
                margin=(16, 30, 86, 110))
    P.xticks([(e, str(e)) for _, e in BEAM]).yticks([(v * 1e-4, f'{v:.1f}') for v in (1.6, 1.8, 2.0, 2.2, 2.4)])
    for arm, col, nm in (('none', GREY, 'no drift diffusion (rc_diffusion)'), ('magboltz', BLUE, 'Magboltz D_T in the fit')):
        xs = [e for _, e in BEAM]
        ys = [R['beam'][l]['y'][arm]['D'] for l, _ in BEAM]
        es = [R['beam'][l]['y'][arm]['D_err'] for l, _ in BEAM]
        P.line(xs, ys, col, 4, dash=None if arm == 'magboltz' else '10 7',
               tips=[f'{nm}, {e} V/cm: D_rc {y * 1e4:.2f} ± {s * 1e4:.2f} ×10⁻⁴; σ0_y {R["beam"][l]["y"][arm]["sig0"]:.3f} mm'
                     for (l, e), y, s in zip(BEAM, ys, es)])
    sD, eD, *_ = T.shift('beam243_noatt', 'beam243_att', 'rc_D', 'y')
    mb = [R['beam'][l]['y']['magboltz']['D'] for l, _ in BEAM]
    corr = np.mean(mb) / (1 + sD)
    P.hline(corr, GREEN, '4 6', 3, label=f'loss-corrected ≈ {corr * 1e4:.2f}', tip=f'mean of the Magboltz arm / (1 {sD:+.2f}) '
            f'(toy shift at the run_71 level): the value with no attachment')
    side = sd.col(
        sd.legend([('Magboltz D_T in the fit', BLUE), ('no drift diffusion', GREY, 'dash')], size=22),
        sd.p(f'det4\'s Y resistive spread is {min(mb) * 1e4:.2f}–{max(mb) * 1e4:.2f}×10⁻⁴ mm²/ns at three fields, whose '
             'depth-to-time mapping differs ×3. Writing the drift diffusion into the fit lowers it 7 %; the attachment '
             f'correction raises it {-sD:.0%}.', 24),
        sd.callout(f'Bench-model value for det4: D_rc ≈ {corr * 1e4:.1f}×10⁻⁴ mm²/ns, ±10 %.', GREEN, 26),
        gap=24, w=640)
    body = sd.title('det4\'s resistive spread on the beam: one number, ±10 %, whatever the field or treatment',
                    'run_71 RAW, Y view head-on, RC fit with and without drift diffusion (rc_drift.py)')
    body += sd.row(P.svg('beamdt'), side, gap=44)
    D.slide('beam-drc', body, '''
<p><code>rc_drift.py</code> is <code>rc_diffusion.fit</code> with drift diffusion written in: the stack's source sample at time t is
taken to have arrived u = t − t_a after the first charge, and lands with width² = σ0² + D_T²·v·u + 2·D_rc·s. D_T and v are
Magboltz at the fitted run_71 composition; t_a is fitted. On the beam t_a runs to its upper bound (the data want even less
depth growth than Magboltz gives), so the Magboltz arm is an upper bound on the diffusion correction.</p>
<p>The toy says the run_71 loss lowers the fitted D_rc by a few per cent; the corrected value divides that out.</p>''',
            short='Beam D_rc')


def s_bench(D, T, R):
    arms = (('none', GREY, 'no drift diffusion'), ('wet_DT275', BLUE, 'D_T 275 (≈1 % water, from v(E))'),
            ('dry_DT480', ORANGE, 'D_T 480 (dry; the bench width growth)'))
    dets = [d for d in BENCH if d in R['bench']]
    P = sd.Plot(1000, 560, x=(-0.6, len(dets) - 0.4), y=(0, 10e-4), ylabel='Y D_rc [10⁻⁴ mm²/ns]',
                margin=(16, 30, 70, 110))
    P.xticks([(i, d) for i, d in enumerate(dets)]).yticks([(v * 1e-4, f'{v:g}') for v in (0, 2, 4, 6, 8, 10)])
    mb = [R['beam'][l]['y']['magboltz']['D'] for l, _ in BEAM]
    P.band([-0.6, len(dets) - 0.4], [min(mb)] * 2, [max(mb)] * 2, RED, 0.18,
           tip=f'det4 on the beam: {min(mb) * 1e4:.2f}–{max(mb) * 1e4:.2f}×10⁻⁴ (three fields, Magboltz D_T)')
    for j, (arm, col, nm) in enumerate(arms):
        for i, d in enumerate(dets):
            q = R['bench'][d]['y'].get(arm)
            if not q:
                continue
            P.points([i + (j - 1) * 0.22], [q['D']], col, 9,
                     tips=[f'{d} bench, {nm}: D_rc {q["D"] * 1e4:.2f}×10⁻⁴, σ0_y {q["sig0"]:.2f} mm, rms {q["rms"]:.4f}'])
    d4 = [R['bench']['det4']['y'][a]['D'] for a, _, _ in arms if a in R['bench'].get('det4', {}).get('y', {})]
    lo_r, hi_r = min(d4) / np.mean(mb), max(d4) / np.mean(mb)
    side = sd.col(
        sd.legend([(n, c, 'dot') for _, c, n in arms] + [('det4 on the beam', RED, 'box')], size=22),
        sd.p('On the bench, D_rc moves by a third with the assumed drift diffusion, and σ0 with it (det2 Y σ0 0.07–0.75 mm '
             'across the arms). The bench cannot separate them; the beam, with 2–3× less drift spread, can.', 24),
        sd.callout(f'det4 measured on both: bench D_rc is {lo_r:.1f}–{hi_r:.1f}× the beam value under every D_T '
                   'assumption. The resistive spread is not yet one number per chamber.', RED, 24),
        gap=22, w=600)
    body = sd.title(f'The bench\'s D_rc depends on the drift diffusion assumed; det4 sits {lo_r:.1f}–{hi_r:.1f}× its beam value',
                    'Head-on Y stacks, five golden bench chambers, RC fit with three drift-diffusion arms (rc_drift.py)')
    body += sd.row(P.svg('bench'), side, gap=40)
    tr = ''.join('<tr><td>' + d + '</td>' + ''.join(
        f'<td>{R["bench"][d]["y"][a]["D"] * 1e4:.2f} / {R["bench"][d]["y"][a]["sig0"]:.2f} / {R["bench"][d]["y"][a]["rms"]:.4f}</td>'
        for a, _, _ in arms if a in R['bench'][d]['y']) + '</tr>' for d in dets)
    D.slide('bench', body, f'''
<p>Cells: D_rc [10⁻⁴ mm²/ns] / σ0_y [mm] / rms.</p>
<table><tr><th>chamber</th>{''.join(f'<th>{n}</th>' for _, _, n in arms)}</tr>{tr}</table>
<p>The bench gas question is itself open: the det3 drift scan's v(E) needs ~1 % water (Magboltz D_T ≈ 270 µm/√cm), while the bench
X width growth prefers dry-size diffusion (X rms best with D_T 480 on det2/det3). A model-free width-growth split by reference
|tan| (<code>width_tan_split.py</code>) was too noisy to settle it.</p>
<p>Not attachment: there is none on the bench, and the toy's bench-gas case puts the diffusion leak into D_rc at only +3…+17 %.
Candidates for det4's bench/beam factor: the resistive layer itself (surface resistivity depends on temperature and humidity; the
two campaigns were five weeks and a different hall apart), the bench Y selection, or a model term both fits lack.</p>''',
            short='Bench D_rc')


def s_apply(D, T, R):
    mb = [R['beam'][l]['y']['magboltz']['D'] for l, _ in BEAM]
    sD, *_ = T.shift('beam243_noatt', 'beam243_att', 'rc_D', 'y')
    sx = [R['beam'][l]['x']['magboltz']['sig0'] for l, _ in BEAM]
    sy = [R['beam'][l]['y']['magboltz']['sig0'] for l, _ in BEAM]
    rows = [
        ['Model structure', 'Y: continuous RC spread; X: none; no discrete copies (c2 &lt; c1 automatic)', 'all chambers',
         'adopt as is'],
        ['Prompt footprint is readout', 'σ0 does not move with gas, field or water; avalanche ~25 µm', 'all chambers',
         'one σ0 per view per chamber, never per setup'],
        ['X wider than Y', f'det4 beam σ0 X {min(sx):.2f}–{max(sx):.2f}, Y {min(sy):.2f}–{max(sy):.2f} mm (RC fit, Magboltz D_T)',
         'all chambers', 'fixed X/Y ratio from the board, not a free knob'],
        ['det4 D_rc', f'{np.mean(mb) / (1 + sD) * 1e4:.1f}×10⁻⁴ mm²/ns ± 10 % (loss-corrected)', 'det4',
         'pin on the bench det4 arm; compare with its bench fit'],
        ['det4 σ0 (leading edge)', 'X 0.42–0.51, Y 0.36–0.38 mm; bench det4 X 0.50', 'det4', 'cross-check, agrees'],
        ['Gain invariance of sharing', 'kernel shape constant to 1–3 % over the run_66 HV scan', 'all chambers', 'no gain term'],
        ['Not transferable', 'v(E), D_T, η of the H4 gas; the 176 µm (≈10 % pessimistic from the loss); beam bundles', '—',
         'bench uses its own gas: v from its drift scan'],
    ]
    body = sd.title('What the bench model takes from SPS', 'Structure for every chamber; numbers for det4; the gas stays behind')
    body += sd.table(['item', 'value / finding', 'applies to', 'how'], rows, size=23, widths=[330, 760, 200, 380],
                     align=['left', 'left', 'left', 'left'])
    D.slide('apply', body, '''
<p>The structural results are the main export: they hold for every chamber and do not depend on any number measured in the beam gas.
The numbers are det4's; the bench measures the other four chambers' σ0 and D_rc itself, with the degeneracy of the bench slide.</p>
<p>The 176 µm intrinsic resolution is a beam-condition performance number. The loss removes on average ~18 % of the charge at 243
V/cm, so a charge-statistics-limited resolution would be ≲ 10 % better without it; it is quoted as measured, in the H4 gas.</p>
<p>The det4 beam calibration bundles (share_lp, beam-pinned) carry the old copy kernel and the H4 v and Dp; they are not used
on the bench.</p>''', short='Apply')


def s_open(D, T, R):
    items = [
        ('det4 bench vs beam D_rc (1.5–2.7×).', 'Not attachment, not drift diffusion within its range. Resistive layer conditions, '
                                           'bench selection, or a model term both fits lack.'),
        ('The bench drift diffusion.', 'v(E) says ~1 % water (D_T ≈ 270); the widths look dry (≈ 480). It sets how well the '
                                       'bench can measure σ0 and D_rc at all.'),
        ('Toy realism below 243 V/cm.', 'The toy\'s widths grow with falling field, the data\'s shrink. Shifts quoted from '
                                        '243 V/cm only.'),
        ('Three-body O₂ scale.', 'Irrelevant here: the test uses the measured η·v, not the ppm.'),
    ]
    lis = ''.join(f'<li style="margin-bottom:14px"><b style="color:{DINK}">{a}</b> {b}</li>' for a, b in items)
    dec = [
        ('Bench arm with det4 D_rc pinned to the beam value', 'paired compare_bench against the bench-fitted rcm'),
        ('Settle the bench D_T', 'drift-scan widths, or a cleaner head-on selection'),
        ('Then freeze the physical kernel', 'σ0 per view, D_rc per chamber, Magboltz D_T, X template'),
    ]
    dl = ''.join(f'<li style="margin-bottom:12px"><b style="color:{DINK}">{a}</b> — {b}</li>' for a, b in dec)
    body = (sd.kicker('What this does not rule out, and what is next')
            + '<div style="display:flex;gap:72px">'
            + f'<div style="flex:1.35"><ul style="font-size:25px;line-height:1.38;color:{DMUT};padding-left:28px">{lis}</ul></div>'
            + f'<div style="flex:1"><p style="font-size:30px;color:{DINK};font-weight:600;margin-bottom:18px">Next</p>'
            + f'<ul style="font-size:26px;line-height:1.4;color:{DMUT};padding-left:28px">{dl}</ul></div></div>')
    D.slide('open', body, '''
<p>Numbers and method: <code>cloud_basics/FINDINGS.md</code> §27. Companion: notes/mx17-attachment-slides (the attachment itself).</p>''',
            dark=True, short='Open')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT, 'mx17-sps-to-bench.html'))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    T = Toy()
    rd = J('rc_drift.json')
    R = {'beam': {l: rd[l] for l, _ in BEAM if l in rd}, 'bench': {d: rd[d] for d in BENCH if d in rd}}
    D = sd.Deck('SPS constants on the bench',
                'Does the O₂ attachment in the SPS beam gas contaminate the det4 model constants, and what carries over '
                'to the cosmic-bench model.')
    s_cover(D, T, R)
    s_inventory(D, T, R)
    s_cancel(D, T, R)
    s_toy(D, T, R)
    s_beam_dt(D, T, R)
    s_bench(D, T, R)
    s_apply(D, T, R)
    s_open(D, T, R)
    p = D.write(a.out, note_meta=dict(
        title='SPS det4 constants on the bench: attachment does not spoil them',
        summary='Injection test of the SPS measurement code with the run_71 O₂ loss on and off: footprint, resistive spread '
                'and c2/c1 move ≤ 7 %. What the bench model takes from SPS, and the bench-side degeneracy that remains.',
        tags='X17, beam test, micromegas, gas, attachment', date='2026-10-11'))
    import re
    ids = [s_[0] for s_ in D.slides]
    txt = p.read_text()
    missing = sorted(set(re.findall(r'§§([a-z0-9-]+)§§', txt)) - set(ids))
    if missing:
        raise SystemExit(f'unknown slide references: {missing}')
    p.write_text(re.sub(r'§§([a-z0-9-]+)§§', lambda m: str(ids.index(m.group(1)) + 1), txt))
    print('wrote', p, f'({len(ids)} slides)')


if __name__ == '__main__':
    main()
