"""Slide note: results of the 2026-10-27 model-audit workstreams (F1-F22), figure first.

Reads each workstream's own outputs under rerun/model_audit/ws_*/ (result JSONs, and the
figures the workstreams rendered, which are gitignored and live in the main checkout), so
rerunning after a workstream updates changes the note with it. The SPS drift-velocity slide
(workstream C, ws_spsv) reads ws_spsv/REPORT.md when it exists and says "pending" until then.

    .venv/bin/python rerun/model_audit/make_audit_deck.py
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        rerun/model_audit/deck/mx17-model-audit.html --slug mx17-model-audit --force --deploy
"""
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, PURPLE, GREY, GREEN, INK, MUT, RULE,  # noqa: E402
                      DBLUE, DRED, DGREEN, DGREY, term)

HERE = Path(__file__).resolve().parent
OUT = HERE / 'deck' / 'mx17-model-audit.html'
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
XC, YC = BLUE, ORANGE          # X view / Y view, every slide


def J(rel):
    return json.loads((HERE / rel).read_text())


def fig(rel, w=None, h=None, tip=None, crop=None, scale=1.3):
    """A workstream's PNG, optionally cropped (fractions l, t, r, b), resized to ~scale x
    its slide box and palette-quantised so the single-file page stays small."""
    im = Image.open(HERE / rel).convert('RGB')
    if crop:
        W, H = im.size
        im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
    box_w = w or (h * im.width / im.height)
    tw = int(min(im.width, box_w * scale))
    if tw < im.width:
        im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(b, 'PNG', optimize=True)
    return sd.img(b.getvalue(), w=w, h=h, tip=tip)


def foot_ws(*ws):
    return 'Source: ' + ', '.join(f'rerun/model_audit/{w}/REPORT.md' for w in ws)


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def s_cover(D):
    body = (sd.kicker('MX17 · cosmic bench + SPS det4 · model audit, 27 Oct 2026') +
            '<h1 style="font-size:76px;font-weight:600;line-height:1.08;letter-spacing:-1.5px;color:#eef0f3">'
            'The angle problems the audit found were mostly calibration, not the physics model, '
            'and they now go through one reconstruction</h1>' +
            sd.row(sd.bignum('22', 'audit findings', DBLUE, 'F1–F22, rerun/model_audit/FINDINGS.md'),
                   sd.bignum('8', 'independent workstreams', DGREEN, 'one per question, all eight done'),
                   sd.bignum('5 + 1', 'chambers', DGREY, 'det2/3/4/6/7 on the bench, det4 at SPS H4'),
                   sd.bignum('8000', 'muons per chamber in the final gate', DRED, 'unified reconstruction vs production'),
                   gap=56))
    notes = """
<p>What each workstream answered, one line each (details on the slides):</p>
<ol>
<li><b>t0 (F1, F2, F5, F6)</b>: the start time of the charge is one DAQ latency for the whole bench, ~300–340 ns; fixing it
improves position by 4–31 %.</li>
<li><b>Drift velocity uncertainty (F8)</b>: the quoted ±0.07 µm/ns was statistics only; the real spread is 2 % (det2/3) to
5–8 % (det4/6/7), because the gas changed during runs.</li>
<li><b>w0 (F9)</b>: a mechanical tilt of the chamber against the telescope, measured independently from positions.</li>
<li><b>Steep tracks read shallow (F4, F18, F21)</b>: a selection cut plus a bowed chamber; a position map fixes 1–27 % of
the resolution. A ~2 % residual is unexplained.</li>
<li><b>SPS (F14, F18, F20)</b>: a 256-ADC offset bug voided the old SPS fits; head-on RAW data fix the prompt footprint
σ0 at 0.32 mm (Y) and ≤ 0.29 mm (X).</li>
<li><b>Physics floor (F3)</b>: a toy says 0.30–0.35°, set by diffusion; we measure 3–6× that.</li>
<li><b>Dead strips and x–y charge balance (F12, F15)</b>: two kinds of fault (chamber vs electronics); the charge balance
is not a gain problem.</li>
<li><b>SPS drift velocity (F16)</b>: the old 14 % was the fit's depth grid; v(243 V/cm) = 14.1 µm/ns from the
column duration; an 8 % slope deficit remains (local gap or fit, not separable).</li>
<li><b>One reconstruction (F22, F7, F10, F11, F13)</b>: no per-dataset switches; finds 1–5 % more tracks.</li>
</ol>
<p>Ledger: <code>rerun/model_audit/FINDINGS.md</code>; each workstream's report: <code>rerun/model_audit/ws_*/REPORT.md</code>.</p>"""
    D.slide('cover', body, notes, dark=True, short='cover')


def fit_diagram(w=1000, h=640):
    """Side view of the drift gap: a track, electrons drifting to the strips, and the fit's parameters."""
    x0, x1, yc, ym = 120, 900, 90, 470          # cathode y, mesh y
    o = []
    o.append(sd.line(x0, yc, x1, yc, INK, 4))
    o.append(sd.T(x0 - 14, yc + 8, 'cathode', 22, MUT, 'end'))
    o.append(sd.line(x0, ym, x1, ym, BLUE, 5))
    o.append(sd.T(x0 - 14, ym + 8, 'mesh', 22, BLUE, 'end'))
    for i in range(26):                           # strips under the mesh
        xs = x0 + 10 + i * 30
        o.append(f'<rect x="{xs}" y="{ym + 22}" width="20" height="12" fill="{GREY}" opacity=".7"/>')
    o.append(sd.T(x0 - 14, ym + 36, 'strips', 22, MUT, 'end'))
    ax, bx = 330, 640                              # track enters at top (ax), exits at mesh (bx)
    o.append(sd.line(ax - 40, yc - 50, bx + 40, ym + 50, RED, 4))
    o.append(sd.T(ax - 46, yc - 58, 'muon', 24, RED, 'end'))
    n = 7
    for k in range(n):
        f = (k + 0.5) / n
        ex, ey = ax + f * (bx - ax), yc + f * (ym - yc)
        o.append(f'<circle cx="{ex:.0f}" cy="{ey:.0f}" r="7" fill="{GOLD}"/>')
        o.append(sd.arrow(ex, ey + 10, ex, ym - 6, GOLD, 2))
    o.append(sd.T(bx + 70, (yc + ym) / 2 - 30, 'electrons drift down at v', 24, GOLD, 'start'))
    o.append(sd.T(bx + 70, (yc + ym) / 2 + 2, 'deeper charge arrives later', 24, GOLD, 'start'))
    # depth arrow
    o.append(sd.arrow(x1 + 40, ym, x1 + 40, yc, MUT, 2))
    o.append(sd.T(x1 + 58, (yc + ym) / 2 + 120, 'drift depth 30 mm', 22, MUT, 'middle', rot=-90))
    # the fit's line in time vs position, below
    ty = ym + 90
    o.append(sd.T(x0, ty + 4, 'The fit:', 26, INK, 'start', 600))
    o.append(sd.T(x0 + 110, ty + 4, 'charge arriving at time t lands at  p0 + w·(t − t0)', 26, INK, 'start',
                  tip='p0: position at the mesh; w: lateral drift speed of the landing point; t0: when the '
                      'mesh-level charge arrives. The depth of a charge is v·(t − t0).'))
    o.append(sd.T(x0 + 110, ty + 44, 'track angle:  tan θ = w / v', 26, INK, 'start',
                  tip='v is the drift velocity from the chamber calibration (the "ladder" v).'))
    return sd.svg(w, h, ''.join(o), 'drift gap diagram')


def s_setup(D):
    params = sd.col(
        sd.p('What the reconstruction fits, per chamber view:', 26, MUT),
        sd.callout(f'{term("t0", "When the charge from the mesh level arrives on the strips. F1, F2, F5, F6 are about it.")}'
                   ': when the charge starts', RED),
        sd.callout(f'{term("p0", "Where the track crosses the mesh, across the strips.")}: where the track crosses the mesh', BLUE),
        sd.callout(f'{term("w", "How fast the landing point moves across the strips as later (deeper) charge arrives.")}'
                   ' / v = the angle', GREEN),
        sd.callout(f'{term("σ0", "The prompt footprint: how wide one point of charge is on the strips before drift diffusion. F18, F20, F21.")}'
                   ', D: how wide the charge is', PURPLE),
        sd.p('Hover any dotted word or data point for its definition and numbers. Details under each slide '
             'carry the method and caveats.', 22, MUT),
        gap=22, w=600)
    body = sd.title('What the fit does, and which handle each finding touches',
                    'Side view of one chamber: a cosmic muon ionises the 30 mm gas gap; the strips record when and '
                    'where the charge lands') + sd.row(fit_diagram(), params, gap=40, align='center')
    notes = """The reconstruction (<code>wft/</code>) fits each view's strip waveforms with a forward model: the track's
charge is cut into depth slices (60 ns bins), each slice is spread by diffusion and the strips' resistive-layer sharing,
and the fit adjusts t0, p0 and w (plus per-slice charges) to match the waveforms. The angle is tan θ = w / v with v the
chamber's calibrated drift velocity. The bench reference is the M3 telescope (four planes, ~0.4 mm pointing).

The calibration bundle per chamber and run condition carries v, the RC kernel, σ0 and Dp, w0/kw (angle offset and
scale against M3), the dead-strip list and t0 information."""
    D.slide('setup', body, notes, short='what the fit does')


def s_t0_float(D):
    body = (sd.title('t0 floated: an earlier t0 draws the same line',
                     'Left: two (t0, p0) pairs give one line; only the modelled cloud width differs. '
                     'Right: per-event χ² vs t0 (det3), side minima one 60 ns bin away') +
            sd.row(fig('ws_t0/figures/t0_explainer.png', w=1100,
                       tip='t0_explainer.png: shifting t0 and p0 together along the track leaves the line unchanged; '
                           'only the modelled diffusion width (right) changes.'),
                   fig('ws_t0/figures/t0_valley.png', w=520,
                       tip='t0_valley.png: per-event chi2 vs t0 offset, det3. Soft, one-sided valley with side minima one '
                           '60 ns bin away.'), gap=44, align='center'))
    notes = """<b>F6.</b> Charge in bin k sits at time t0 + u_k and position p0 + w·u_k. Moving t0 later and p0 along the
track by w·Δt0 reproduces the same line: the only things that pin t0 are the growth of the diffusion width with depth and
"no charge arrives before the mesh". So the free t0 wanders ±53–140 ns per event, with side minima one 60 ns bin away;
the reported t0 error of 1–2 ns is meaningless, and because p0 slides with it, positions of inclined tracks suffer.

The old ±5 ns prior on t0 was invisible: next to χ² ~ 10⁴ it carried no weight. Source: <code>ws_t0/REPORT.md</code>."""
    D.slide('t0-float', body, notes, foot=foot_ws('ws_t0'), short='t0 floats')


def s_t0_latency(D):
    le = J('ws_t0/lead_edge.json')
    pin = J('ws_t0/pin_compare_lat.json')
    P = sd.Plot(800, 560, x=(-0.5, 4.5, 'lin'), y=(270, 370, 'lin'), ylabel='earliest pulse, phase-corrected [ns]',
                title='Charge arrival, measured without any fit')
    P.xticks([(i, d) for i, d in enumerate(DETS)])
    P.yticks([(v, str(v)) for v in range(280, 371, 20)])
    P.band([-0.5, 4.5], [300] * 2, [340] * 2, GREY, 0.12, tip='300–340 ns: the one bench latency C')
    for v, c, dx in (('x', XC, -0.12), ('y', YC, 0.12)):
        xs, ys, tips = [], [], []
        for i, d in enumerate(DETS):
            r = le[d][v]
            xs.append(i + dx); ys.append(r['t_lead'])
            tips.append(f'{d} {v.upper()} (FEU {r["feu"]}): {r["t_lead"]:.0f} ns, spread {r["t_lead_rsig"]:.0f} ns, '
                        f'n={r["n"]}; ftst slope {r["ftst_slope_uncorrected"]:.1f} ns/class')
        P.points(xs, ys, c, r=11, tips=tips)
    P.text(0.0, 362, 'one latency, every chamber and view', 26, INK, 'start', 600)
    lat = P.svg('latency per chamber')

    Q = sd.Plot(800, 560, x=(-0.5, 4.5, 'lin'), y=(-40, 8, 'lin'), ylabel='mesh position resolution change [%]',
                title='Pin t0 at the latency: position improves')
    Q.xticks([(i, d) for i, d in enumerate(DETS)])
    Q.yticks([(v, f'{v}') for v in range(-40, 6, 10)])
    Q.hline(0, MUT, None, 1.5)
    for v, c, dx in (('x', XC, -0.18), ('y', YC, 0.18)):
        for i, d in enumerate(DETS):
            r = pin[d][v]
            pc = 100 * r['delta'] / r['base']
            Q.vbar(i + dx, pc, 46, c, tip=f'{d} {v.upper()}: {r["base"]:.3f} → {r["arm"]:.3f} mm ({pc:+.0f} %, '
                                            f'±{100 * r["err"] / r["base"]:.0f} %), n={r["n"]} held-out tracks; '
                                            f'M3 ~0.40 mm still included')
    Q.text(-0.3, 3, 'angles unchanged (±0.03°)', 24, INK, 'start', 600)
    body = (sd.title('t0 is one DAQ latency; pinning it improves position',
                     'Left: earliest charge on the strips, every golden chamber. Right: held-out tracks, pinned t0 '
                     'vs free t0') +
            sd.row(lat, Q.svg('pin effect'), gap=44) +
            sd.legend([('X view', XC, 'dot'), ('Y view', YC, 'dot'), ('300–340 ns band', '#d0d3d8', 'box')]))
    notes = """<b>F1: yes, one latency.</b> All golden rows use the same DREAM configuration and the scintillator
trigger. The fine timestamp ftst is the trigger phase in the 60 ns clock (−10 ns per class, measured −7.7…−12.0), so
t0 = C − 10 ns·ftst. C measured fit-free (earliest 30 % pulse crossing, phase-corrected), X / Y: det2 300/309, det3
319/327, det4 334/339, det6 323/327, det7 304/304 ns. No X–Y FEU offset.

<b>Pinning</b> at C − 10·ftst (1320 held-out tracks per chamber): mesh position vs M3 improves 4–31 %; angle s68
within ±0.03°; χ²/dof rises 4–20 % (model error the free t0 had absorbed). A scan of C shows the angle scale moves
+5 % per 100 ns of C and matches the timing v on all ten planes at C ≈ 300–330 ns: the old stored t0s were too late,
which is the +2…+5 % ladder shift of F6.

<b>F5:</b> once t0 is pinned, the x–y t0 coincidence checks nothing. Compare column mid-times (t0 + u50) instead: they
centre at 0 with no table. Not yet implemented.

<b>Never pin on n_TOF</b>: there the trigger is not the particle. Code: <code>wft/calibrate.py measure_latency</code>,
<code>set_w0kw.py --latency-cache --pin-t0</code>. No bundle uses the pin yet."""
    D.slide('t0-latency', body, notes, foot=foot_ws('ws_t0'), short='t0 = latency')


def s_v(D):
    body = (sd.title('The v error was statistics; the gas changed in runs',
                     'Left: four sizes of v uncertainty per chamber view. Right: v from the angle route vs v from '
                     'the timing route, fifths of each run') +
            sd.row(fig('ws_v_w0/figures/v_budget.png', w=820,
                       tip='v_budget.png: grey = ledger (statistics), blue = real error on the run mean, orange = '
                           'local variation in time and position, red = half the +tan / −tan difference.'),
                   fig('ws_v_w0/figures/v_time_two_routes.png', h=600,
                       tip='v_time_two_routes.png: two independent routes move together along the diagonal, so '
                           'the change is real gas, not the fit.'), gap=36, align='center'))
    notes = """<b>F8: you were right, ±0.07 µm/ns (0.2 %) was too small.</b> Three different numbers:
<ol><li>Error on the run-mean v: ±0.16–0.27 µm/ns (0.4–0.7 %) for det2/det3, ±0.4–1.2 for det4/6/7. Includes time-block
bootstrap, estimator and window choice, and the M3 angle scale (~0.25 %, irreducible).</li>
<li>Local spread around it, which is what matters when v is used elsewhere: 1.7–2.1 % (det2/3), 5–8 % (det4/6/7).
In time: det4 −9…−10 % over its day run, det6 +12 %, det7 +4 % over the shared night. The independent column-duration
route moves the same way (right panel), so the gas really changed.</li>
<li>+tan vs −tan agree to ≤ 1 % (det6 Y 2.7 %); the apparent 2–7 % came from a biased head-on offset in
<code>fleet_tables.py</code>, still to fix.</li></ol>
<b>det3 X vs Y</b> differ by +2.1 ± 0.4 % on the same events. The sign changes between chambers (det4 −3.0 %, det6
+1.9 %), so it is not strip-layer geometry; it is an angle-scale error of the size of the width–tilt trade. On the bench
kw absorbs it; elsewhere it is a 2–3 % per-view systematic."""
    D.slide('v', body, notes, foot=foot_ws('ws_v_w0'), short='v uncertainty')


def s_w0(D):
    body = (sd.title('w0 is the chamber sitting tilted on the bench, measured twice',
                     'Left: what w0 is. Right: angle offset from the fit vs tilt from position residuals alone, '
                     '16 chamber views') +
            sd.row(fig('ws_v_w0/figures/w0_tilt_diagram.png', w=840,
                       tip='w0_tilt_diagram.png (tilt drawn ×20). A vertical muon in a tilted chamber reads a small '
                           'angle; the mesh height also changes across the chamber.'),
                   fig('ws_v_w0/figures/w0_vs_tilt.png', h=620,
                       tip='w0_vs_tilt.png: r = 0.81 over 16 views; the dashed line is "tilt explains it all".'),
                   gap=36, align='center'))
    notes = """<b>F9.</b> w0 is the fitted lateral speed w for a track that M3 calls perpendicular, i.e. an angle
offset α = w0/v. Negative is not unphysical: it is a direction. A tilted chamber also gives a position residual that
grows with tan × position, which measures the same tilt from positions alone. The two agree (r = 0.81 over 16 chamber
views); rms offset 0.34° → 0.18° after subtracting the position tilt. det6 Y −0.86° vs −0.89°. det3 moved to the bottom
slot (6-25) changed its offset with the mounting.

<b>Sign</b>: every Y offset is negative (−0.27 … −0.89°): the bench slots are not level along M3 x. Fixed by M3's z
axis pointing up (upper plane at 1302 mm), an external fact.

The apparent det3 field dependence was an artefact of <code>set_w0kw.py</code>; as an angle it is constant (−0.28 /
−0.24 / −0.28° at 1000 / 700 / 500 V).

<b>Risk</b>: bench w0 is mounting, so it must not be used at n_TOF. The n_TOF bundles were bench transfers including
w0/kw; which w0 stage 3 actually used is not checked."""
    D.slide('w0', body, notes, foot=foot_ws('ws_v_w0'), short='w0 = tilt')


def s_f4_cut(D):
    body = (sd.title('Most of the steep-track fall was a selection cut',
                     'Measured slope / true slope vs M3 angle. Dashed: the F4 analysis (cut on the fitted slope). '
                     'Solid: no cut, tilt removed') +
            fig('ws_widthtilt/figures/fig_f4_gate.png', w=1664,
                tip='fig_f4_gate.png: the cut |tan_fit| < 0.08 removed fits that came out too shallow in the '
                    'shallowest bin, inflating it 3–13 %.'))
    notes = """<b>F4.</b> <code>03_angles.py</code> dropped fits with |tan_fit| &lt; 0.08. In the shallowest reference bin
that removes exactly the fits that came out too shallow, so that bin is inflated by 3–13 % and the curve appears to fall
with angle. Without the cut (w0 removed, ± sides averaged), the steepest bin (|tan| ≈ 0.36) is within 0–6 % of the
shallow ones: det7 0, det3 ~1.5, det4 3–4, det2 Y 4.5, det6 X 6 %.

The rest of the fall is the chamber bow (next slide). Toys with the real windows, noise and gains close to ≤ 0.3 % at
every angle, so the fitter itself is unbiased on its own model."""
    D.slide('f4-cut', body, notes, foot=foot_ws('ws_widthtilt'), short='F4: selection cut')


def s_f4_bow(D):
    body = (sd.title('The rest is a bow: the tilt varies across each chamber',
                     'Left: tilt offset w0 vs position, X view (ramps ±0.6 µm/ns). Right: mesh height from M3 '
                     'positions alone (points) vs the bow the w0 ramp predicts (dashed)') +
            sd.row(fig('ws_widthtilt/figures/fig_w0_profile.png', w=620, crop=(0, 0.035, 0.5, 0.515),
                       tip='fig_w0_profile.png (top half): lateral drift offset w0 vs track position, all chambers.'),
                   fig('ws_widthtilt/figures/fig_bow_test.png', w=1020,
                       tip='fig_bow_test.png: points = mesh height fitted from M3 position residuals; dashed = bow '
                           'predicted from the w0 ramp. Same sign on 10/10 views, 1–3 mm dish.'),
                   gap=24, align='center'))
    notes = """w0 is not constant: on every chamber and both views it ramps from about −0.6 to +0.6 µm/ns over 300 mm,
zero near the centre. Through the telescope's acceptance, position and angle are correlated (strongly on det6), so a
ramp in position looks like an angle non-linearity.

Independent check from M3 positions only: the mesh-height curvature fitted from position residuals vs tan has the same
sign on all 10 chamber views, 120–250 /km (40–200 /km predicted from the w0 ramp), i.e. a 1–3 mm dish. Gravity sag of
an edge-supported board is the candidate; which way the centre sits is not pinned.

The bow also biases position by tan × height (~±0.45 mm at tan 0.3), likely part of F14's 0.5–0.76 mm bench position
resolution. Bench geometry: never transfer this map to n_TOF."""
    D.slide('f4-bow', body, notes, foot=foot_ws('ws_widthtilt'), short='F4: chamber bow')


def s_f4_fix(D):
    body = (sd.title('A position map of tilt and scale: angles 1–27 % better',
                     'Angle resolution s68, all angles; grey = production (one w0, one scale), green = position map '
                     'measured on the other half of the data') +
            sd.row(fig('ws_widthtilt/figures/fig_resolution_map.png', w=1160,
                       tip='fig_resolution_map.png: cross-validated (measure on one half, apply to the other); far '
                           'fractions unchanged.'),
                   sd.col(sd.callout('Measured on the bench from M3, applied in the angle conversion', GREEN),
                          sd.callout('Remaining: ~2 % symmetric compression (det4 Y 7 %), unexplained', RED),
                          sd.callout('Bench geometry: not for n_TOF', GREY), gap=28, w=460),
                   gap=44, align='center'))
    notes = """Cross-validated s68 change X / Y: det2 −12/−26 %, det3 −12/−16, det4 −3/−25, det6 −26/−27, det7 −1/−18 %.
Code: optional <code>angle_map</code> in <code>wft/calib.py</code>, applied by <code>tan_of()</code> in
<code>wft/reco.py</code>; test <code>wft/tests/test_angle_map.py</code>. The map is bundle content, so it fits the single
reconstruction. Not yet written into the production bundles.

Residual after the correction: a symmetric compression between |tan| 0.15 and 0.36 of ~2 ± 0.5 % (det3, det4 X, det6 X),
4 % det2 Y, 7 % det4 Y, none on det7. Unexplained. An M3 plane-scale error is excluded only by plausibility (it would need
a 5–10 % scale error). Rim bins (&lt; 30 mm from the edge) flip sign on det3 X and det2, unchecked."""
    D.slide('f4-fix', body, notes, foot=foot_ws('ws_widthtilt'), short='F4: the fix')


def s_width(D):
    body = (sd.title('The width–tilt trade is real, but only a percent',
                     'Left: fitted slope of steep tracks vs the footprint σ0 the model assumes (data). Right: toys, '
                     'a model wider than the truth compresses ~2 %, mostly at shallow angles') +
            sd.row(fig('ws_widthtilt/figures/fig_width_trade.png', w=810,
                       tip='fig_width_trade.png: σ0 0.24 → 0.54 mm lowers the slope 1–3 % on steep tracks (det4, det6).'),
                   fig('ws_widthtilt/figures/fig_toys.png', w=810,
                       tip='fig_toys.png: fitter on its own model = 1.000; footprint 0.8× or diffusion 0.5× in truth '
                           '→ 1–2 % compression, largest at shallow angles (the opposite of F4).'),
                   gap=44))
    notes = """<b>F21.</b> On data (det4, det6) changing the assumed footprint σ0 from 0.24 to 0.54 mm lowers the
slope 3–10 %, most at shallow angles. No (σ0, Dp) pair on the grid makes the M3 angle the χ² minimum, so the trade is
real but percent-level, not the cause of the steep-track compression.

<b>F18.</b> The per-event wide/narrow footprint choice is harmful: in toys a truly wider footprint picked this way is
unbiased, but in data the wide-pick fits read 5–15 % shallower even in the chamber centre and are 1.3–5× brighter. With
the choice off, bright tracks still read 5–10 % shallower on det6/det7: a brightness-dependent mismatch the model does
not describe.

<b>Resolution of the trade</b>: fix σ0 from SPS head-on data (next slides) fleet-wide, drop the per-event choice, and keep
the bow correction separate as geometry. Not yet done or gated in the single reconstruction.

Width probes on det2/det3/det7 are still on condor; the data trade rests on det4/det6 only."""
    D.slide('width', body, notes, foot=foot_ws('ws_widthtilt'), short='width–tilt trade')


def s_sps_sigma0(D):
    h = J('ws_sps/results/sigma0_dp_headline.json')
    F = h['fields_Vcm']
    P = sd.Plot(900, 600, x=(80, 260, 'lin'), y=(0.15, 0.55, 'lin'), xlabel='drift field [V/cm]',
                ylabel='prompt footprint σ0 [mm]', title='det4 at SPS, head-on RAW tracks')
    P.xticks([(v, str(v)) for v in (100, 150, 200, 250)])
    P.yticks([(v, f'{v:.2f}') for v in (0.2, 0.3, 0.4, 0.5)])
    P.hline(0.343, YC, '8 6', 2, label='bench production Y 0.343', tip='σ0_Y used in production bundles (det3)')
    P.hline(0.24, XC, '8 6', 2, label='bench production X 0.24', where='below')
    P.hline(0.50, GREY, '3 6', 2, label='per-event "wide" choice (Y 0.50, X 0.42)', tip='The per-event profile choice; no physical basis')
    for k, c, dx in (('x', XC, -2.5), ('y', YC, 2.5)):
        ys, es = h[f'sigma0_{k}_mm'], h[f'sigma0_{k}_err_mm']
        tips = [f'{k.upper()} at {f} V/cm: σ0 = {y:.3f} ± {e:.3f} mm (1200 clean tracks, bootstrap)' for f, y, e in zip(F, ys, es)]
        for f, y, e in zip(F, ys, es):
            P.raw(sd.line(P.X(f + dx), P.Y(y - e), P.X(f + dx), P.Y(y + e), c, 3))
        P.points([f + dx for f in F], ys, c, r=10, tips=tips)
    P.text(90, 0.18, 'σ0 is flat with field and with pulse height', 26, INK, 'start', 600)
    body = (sd.title('SPS pins the footprint: σ0 0.32 mm (Y), ≤ 0.29 mm (X)',
                     'Best-fit σ0 per view and drift field, det4 RAW run_71, with Dp and the Y RC spread profiled') +
            sd.row(sd.col(P.svg('sigma0 vs field'), sd.legend([('X view', XC, 'dot'), ('Y view', YC, 'dot')]), gap=8, w=900),
                   fig('ws_sps/figures/sigma0_vs_amplitude.png', w=720,
                       tip='sigma0_vs_amplitude.png: best σ0 in thirds of pulse height; flat within ±0.03 mm.'),
                   gap=44, align='start'))
    notes = """<b>F18 status: the idea works for Y, and as a bound for X.</b> 1200 clean head-on tracks per view and
field, per-event free (p0, w, t0) then χ² on a (σ0, Dp, D_rc) grid.
<table><tr><th>field</th><th>σ0 X</th><th>σ0 Y</th><th>Dp X</th><th>Dp Y</th><th>Magboltz Dp</th></tr>
<tr><td>243 V/cm</td><td>0.306 ± 0.007</td><td>0.297 ± 0.020</td><td>0.0109</td><td>0.0086</td><td>0.0057</td></tr>
<tr><td>156 V/cm</td><td>0.287 ± 0.006</td><td>0.332 ± 0.012</td><td>0.0100</td><td>0.0035</td><td>0.0052</td></tr>
<tr><td>96 V/cm</td><td>0.285 ± 0.008</td><td>0.324 ± 0.014</td><td>0.0100</td><td>0.0046</td><td>0.0050</td></tr></table>
σ0_Y 0.32 ± 0.02 vs bench production 0.343: within 7 %, so one physical Y footprint is supported. σ0_X 0.29 is an upper
bound: the X width grows with drift time at twice Magboltz, Y does not; this is a detector effect (likely the unmodelled
one-sided X copy). Defensible X range 0.24–0.29 mm.

σ0 is flat ±0.03 mm with amplitude at every field. Per event σ0 is unconstrained (0.19–0.41) and 8–14 % pick ≥ 0.42 by
scatter alone, so the per-event wide-footprint choice has no physical basis for unsaturated pulses. Saturation untested
(beam max 3342 ADC). Proposal: σ0_Y ≈ 0.33, σ0_X 0.24–0.29 fleet-wide; drop the per-event profile."""
    D.slide('sps-sigma0', body, notes, foot=foot_ws('ws_sps'), short='SPS: σ0')


def s_sps_zs(D):
    body = (sd.title('Zero-suppressed data work if zeros count as measured',
                     'Same events, RAW vs zero-suppression emulated on RAW: shift of the fitted track time (top) '
                     'and position (bottom) for three treatments') +
            sd.row(fig('ws_sps/figures/zs_treatment.png', h=630,
                       tip='zs_treatment.png. Grey: suppressed = measured zero (best). Red: censored, what the F16/F20 fits '
                           'used (worst). Blue: zero-or-below-threshold bound.'),
                   sd.col(sd.callout('Measured zeros: t0 within 27 ns, position within 15–27 µm of RAW', GREEN),
                          sd.callout('Censored (old F16/F20 fits): 2–3× worse', RED),
                          sd.callout('Separately: every old SPS cache sat 256 ADC too high; F16 is redone (next slides)', GOLD),
                          gap=28, w=520), gap=40, align='center'))
    notes = """<b>F20.</b> Run_71 RAW gives truth; zero suppression is emulated on the same events. Treating suppressed
samples as measured zeros (they are: the DAQ measured them below threshold): t0 ≤ 27 ns, position 15–27 µm, slope
0.03–0.07 µm/ns from RAW. Censoring them (what F16/F20 used): 44–66 ns, 45–86 µm, 0.14–0.31 µm/ns. The bound treatment
sits in between. With zeros measured, ZS recovers σ0_X and Dp_X; it cannot separate σ0 from RC spread in Y. Now the
default in the single reconstruction.

<b>Cache bug</b>: the EOS-built ZS caches never subtracted the ZS baseline, so kept samples sat 256 ADC above the
suppressed zeros, and RAW rows sat on raw pedestals of 300–700 ADC. Fixed in 7505338. F16 and F20 numbers came from those
caches and are void; F16 is redone on the fixed caches (next slides), the 25.64° angle numbers of F20 are not yet.

Also fixed: front-end-dropped RAW samples (23 %) had been fitted as zeros (χ² +30 %); they are now marked missing."""
    D.slide('sps-zs', body, notes, foot=foot_ws('ws_sps'), short='SPS: zero suppression')


def s_sps_what(D):
    G, R, Y_ = GREEN, RED, GOLD
    rows = [('prompt footprint σ0, Y', '0.32 ± 0.02 mm', 'yes (readout)', 'fix ~0.33 fleet-wide', G),
            ('prompt footprint σ0, X', '≤ 0.29 mm', 'as a bound', 'range 0.24–0.29', G),
            ('σ0 vs pulse height', 'flat ±0.03 mm', 'yes', 'drop the per-event choice', G),
            ('charge sharing shape', 'Y RC, X none, ±1 delay', 'yes', 'already in the model', G),
            ('electronics pulse shape', 'same as bench', 'yes', 'bench templates', G),
            ('position at 0°', '176 µm (det4)', 'technology bound', 'M3 deconvolution', Y_),
            ('position at 25.6°', '+0.6 mm (inclined view)', 'det4 only', 'none', Y_),
            ('drift diffusion Dp', 'X 2× Magboltz', 'no (different gas)', "don't force Magboltz on X", R),
            ('drift velocity', '14.1 µm/ns at 243 V/cm', 'no (different gas)', 'none', R),
            ('gain', '×20 swings in bands', 'no', 'none', R)]
    tb = sd.table(['quantity', 'SPS gives', 'transfers to bench?', 'constraint on bench'],
                  [[f'<span style="color:{c};font-weight:600">●</span> {a}', b, t, k] for a, b, t, k, c in rows],
                  size=25, align=['left', 'left', 'left', 'left'], widths=[480, 380, 330, 420])
    body = (sd.title('SPS constrains the readout on the bench, not the gas',
                     'det4 at H4. Bench gas is Ar/iso 95/5; SPS ran wet Ar/CO₂/iso and Ar/CF₄/iso with O₂, so no gas '
                     'property transfers') + tb +
            sd.legend([('transfers: use as a constraint', GREEN, 'dot'), ('det4 / technology number only', GOLD, 'dot'),
                       ('gas-dependent: no transfer', RED, 'dot')]))
    notes = """<b>F20, the summary.</b> Things set by the strips and electronics (footprint, sharing, pulse shape)
transfer. Things set by the gas (drift velocity, diffusion, attachment) do not, because the gases differ. Gain does not.

F14: SPS measured det4's intrinsic resolution at 0° as 176 µm (earlier, 3-pitch method). In this workstream, position minus
the uRWELL prediction gives 391 / 311 µm (X / Y) at 0°, upper bounds including a ~250 µm reference term. The bench's
~0.5 mm is consistent with M3 pointing (~0.4 mm) plus angle, plus the bow's ±0.45 mm. At 25.64° the inclined view adds
~0.6 mm in quadrature (pessimistic: censored ZS, low-field plateaus).

The x–y t0 difference on clean RAW head-on tracks has σ 84–118 ns: t0 is uncertain at 60–80 ns per plane, intrinsically.

Still to run: the SPS gate with the measured H4 dead-strip mask; the angle scale and resolution at 25.64° on the fixed
caches."""
    D.slide('sps-what', body, notes, foot=foot_ws('ws_sps'), short='SPS: what transfers')


def s_sps_v_how(D):
    body = (sd.title('SPS drift velocity, two ways: column duration and ladder',
                     'det4 at H4, 25.64° mount. Left: one track. Middle: the summed strip signal lasts T = gap / v. '
                     'Right: the sideways speed w grows as v·tan θ') +
            fig('ws_spsv/figures/f16_concepts.png', w=1664,
                tip='f16_concepts.png. Multiplying the two, w·T = gap·tan θ: the sideways width of the column, with no '
                    'velocity in it.'))
    notes = """<b>F16.</b> The particle ionises the whole gap at once: electrons next to the mesh arrive first, those
from the cathode last. Summed over all strips the signal is a box of length T = gap / v, smeared by the electronics, and
the angle terms cancel, so T is measured without the track model.

In the tilted view the landing point slides sideways at w = v·tan θ; with a known mount angle, w / tan θ is the
"ladder" velocity. The product w·T = gap·tan θ contains no velocity: it checks the gap (or the fit) directly.

The old <i>span route</i> (spread of single-strip hit times anchored to one velocity from run_71) is retired: it was built
on combined_hits times, and its anchor carried a field label computed with 30 mm instead of the registry's 28.8 mm."""
    D.slide('sps-v-how', body, notes, foot=foot_ws('ws_spsv'), short='SPS v: how')


def s_sps_v_grid(D):
    r = J('ws_spsv/results/results.json') if (HERE / 'ws_spsv/results/results.json').exists() else {}
    body = (sd.title('The old "14 %" was the depth grid; T = 2.05 µs',
                     'Left: the old fits\' column end sat on their depth-grid end, fixing the ratio. Right: strip-sum '
                     'waveforms at 243 V/cm with box fits') +
            sd.row(sd.col(fig('ws_spsv/figures/f16_grid_artefact.png', w=480, crop=(0, 0, 0.5, 1),
                              tip='f16_grid_artefact.png: the grid length came from the span velocity, and q_uend = grid '
                                  'end at every plateau (2400–2880 ns).'),
                          fig('ws_spsv/figures/f16_grid_artefact.png', w=480, crop=(0.5, 0, 1, 1),
                              tip='so v_span / v_col = 1.155–1.167 by construction: the "14 %".'), gap=16, w=480),
                   fig('ws_spsv/figures/f16_column_243.png', w=1140,
                       tip='f16_column_243.png: head-on RAW 1970 ns, head-on ZS 2060 ns, 25.64° ZS 2040 ns (Y), 2130 ns '
                           '(X). T = 2.05 ± 0.07 µs.'),
                   gap=44, align='center'))
    notes = """The old F16 fits sized their depth grid (K bins × 60 ns) from the span-route velocity. The fitted column
end q_uend sat exactly on the grid end at every plateau (2400, 2520, 2700, 2820, 2880, 2880 ns), so v_span divided by
28.8 mm / q_uend is 1.155–1.167 from arithmetic alone: the "14 %". The "+4 % self-consistency" compared the slope with the
grid end, so it measured nothing either. On top, those fits used the caches that sat 256 ADC high.

Model-free column duration at 243 V/cm: head-on RAW (run_71) 1970 ns, head-on ZS (run_63) 2050, 25.64° ZS (run_62)
2040 (Y) / 2130 (X): <b>T = 2.05 ± 0.07 µs</b>. A flat box reads 2–5 % shorter than a box with a charge gradient; the
charge falls ~50 % along the column (late charge lost, consistent with O₂ attachment).

Caveats: the box estimator assumes the bench det4 X template holds at H4 (same DREAM registers, not checked further);
~11 % of inclined events are cut by the ±16-strip window (not corrected); run_71 RAW caches use a per-strip pre-window
baseline without common-mode correction."""
    D.slide('sps-v-grid', body, notes, short='SPS v: the 14 %')


def s_sps_v_result(D):
    body = (sd.title('What remains: the slope is 8 % short of gap / T at every field',
                     'Left: v(E) from 28.8 mm / T (model-free) and from the fitted slope. Right: the gap implied by '
                     'slope × duration, which contains no velocity') +
            sd.row(fig('ws_spsv/figures/f16_v_of_E.png', w=900,
                       tip='f16_v_of_E.png: v(243 V/cm) = 14.1 +0.6 −1.1 µm/ns (28.8 mm / T); ~1.4–1.6 % water against '
                           'Magboltz for the SPS gas. Below ~170 V/cm the column overruns the 3.84 µs window (bounds only).'),
                   fig('ws_spsv/figures/f16_gap_eff.png', w=720,
                       tip='f16_gap_eff.png: |w|·T / tan = 26.2–27.0 mm (25.0–26.4 flat box). Measured SPS diffusion moves '
                           'it ≤ 0.7 %, zero treatment ≤ 0.4 %: not the model widths.'),
                   gap=44))
    notes = """<b>Absolute drift velocity, det4 at SPS (gap / T)</b>, µm/ns with 28.8 mm (30 mm): 243 V/cm 14.1 (14.7);
235: 13.5 (14.1); 217: 12.7 (13.3); 200: 11.2 (11.6); 174–182: 9.6–10.4 (10.0–10.8). Defensible:
<b>v(243 V/cm) = 14.1 +0.6 −1.1 µm/ns</b> (box estimator, gap 26.5–30 mm, run-to-run). Fields are HV / 28.8 mm; whether
the mesh voltage adds to the drift field is not in any record.

<b>Residual</b>: the fitted slope |w|/tan is ~8 % below 28.8 mm / T at every field 174–243 V/cm (8 plateaus, 2 runs,
both ZS treatments). Equivalently the implied gap is 26.2–27.0 mm. Two explanations, and the existing data cannot split
them:
<ul><li>the <b>local gap</b> really is ~26.5 mm at the beam spot: det4's bench gap map is unusable (25.5 mm overall,
charge quartiles 24.4–33.5 mm), and its ±3 mm bow allows it;</li>
<li>the <b>fit compresses tilted tracks</b>: in run_61 at the same field the slope reads 11.38 at 15.465° but 12.67 at
25.64°, which a gap error cannot do (or a ~1.5° error in the 15° mount, or 32-sample truncation).</li></ul>
Consequence: with v = 28.8 mm / T, angles at 25.64° read 8 % low in tan (23.8°). The old +4 % is not the truth; the
truth is <b>−8 % ± 3 %</b>, owed to the local gap or angle-dependent compression. Stated limitation. Not a model-width
effect: setting Dp and σ0 to the measured SPS values moves the slope ≤ 0.7 % (χ²/dof 7.3 → 5.9)."""
    D.slide('sps-v', body, notes, foot=foot_ws('ws_spsv'), short='SPS v: the 8 %')


def s_floor(D):
    s = J('ws_floor/results/summary.json')
    m = J('ws_floor/results/measured.json')
    d3 = s['det3']
    angs = [int(a) for a in d3['full']]
    P = sd.Plot(1000, 620, x=(-1, 46, 'lin'), y=(0, 1.6, 'lin'), xlabel='track angle in the view [deg]',
                ylabel='angle resolution s68 [deg]', title='det3: how far the measurement is from the physics floor')
    P.xticks([(a, str(a)) for a in range(0, 46, 10)])
    P.yticks([(v, f'{v:.1f}') for v in (0, 0.5, 1.0, 1.5)])

    def series(key, color, dash=None, lab=''):
        ys = [d3[key][str(a)]['s68'] for a in angs]
        P.line(angs, ys, color, dash=dash, tips=[f'{lab}, {a}°: {y:.2f}°' for a, y in zip(angs, ys)])
    P.line(angs, [d3['full'][str(a)]['oracle'] for a in angs], GREY, dash='6 6',
           tips=[f'every electron seen, {a}°: {d3["full"][str(a)]["oracle"]:.2f}°' for a in angs])
    series('full_bin20', GREEN, lab='perfect model, 20 ns bins')
    series('full', RED, lab='perfect model, 60 ns bins (production grid)')
    for v, c, mk in (('x', XC, 'circle'), ('y', YC, 'square')):
        b = [r for r in m[f'det3_{v}']['bins'] if r['n'] >= 100]
        P.points([r['mid'] for r in b], [r['s68'] for r in b], c, r=10, marker=mk,
                 tips=[f'measured det3 {v.upper()}, {r["lo"]}–{r["hi"]}°: {r["s68"]:.2f}° (n={r["n"]}, vs M3)' for r in b])
    P.text(2, 1.48, 'measured: 3× the floor', 26, INK, 'start', 600)
    P.text(1, 0.50, 'floor 0.30–0.35°, set by diffusion', 24, GREEN, 'start', 600)

    fl = s['fleet_5_15deg']
    Q = sd.Plot(600, 620, x=(-0.6, 4.6, 'lin'), y=(0, 2.0, 'lin'), ylabel='s68 [deg]', title='every chamber, 5–15°')
    Q.xticks([(i, d) for i, d in enumerate(DETS)])
    Q.yticks([(v, f'{v:.1f}') for v in (0, 0.5, 1.0, 1.5, 2.0)])
    for i, d in enumerate(DETS):
        r = fl[d]
        Q.vbar(i - 0.22, r['production_fit'], 30, GREEN, tip=f'{d} floor (perfect model): {r["production_fit"]:.2f}°')
        Q.vbar(i + 0.02, r['meas_x'], 26, XC, tip=f'{d} measured X: {r["meas_x"]:.2f}°')
        Q.vbar(i + 0.24, r['meas_y'], 26, YC, tip=f'{d} measured Y: {r["meas_y"]:.2f}°')
    body = (sd.title('Physics floor 0.30–0.35°; we measure 3–6× that',
                     'Electron-by-electron toy fitted with its own exact model, vs the measured resolution against M3') +
            sd.row(P.svg('floor vs angle'), Q.svg('fleet floor'), gap=40) +
            sd.legend([('every electron seen', GREY, 'dash'), ('floor: perfect model, 20 ns bins', GREEN),
                       ('perfect model, 60 ns bins (production)', RED), ('measured X', XC, 'dot'), ('measured Y', YC, 'dot')], size=22))
    notes = """<b>F3, the toy for the paper.</b> Electron by electron: Fischle cluster sizes with δ-ray ranges, Magboltz
diffusion, Polya gain, the measured footprint, strip pitch, pulse shape, sampling, noise and ADC ceiling; fitted with the
toy's own exact mean model, started at truth. Floor 0.30–0.35° (s68) at every angle 0–45°, all five chambers.

Which physics sets it (det3, 20 ns, one effect off at 5/20/35°, all on 0.34/0.30/0.30): no diffusion 0.02/0.05/0.09, so
<b>diffusion is the floor</b>; no cluster-size fluctuation 0.25/0.20/0.19; no gain fluctuation 0.29/0.29/0.24; electronic
noise negligible. Closed form σ_θ ≈ D_T √(6(1+f)/(N·G)), N drift electrons, G gap, f = 1/(1+θ_Polya): 0.29° for det3.
D_T is the same on all bench chambers, so the floor is too.

The old "~1° physics floor" (script 18, <code>FLOOR_TAN = 0.018</code>) is withdrawn: it assumed 0.30 mm centroid jitter
per 60 ns bin; the physical value is 0.185 mm. The measured 1.0° (best chamber) is model error of the charge-dispersion
response, not a detector limit.

<b>Red curve:</b> with 60 ns charge bins even a perfect model degrades past 25° and reads steep tracks shallow (−0.5° at
40°), because v·Δt·tan θ exceeds a strip pitch. 20 ns bins fix it. Small on the bench (−0.05° at 20°), the main term for
n_TOF pairs at 30–45°. Untested on data. Proposed paper paragraph: <code>ws_floor/REPORT.md</code>.
Limits: X view only, bench gas only (n_TOF gas has about half the D_T, so a lower floor; not run)."""
    D.slide('floor', body, notes, short='physics floor')


def s_dead(D):
    body = (sd.title('Dead strips: the chamber\'s own, or the electronics\'',
                     'det6 X and det7 X: every dataset (rows) against strip position. Green squares = the current '
                     'bench mask; other markers = strips found dead by three tests') +
            sd.row(fig('ws_dead_charge/figures/f12_dead_map.png', w=810, crop=(0, 0.603, 0.5, 0.795),
                       tip='f12_dead_map.png, det6 X: the 0.8–18 mm, 150–186 and 342–350 mm blocks recur in every bench '
                           'and n_TOF dataset (chamber-side). The mask misses the 177–185 mm block.'),
                   fig('ws_dead_charge/figures/f12_dead_map.png', w=810, crop=(0, 0.793, 0.5, 1.0),
                       tip='f12_dead_map.png, det7 X: connectors 0–3 dead on the bench and again at n_TOF D on another '
                           'FEU, so it is the chamber, not the seating.'), gap=44) +
            sd.legend([('census dead (M3 crossing)', RED, 'dot'), ('census short', PURPLE, 'dot'),
                       ('flank dead (neighbours fire, it does not)', BLUE, 'dot'), ('silent', GREY, 'dot'),
                       ('bench mask', GREEN, 'open')], size=22))
    notes = """<b>F12.</b> 17 bench rows, 3 SPS flat sets and n_TOF runs 55/79/145, in strip position. Three tests:
census (M3/uRWELL crossing, no hit), flank (silent strip between two strips above 300 ADC) and silent (&lt; 5 % of the
neighbourhood's occupancy). Two fault classes, consistent across datasets:
<ul><li><b>Chamber-side</b>, follows the chamber across FEUs: det6 X 0.8–18 mm alternate, 150–186, 342–350; det7 X
connectors 0–3 (reappear at n_TOF D on FEU 1); det6/det7 Y 40.6–43.7; det2 X 7.0, 53; det4 X dead stripes (confirmed at
SPS).</li>
<li><b>Electronics-side</b>, follows an FEU channel: bench FEU 6 ch 269 (det2/4/7); FEU 4 ch 50 (det3, det6, n_TOF A);
FEU 7 ch 307/398/400.</li></ul>
det3's whole bench mask is FEU-7 electronics: moved to n_TOF A (FEU 3) it would censor 3 live strips and miss FEU 4 ch 50.
det6/det7 masks miss 15–18 chamber-side strips; det4 over-masks (census statistics at ~30 crossings/channel). <b>n_TOF
bundles carry no dead list at all.</b> Rule proposed: chamber faults by position + each dataset's own electronics faults.

SPS: det4 at H4 was read on FEU 03, connectors 3–6, so the bench mask by channel number censored the wrong strips. Fixed:
<code>CalibrationBundle.remap_channels</code> and an H4 mask measured with the uRWELL. The SPS gate with that mask is not
run; the golden mask corrections are not gated."""
    D.slide('dead', body, notes, foot=foot_ws('ws_dead_charge'), short='dead strips')


def s_charge(D):
    f = J('ws_dead_charge/f15.json')
    steps = [('all', 'all fitted'), ('m3', 'M3 track, both views'), ('live', 'no dead strip in window'),
             ('unsat', 'unsaturated')]
    cols = [GREY, '#8fb3dc', BLUE, GREEN]
    P = sd.Plot(1100, 560, x=(-0.6, 4.6, 'lin'), y=(0, 0.34, 'lin'), ylabel='width of ln(qx / qy)',
                title='x–y charge balance width, removing one cause at a time')
    P.xticks([(i, d) for i, d in enumerate(DETS)])
    P.yticks([(v, f'{v:.1f}') for v in (0, 0.1, 0.2, 0.3)])
    for i, d in enumerate(DETS):
        r = f.get(d, {})
        for j, (k, lab) in enumerate(steps):
            if k in r:
                P.vbar(i - 0.3 + 0.2 * j, r[k], 34, cols[j], tip=f'{d}, {lab}: {r[k]:.3f}' +
                       (f' (saturated fraction {r["frac_sat"]:.0%})' if k == 'unsat' and 'frac_sat' in r else ''))
    P.text(0.0, 0.30, 'mispairing and saturation, not gain', 26, INK, 'start', 600)
    body = (sd.title('x–y charge balance: mispairing and saturation, not gain',
                     'Robust width of ln(qx/qy); each bar is a subset of the previous one') +
            sd.row(sd.col(P.svg('charge cascade'), sd.legend([(lab, c, 'box') for (k, lab), c in zip(steps, cols)], size=22), gap=6, w=1100),
                   sd.col(sd.callout('Per-connector X/Y gains: equal to ±2 %; correcting them moves the width ≤ 0.004', BLUE),
                          sd.callout('So: charge pairing needs a width per event type, not one per chamber', GREEN),
                          gap=28, w=500), gap=48, align='center'))
    notes = """<b>F15: neither dead strips nor missing normalisation.</b> X and Y share one amplification gap, so gain
cancels in qx/qy. Per-connector X/Y gains (split-half) are equal to ±2 % (≤ ±5 %); applying them changes the width by
≤ 0.004, and the ln(qx/qy) map in 25 mm cells is flat to 0.02–0.03. Censoring restores dead-strip charge on average but
those windows are 1.3–3× wider.

What does drive the width: wrong x–y pairing (det7 0.32 → 0.19 when M3 confirms both views), saturation (16 % of det3
events, 30 % det2, 69 % det6/det7), and censored windows. The residual on det6/det7 tracks their X–Y χ²/dof difference
(r 0.44–0.46), a model-side effect of the F4/F18 family.

Consequence for the single reconstruction: pairing tolerance per event class (saturated / censored / clean). Not yet
implemented."""
    D.slide('charge', body, notes, foot=foot_ws('ws_dead_charge'), short='x–y charge balance')


def s_unify(D):
    steps = [dict(label='code constants', sub='one algorithm, every dataset', color=BLUE,
                  tip='Every former WFT_* switch is either always on or a fixed constant'),
             dict(label='+ calibration bundle', sub='per chamber + run condition: v, kernel, σ0, mask, t0, angle map', color=GREEN,
                  tip='Everything that differs between datasets is measured and stored in the bundle'),
             dict(label='= reconstruction', sub='same for bench, SPS, n_TOF', color=INK),
             dict(label='per-dataset switches', sub='WFT_* environment variables', color=RED, strike=True,
                  tip='No longer read by wft. Study variants go through wft/diag.py; production refuses them unless --diag')]
    body = (sd.title('One reconstruction: code + measured calibration',
                     'Unified reconstruction minus production, 8000 M3 muons per chamber') +
            sd.flow(steps, size=24) +
            fig('ws_unify/figures/gate_delta.png', w=1400,
                tip='gate_delta.png: detected +1.1 to +5.5 points; far tracks ±0.2 pt; overall s68 worse only on det4 and '
                    'det7 X, from the newly found lower-quality tracks. Tracks production already found improve on every view.'))
    notes = """<b>F22.</b> <code>wft</code> no longer reads any <code>WFT_*</code> environment variable. Study variants go
through one hook, <code>wft/diag.py</code>, which records any override in each table's metadata;
<code>rerun/stage_job.py</code> refuses a production stage with an override unless <code>--diag</code>.
<table><tr><th>former switch</th><th>now</th></tr>
<tr><td>pivot start, column guard, dead-strip seeding</td><td>always on</td></tr>
<tr><td>noise-band veto, head-on seeds + rescue</td><td>always on (fleet gate passed)</td></tr>
<tr><td>continuous t0, saturated-sample re-solve</td><td>always on (correctness)</td></tr>
<tr><td>two-track fit</td><td>always on; minimum separation 4 strip pitches (3.1 mm)</td></tr>
<tr><td>x–y t0 window, head-on charge tolerance, angle error floor</td><td>from the bundle's own measurements</td></tr>
<tr><td>zero-suppressed samples</td><td>measured zeros</td></tr>
<tr><td>WFT_U_FIXED</td><td>deleted</td></tr></table>
<b>F7: neither hard-coded nor loosened ad hoc.</b> The plausibility window is (0.30–1.60) × 30 mm / v of the bundle
(det3 250–1330 ns, det6 375–2000 ns). The old upper edge rejected 7–17 % of M3-confirmed single tracks.

Gate: detected +1.12 / +1.15 / +3.70 / +1.85 / +5.45 points (det2/3/4/6/7). On the tracks production already found, s68
improves slightly on every view.

<b>Not yet in the single reconstruction</b> (all bundle content, not switches): fixed fleet σ0 and dropping the per-event
choice; the pinned t0; per-class pairing widths; the 20 ns grid. The bundles must be re-finalised under the new algorithm
before a production pass."""
    D.slide('unify', body, notes, short='one reconstruction')


def s_split(D):
    body = (sd.title('det6\'s two-track "pairs" were bright single muons',
                     'Left: separation of the two fitted tracks. Right: fraction of clean single muons split in two, '
                     'old and new minimum') +
            sd.row(fig('ws_unify/figures/split_sep.png', w=810,
                       tip='split_sep.png: false splits sit at ~2 mm, inside one sharing kernel; real two-muon overlays '
                           '(det3) spread to 16 mm.'),
                   fig('ws_unify/figures/split_rate.png', w=810,
                       tip='split_rate.png: with the 3.1 mm minimum every view is under the 1 % contract (det6 Y 0.6 %).'),
                   gap=44))
    notes = """The 417 det6 splits were the brightest single muons with the worst χ² (p94–p99), refitted as two parallel
tracks ~2 mm apart sharing charge ~40/60: 4.1 % of clean singles on det6 Y. A real two-muon overlay spreads out to 16 mm.
With a minimum separation of 4 strip pitches (3.11 mm), false splits are 0.05–0.72 % on every chamber and view, inside
the 1 % bound.

Risk: the n_TOF pipeline (<code>wft_beam</code>) inherits the 4-pitch minimum; pair efficiency at small separation there
has not been re-measured."""
    D.slide('split', body, notes, foot=foot_ws('ws_unify'), short='two-track splits')


def s_close(D):
    fg = '#eef0f3'

    def li(t):
        return f'<li style="font-size:28px;line-height:1.35;color:{fg};margin-bottom:10px">{t}</li>'
    left = ('<ul style="padding-left:28px">' + ''.join(li(t) for t in [
        'A ~2 % compression of steep tracks after every correction (det4 Y 7 %)',
        'Bright tracks reading 5–10 % shallow on det6/det7: a model mismatch',
        'Measured angles are 3–6× the physics floor: the response model is the limit',
        'Bow and tilt maps are bench mounting; n_TOF needs its own (which w0 stage 3 used is unchecked)',
        'SPS: the fitted slope is 8 % short of gap / T, either the local gap or the fit compressing tilted tracks']) + '</ul>')
    right = ('<ul style="padding-left:28px">' + ''.join(li(t) for t in [
        'Fix σ0 fleet-wide (Y 0.33, X 0.24–0.29) and drop the per-event choice: needs its gate',
        'Pin t0 on bench bundles; switch the x–y check to column time',
        'Per-event-class pairing widths; corrected golden dead masks; dead lists for n_TOF',
        '20 ns charge grid (untested on data)',
        'Re-finalise and promote every bundle under the unified algorithm; re-measure n_TOF pair efficiency']) + '</ul>')
    body = (sd.kicker('What this does not rule out, and what is left to do') +
            sd.row(sd.col(sd.p('Not explained yet', 34, '#e07a86', 600), left, gap=12),
                   sd.col(sd.p('Decided, not yet in production', 34, '#6cc79a', 600), right, gap=12), gap=64))
    notes = """Condor probes still out: width probes on det2/det3/det7 and toy-mode probes (widthtilt), the X footprint grid
on 22 ZS flat rows (ws_sps). The SPS changes did not go through the before/after gate.
det7's promoted bundle has empty run conditions (F10; <code>promote_phys.py --fill-conditions</code> fixes it at the next
promotion)."""
    D.slide('close', body, notes, dark=True, short='open')


def main():
    D = sd.Deck('MX17 model audit: results', 'What the eight model-audit workstreams found, 27 Oct 2026')
    for f in (s_cover, s_setup, s_t0_float, s_t0_latency, s_v, s_w0, s_f4_cut, s_f4_bow, s_f4_fix, s_width,
              s_sps_sigma0, s_sps_zs, s_sps_what, s_sps_v_how, s_sps_v_grid, s_sps_v_result, s_floor, s_dead, s_charge, s_unify, s_split, s_close):
        f(D)
    OUT.parent.mkdir(exist_ok=True)
    D.write(OUT, note_meta=dict(
        title='MX17 model audit: results',
        summary='The bench angle problems were mostly calibration (t0, tilt, bow, a selection cut), not the physics '
                'model; SPS fixes the footprint; physics floor 0.3°; one reconstruction.',
        tags='Cosmic bench, Micromegas, Reconstruction, SPS', date='2026-10-27'))
    print(OUT, f'{OUT.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
