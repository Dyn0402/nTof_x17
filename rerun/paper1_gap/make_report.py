#!/usr/bin/env python3
"""make_report.py -- report.html of the chamber-A gap map from results/compare.json and the figures (gapfig.py)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(f'{HERE}/results/compare.json'))
B = {'kuu': 0.8818883015769696, 'kww': 1.4923582013350711, 'kuw': -0.06296886468393921}


def fit_dish_tilt(m):
    """dg = c0 + a U + b W + s * dish (dish = flat cathode on the measured board dish)."""
    t = np.array(C['grids']['4x4'][m]['comb']); t = t[np.isfinite(t[:, 2]) & (t[:, 3] > 0)]
    U, W = (t[:, 0] - 200) / 100, (t[:, 1] - 200) / 100
    dish = -(B['kuu'] * U * U / 2 + B['kww'] * W * W / 2 + B['kuw'] * U * W)
    A = np.vstack([np.ones_like(U), U, W, dish]).T; w = 1 / t[:, 3]
    c = np.linalg.lstsq(A * w[:, None], t[:, 2] * w, rcond=None)[0]
    cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    chi2 = float(np.sum(((A @ c - t[:, 2]) * w) ** 2))
    return c, np.sqrt(np.diag(cov)), chi2, len(t) - 4


def f(x, e=None, d=2):
    return f'{x:.{d}f}' if e is None else f'{x:.{d}f} ± {e:.{d}f}'


def surfaces_html():
    JJ = json.load(open(f'{HERE}/results/joint.json'))
    TG = json.load(open(f'{HERE}/results/track_grid.json'))
    TT = json.load(open(f'{HERE}/results/track_test.json'))['metrics']
    lab = {'all': 'all', 'tilt': 'positions + drift tilt (w0, sign asymmetry)', 'gap': 'positions + gap (G, GY, T, v)',
           'geo': 'no timing (positions, G, GY, sign asymmetry)', 'timing': 'timing (positions, w0, T, v)',
           'no_hp': 'no positions', 'm3dish': 'all + free telescope dish', 'beta068': 'all, beta = 0.68',
           'beta087': 'all, beta = 0.87'}
    rows = []
    for a, l in lab.items():
        F = JJ['arms'][a]; P, E = F['params'], F['errors']
        d = lambda k: f"{1.125 * P[k]:.2f} ± {1.125 * E[k]:.2f}" if np.isfinite(E[k]) else f"{1.125 * P[k]:.2f}"
        tilt = P['c_lu'] - P['h_lu']
        rows.append(f"<tr><td>{l}</td><td>{F['chi2']:.0f} / {F['n'] - F['nfree']}</td><td>{d('h_kuu')}</td><td>{d('h_kww')}</td>"
                    f"<td>{d('c_kuu')}</td><td>{d('c_kww')}</td><td>{tilt:.2f}</td><td>{P['beta']:.2f}</td></tr>")
    fl = JJ['floors']
    trk = []
    for arm, src in (('pass 1', None), ('two-surface model', 'model'), ('map 85 mm', '4x4 w0+v'), ('map 57 mm', '6x6 w0+v'),
                     ('map 42 mm', '8x8 w0+v'), ('map 42 mm, offset only', '8x8 w0'), ('map 42 mm, velocity only', '8x8 v'),
                     ('map 34 mm', '10x10 w0+v')):
        vals = []
        for p in 'xy':
            for r in ('run1', 'run2'):
                vals.append(TG[r][p]['pass1']['s68'] if src is None else (TT[r][p]['model']['s68'] if src == 'model' else TG[r][p][src]['s68']))
        trk.append(f"<tr><td>{arm}</td>" + ''.join(f'<td>{v:.3f}</td>' for v in vals) + '</tr>')
    return f"""
<h2>Two surfaces: the mesh and the cathode (joint fit, joint.py)</h2>
<p>One quadratic mesh h and one quadratic cathode c predict every observable (cellobs.py, 4 × 4 cells, both runs): mesh height
from positions (X, Y), the drift tilt from the ladder intercept w0 = v (∂h + ∂c)/2 and, timing-free, from the ±tan asymmetry of the
geometric gap; the gap g = c − h (G, GY); T ∝ g<sup>1+β</sup>; v ∝ g<sup>−β</sup>. Error floors added in quadrature so each class has
reduced χ² = 1: hp {fl['hp']:.2f} mm, w0 {fl['w0']:.2f} µm/ns, sign-asymmetry tilt {fl['d']:.4f}, G {fl['G']:.2f} mm, T {100 * fl['T']:.1f} %,
v {100 * fl['v']:.1f} % (what a quadratic surface does not describe). Depths at 150 mm (mm, positive = centre lower):</p>
<table><tr><th>observables</th><th>χ² / ndf</th><th>mesh x</th><th>mesh y</th><th>cathode x</th><th>cathode y</th><th>gap tilt x (mm/100 mm)</th><th>β</th></tr>{''.join(rows)}</table>
<ul><li><b>The board dish is measured twice:</b> without any position information the drift tilt (h + c) and the gap (c − h) give a
mesh 1.01 / 1.55 mm deep (x / y) against 0.77–0.88 / 1.38–1.47 from positions; a free telescope dish is consistent with zero.</li>
<li><b>Along y the cathode is nearly flat</b> in every subset; the gap is ~1.2 mm longer at the centre than 150 mm out.</li>
<li><b>Along x the cathode follows the board</b>; the gap rises ~0.4 mm per 100 mm. Only the no-timing subset (geometric gap covers
x ≈ 100–320 mm only) prefers a flat cathode along x (2.5σ). The w0 ramp along x is steeper at the edges (0.39 µm/ns per 100 mm over
the interior, 0.31 in the central 200 mm), so r_f along x reads 1.2 > 1: the board is not a pure paraboloid.</li>
<li>β = 1.0 ± 0.2 (Magboltz 0.68: Δχ² = +5; drift-scan 0.87: +0.7); the surfaces do not depend on it.</li></ul>

<h2>Tracking: a finer drift-direction map (track_test.py, track_grid.py)</h2>
<p>Maps of the ladder intercept w0 and slope v per cell are made on one run and applied to the other (bilinear, indexed by the
reconstructed mid-gap position); only two constants are re-measured on the test run. s68 of θ − θ_ref (deg), interior single tracks,
|tan θ_ref| ≤ 0.6:</p>
<table><tr><th>conversion</th><th>X Run 1</th><th>X Run 2</th><th>Y Run 1</th><th>Y Run 2</th></tr>{''.join(trk)}</table>
<ul><li>42 mm cells improve the angles by 0.15–0.17° (16–21 %), both views, both runs. Most is the offset (drift tilt); the velocity
map adds 0.02–0.04°. At 34 mm the map from the smaller Run 1 is noise-limited.</li>
<li>The structure is 2-D and smooth: a map in the view's own coordinate alone recovers a fifth to a third of the gain, the other
coordinate alone none, and there are no steps at the 64-strip (50 mm) chip boundaries. So it is the drift field, not chip timing.</li>
<li>Not tested at n_TOF (boards vertical: no gravity sag; no reference tracks).</li></ul>
<figure><img src="figures/gap_surfaces.png"><figcaption>Paper I appendix figure (app:gapmap): gap profiles with the joint model, and the
angle resolution against the map cell size.</figcaption></figure>
"""


def main():
    ch = C['chamber']
    G4 = C['grids']['4x4']
    names = {'G': 'geometric, X strips (S4 estimator, ±tan averaged)', 'GY': 'geometric, Y strips (peak, relative)',
             'VT': 'angle-scale v × column duration T (no β)', 'T': 'column duration T only (β = 0.68)',
             'V': 'angle-scale v only (β = 0.68)'}
    rows = []
    for m in names:
        r = G4[m]
        c, e, chi2, ndf = fit_dish_tilt(m)
        rows.append(f"<tr><td>{names[m]}</td><td>{f(r['run_diff_chi2'], d=1)} / {r['run_diff_n']}</td>"
                    f"<td>{f(r['flat_chi2'], d=1)} / {r['flat_ndf']}</td><td>{f(r['rms'])}</td>"
                    f"<td>{f(c[1], e[1])}</td><td>{f(c[2], e[2])}</td><td>{f(c[3], e[3])}</td>"
                    f"<td>{f(chi2, d=1)} / {ndf}</td></tr>")
    corr = G4['corr']; k = G4['k']
    cg = lambda a, b: f"{corr[f'{a}~{b}'][0]:.2f} ({corr[f'{a}~{b}'][1]} cells)"
    sl = G4['slope_vs_G']
    prof = []
    for grid, lab in (('x5', 'x'), ('y5', 'y')):
        t = {m: np.array(C['grids'][grid][m]['comb']) for m in ('G', 'GY', 'VT', 'T', 'bow')}
        axis = 0 if lab == 'x' else 1
        for i in range(len(t['G'])):
            prof.append(f"<tr><td>{lab} = {t['G'][i, axis]:.0f}</td>" + ''.join(
                f"<td>{'–' if not np.isfinite(t[m][i, 2]) else (f(t[m][i, 2]) if m == 'bow' else f(t[m][i, 2], t[m][i, 3]))}</td>"
                for m in ('G', 'GY', 'VT', 'T', 'bow')) + '</tr>')
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Chamber A gap map</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;color:#1b2430;line-height:1.45}}
table{{border-collapse:collapse;margin:1em 0;font-size:0.9em}}td,th{{border:1px solid #d5dae1;padding:3px 7px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}.v{{background:#eef4fb;border-left:4px solid #0b3d91;padding:0.6em 1em}}
figcaption{{font-size:0.9em;color:#4a5563}}</style></head><body>
<h1>The drift gap of chamber A across its surface (pass 1)</h1>
<p>Paper I Sec. 5.1. Code: <code>rerun/paper1_gap/</code>. Inputs: the pass-1 tables and the rr1 waveforms of the two det3
golden rows (Run 1 = sat_det3, 6-27; Run 2 = the 6-28 p2 overnight), 490 V mesh, 1000 V drift, top slot.</p>
<div class="v"><b>Verdict.</b> The gap of chamber A is not uniform. Three independent measurements (geometric with the X strips,
geometric with the Y strips, and angle-scale velocity × drift time) show the same pattern in both runs. The gap rises along x
by 0.5–0.8 mm per 100 mm (a cathode tilt of about 0.3–0.5° relative to the mesh). Along y it is largest near the centre and
falls toward both ends, with the shape of the measured dish of the readout board under a flat cathode (0.5–0.8 of the dish).
Over the interior the gap runs from about 27 to about 29 mm around the mean of 28.2 mm, with a cell scatter of 0.5–0.7 mm.
The timing maps carry the signature of a gap change, not a gas change (d ln v / d ln T = {f(k['k'], k['err'])}; gap
{k['pure_gap']:.2f}, gas −1).</div>

<h2>Headline numbers</h2>
<table><tr><th></th><th>Run 1</th><th>Run 2</th></tr>
<tr><td>chamber gap, geometric X, interior (mm)</td><td>{f(ch['run1']['G'], ch['run1']['G_err'])}</td><td>{f(ch['run2']['G'], ch['run2']['G_err'])}</td></tr>
<tr><td>+tan / −tan halves (mm)</td><td>{f(ch['run1']['G_plus'])} / {f(ch['run1']['G_minus'])}</td><td>{f(ch['run2']['G_plus'])} / {f(ch['run2']['G_minus'])}</td></tr>
<tr><td>inclined tracks (geometric X)</td><td>{ch['run1']['n_G']}</td><td>{ch['run2']['n_G']}</td></tr>
<tr><td>angle-scale v, X / Y (µm/ns)</td><td>{f(ch['run1']['vX'])} / {f(ch['run1']['vY'])}</td><td>{f(ch['run2']['vX'])} / {f(ch['run2']['vY'])}</td></tr>
<tr><td>column duration T, box fit, 20 % trimmed mean (ns); tracks</td><td>{f(ch['run1']['T'], d=1)}; {ch['run1']['n_T']}</td><td>{f(ch['run2']['T'], d=1)}; {ch['run2']['n_T']}</td></tr></table>
<p>The S4 re-check gave 28.19 ± 0.19 mm on Run 1 with the reference cut at the mesh; the cut on the track at mid-gap used
here gives the same value, and Run 2 reproduces it independently. The colsum box T is ~3 % longer than the stack T of
the paper (a different end model), so every timing map is used relative to its own chamber value.</p>

<h2>Methods, per cell (track position at mid-gap, ≥ 30 mm inside the active area)</h2>
<ul>
<li><b>G, geometric, X strips</b> (no time): the half-height edges of the X charge of inclined tracks (|tan θ| 0.2–0.8)
against z = (x − x_mesh)/tan θ_ref; mask = pass dead list ∪ audit flags; mean of the unsaturated and saturation-masked
selections (the S4 recipe); the +tan and −tan halves averaged per cell, which removes a local drift tilt to first order.
Bootstrap errors (40).</li>
<li><b>GY, geometric, Y strips</b>: the same on the Y view with the peak sample (the Y integral drains through the RC
spreading). The peak version reads high by a blur-dependent amount, so GY is used relative to its own chamber value.</li>
<li><b>T, column duration</b>: colsum.py's per-trigger box fit of the strip sum of the track's cluster (valid for any
angle), matched to the M3 table; unsaturated single clusters; amplitude walk removed globally (it moves the map by
≤ 0.06 mm); 20 % trimmed mean.</li>
<li><b>V, angle-scale velocity</b>: per cell and view, a robust line w = a + v tan θ_ref (|tan θ_ref| ≤ 0.6); no gap enters.</li>
<li><b>VT</b>: g = v × T per cell, the paper's second gap measurement made locally. The field dependence cancels:
T ∝ g<sup>1+β</sup>, v ∝ g<sup>−β</sup>.</li>
<li><b>bow</b>: a flat cathode above the measured dish of the board, −(curvature part of the pass-1 bow surface). The linear
part of the bow surface is the rigid tilt of the whole chamber, which the cathode shares, so it is left out.</li></ul>
<p><b>Coverage.</b> Inclined tracks reach only the centre of the chamber through the M3 telescope: the X-strip geometric
method covers x ≈ 100–320 mm (all y), the Y-strip one y ≈ 60–290 mm (all x). The timing methods use every angle and cover
the whole interior.</p>

<h2>Profiles</h2>
<figure><img src="figures/gap_profiles.png"><figcaption>The gap along x (a, all y in the interior) and along y (b, all x),
both runs combined. Filled circles: geometric, X strips. Open circles: geometric, Y strips (relative, placed on 28.19).
Squares: angle-scale velocity × column duration. Triangles: column duration only, read as a gap change with β = 0.68.
Dashed: a flat cathode above the measured dish of the board.</figcaption></figure>
<table><tr><th>cell</th><th>G</th><th>GY</th><th>VT</th><th>T</th><th>bow</th></tr>{''.join(prof)}</table>
<p>Deviations in mm from each method's chamber value.</p>

<h2>Maps and fits (4 × 4 cells of ~85 × 75 mm)</h2>
<figure><img src="figures/gap_maps.png"><figcaption>Gap change per cell (mm), both runs combined, for every method and for
the flat-cathode-on-the-dish prediction. Empty cells: no coverage.</figcaption></figure>
<table><tr><th>method</th><th>Run 1 − Run 2 χ² / cells</th><th>flat χ² / ndf</th><th>rms (mm)</th><th>tilt along x (mm/100 mm)</th>
<th>tilt along y (mm/100 mm)</th><th>dish scale</th><th>dish + tilt χ² / ndf</th></tr>{''.join(rows)}</table>
<p>The last four columns fit dg = c0 + a U + b W + s × dish, U and W in units of 100 mm from the chamber centre.
Correlations across cells: G–VT {cg('G', 'VT')}, G–GY {cg('G', 'GY')}, GY–VT {cg('GY', 'VT')}, G–T {cg('G', 'T')}.
Amplitude relative to G (fit through the origin): VT {f(sl['VT'][0], sl['VT'][1])}, T {f(sl['T'][0], sl['T'][1])},
V {f(sl['V'][0], sl['V'][1])}, GY {f(sl['GY'][0], sl['GY'][1])}.</p>

{surfaces_html()}
<h2>Checks</h2>
<figure><img src="figures/gap_runs.png" style="max-width:420px"><figcaption>Run 1 against Run 2, every method and cell:
the runs agree within their errors (χ² per cell ≈ 1 for each method, table above).</figcaption></figure>
<figure><img src="figures/gap_signature.png" style="max-width:420px"><figcaption>Δ ln v against Δ ln T per cell and run.
A gap change at fixed voltage moves them along −β/(1+β) = {k['pure_gap']:.2f} (β = d ln v/d ln E = 0.68, Magboltz at 0.97 %
water, 355 V/cm); a gas change along −1. Fit {f(k['k'], k['err'])} (χ² {k['chi2']:.0f}/{k['ndf']}).</figcaption></figure>
<ul><li>Staging: the Run 1 inclined-track cache rebuilt here is identical, event by event and sample by sample, to S4's pass-1 cache.</li>
<li>Gain: removing the amplitude walk of T changes the T map by ≤ 0.06 mm.</li>
<li>Run 1 and Run 2 were taken in the same mounting, hours apart; the pattern is not a time effect within a run.</li></ul>

<h2>What this does not rule out</h2>
<ul><li>The timing product reproduces the pattern with about 0.7 of the geometric amplitude ({f(sl['VT'][0], sl['VT'][1])}),
and the v-only map varies more than a pure gap change predicts (k = {k['k']:.2f} against {k['pure_gap']:.2f}). So the
amplitude of the variation is known to about 30 %, and part of the velocity map may come from something other than the
gap (β itself is uncertain: 0.68 from Magboltz, 0.87 from the drift-time scan alone).</li>
<li>Structure beyond tilt + dish remains (dish + tilt χ² above ndf on the timing maps); its origin is not identified.</li>
<li>The geometric methods do not see the outer x columns (X strips) or the high-y row (Y strips); there the map rests on timing.</li>
<li>The mean gap (28.2 mm, 1.8 mm short of the 30 mm design) is not explained by this map; we have no measurement of the cathode position.</li>
<li>Only one mounting of chamber A at the operating point is usable; the earlier bottom-slot runs are excluded (pathological drift).</li></ul>
<p>Superseded: the July map (mx_june_wft/GAP_STUDY_2026-07-30.md, 25.7–29.2 mm, drift-time × v on the old kernel) and
the FA follow-up maps (pre-pass bundles, Run 1 only).</p>
</body></html>"""
    open(f'{HERE}/report.html', 'w').write(html)
    print('wrote', f'{HERE}/report.html')


if __name__ == '__main__':
    main()
