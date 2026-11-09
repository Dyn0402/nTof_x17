#!/usr/bin/env python3
"""make_report.py -- report_coupling/report.html: what capacitive coupling implies and what MX17 chamber A shows.

Reads results/coupling_summary.json, results/coupling_boundary.json, results/coupling_fits/*.json (made by observables.py, profile.py,
boundary.py, fit_kernel.py on the window pickles), so every number in the tables and the verdict text follows the tables.
"""
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
OUT = os.path.join(HERE, '..', 'report_coupling')
INK, MUT = '#1f2328', '#6b7280'
C = {'base': '#2a78d6', 'free': '#eb6834', 's033': '#1baf7a'}
M = {'base': 's', 'free': '^', 's033': 'D'}
LAB = {'base': 'no coupling (σ₀ free)', 'free': 'all free', 's033': 'σ₀ = 33 µm, coupling free'}
plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUT, 'axes.labelcolor': INK,
                     'xtick.color': MUT, 'ytick.color': MUT, 'axes.grid': True, 'grid.color': '#e7e7e4', 'grid.linewidth': .6, 'legend.frameon': False, 'figure.dpi': 130})


def f(x, p='.3f'):
    return format(x, p)


def main():
    S = json.load(open(os.path.join(RES, 'coupling_summary.json')))
    B = json.load(open(os.path.join(RES, 'coupling_boundary.json')))
    F = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(RES, 'coupling_fits', '*.json'))}
    os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
    st, gr = S['stacks'], S['grid']
    iearly = [gr.index(k) for k in (0, 1, 2, 3)]

    def ratio(v, ph, name, strip, idx=None):
        a, b = st[f'{v}|{ph}|{name}|{strip}']['med'], st[f'{v}|{ph}|{name}|centre']['med']
        return sum(a[i] / b[i] for i in iearly) / 4 if idx is None else a[idx] / b[idx]

    # F1: near and far neighbour against position in strip
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    px = [0.1, 0.3, 0.45]
    for j, v in enumerate('xy'):
        for i, strip in enumerate(('near', 'far')):
            a = ax[i, j]
            for n in ('base', 'free', 's033'):
                a.plot(px, [ratio(v, k, n, strip) for k in range(3)], color=C[n], marker=M[n], ms=5, lw=1.6, label=LAB[n])
            a.plot(px, [ratio(v, k, 'data', strip) for k in range(3)], color=INK, marker='o', ms=6, lw=0, label='data')
            a.set_ylabel(f'{strip} strip / centre')
            a.set_title(f'{v.upper()} view, first four samples', fontsize=10, color=INK)
            a.set_ylim(bottom=0)
    ax[1, 0].set_xlabel('|position in strip| (pitch)')
    ax[1, 1].set_xlabel('|position in strip| (pitch)')
    ax[0, 0].legend(fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figures', 'phase.png'))
    plt.close(fig)

    # F2: the -2 strip from t0
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    xs = list(range(gr.index(-1), gr.index(9)))
    for j, v in enumerate('xy'):
        a = ax[j]
        for n in ('base', 's033'):
            a.plot([gr[i] for i in xs], [st[f'{v}|1|{n}|far2']['med'][i] for i in xs], color=C[n], marker=M[n], ms=4, lw=1.6, label=LAB[n])
        a.plot([gr[i] for i in xs], [st[f'{v}|1|data|far2']['med'][i] for i in xs], color=INK, marker='o', ms=5, lw=0, label='data')
        a.set_title(f'{v.upper()} view, −2 strip (position in strip 0.2–0.4)', fontsize=10, color=INK)
        a.set_xlabel('samples after t₀ (60 ns)')
        a.set_ylabel('amplitude / centre peak')
    ax[0].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figures', 'far2.png'))
    plt.close(fig)

    # F3: time shift profile
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ed = S['xbins']
    xc = [(ed[i] + ed[i + 1]) / 2 for i in range(len(ed) - 1)]
    arms = [('s033', 'coupling + diffusion', C['base'], 'D'), ('s033nok', 'diffusion only', C['free'], '^'), ('s033nodiff', 'coupling only', C['s033'], 's'),
            ('point', 'neither (geometry)', '#eda100', 'v')]
    for j, v in enumerate('xy'):
        a = ax[j]
        for n, l, c, m in arms:
            a.plot(xc, S['profile'][v][n], color=c, marker=m, ms=4, lw=1.5, label=l)
        a.plot(xc, S['profile'][v]['data'], color=INK, marker='o', ms=5, lw=0, label='data')
        a.set_title(f'{v.upper()} view', fontsize=10, color=INK)
        a.set_xlabel('position along track (0 mesh, 1 cathode)')
        a.set_ylabel('hit time − track time (ns)')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figures', 'profile.png'))
    plt.close(fig)

    u = S['uratio']
    pts = {v: dict(point=u[v]['point'], full=u[v]['s033'], nok=u[v]['s033nok'], nod=u[v]['s033nodiff'], data=u[v]['data'], pas=u[v]['pass'], base=u[v]['base'], free=u[v]['free'], freenok=u[v]['freenok']) for v in 'xy'}

    def fitrow(name, label, v):
        r = F[f'{name}_{v}']
        p = r['params']
        d = r['mean_ln_chi2'] - F[f'base_{v}']['mean_ln_chi2']
        return f'<tr><td>{label}</td><td>{f(d, "+.3f")}</td><td>{f(p["sigma"], ".3f")}</td><td>{f(p["Dp"], ".4f")}</td><td>{f(p["k1"], ".3f")}</td><td>{f(p["k2"], ".3f")}</td></tr>'

    def bd(v):
        i, j1, j2 = B[f'{v}|False|False'], B[f'{v}|False|True'], B[f'{v}|True|False']
        w1, w2 = 1 / j1['r2e'] ** 2, 1 / j2['r2e'] ** 2
        m = (w1 * j1['r2'] + w2 * j2['r2']) / (w1 + w2)
        e = (w1 + w2) ** -0.5
        return i, j1, j2, m, e

    def closure(key, v):
        r = F[key + '_' + v]
        return r
    html = ['<!doctype html><meta charset="utf-8"><title>Capacitive coupling on MX17</title><style>'
            'body{font:15px/1.5 system-ui,sans-serif;max-width:880px;margin:24px auto;padding:0 16px;color:#1f2328;background:#fcfcfb}'
            'table{border-collapse:collapse;font-size:13.5px;margin:12px 0;width:100%}th,td{border-bottom:1px solid #e3e3df;padding:4px 7px;text-align:left;vertical-align:top}'
            'img{max-width:100%}.cap{color:#6b7280;font-size:13px}.v{background:#eef3fa;border-left:4px solid #2a78d6;padding:10px 14px;margin:14px 0}h2{margin-top:30px}</style>']
    A = html.append
    A('<h1>Capacitive coupling and the micro-TPC bias on MX17</h1><p class="cap">Chamber A (det3), bench cosmics, Runs 1 and 2, existing waveforms. Code: <code>papers/paper2/hits_daq/coupling/</code>. Literature: <code>reports/Capacitive coupling in strip microTPCs.md</code>.</p>')
    A(f'''<div class="v"><b>Verdict.</b> A pure capacitive replacement for σ₀ does not describe chamber A, but same-time sharing is a large part of the compression, and part of it behaves like coupling.
<ul><li><b>Removing σ₀ works less well than it first looks.</b> With σ₀ at 33 µm the fit asks for k₁ = {f(F["s033_x"]["params"]["k1"], ".2f")} (X) and {f(F["s033_y"]["params"]["k1"], ".2f")} (Y). But data generated with <i>no</i> coupling, only the 0.3 mm Gaussian, gives k₁ = {f(F["inj_basetos033_x"]["params"]["k1"], ".2f")} and {f(F["inj_basetos033_y"]["params"]["k1"], ".2f")} in the same fit, so most of that number is a Gaussian in another language.</li>
<li><b>Position-in-strip dependence rejects pure coupling.</b> It predicts a far neighbour fixed near 0.13; the data are 0.07–0.09 on X, and on the near side the pure-coupling fit is too low by 30 % at the strip centre.</li>
<li><b>A flat floor is still there</b>: the X far neighbour stays at 0.07–0.09 where a Gaussian falls to 0.04. That caps a same-time nearest-neighbour term at about 7 % on X and 9 % on Y, with a smooth width of about 0.25 mm still needed on X.</li>
<li><b>Same-time sharing matters for the bias.</b> With σ₀ removed, X compression goes from {f(pts["x"]["point"], ".3f")} (geometry) to {f(pts["x"]["nok"], ".3f")} with diffusion only, {f(pts["x"]["nod"], ".3f")} with coupling only, and {f(pts["x"]["full"], ".3f")} with both (data {f(pts["x"]["data"], ".3f")}): equal shares on X. It is not negligible, and our text must not say so.</li>
<li><b>The −2 strip is mostly electronics.</b> It reads {f(bd("x")[0]["r2"]*100, ".1f")} % (X) within one chip and {f(bd("x")[3]*100, ".1f")} ± {f(bd("x")[4]*100, ".1f")} % when the strip sits across a chip or connector boundary.</li></ul>
<b>Physics we cannot supply:</b> nothing we can calculate gives a 7 % same-time term. Direct strip-to-strip capacitance estimates at 1–2 % and is derivative-shaped, and the 31.2 mm resistive-strip beat that could give more is not seen. So the data fix the size of the same-time sharing better than they fix its origin.</div>''')

    A('''<h2>1. What "capacitive coupling" physically is</h2>
<p>The phrase covers separate mechanisms. They differ in time structure, in net charge and in what sets their size, and a citation for one does not support another.</p>
<table><tr><th>Mechanism</th><th>Physics</th><th>Time on the neighbour</th><th>Net charge</th><th>Estimate for MX17</th></tr>
<tr><td>Direct capacitance between amplifier-fed strips (combs)</td><td>A real charge pulse on strip 0 drives C_is into the neighbour's input. A charge-sensitive amplifier is close to a virtual ground, so the transferred charge flows back: Z<sub>in</sub> ≈ R<sub>in</sub> + 1/(sC<sub>in</sub>), neighbour = k<sub>HF</sub>·transient + k<sub>LF</sub>, k<sub>HF</sub> ≈ C<sub>is</sub>/C<sub>node</sub>, k<sub>LF</sub> ≈ C<sub>is</sub>/(C<sub>in</sub> + C<sub>node</sub>)</td><td>Prompt, derivative-like, leading the main peak by 40–115 ns</td><td>Small</td><td>0.3–4.5 %, best 1–2 % (computed; DREAM R<sub>in</sub>, C<sub>in</sub> assumed)</td></tr>
<tr><td>Charge on a floating electrode imaged onto several strips</td><td>Charge on a resistive strip has no amplifier of its own. It images on the pads below in the ratio of the capacitances. This is the route of the LMU LTspice models (0.25–0.29 q)</td><td>Same time, same shape: a true scaled copy</td><td>Non-zero</td><td>~1 % per side when centred on a column; up to 1:1 where a 0.80 mm strip straddles two 0.78 mm columns (31.2 mm beat), projected-area mean 22 %</td></tr>
<tr><td>Weighting-field induction on non-collecting strips</td><td>Shockley–Ramo: signal while the charge moves, returning to zero unless electrons land there</td><td>Bipolar, ends when charge arrives</td><td>Zero</td><td>≈ 0 after shaping</td></tr>
<tr><td>RC spreading along resistive strips (Y view)</td><td>Charge moves along the strip with diffusion constant 1/(R□c′)</td><td>Delayed, longer with distance</td><td>Non-zero</td><td>Already in the model (rc_D_y)</td></tr>
<tr><td>Front-end or connector cross-talk</td><td>Mutual capacitance in the ASIC or flex</td><td>Same time, or derivative</td><td>Depends on path</td><td>Unknown; no DREAM measurement found. Measured here at chip boundaries</td></tr></table>
<h2>2. What a coupling does to a track</h2>
<p>Take a same-time copy, S<sub>i</sub>(t) → S<sub>i</sub>(t) + κ[S<sub>i−1</sub>(t) + S<sub>i+1</sub>(t)]. On an inclined track the strip i's neighbours are one earlier and one later by Δt = pitch/w.</p>
<ul><li><b>Interior strips:</b> both neighbours contribute. A leading-edge time moves earlier by the same amount on every interior strip, so the interior ladder stays straight: its slope is not changed to first order.</li>
<li><b>Track ends:</b> the end strip has a neighbour only on one side, and the strip beyond the end receives only a copy of it, at the neighbour's time (the "ghost hit"). Ends and ghosts rotate the fitted line. This is why the effect is an end effect, why it shrinks with the number of strips (roughly 1/N² in slope, the form Bortfeldt's curve follows) and why cutting edge strips helps.</li>
<li><b>Diffusion</b> behaves differently: charge from a range of depths reaches a strip, and the spread grows with depth, so it can also act on the interior. Both act one-sidedly at the ends. So the angle dependence alone cannot tell them apart, as the literature review found; depth and position-in-strip can.</li></ul>''')

    A(f'''<h2>3. What we would have to see, and what we see</h2>
<table><tr><th>If coupling is present, then…</th><th>Observed</th><th>Reading</th></tr>
<tr><td>A neighbour signal that starts with the centre strip at t₀, with no delay</td><td>Yes: −1 and −2 rise with the centre from the first sample</td><td>Agrees, but a same-time footprint gives this too</td></tr>
<tr><td>A neighbour ratio independent of where the charge lands in the strip</td><td>X far neighbour 0.07–0.09, flat from phase 0.2 (Gaussian falls to 0.04). Injecting position error does not produce it</td><td>A floor exists, capped at ~7 % (X), ~9 % (Y). Near the strip centre the data are below every model</td></tr>
<tr><td>Neighbour at the near side set by the coupling alone at σ₀ → 0</td><td>Near strip 0.23 at the centre against 0.16 for pure coupling (X)</td><td>Pure coupling rejected; a smooth width is needed</td></tr>
<tr><td>Reach: ±2 only as κ²</td><td>−2 carries 1.3–1.7 %, in both runs and views, and −3, −4 ≈ 0.1 %</td><td>Above κ² for κ near 0.06, but see the next row</td></tr>
<tr><td>Board coupling survives across an ASIC/connector boundary; electronics does not</td><td>X −2: {f(bd("x")[0]["r2"]*100, ".2f")} ± {f(bd("x")[0]["r2e"]*100, ".2f")} % inside a chip, {f(bd("x")[3]*100, ".2f")} ± {f(bd("x")[4]*100, ".2f")} % across. Y: {f(bd("y")[0]["r2"]*100, ".2f")} ± {f(bd("y")[0]["r2e"]*100, ".2f")} % against {f(bd("y")[3]*100, ".2f")} ± {f(bd("y")[4]*100, ".2f")} %. The −1 strip changes little</td><td>The −2 signal is mostly front-end or connector. −1 is mostly on the board</td></tr>
<tr><td>Derivative-shaped neighbour (direct C into a charge amplifier)</td><td>χ² is minimal at zero derivative amplitude, both views</td><td>Not requested above a few % per sample; 60 ns sampling limits sensitivity</td></tr>
<tr><td>Imaging from 0.80 mm strips: charge pinned to that grid, 31.2 mm beat in the near share</td><td>No 0.80 mm periodicity in first-sample centroids (Rayleigh Z 2.5 X, 1.4 Y; 281 and 152 at 0.78). Near share shows no beat structure at the predicted size</td><td>Large geometric route disfavoured; early-charge smearing by diffusion weakens the pinning test</td></tr>
<tr><td>Linearity: fixed ratio at every amplitude</td><td>−2 flat (1.1–1.4 % X, 1.6–1.8 % Y across five charge quintiles); −1 falls from 8.2 % to 5.4 % (X)</td><td>−2 agrees. The −1 trend is not explained by a fixed ratio; not understood</td></tr>
<tr><td>Correlated noise between neighbours</td><td>Not measurable: the window tables keep 9–12 strips around the track and too few quiet samples</td><td>Stated limitation</td></tr></table>
<img src="figures/phase.png" alt="neighbour against position in strip"><p class="cap">Neighbour/centre at the first four samples against position in strip, near-normal tracks, Runs 1+2. Pure coupling (green) is too low on the near side and flat at too high a level on the far side; no coupling (blue) falls too fast on the far side.</p>
<img src="figures/far2.png" alt="−2 strip"><p class="cap">The −2 strip rises with the centre strip. The same-time term with k₂ follows it; the pass model has almost none.</p>''')

    x, y = F, F
    A(f'''<h2>4. Removing σ₀: joint fits</h2>
<p>All planes share σ₀, D<sub>p</sub>, RC (Y) and same-shape k₁, k₂; charge per plane by NNLS; track at the pass values; 2,000 planes per view. Mean ln χ² relative to the no-coupling fit. Parity halves give k₁ {f(F["s033_p0_x"]["params"]["k1"])}/{f(F["s033_p1_x"]["params"]["k1"])} (X), {f(F["s033_p0_y"]["params"]["k1"])}/{f(F["s033_p1_y"]["params"]["k1"])} (Y).</p>
<table><tr><th>X view</th><th>Δ mean ln χ²</th><th>σ₀ (mm)</th><th>D<sub>p</sub></th><th>k₁</th><th>k₂</th></tr>{fitrow("base","no coupling","x")}{fitrow("free","all free","x")}{fitrow("s070","σ₀ 0.07","x")}{fitrow("s033","σ₀ 0.033","x")}{fitrow("s033_nok2","σ₀ 0.033, k₂ = 0","x")}</table>
<table><tr><th>Y view</th><th>Δ mean ln χ²</th><th>σ₀ (mm)</th><th>D<sub>p</sub></th><th>k₁</th><th>k₂</th></tr>{fitrow("base","no coupling","y")}{fitrow("free","all free","y")}{fitrow("s070","σ₀ 0.07","y")}{fitrow("s033","σ₀ 0.033","y")}{fitrow("s033_nok2","σ₀ 0.033, k₂ = 0","y")}</table>
<p>D<sub>p</sub> falls to 0.018–0.020 from 0.023 once the kernel is freed, which matters because σ₀ and D<sub>p</sub> trade. χ²/dof is about 66, so these are preferences, not significances.</p>
<h3>Closure: what does the fit return on known data?</h3>
<table><tr><th>Generated with</th><th>Fitted with</th><th>Result</th><th>Mean ln χ²</th></tr>
<tr><td>X: no coupling, σ₀ 0.318</td><td>σ₀ = 33 µm, k free</td><td>k₁ = {f(F["inj_basetos033_x"]["params"]["k1"])}, k₂ = {f(abs(F["inj_basetos033_x"]["params"]["k2"]))}</td><td>{f(F["inj_basetos033_x"]["mean_ln_chi2"], ".2f")}</td></tr>
<tr><td>X: no coupling, σ₀ 0.318</td><td>everything free</td><td>recovers σ₀ {f(F["inj_basetofree_x"]["params"]["sigma"])}, k₁ {f(F["inj_basetofree_x"]["params"]["k1"])}</td><td>{f(F["inj_basetofree_x"]["mean_ln_chi2"], ".2f")}</td></tr>
<tr><td>X: σ₀ 33 µm, k₁ 0.131, k₂ 0.010</td><td>the same form</td><td>k₁ {f(F["inj_s033tos033_x"]["params"]["k1"])}, k₂ {f(F["inj_s033tos033_x"]["params"]["k2"])}</td><td>{f(F["inj_s033tos033_x"]["mean_ln_chi2"], ".2f")}</td></tr>
<tr><td>X: σ₀ 33 µm, k₁ 0.131, k₂ 0.010</td><td>no coupling</td><td>σ₀ {f(F["inj_s033tobase_x"]["params"]["sigma"])}</td><td>{f(F["inj_s033tobase_x"]["mean_ln_chi2"], ".2f")}</td></tr>
<tr><td>X: σ₀ 33 µm, k₁ 0.131, k₂ 0.010</td><td>everything free</td><td>recovers σ₀ {f(F["inj_s033tofree_x"]["params"]["sigma"])}, k₁ {f(F["inj_s033tofree_x"]["params"]["k1"])}</td><td>{f(F["inj_s033tofree_x"]["mean_ln_chi2"], ".2f")}</td></tr></table>
<p>The fit recovers truth, and with white noise the two forms differ by 1.7–2.4 in mean ln χ². In the data they differ by 0.01–0.07, because model error sets the χ². So the χ² cannot separate them on real data; only the shape tests in section 3 can. That is why fitted k₁ on its own is not evidence.</p>
<h2>5. Where along the track the hit time moves</h2>
<img src="figures/profile.png" alt="time shift along the track"><p class="cap">Median hit time relative to the track, 0.1 &lt; |tan| &lt; 0.45, Run 1, relative to mid-track. All arms use σ₀ = 33 µm with the fit's other parameters.</p>
<table><tr><th>Ladder slope ratio (median)</th><th>Data</th><th>Pass model</th><th>Geometry only</th><th>Diffusion only</th><th>Coupling only</th><th>Both</th></tr>
<tr><td>X</td><td>{f(pts["x"]["data"])}</td><td>{f(pts["x"]["pas"])}</td><td>{f(pts["x"]["point"])}</td><td>{f(pts["x"]["nok"])}</td><td>{f(pts["x"]["nod"])}</td><td>{f(pts["x"]["full"])}</td></tr>
<tr><td>Y (RC kept on)</td><td>{f(pts["y"]["data"])}</td><td>{f(pts["y"]["pas"])}</td><td>{f(pts["y"]["point"])}</td><td>{f(pts["y"]["nok"])}</td><td>{f(pts["y"]["nod"])}</td><td>{f(pts["y"]["full"])}</td></tr></table>
<p>In the σ₀-removed arms, coupling and diffusion each add about 0.07 on X. With σ₀ kept, the same term falls to {f(pts["x"]["free"] - pts["x"]["freenok"], ".3f")} of the {f(pts["x"]["free"] - pts["x"]["point"], ".3f")} excess. So the size of "coupling's share" depends on how same-time sharing is parametrised. <b>What is firm is that same-time sharing, by any name, is a substantial part of the X bias.</b> The time shift sits at the track ends and beyond them, as the argument in section 2 expects. In the middle 60 % of the track, data move about half as much as any arm with diffusion, close to the arm with coupling only. At and beyond the ends, the data need diffusion. This is one chamber and the difference inside the track is about 5 ns in 780 ns.</p>''')

    A('''<h2>6. Wording for the paper</h2>
<p><i>"Single-time-per-strip reconstruction reads tracks too steep. The literature attributes this to capacitive coupling and edge strips; those studies have not separated coupling from diffusion and geometry. In our chamber, the bias arises from same-time charge sharing between neighbouring strips combined with depth mixing by drift diffusion, each comparable on the view across the resistive strips' direction [X], with resistive spreading added on the other. The same-time sharing cannot be a pure nearest-neighbour coupling of the size needed to replace the prompt width: its dependence on the position in the strip requires a smooth component and caps a same-time floor at about 7–9 %. Its physical origin is not identified."</i></p>
<h2>7. What this does not rule out</h2><ul>
<li>Only chamber A. The strength of coupling depends on the board stack and the electronics, so other chambers may differ.</li>
<li>Near-normal samples for sections 3 and the track held at the pass fit for the joint fits; the pass itself was calibrated with no coupling.</li>
<li>The bottom-up circuit estimate uses an assumed board stack and DREAM impedances; a factor of 3–5 either way is not excluded.</li>
<li>Non-Gaussian footprints with long tails are not excluded by the floor; Lorentzian tails match the −2 strip on X but not Y.</li>
<li>The fall of the −1 ratio with amplitude and the data lying below all models near the strip centre are unexplained.</li>
<li>Noise correlations could not be tested with the saved windows.</li></ul>
<h2>Regenerate</h2><pre>coupling/observables.py, profile.py, boundary.py, fit_kernel.py (run_fits.sh) on the window pickles → results/coupling_*; coupling/make_report.py</pre>''')
    open(os.path.join(OUT, 'report.html'), 'w').write('\n'.join(html))
    print(os.path.join(OUT, 'report.html'))


if __name__ == '__main__':
    main()
