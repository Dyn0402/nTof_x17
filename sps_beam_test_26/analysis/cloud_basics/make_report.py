#!/usr/bin/env python3
"""make_report.py -- report.html for the late-charge (attachment) study.

Reads results/*.json and <OUT>/fits.json (make_figures.py) and writes
<OUT>/report.html with relative figure links (figures/x.png), so the file works
from disk, from the DAQ page's Analysis tab, or copied with its figures/.
Generated: re-run make_figures.py then this after any analysis update.

    make_report.py [--out DIR]
"""
import argparse
import html
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
OUT = '/home/dylan/x17/cosmic_bench/cloud_basics/attachment'


def J(p):
    return json.load(open(p))


def f2(x, n=3):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def table(head, rows, cls=''):
    h = ''.join(f'<th>{c}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--inline', default='',
                    help='also write this self-contained copy with the figures embedded as data: URIs '
                         '(for the offline notes site)')
    a = ap.parse_args()
    F = J(os.path.join(a.out, 'fits.json'))
    H = J(os.path.join(RES, 'headon_masked_k12.json'))
    L = J(os.path.join(RES, 'ladder_profile.json'))
    G = J(os.path.join(RES, 'gain_vs_loss.json'))
    S = J(os.path.join(RES, 'headon_split.json'))
    T = J(os.path.join(RES, 'split_toy.json'))
    E = J(os.path.join(RES, 'zs_emulate.json'))
    Zt = J(os.path.join(RES, 'zs_timestack.json'))
    old = J(os.path.join(RES, 'headon_oldblock_k4.json'))
    PL = (('raw700', 243), ('raw450', 150), ('raw275', 92))

    rates = [F[l]['r'] * 1e4 for l, _ in PL]
    cfp0 = os.path.join(a.out, 'compositions.json')
    _b = J(cfp0)['beam'] if os.path.exists(cfp0) else dict(water=np.nan, air=np.nan, o2_ppm=np.nan)
    AB = dict(water_pct=_b['water'], air_pct=_b['air'], o2_ppm=_b['o2_ppm'])
    # O2 equivalent: Magboltz Ar/CF4/iso + 1.7 % H2O + 0.1 % O2, eta*v at each field
    MB = {p['E_Vcm']: p for p in J(os.path.join(RES, 'magboltz_beam_w1p7_o0p1.json'))['points']}
    ppm = [1000 * F[l]['r'] / (MB[Ev]['eta_per_cm'] * MB[Ev]['v_true_um_ns'] * 1e-4) for l, Ev in PL]
    ZH = J(os.path.join(RES, 'zs_headon.json')); tzh = np.array(ZH['t'])
    sx = np.array(ZH['r63_flat700']['x']['sum']['1'])
    r63flat_x1 = sx[(tzh >= 2400) & (tzh < 2700)].mean() / sx[(tzh >= 1080) & (tzh <= 1260)].mean()
    us = F['undershoot']

    # -------- tables
    t_raw = []
    for lab, Ev in PL:
        hx, hy = H[lab]['x'], H[lab]['y']
        t_raw.append([f'{Ev} V/cm', f'{H[lab]["n"]}',
                      f'{f2(hx["metrics"]["2"]["r1800"])} ± {f2(hx["metrics_err"]["2"]["r1800"])}',
                      f'{f2(hx["metrics"]["2"]["r2400"])} ± {f2(hx["metrics_err"]["2"]["r2400"])}',
                      f'{f2(hy["metrics"]["2"]["r2400"])}',
                      f'{f2(hy["metrics"]["8"]["r2400"])} ± {f2(hy["metrics_err"]["8"]["r2400"])}',
                      f'{f2(hy["metrics"]["12"]["r2400"])}',
                      f'<b>{F[lab]["r"] * 1e4:.2f} ± {F[lab]["r_err"] * 1e4:.2f}</b>',
                      f'{F[lab]["chi2"]:.0f} / {F[lab]["chi2_r0"]:.0f}'])
    t_lad = []
    for arm, Ev in (('rot_d425', 142), ('rot_d325', 108), ('rot_d225', 75)):
        f = L[arm]['fit']
        t_lad.append([f'{Ev} V/cm', f'{L[arm]["n_events"]}', f'{f["t_range"][0] / 1e3:.2f}–{f["t_range"][1] / 1e3:.2f} µs',
                      f'{f["depth_span_mm"]:.1f} mm', f'{f["survival_over_span"]:.2f}',
                      f'{f["per_mm"][0]:.3f} ± {f["per_mm"][1]:.3f}',
                      f'{f["per_ns"][0] * 1e4:.2f} ± {f["per_ns"][1] * 1e4:.2f}'])
    t_zs = []
    for k in ('r63_flat700', 'r63_d425', 'r63_d325', 'r56_625V', 'r56_590V'):
        z = Zt[k]
        t_zs.append([html.escape(z['label']), f'{z["n"]}', f'{z["R_x_all"][0]:.3f}', f'{z["R_x_mid"][0]:.3f}',
                     f'{z["R_y_all"][0]:.3f}', f'{z["R_y_mid"][0]:.3f}'])
    t_art = []
    for lab, Ev in PL:
        t_art.append([f'{Ev} V/cm', f2(old[lab]['x']['metrics']['2']['r2400']), f2(H[lab]['x']['metrics']['2']['r2400']),
                      f2(E[lab]['x']['raw']), f2(E[lab]['x']['zs4']), f2(E[lab]['x']['zs5'])])
    t_disc = []
    for lab, Ev in PL:
        for key in ('by_x', 'by_y'):
            g = G[lab][key]
            gs = [r['gain'] for r in g['rows']]
            t_disc.append([f'{Ev} V/cm', key.replace('by_', 'position in '), f'×{max(gs) / min(gs):.1f}',
                           f'{g["slope_Rx_per_relgain"][0]:+.3f} ± {g["slope_Rx_per_relgain"][1]:.3f}'])
    t_rate = []
    for lab, Ev in PL:
        for key in ('rate', 'spill'):
            t_rate.append([f'{Ev} V/cm', key] + [f'{r["x"][0]:.3f} ± {r["x"][1]:.3f}' for r in S[lab][key]])
    t_toy = [['data, 243 V/cm'] + [f'{r["x"][0]:.3f} ± {r["x"][1]:.3f}' for r in S['raw700']['charge']] +
             [f2(H['raw700']['x']['metrics']['2']['r2400'])]]
    for rk, nm in (('0.0', 'toy, r = 0'), ('0.0001', 'toy, r = 1.0e-4/ns'), ('0.00019', 'toy, r = 1.9e-4/ns'),
                   ('0.0003', 'toy, r = 3.0e-4/ns')):
        t_toy.append([nm] + [f2(x) for x in T[rk]['terciles']] + [f2(T[rk]['all'])])

    figs = [
        ('f1_headon_xy.png', 'The observable',
         'run_71 RAW, both views head-on, 20 000 events per field. Time is the DREAM sample index (the window '
         'opens on the beam trigger), missing RAW samples are excluded, per-strip pre-trigger baselines, masked '
         'common mode. X summed over ±2 strips, Y over ±8 (Y needs the width to contain its resistive-strip '
         'spread). Shaded bands: bootstrap over events. Black: box(T)·e<sup>−rt</sup> ⊗ the measured electronics '
         'template; dotted grey: the same fit with r set to zero. The grey area is the charge that did not arrive.'),
        ('f2_time_not_depth.png', 'A loss per unit time, not per unit depth',
         'X ±2 at the three fields against drift time (left) and drift depth (right). The curves coincide in '
         'time and separate in depth. Dotted: what a loss with a fixed attenuation length per mm (taken from '
         '243 V/cm) would look like in time at 150 and 92 V/cm.'),
        ('f3_undershoot.png', 'Not the readout',
         'At 243 V/cm the drift ends inside the window. A readout high-pass with the time constant that makes '
         'the same plateau sag must swing to ≈ −0.3 when the current stops; the data settle at '
         f'{us["data_x"]:+.3f} (X) and {us["data_y"]:+.3f} (Y).'),
        ('f5_ladder_depth.png', 'Template-free, depth-resolved: charge per strip along the beam drift ladder',
         'Left: run_63 at 25.64°, the Y view is the drift ladder (each strip collects one depth slice). '
         'Per-strip peak amplitude, 70th percentile over ALL events with an unfired strip counted as 0 '
         '(immune to zero-suppression censoring), from the first full strip past the mesh edge to the last '
         'strip whose pulse is inside the 3.84 µs window. Right: the same per-strip charge on inclined bench '
         'cosmics (both views, Ar/iso), normalised at 6–12 mm; det4 X omitted (its amplification stripes run '
         'across X). The bench interior is flat to ±5 %; the fall beyond ~22 mm is the gap end and M3 smearing.'),
        ('f4_rot_xy_time.png', 'Same events, both views: the ladder view loses charge in time like the head-on view',
         'run_63 rotated mount. X is head-on (±2); Y is the ladder, summed over all its strips, which is the '
         'same arriving current. ZS 4σ, middle 60 % of events by total charge. An unexplained ~0.7 µs ripple, '
         'anti-phased between X and Y, rides on both (not present in RAW run_71; likely on-board common-mode '
         'handling in ZS mode on the shared FEU) — it does not change the trend.'),
        ('f6_all_datasets.png', 'Every beam dataset, both views',
         'R = level(2.4–2.7 µs) / level(1.08–1.26 µs). RAW rows: bootstrap errors. ZS rows: the bar is the '
         'range between all events and the middle 60 % by charge (ZS censoring makes R selection-dependent at '
         'the ±0.05 level); ZS Y reads low because Y\'s spread signal is censored more (in RAW, Y = X).'),
        ('f7_discriminators.png', 'What it does not depend on',
         'Left: R in position bins against the local gain (det4\'s amplification stripes, ×1.4–2.8); '
         'resistive-layer charging would follow the dashed line. Middle: beam-rate and spill-phase terciles. '
         'Right: split by each event\'s total charge — attachment itself makes bright events read lower '
         '(their big clusters were preferentially early); the toy reproduces mid and bright.'),
    ]
    air_html = ''
    cfp = os.path.join(a.out, 'compositions.json')
    if os.path.exists(cfp):
        Cm = J(cfp); bm = Cm['beam']; zs = Cm['zs_arms']; co2 = Cm['co2_period']
        DSf = J(os.path.join(RES, 'driftscan_fit_air_hs_x.json')); BCf = J(os.path.join(RES, 'beam_comp_fit_air_hs.json'))
        BFf = J(os.path.join(RES, 'bench_fit_x_free.json'))
        rows = [['run_63 25.6°, 142 V/cm (ZS)', 'Aug 3 00:22–00:30', f'({zs["r63_d425"]["water_68"][0]:.1f}–{zs["r63_d425"]["water_68"][1]:.1f})',
                 f'{zs["r63_d425"]["air"]:.3f}', f'{zs["r63_d425"]["o2_ppm"]:.0f}', f'{zs["r63_d425"]["chi2"]:.0f} / {zs["r63_d425"]["n"]}',
                 f'{zs["r63_d425"]["chi2_noair"]:.0f}'],
                ['run_63 25.6°, 108 V/cm (ZS)', 'Aug 3 00:30–00:37', f'({zs["r63_d325"]["water_68"][0]:.1f}–{zs["r63_d325"]["water_68"][1]:.1f})',
                 f'{zs["r63_d325"]["air"]:.3f}', f'{zs["r63_d325"]["o2_ppm"]:.0f}', f'{zs["r63_d325"]["chi2"]:.0f} / {zs["r63_d325"]["n"]}',
                 f'{zs["r63_d325"]["chi2_noair"]:.0f}'],
                ['run_63 flat, 243 V/cm (ZS)', 'Aug 3 01:00–01:54', f'{zs["r63_flat700"]["water"]:.2f}', f'{zs["r63_flat700"]["air"]:.3f}',
                 f'{zs["r63_flat700"]["o2_ppm"]:.0f}', f'{zs["r63_flat700"]["chi2"]:.0f} / {zs["r63_flat700"]["n"]}',
                 f'{zs["r63_flat700"]["chi2_noair"]:.0f}'],
                ['<b>run_71 RAW, 3 fields × 2 views + ladder v</b>', 'Aug 3 05:22–05:52', f'<b>{bm["water"]:.2f}</b>', f'<b>{bm["air"]:.3f}</b>',
                 f'<b>{bm["o2_ppm"]:.0f}</b>', f'{sum(bm["chi2"].values()):.0f} / {55 * len(bm["chi2"])}',
                 f'{sum(bm["chi2_noair"].values()):.0f}'],
                ['run_56 flat, CO<sub>2</sub> gas (ZS 5σ, rough)', 'Aug 1 15:47', f'{co2["water"]:.2f}', f'{co2["air"]:.3f}',
                 f'{co2["o2_ppm"]:.0f} ({co2["o2_range"][0]:.0f}–{co2["o2_range"][1]:.0f})', '—', '—'],
                ['<b>bench det3, 6 fields (35–382 V/cm)</b>', 'Jun 27', f'<b>{DSf["water"]:.2f}</b>', '<b>0</b>', '<b>≲ 2 (stat.), ≲ 10</b>',
                 f'{sum(v["chi2"] for v in DSf["per_field"].values()):.0f} / {sum(v["n"] for v in DSf["per_field"].values())}', '—']]
        for d in ('det2', 'det4', 'det7'):
            f = BFf[d]
            rows.append([f'bench {d}, {f["E"]:.0f} V/cm', 'Jun', f'{f["water"]:.2f}', f'≤ {max(f["air_68"][1], 0.005):.3f}',
                         f'≤ {max(f["air_68"][1] * 2095, 10):.0f}', f'{f["chi2"]:.0f} / {f["n"]}', '—'])
        air_html = f"""
<h2>Gas compositions, shown as model curves on the data</h2>
<p>One physics model for both setups (<code>predict.current_field</code>): uniform ionisation over the gap, drift with
Magboltz v, η and D<sub>L</sub> for the gas (high-statistics grid, condor 4410759/4410787, 3e8 collisions; η smoothed in E
because Magboltz's attachment Monte Carlo still scatters ±10–15 % between neighbouring fields), an optional linear
drift-field profile and gap spread (geometry), a parametric shaper and trigger jitter. <b>No free loss rate</b>: the loss
follows from the composition. Shared across every field and view of a dataset: composition, geometry, shaper. Free per
stack: amplitude and t0. Gases: beam Ar/CF<sub>4</sub>/iso 88/10/2 (CO<sub>2</sub> period Ar/CO<sub>2</sub>/iso 95/3/2),
bench Ar/iso 95/5; water and air (N<sub>2</sub>/O<sub>2</sub>/Ar 78.08/20.95/0.93) replace argon.</p>
@@COMP_TABLE@@
<figure><img src="figures/f9_beam_composition.png" alt="beam composition" loading="lazy"><figcaption><b>Beam.</b> run_71 RAW,
both views at three fields and the drift velocity at four, all from one composition. Grey dotted: the same gas without
air. Residuals in units of the bootstrap error.</figcaption></figure>
<figure><img src="figures/f10_bench_driftscan.png" alt="bench drift scan" loading="lazy"><figcaption><b>Bench, one gas fill,
six fields.</b> det3 on 6-27. The same model with no air describes all six fields; grey dotted: the same gas with
0.04 % air added (84 ppm O<sub>2</sub>, half the beam's), which the low fields reject (χ² up to ×13).</figcaption></figure>
<figure><img src="figures/f11_bench_chambers.png" alt="bench chambers" loading="lazy"><figcaption><b>Bench, four chambers at
their operating field.</b> det4 X head-on only (its amplification stripes run across X). det7 fits poorly (χ² 191/86): its
plateau rises ~8 %, which needs a field gradient beyond what det3's drift scan supports or a gap different from its
marginal 27.5 mm; it wants no air either way.</figcaption></figure>
<figure><img src="figures/f13_zs_arms.png" alt="ZS arms" loading="lazy"><figcaption><b>The night before run_71.</b> Zero-suppressed
run_63 blocks, composition fitted per block (ZS distortion from RAW run_71 at the nearest field, additive; +1 % systematic).
The run_71 composition (dashed) does not describe them: the gas changed over hours.</figcaption></figure>
<figure><img src="figures/f12_composition_map.png" alt="composition map" loading="lazy"><figcaption><b>The compositions.</b>
Beam points carry 68 % ranges (the rotated blocks cannot pin the water: no drift end in the window); bench points are upper
limits on O<sub>2</sub>.</figcaption></figure>
<ul>
<li><b>Bench and beam are explained by the same model.</b> The beam gas carries 1.5–1.6 % water and 150–230 ppm O<sub>2</sub>
(falling through the night of Aug 2–3; ~290 ppm two days earlier in the CO<sub>2</sub> gas). The bench gas carries 0.6–1.0 %
water and ≲ 10 ppm O<sub>2</sub>. The bench null is not an insensitivity: at its low fields the bench rejects even half the
beam's air.</li>
<li><b>The water and the O<sub>2</sub> do not come in together as room air.</b> Room air carries H<sub>2</sub>O/O<sub>2</sub>
≈ 0.05; the beam gas ≈ 100, the bench gas ≥ 1000. A leak that supplied the water would bring percent-level O<sub>2</sub>
and stop every signal. Water enters on both setups by another route (permeation through tubing, or wet gas); the O<sub>2</sub>
is specific to the beam line and varied over hours, as air ingress would.</li>
<li><b>Caveats.</b> The O<sub>2</sub> scale inherits Magboltz's three-body attachment (H<sub>2</sub>O and isobutane as third
bodies are not well modelled): a factor of a few in absolute ppm, not in the beam/bench contrast or the time trend. The water
fractions assume the nominal drift fields and the measured (bench) or nominal 30 mm (beam) gaps. The fitted shapers are
effective responses (they include the ion tail), different on bench and beam although the DREAM settings are identical.</li>
</ul>"""
        air_html = air_html.replace('@@COMP_TABLE@@', table(['dataset', 'when', 'water [%]', 'air [%]', 'O<sub>2</sub> [ppm]', 'χ² / points', 'χ² with no air'], rows))

    fig_html = ''.join(
        f'<figure><img src="figures/{f}" alt="{html.escape(t)}" loading="lazy">'
        f'<figcaption><b>{html.escape(t)}.</b> {c}</figcaption></figure>' for f, t, c in figs)

    body = f'''
<h1>Late drift charge is lost in the beam gas: the attachment observable, measured</h1>
<p class="sub">det4 at H4 (Aug 2026) against the June bench · MX17 detector-model basics · generated by
<code>cloud_basics/make_report.py</code></p>

<div class="verdict">
<p><b>Verdict.</b> Every beam dataset with waveforms loses late drift charge, in <b>both</b> readout views:
Ar/CF<sub>4</sub>/iso flat (run_71 RAW, three fields; run_63 ZS), Ar/CF<sub>4</sub>/iso at 25.64° (run_63, head-on X
and drift-ladder Y), and Ar/CO<sub>2</sub>/iso flat (run_56, larger loss). In the one dataset that is fully clean
(run_71 RAW) the loss is <b>{rates[0]:.2f}, {rates[1]:.2f} and {rates[2]:.2f} × 10<sup>−4</sup> ns<sup>−1</sup></b> at 243, 150
and 92 V/cm — the same per unit <i>time</i> while the depth reached differs by ~3×, identical in X and Y, with
no undershoot when the drift ends, and independent (within ~2σ) of beam rate, spill phase, position and local
gain. Of the mechanisms tested — readout high-pass, a per-depth (field/geometry) loss, resistive-layer charging,
space charge, alignment and zero-suppression artefacts — only electrons removed from the drifting cloud
(attachment) fits all of it.</p>
<p><b>The gas compositions, as model curves on the data.</b> One physics model — Magboltz drift and attachment for
the actual gas, the measured geometry and electronics, <i>no free loss rate</i> — describes run_71 at three fields
in both views with <b>{AB["water_pct"]:.2f} % water and {AB["air_pct"]:.3f} % air ({AB["o2_ppm"]:.0f} ppm O<sub>2</sub>)</b>,
the run_63 blocks of the night before with 150–230 ppm O<sub>2</sub> (falling over hours), and the bench (det3 at six
fields, 35–382 V/cm) with <b>0.95 % water and no air (≲ 10 ppm O<sub>2</sub>)</b>; the bench rejects even half the beam's air.
Water and O<sub>2</sub> are not in room-air proportion on either setup: the O<sub>2</sub> is specific to the beam line.</p>
<p>It is <b>not</b> what earlier passes claimed: §11's field-ordered "spike" and §13's 5× O<sub>2</sub> spread were
threshold-alignment artefacts, and every earlier stack counted the FEU's dropped RAW samples as zeros (worth
~6 % of fake late loss).</p>
</div>

<div class="kpis">
<div><span>{rates[0]:.2f}</span>e-4 /ns at 243 V/cm</div>
<div><span>{rates[1]:.2f}</span>e-4 /ns at 150 V/cm</div>
<div><span>{rates[2]:.2f}</span>e-4 /ns at 92 V/cm</div>
<div><span>{us["data_x"]:+.3f}</span>undershoot (high-pass would give {us["highpass"]:+.2f})</div>
</div>

<h2>The observable</h2>
<p>A head-on track deposits ionisation uniformly through the 30 mm gap, so the current reaching the mesh is
flat until the drift ends: charge arriving at time t started at depth v·t. Electrons removed during the drift
(attachment) make the current fall with time, by e<sup>−η v t</sup>. The stack has to be built without any
alignment on the pulse itself and without counting missing samples as zero; both mistakes produce structure
of this size (§10, §14, this report's artefact table).</p>
{fig_html}

<h2>run_71 RAW: both views, three fields</h2>
<p>Ratios to the 1.08–1.26 µs level, bootstrap errors (100 resamples of events). r: forward fit of
box(T)·e<sup>−rt</sup> ⊗ the measured template to X ±2 (T free at 243 V/cm, beyond the window otherwise).</p>
{table(['field', 'events', 'X ±2, 1.8–2.1 µs', 'X ±2, 2.4–2.7 µs', 'Y ±2', 'Y ±8, 2.4–2.7 µs', 'Y ±12', 'r [e-4/ns]', 'χ² / χ²(r=0), ndf 55'], t_raw)}
<p>X converges at ±2 (no resistive spread); Y converges by ±8, where it equals X. The fitted drift length at
243 V/cm is {F["raw700"]["T"]:.0f} ns, i.e. v ≈ {30e3 / F["raw700"]["T"]:.1f} µm/ns for a 30 mm gap.</p>

<h2>Template-free depth profile (run_63, 25.64°, Y ladder)</h2>
{table(['field', 'events', 'drift-time span', 'depth span', 'survival over span', 'loss per mm', 'loss per ns [e-4]'], t_lad)}
<p>The per-strip peak estimator reads ~30 % steeper than the time stacks (3.2–3.3 vs ~2.4 × 10<sup>−4</sup>/ns on
the same events): with ZS, deep strips carrying a single depth slice sit closer to threshold, and transverse
diffusion lowers per-strip peaks slightly with depth. Treat it as the depth-resolved, template-free
confirmation; the rate comes from the time stacks.</p>

<h2>Zero-suppressed arms</h2>
{table(['arm', 'events', 'R X, all', 'R X, mid 60 %', 'R Y, all', 'R Y, mid 60 %'], t_zs)}

<h2>Artefact budget (243/150/92 V/cm, X)</h2>
{table(['field', 'old cache: block CM, ±4 (NaN-aware)', 'this report: masked CM ±2', 'RAW X ±1', 'RAW → ZS 4σ', 'RAW → ZS 5σ'], t_art)}
<p>Block-median common mode biases R by &lt; 1 %. Zero suppression lowers X ±1 by 0.01–0.04 (4σ) and 0.03–0.06 (5σ).
run_63 flat (real ZS 4σ, all events, X ±1: {f2(r63flat_x1)}) matches RAW run_71
emulated at 4σ ({f2(E['raw700']['x']['zs4'])}) five hours later. Zero-filling the dropped RAW packets (as §14's
spike_test did) adds ~6 %.</p>

<h2>Discriminators</h2>
<h3>Local gain (resistive-layer charging would scale with it)</h3>
{table(['field', 'binning', 'gain range', 'dR<sub>X</sub> / d(gain / mean)'], t_disc)}
<p>Charging that produced the whole loss predicts a slope of ≈ −0.23. No bin set shows a negative slope; at 2σ
charging is at most ~20 % of the effect. Independently: the plateau current falls ×2.0 from 243 to 92 V/cm
while the loss per ns does not change.</p>
<h3>Beam rate and spill phase (space charge would grow with them)</h3>
{table(['field', 'split', 'low / early', 'mid', 'high / late'], t_rate)}
<h3>Event charge: a selection effect that attachment itself creates</h3>
{table(['', 'faint', 'mid', 'bright', 'all'], t_toy)}
<p>Toy: Poisson clusters with a 1/n² size tail, per-electron survival e<sup>−rt</sup>, per-event gain
scatter, the measured template, random sampling phase; split exactly like the data. Mid and bright agree with
r = 1.9 × 10<sup>−4</sup>/ns. The data's faint tercile first read 3.5σ above the toy: that was a selection artefact —
classifying by a sum over <i>present</i> samples put events that had lost readout packets on their high-signal part in
the faint class. Classified by the per-event mean over present samples, normalised by the local gain, the terciles agree
with the toy within 2σ (0.837 / 0.790 / 0.757 against 0.808 / 0.799 / 0.732; <code>faint_test.py</code>).</p>

{air_html}

<h2>The bench</h2>
<p>At the operating field the bench drift lasts only ~800 ns, too short for a single-field null: a forward fit of the
plateau cannot separate a small loss from the electronics undershoot or a field gradient. The det3 <b>drift scan</b>
(6-27, 100–1100 V = 35–382 V/cm, pulled from EOS, <code>bench_driftscan.py</code>) removes that: at 35 and 104 V/cm the
plateau stays flat to ±2 % over 1.2 µs of drift, and one composition with no air fits all six fields (F10). The same
fits exclude the field gradients the single-field fits had wandered to. The template-free bench ladder (F5) is flat to
±5 % from 6 to 20 mm in both views — consistent.</p>

<h2>The X view (for the model, not the gas)</h2>
<ul>
<li><b>No snapping to the resistive strips</b> (<code>snap_test.py</code>): X charge centroids show no 0.80 mm (resistive
pitch) periodicity on any chamber (Rayleigh power 0.2–1.7, ~1 for noise); Y shows the 0.78 mm readout non-linearity
strongly (12–40). Snapping is below ~10 %.</li>
<li><b>X's footprint has tails beyond the depth mixture</b> (<code>footprint_test.py</code>), present in the median event
(not only δ-electrons); a pseudo-Voigt (10–35 % Lorentzian, γ 0.5–0.9 mm) describes them. Implemented as opt-in wft keys
(<code>lor_frac_x</code>, <code>lor_gamma_x</code>), bit-identical when absent. It does <b>not</b> change the X angle
resolution (benched paired against production on all five chambers, within errors).</li>
<li>The remaining X deficit is det4's: its amplification stripes run across X, which a uniform-gain model cannot represent.
That is the next hypothesis, not tested here.</li>
</ul>

<h2>What this does not rule out</h2>
<ul>
<li><b>Which gas is responsible.</b> The observable fixes η·v; Magboltz says water alone attaches nothing and
O<sub>2</sub> does, but Magboltz's three-body O<sub>2</sub> attachment with H<sub>2</sub>O / isobutane as third body is
not well modelled, so the ppm scale is uncertain by a factor of a few. The beam/bench contrast and the time trend do
not depend on it.</li>
<li><b>Magboltz η is smoothed in field.</b> Even at 3e8 collisions its attachment estimate scatters ±10–15 % between
neighbouring fields; the fits use a quadratic in log E per mixture. With that, the 92 V/cm stacks fit (χ² 76 and
73 / 55); with the raw values they would not.</li>
<li><b>The shapers are effective.</b> Fitted per setup (they include the gas-dependent ion tail), not the literal DREAM
CR-RC<sup>n</sup>; the DREAM settings are the same on bench and beam.</li>
<li><b>det7</b> (bench) fits poorly at its single field (a rising plateau); its composition is the least certain.</li>
<li><b>A time-dependent loss inside the amplification stage</b> that is independent of gain, rate and position,
identical in both views and leaves no undershoot. None is known; it is the residual alternative.</li>
<li><b>The CO<sub>2</sub> period</b> is measured on ZS data only (no RAW, no ladder locally, drift field assumed 243 V/cm);
its O<sub>2</sub> (≈ 290 ppm) is rough.</li>
<li><b>The ~0.7 µs ripple</b> in the lower-drift-voltage runs is a trigger-locked coherent pickup (it is on signal-free
strips in RAW too); it oscillates around zero and cannot make a loss. Its source (likely the drift HV network ringing)
is not confirmed.</li>
</ul>
'''
    css = '''
:root{color-scheme:light dark;--bg:#fbfbfa;--surface:#fff;--line:#e3e2df;--ink:#14140f;--ink2:#55534c;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15151a;--surface:#1c1c22;--line:#33333c;--ink:#f2f2ef;--ink2:#b6b4ab;--acc:#3987e5}}
:root[data-theme="dark"]{--bg:#15151a;--surface:#1c1c22;--line:#33333c;--ink:#f2f2ef;--ink2:#b6b4ab;--acc:#3987e5}
*{box-sizing:border-box}
body{margin:0;padding:28px 16px 64px;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.55rem;line-height:1.25;margin:0 0 6px}
h2{font-size:1.12rem;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:20px 0 8px}
.sub{color:var(--ink2);margin:0 0 18px}
.verdict{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:12px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:16px 0}
.kpis div{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px;color:var(--ink2);font-size:.85rem}
.kpis span{display:block;font-size:1.5rem;color:var(--ink);font-weight:600}
.tw{overflow-x:auto}
table{border-collapse:collapse;font-size:.86rem;margin:8px 0;background:var(--surface)}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600}
td:first-child,th:first-child{text-align:left}
figure{margin:22px 0}
figure img{width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px}
figcaption{color:var(--ink2);font-size:.88rem;margin-top:6px}
code{font-size:.85em}
'''
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Beam attachment observable</title><style>{css}</style></head>'
            f'<body><div class="wrap">{body}</div></body></html>')
    out = os.path.join(a.out, 'report.html')
    open(out, 'w').write(page)
    print('wrote', out)
    if a.inline:
        import base64
        import re

        def emb(m):
            b = base64.b64encode(open(os.path.join(a.out, m.group(1)), 'rb').read()).decode()
            return f'src="data:image/png;base64,{b}"'
        desc = ('det4 at the SPS: every beam dataset loses late drift charge in both views, '
                'per unit time, with no readout undershoot; bench and discriminators.')
        inl = re.sub(r'src="(figures/[^"]+)"', emb, page).replace(
            '<title>', f'<meta name="description" content="{desc}"><title>', 1)
        open(a.inline, 'w').write(inl)
        print('wrote', a.inline, f'({len(inl) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
