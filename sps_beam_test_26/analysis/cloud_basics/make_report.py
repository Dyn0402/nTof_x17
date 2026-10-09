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
(attachment) fits all of it. The rate is what Magboltz gives for
{min(ppm):.0f}–{max(ppm):.0f} ppm O<sub>2</sub> in this gas (with the 1.7 % water); air is the obvious carrier and is being modelled next.</p>
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
r = 1.9 × 10<sup>−4</sup>/ns; the data's faint tercile loses less than the toy (≈ 3.5σ at 243 V/cm, also at 150, not at
92) — open.</p>

<h2>The bench</h2>
<p>The bench head-on drift lasts only ~800 ns, so a beam-like 2 × 10<sup>−4</sup>/ns would remove ~15 % over the
whole drift — comparable to the template/diffusion systematics of a forward fit on so short a box (the X fit
moves by ±2 × 10<sup>−4</sup>/ns between ±4 and ±7 strip sums; <code>bench_trig.py</code>). The bench therefore does
<b>not</b> provide a precise null on the head-on stack. The template-free ladder (F5, right) is flat to ±5 % from
6 to 20 mm in both views on det2/3/6/7 and det4 Y — no beam-like loss, at modest significance.</p>

<h2>What this does not rule out</h2>
<ul>
<li><b>Which gas is responsible.</b> The observable fixes η·v; Magboltz says water alone attaches nothing and
O<sub>2</sub> does, but Magboltz's three-body O<sub>2</sub> attachment with H<sub>2</sub>O / isobutane as third body is
not well modelled, so the ppm scale is uncertain by a factor of a few. Air (N<sub>2</sub>/O<sub>2</sub>) is being
simulated: condor 4410646.</li>
<li><b>Mild field-shape tension.</b> Magboltz O<sub>2</sub> expects ~17 % less loss per ns at 92 than at 243 V/cm;
the data have it equal or slightly larger.</li>
<li><b>A time-dependent loss inside the amplification stage</b> that is independent of gain, rate and position,
identical in both views and leaves no undershoot. None is known; it is the residual alternative.</li>
<li><b>Whether the O<sub>2</sub> level drifted</b> over the campaign: the rotated run_63 block reads a little more
loss than the flat blocks; ZS selection systematics (±0.05) cover the difference.</li>
<li><b>The CO<sub>2</sub> period</b> is measured on ZS data only (no RAW, no ladder locally); its loss is larger but
its absolute value carries the ZS caveats above.</li>
<li><b>The faint-tercile excess</b> and the <b>ZS ~0.7 µs ripple</b> in the rotated mount are unexplained.</li>
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


if __name__ == '__main__':
    main()
